"""Cross-sectional forecasting: score every name every day, evaluate the forecasts
before any portfolio touches them, then build a portfolio from them and attribute.

Why this exists. In the trading replay each agent makes one decision a day, so a
60-day window yields 4-17 trades per agent — too few to measure skill. Here a
forecaster returns, for each of the 50 names, the probability that the name beats
the equal-weight universe over the next `HORIZON` trading days. Fifty forecasts a
day, ~3,000 per window, scored directly:

    rank IC      Spearman(p, realised excess return) per day; mean, Newey-West t-stat
    hit rate     share of (p > 0.5) == (name beat the universe)
    Brier        mean (p - beat)^2; 0.25 is the score of "0.5 for everything"
    calibration  realised beat rate per stated-probability bucket
    spread       realised excess return, top quintile of p minus bottom quintile

A forecaster is anything with `.name` and `.forecast(payload) -> dict[symbol, p]`.
`LLMForecaster` asks a model; `MomentumForecaster`, `RandomForecaster` and
`ConstantForecaster` are the controls. `forecast_portfolio()` turns the forecasts
into an overlapping-cohort long/short (or long-only) book with fees, so signal
quality and implementation can be separated.

Point-in-time: forecasts are built from `md.asof(t)` payloads (the same guard as the
replay); targets are computed from the full data and joined afterwards.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .agents.llm import Backend, ResponseCache, DEFAULT_CACHE_DIR, _strip_fences
from .market import MarketData
from .prompt import build_base, universe_rows
from .universe import BENCHMARK, TICKERS

log = logging.getLogger(__name__)

HORIZON = 10          # trading days
FEE_RATE = 0.001      # per side, as in the engine


# ---- targets ------------------------------------------------------------------
def forward_targets(md: MarketData, dates: pd.DatetimeIndex, horizon: int = HORIZON,
                    tickers: list[str] = TICKERS) -> dict[str, pd.DataFrame]:
    """For each decision date t: return close_t -> close_{t+horizon} per name, its excess
    over the cross-sectional mean (the equal-weight universe), and beat = excess > 0.
    Dates whose horizon runs past the data are NaN (and are excluded from every metric)."""
    close = md.close[tickers]
    idx = close.index
    pos = idx.searchsorted(dates)
    fwd = pd.DataFrame(np.nan, index=dates, columns=tickers)
    for d, i in zip(dates, pos):
        if i + horizon < len(idx):
            fwd.loc[d] = (close.iloc[i + horizon] / close.iloc[i] - 1.0).to_numpy()
    excess = fwd.sub(fwd.mean(axis=1), axis=0)
    beat = (excess > 0).astype(float).where(excess.notna())
    return {"fwd_return": fwd, "excess": excess, "beat": beat}


# ---- forecasters ----------------------------------------------------------------
class Forecaster:
    name = "forecaster"

    def forecast(self, payload: dict) -> dict[str, float]:  # pragma: no cover - interface
        raise NotImplementedError

    def reset(self) -> None:
        pass


def _rank_to_prob(values: dict[str, float], lo: float = 0.2, hi: float = 0.8) -> dict[str, float]:
    """Cross-sectional rank mapped linearly onto [lo, hi] — a control that carries
    ordering information but claims only moderate confidence."""
    s = pd.Series(values, dtype=float).dropna()
    if s.empty:
        return {}
    r = (s.rank() - 1) / max(len(s) - 1, 1)
    return (lo + (hi - lo) * r).round(3).to_dict()


class MomentumForecaster(Forecaster):
    """p from the cross-sectional rank of a universe-table column (default ret_10d)."""
    def __init__(self, column: str = "ret_10d", reverse: bool = False):
        self.column, self.reverse = column, reverse
        self.name = f"control_{'reversal' if reverse else 'momentum'}_{column}"

    def forecast(self, payload: dict) -> dict[str, float]:
        vals = {r["symbol"]: float(r[self.column]) for r in universe_rows(payload)
                if r.get(self.column) is not None and r[self.column] == r[self.column]}
        if self.reverse:
            vals = {k: -v for k, v in vals.items()}
        return _rank_to_prob(vals)


class RandomForecaster(Forecaster):
    def __init__(self, seed: int = 0):
        self.seed, self.rng = seed, np.random.default_rng(seed)
        self.name = f"control_random_{seed}"

    def reset(self) -> None:
        self.rng = np.random.default_rng(self.seed)

    def forecast(self, payload: dict) -> dict[str, float]:
        syms = [r["symbol"] for r in universe_rows(payload)]
        return {s: float(v) for s, v in zip(syms, self.rng.uniform(0.0, 1.0, len(syms)).round(3))}


class ConstantForecaster(Forecaster):
    """0.5 for everything: no ordering, perfect calibration, Brier 0.25. The floor."""
    name = "control_constant_0.5"

    def forecast(self, payload: dict) -> dict[str, float]:
        return {r["symbol"]: 0.5 for r in universe_rows(payload)}


# ---- the LLM forecaster -------------------------------------------------------------
FORECAST_MANDATE = """You are a quantitative analyst covering 50 large US stocks. Every trading day, after the close, you are shown one row per name (price, recent returns, volatility, RSI, liquidity, sector, and derived features) and the S&P 500's recent candles.

TASK. For EVERY name in the universe table, state the probability that its total return over the next {horizon} trading days will be HIGHER than the average return of all 50 names over the same {horizon} days. This is a relative forecast: by construction about half the names beat the average, so your probabilities should be spread on both sides of 0.5 and average roughly 0.5 across the table. Use values away from 0.5 only where the data gives you a reason; 0.5 means "no view".

You are scored on three things: the ordering of your probabilities against what actually happens (rank correlation), whether names you put above 0.5 beat the average more often than not, and calibration — of the names you give 0.7, about 70% should beat the average. Overconfidence is penalised.

Read the data carefully: returns and distances in the universe table are decimals (0.052 = +5.2%); the features block is in percent (5.2 = +5.2%). Large caps mostly move together, so think about which names are likely to lead or lag the group, not about the market's direction."""

FORECAST_CONTRACT = ('Respond with a single JSON object and nothing else: {"reasoning": string (<= 80 words, what drove your ordering), '
                     '"forecasts": [{"symbol": string, "p_beat": number between 0 and 1}, ...]}. '
                     'Include every symbol from the universe table exactly once. No other keys, no prose outside the JSON.')


def forecast_schema(tickers: list[str] = TICKERS) -> dict:
    """JSON schema for grammar-constrained backends: symbols restricted to the universe."""
    return {"type": "object",
            "properties": {"reasoning": {"type": "string"},
                           "forecasts": {"type": "array", "minItems": 1,
                                         "items": {"type": "object",
                                                   "properties": {"symbol": {"type": "string", "enum": list(tickers)},
                                                                  "p_beat": {"type": "number", "minimum": 0.0, "maximum": 1.0}},
                                                   "required": ["symbol", "p_beat"]}}},
            "required": ["reasoning", "forecasts"]}


@dataclass
class ForecastCall:
    decision_date: str
    attempt: int
    ok: bool
    n_returned: int
    n_valid: int
    error: str | None
    prompt_tokens: int | None
    completion_tokens: int | None
    latency_s: float
    cached: bool
    text: str = ""
    reasoning: str = ""


def render_forecast_prompt(payload: dict, horizon: int = HORIZON, system_prompt: str | None = None) -> tuple[str, str]:
    """(system, user) for a forecast call. The payload is a `build_base(..., candidates="universe")`
    dict; only the universe table, the benchmark candles and the format notes are shown."""
    system = (system_prompt or FORECAST_MANDATE).format(horizon=horizon) + "\n\n" + FORECAST_CONTRACT
    body = {"decision_date": payload["decision_date"], "universe": payload["universe"],
            "candle_format": payload["candle_format"], "benchmark": payload["benchmark"]}
    user = f"DATA (as of {payload['decision_date']} close):\n" + json.dumps(body, separators=(",", ":"), default=str)
    return system, user


def parse_forecasts(text: str, tickers: list[str]) -> tuple[dict[str, float], int, str, str | None]:
    """-> (forecasts for known tickers, n items returned, reasoning, error). Probabilities given
    in percent (e.g. 65) are read as 0.65; values outside [0, 100] and unknown symbols are dropped."""
    try:
        obj = json.loads(_strip_fences(text))
    except json.JSONDecodeError as e:
        return {}, 0, "", f"not JSON: {e}"
    if not isinstance(obj, dict) or not isinstance(obj.get("forecasts"), list):
        return {}, 0, "", "no 'forecasts' list"
    out: dict[str, float] = {}
    known = set(tickers)
    for it in obj["forecasts"]:
        if not isinstance(it, dict):
            continue
        s, p = it.get("symbol"), it.get("p_beat")
        if s not in known or not isinstance(p, (int, float)) or p != p:
            continue
        p = float(p)
        if 1.0 < p <= 100.0:                           # "65" for 0.65
            p /= 100.0
        if not 0.0 <= p <= 1.0:
            continue
        out.setdefault(s, p)                           # first occurrence wins
    err = None if out else "no usable forecasts"
    return out, len(obj["forecasts"]), str(obj.get("reasoning", "")), err


class LLMForecaster(Forecaster):
    """Asks a model for p_beat on every name. Same backend, cache and reproducibility
    contract as `LLMAgent`: temperature 0, fixed seed, every call hashed and cached."""

    def __init__(self, name: str, backend: Backend, model: str, horizon: int = HORIZON, temperature: float = 0.0,
                 seed: int | None = 7, system_prompt: str | None = None, use_schema: bool = True,
                 cache_dir: Path | None = None, max_tokens: int = 2500, tickers: list[str] = TICKERS,
                 min_coverage: float = 0.5, price_per_mtok: tuple[float, float] = (0.0, 0.0)):
        self.name, self.backend, self.model, self.horizon = name, backend, model, horizon
        self.temperature, self.seed, self.system_prompt, self.max_tokens = temperature, seed, system_prompt, max_tokens
        self.tickers, self.min_coverage = list(tickers), min_coverage
        self.price_in, self.price_out = price_per_mtok
        self.schema = forecast_schema(self.tickers) if use_schema else None
        self.cache = ResponseCache(Path(cache_dir or DEFAULT_CACHE_DIR) / name)
        self.records: list[ForecastCall] = []

    def _call(self, system: str, user: str, decision_date: str, attempt: int) -> tuple[dict[str, float], str | None]:
        key = self.cache.key(backend=self.backend.name, model=self.model, temperature=self.temperature, seed=self.seed,
                             system=system, user=user, schema=self.schema, max_tokens=self.max_tokens,
                             **({"backend_opts": self.backend.signature} if getattr(self.backend, "signature", None) else {}))
        resp = self.cache.get(key)
        if resp is None:
            resp = self.backend.complete(system, user, self.model, self.temperature, self.schema, self.seed, self.max_tokens)
            self.cache.put(key, system, user, self.model, resp, {"forecaster": self.name, "decision_date": decision_date, "attempt": attempt})
        fc, n_ret, reasoning, err = parse_forecasts(resp.text, self.tickers)
        if err is None and len(fc) < self.min_coverage * len(self.tickers):
            err = f"only {len(fc)} of {len(self.tickers)} names forecast"
        self.records.append(ForecastCall(decision_date, attempt, err is None, n_ret, len(fc), err, resp.prompt_tokens,
                                         resp.completion_tokens, resp.latency_s, resp.cached, resp.text, reasoning))
        return fc, err

    def forecast(self, payload: dict) -> dict[str, float]:
        system, user = render_forecast_prompt(payload, self.horizon, self.system_prompt)
        d = payload["decision_date"]
        fc, err = self._call(system, user, d, attempt=1)
        if err is not None:
            repair = (user + "\n\nYour previous output was invalid: " + err[:500]
                      + "\n\nReturn the corrected JSON object only, with one entry per symbol in the universe table.")
            fc2, err2 = self._call(system, repair, d, attempt=2)
            if len(fc2) > len(fc):
                fc, err = fc2, err2
        if not fc:
            log.warning("%s %s: no usable forecasts → all NaN", self.name, d)
        return fc

    def stats(self) -> dict:
        r = self.records
        if not r:
            return {}
        live = [x for x in r if not x.cached]
        pin = sum(x.prompt_tokens or 0 for x in r); pout = sum(x.completion_tokens or 0 for x in r)
        return {"calls": len(r), "cached": sum(x.cached for x in r), "repairs": sum(x.attempt == 2 for x in r),
                "invalid_final": sum(1 for x in r if x.attempt == 2 and not x.ok),
                "avg_names_returned": float(np.mean([x.n_valid for x in r])),
                "prompt_tokens": pin, "completion_tokens": pout,
                "avg_prompt_tokens": pin / len(r), "avg_completion_tokens": pout / len(r),
                "avg_latency_s": (sum(x.latency_s for x in live) / len(live)) if live else 0.0,
                "cost_usd": pin / 1e6 * self.price_in + pout / 1e6 * self.price_out}


# ---- running ----------------------------------------------------------------------
def run_forecasts(md: MarketData, forecasters: list[Forecaster], dates: pd.DatetimeIndex, tickers: list[str] = TICKERS,
                  out_dir: Path | None = None, on_date=None) -> dict[str, pd.DataFrame]:
    """Every forecaster sees the same payload per date (built once). Returns name -> DataFrame
    (date x symbol) of probabilities, NaN where a forecaster gave none. Saved as parquet per
    forecaster when out_dir is given. on_date(i, n, date, elapsed) is called after each date."""
    frames = {f.name: pd.DataFrame(np.nan, index=dates, columns=tickers) for f in forecasters}
    for f in forecasters:
        f.reset()
    t0 = time.time()
    for i, t in enumerate(dates):
        base = build_base(md.asof(t), t, tickers=tickers, candidates="universe")
        for f in forecasters:
            try:
                fc = f.forecast(base)
            except Exception as e:
                log.warning("%s on %s failed: %s", f.name, t.date(), e)
                fc = {}
            for s, p in fc.items():
                if s in frames[f.name].columns:
                    frames[f.name].at[t, s] = p
        if on_date is not None:
            on_date(i + 1, len(dates), t, time.time() - t0)
    if out_dir is not None:
        out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        for name, df in frames.items():
            df.to_parquet(out_dir / f"{name}.parquet")
    return frames


def load_forecasts(out_dir: Path) -> dict[str, pd.DataFrame]:
    return {p.stem: pd.read_parquet(p) for p in sorted(Path(out_dir).glob("*.parquet"))}


# ---- evaluation ---------------------------------------------------------------------
def newey_west_tstat(x: pd.Series | np.ndarray, lag: int) -> float:
    """t-stat of the mean with a Newey-West (Bartlett) variance, for series such as daily
    ICs on overlapping horizons; lag = horizon - 1 is the usual choice."""
    x = np.asarray(pd.Series(x).dropna(), dtype=float)
    n = len(x)
    if n < 3:
        return np.nan
    e = x - x.mean()
    var = float(e @ e) / n
    for l in range(1, min(lag, n - 1) + 1):
        var += 2.0 * (1.0 - l / (lag + 1.0)) * float(e[l:] @ e[:-l]) / n
    se = np.sqrt(max(var, 1e-18) / n)
    return float(x.mean() / se)


def daily_ic(p: pd.DataFrame, excess: pd.DataFrame, min_names: int = 10) -> pd.Series:
    """Spearman rank correlation between forecast and realised excess return, per date."""
    out = {}
    for d in p.index:
        if d not in excess.index:
            continue
        a, b = p.loc[d], excess.loc[d]
        m = a.notna() & b.notna()
        if m.sum() < min_names or a[m].nunique() < 2:
            out[d] = np.nan
        else:
            out[d] = a[m].rank().corr(b[m].rank())
    return pd.Series(out, name="ic")


def quintile_spread(p: pd.DataFrame, excess: pd.DataFrame, q: float = 0.2) -> pd.Series:
    """Per date: mean realised excess return of the top q of names by p minus the bottom q.
    NaN on dates where p has no ordering (all equal)."""
    out = {}
    for d in p.index:
        if d not in excess.index:
            continue
        a, b = p.loc[d], excess.loc[d]
        m = a.notna() & b.notna()
        a, b = a[m], b[m]
        k = int(round(q * len(a)))
        if k < 1 or a.nunique() < 2:
            out[d] = np.nan; continue
        order = a.sort_values(kind="mergesort")
        out[d] = float(b[order.index[-k:]].mean() - b[order.index[:k]].mean())
    return pd.Series(out, name="spread")


def calibration_table(p: pd.DataFrame, beat: pd.DataFrame, edges=(0.0, 0.3, 0.45, 0.55, 0.7, 1.0)) -> pd.DataFrame:
    a, b = p.stack(future_stack=True), beat.reindex_like(p).stack(future_stack=True)
    m = a.notna() & b.notna()
    a, b = a[m], b[m]
    bucket = pd.cut(a, list(edges), include_lowest=True)
    t = pd.DataFrame({"n": b.groupby(bucket, observed=True).size(),
                      "stated": a.groupby(bucket, observed=True).mean(),
                      "realised": b.groupby(bucket, observed=True).mean()})
    t.index.name = "p_beat bucket"
    return t


def evaluate(p: pd.DataFrame, targets: dict[str, pd.DataFrame], horizon: int = HORIZON) -> dict:
    """All forecast metrics for one forecaster. Only cells with both a forecast and a
    realised target count."""
    excess, beat = targets["excess"].reindex_like(p), targets["beat"].reindex_like(p)
    a, y = p.stack(future_stack=True), beat.stack(future_stack=True)
    m = a.notna() & y.notna()
    a, y = a[m], y[m]
    ic = daily_ic(p, excess)
    spread = quintile_spread(p, excess)
    decided = a[(a - 0.5).abs() > 1e-9]
    hits = ((decided > 0.5).astype(float) == y[decided.index]).mean() if len(decided) else np.nan
    brier = float(((a - y) ** 2).mean()) if len(a) else np.nan
    n_dates = int(p.notna().any(axis=1).sum())
    scored_dates = int(excess.notna().any(axis=1).sum())
    return {
        "n_forecasts": int(len(a)), "n_dates_scored": int(ic.notna().sum()), "n_dates_forecast": n_dates,
        "coverage": float(p.notna().sum().sum() / (p.shape[0] * p.shape[1])) if p.size else np.nan,
        "ic_mean": float(ic.mean()), "ic_std": float(ic.std()), "ic_tstat_nw": newey_west_tstat(ic, horizon - 1),
        "ic_share_positive": float((ic.dropna() > 0).mean()) if ic.notna().any() else np.nan,
        "hit_rate": float(hits), "n_decided": int(len(decided)),
        "brier": brier, "brier_skill": (1.0 - brier / 0.25) if brier == brier else np.nan,
        "spread_mean": float(spread.mean()), "spread_tstat_nw": newey_west_tstat(spread, horizon - 1),
        "p_mean": float(a.mean()) if len(a) else np.nan, "p_std": float(a.std()) if len(a) else np.nan,
        "share_at_0.5": float(((a - 0.5).abs() <= 1e-9).mean()) if len(a) else np.nan,
        "share_confident": float(((a - 0.5).abs() > 0.2).mean()) if len(a) else np.nan,
        "_ic": ic, "_spread": spread, "_scored_dates_available": scored_dates,
    }


def scorecard(forecasts: dict[str, pd.DataFrame], targets: dict[str, pd.DataFrame], horizon: int = HORIZON) -> pd.DataFrame:
    rows = []
    for name, p in forecasts.items():
        m = evaluate(p, targets, horizon)
        rows.append({"forecaster": name, **{k: v for k, v in m.items() if not k.startswith("_")}})
    return pd.DataFrame(rows).set_index("forecaster")


# ---- portfolio from forecasts ----------------------------------------------------------
def forecast_portfolio(p: pd.DataFrame, md: MarketData, horizon: int = HORIZON, k: int = 5, long_only: bool = False,
                       fee_rate: float = FEE_RATE, tickers: list[str] = TICKERS, initial_capital: float = 10_000.0) -> dict:
    """Overlapping-cohort book: on each forecast date, the top-k names by p go long (and the
    bottom-k short unless long_only), equal-weighted, entered at the NEXT close and held
    `horizon` days. `horizon` cohorts are live at once, each with 1/horizon of the capital.
    Entering one close after the decision is deliberately conservative (the engine fills at
    the next open). Fees are charged per side on every entry and exit. Returns equity (net and
    gross), daily returns, cohort weights and turnover."""
    close = md.close[tickers]
    rets = close.pct_change()
    idx = close.index
    dates = [d for d in p.index if p.loc[d].notna().sum() >= 2 * k]
    daily_gross = pd.Series(0.0, index=idx)
    daily_cost = pd.Series(0.0, index=idx)
    active = pd.Series(0, index=idx)
    weights = {}
    for d in dates:
        s = p.loc[d].dropna()
        if s.nunique() < 2:
            continue
        order = s.sort_values(kind="mergesort")
        w = pd.Series(0.0, index=tickers)
        w[order.index[-k:]] = 1.0 / k
        if not long_only:
            w[order.index[:k]] = -1.0 / k
        weights[d] = w
        i = idx.searchsorted(d)
        entry_i, exit_i = i + 1, min(i + 1 + horizon, len(idx) - 1)       # enter at close of entry_i, exit at close of exit_i
        if entry_i >= len(idx) - 1:
            continue
        span = idx[entry_i + 1: exit_i + 1]
        cohort_r = rets.loc[span, tickers].fillna(0.0) @ w
        daily_gross.loc[span] += cohort_r
        active.loc[span] += 1
        gross_notional = float(w.abs().sum())
        daily_cost.loc[idx[entry_i]] += fee_rate * gross_notional
        daily_cost.loc[idx[exit_i]] += fee_rate * gross_notional
    n_live = active.replace(0, np.nan)
    r_gross = (daily_gross / horizon).fillna(0.0)                    # each cohort runs 1/horizon of the capital
    r_net = r_gross - daily_cost / horizon
    live = active > 0
    first = idx[max(idx.searchsorted(dates[0]) if dates else 0, 0)]
    r_gross, r_net = r_gross.loc[first:], r_net.loc[first:]
    last = idx[min(idx.searchsorted(dates[-1]) + 1 + horizon, len(idx) - 1)] if dates else first
    r_gross, r_net = r_gross.loc[:last], r_net.loc[:last]
    eq_net = (1.0 + r_net).cumprod() * initial_capital
    eq_gross = (1.0 + r_gross).cumprod() * initial_capital
    turnover = float(sum(w.abs().sum() for w in weights.values()) / max(len(weights), 1))
    return {"equity": eq_net.rename("net"), "equity_gross": eq_gross.rename("gross"), "returns": r_net, "returns_gross": r_gross,
            "weights": pd.DataFrame(weights).T, "n_cohorts": len(weights), "avg_live_cohorts": float(n_live.loc[first:last].mean()),
            "gross_notional_per_cohort": (1.0 if long_only else 2.0), "fees_total_pct": float(daily_cost.loc[first:last].sum() / horizon),
            "turnover_per_cohort": turnover}


def attribution(forecasts: dict[str, pd.DataFrame], targets: dict[str, pd.DataFrame], md: MarketData, horizon: int = HORIZON,
                k: int = 5, long_only: bool = False, benchmark: pd.Series | None = None) -> pd.DataFrame:
    """Signal vs implementation, one row per forecaster: forecast quality (IC, hit rate,
    Brier, quintile spread) next to what a mechanical portfolio built from the same
    forecasts made, gross and net of fees."""
    from .metrics import equity_metrics
    rows = []
    for name, p in forecasts.items():
        m = evaluate(p, targets, horizon)
        pf = forecast_portfolio(p, md, horizon, k, long_only)
        em_net = equity_metrics(pf["equity"], benchmark)
        em_gross = equity_metrics(pf["equity_gross"])
        rows.append({"forecaster": name, "ic_mean": m["ic_mean"], "ic_tstat_nw": m["ic_tstat_nw"], "hit_rate": m["hit_rate"],
                     "brier": m["brier"], "spread_mean": m["spread_mean"], "spread_tstat_nw": m["spread_tstat_nw"],
                     "portfolio_gross": em_gross.get("total_return", np.nan), "portfolio_net": em_net.get("total_return", np.nan),
                     "sharpe_net": em_net.get("sharpe", np.nan), "max_drawdown_net": em_net.get("max_drawdown", np.nan),
                     "fees_pct": pf["fees_total_pct"], "n_cohorts": pf["n_cohorts"],
                     **({"excess_vs_benchmark": em_net.get("excess_vs_benchmark", np.nan)} if benchmark is not None else {})})
    return pd.DataFrame(rows).set_index("forecaster")
