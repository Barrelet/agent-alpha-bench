"""Leaderboard metrics from an equity curve and a trade log."""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def equal_weight_index(close: pd.DataFrame, start=None, base: float = 10_000.0) -> pd.Series:
    """Equal-weight, daily-rebalanced index of every column in `close` (the
    universe benchmark). Not run through the engine, so it is not bound by the
    position cap — it answers "what did the universe itself do?"."""
    r = close.pct_change().mean(axis=1)  # cross-sectional mean of daily returns = EW, daily rebalanced
    if start is not None:
        r = r.loc[pd.Timestamp(start):]
    idx = (1.0 + r.fillna(0.0)).cumprod() * base
    idx.iloc[0] = base
    return idx.rename("equal_weight_universe")


def equity_metrics(eq: pd.Series, benchmark: pd.Series | dict[str, pd.Series] | None = None) -> dict:
    eq = eq.dropna()
    if len(eq) < 2:
        return {}
    r = eq.pct_change().dropna()
    total = eq.iloc[-1] / eq.iloc[0] - 1.0
    years = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9)
    dd = eq / eq.cummax() - 1.0
    out = {
        "total_return": total,
        "cagr": (1.0 + total) ** (1.0 / years) - 1.0 if total > -1 else -1.0,
        "sharpe": float(r.mean() / r.std() * np.sqrt(TRADING_DAYS)) if r.std() > 0 else 0.0,
        "max_drawdown": float(dd.min()),
        "volatility_ann": float(r.std() * np.sqrt(TRADING_DAYS)),
        "days": int(len(eq)),
    }
    benchmarks = benchmark if isinstance(benchmark, dict) else ({"benchmark": benchmark} if benchmark is not None else {})
    for name, b in benchmarks.items():
        b = b.reindex(eq.index).ffill()
        out[f"excess_vs_{name}"] = total - (b.iloc[-1] / b.iloc[0] - 1.0)
    return out


def trade_metrics(trades: pd.DataFrame) -> dict:
    if trades is None or trades.empty:
        return {"n_trades": 0, "n_open_at_end": 0, "win_rate": np.nan, "avg_holding_days": np.nan, "invalidation_rate": np.nan}
    wins = trades["pnl"] > 0
    return {
        "n_trades": int(len(trades)),                       # includes end_of_window forced closes
        "n_open_at_end": int((trades["reason"] == "end_of_window").sum()),
        "win_rate": float(wins.mean()),
        "avg_holding_days": float(trades["holding_days"].mean()),
        "avg_pnl_pct": float(trades["pnl_pct"].mean()),
        "invalidation_rate": float((trades["reason"] == "invalidation").mean()),
        "closed_by_choice_rate": float((trades["reason"] == "close").mean()),
        "total_fees": float(trades["fees"].sum()),
    }


def concentration(fills: pd.DataFrame, trades: pd.DataFrame, equity: pd.Series) -> dict:
    """How concentrated was the book? Position weights are taken right after each fill
    (running signed quantity per symbol × fill price, over that day's equity), so adds
    that pyramid one name into most of the account show up as max_position_weight.
    best_trade_share = largest trade P&L over total P&L (only meaningful when total > 0)."""
    out = {"max_position_weight": np.nan, "max_gross_exposure": np.nan, "n_adds": 0, "best_trade_share": np.nan, "best_trade_pnl": np.nan}
    if fills is None or fills.empty:
        return out
    f = fills.sort_values("date").copy()
    f["signed_qty"] = f["side"] * f["qty"]
    eq = equity.reindex(pd.to_datetime(f["date"])).ffill().to_numpy()
    running: dict[str, float] = {}
    last_px: dict[str, float] = {}
    weights, gross = [], []
    for (sym, sq, px), e in zip(f[["symbol", "signed_qty", "price"]].itertuples(index=False), eq):
        running[sym] = running.get(sym, 0.0) + sq
        last_px[sym] = px
        weights.append(abs(running[sym]) * px / e if e else np.nan)
        gross.append(sum(abs(q) * last_px[s_] for s_, q in running.items()) / e if e else np.nan)
    out["max_position_weight"] = float(np.nanmax(weights))
    out["max_gross_exposure"] = float(np.nanmax(gross))
    out["n_adds"] = int((f["action"] == "add").sum())
    if trades is not None and not trades.empty:
        total = float(trades["pnl"].sum())
        out["best_trade_pnl"] = float(trades["pnl"].max())
        out["best_trade_share"] = float(trades["pnl"].max() / total) if total > 0 else np.nan
    return out


def calibration(trades: pd.DataFrame, bins: int = 4) -> tuple[pd.DataFrame, float]:
    """Reliability table and Brier score of stated confidence vs realised win.
    Confidence lives in [0.80, 1.0] by construction, so bins cover that range."""
    if trades is None or trades.empty or trades["confidence"].isna().all():
        return pd.DataFrame(), np.nan
    t = trades.dropna(subset=["confidence"]).copy()
    t["win"] = (t["pnl"] > 0).astype(float)
    edges = np.linspace(0.80, 1.0 + 1e-9, bins + 1)
    t["bucket"] = pd.cut(t["confidence"], edges, include_lowest=True)
    table = t.groupby("bucket", observed=True).agg(n=("win", "size"), stated=("confidence", "mean"), realised=("win", "mean"))
    brier = float(((t["confidence"] - t["win"]) ** 2).mean())
    return table, brier


def leaderboard(results: dict[str, dict], benchmark: pd.Series | dict[str, pd.Series] | None = None) -> pd.DataFrame:
    """results: name -> {'equity': Series, 'trades': DataFrame}.
    benchmark: one Series or a dict name -> Series (adds an excess_vs_<name> column each)."""
    rows = []
    for name, res in results.items():
        m = {"agent": name, **equity_metrics(res["equity"], benchmark), **trade_metrics(res["trades"])}
        _, m["brier"] = calibration(res["trades"])
        rows.append(m)
    lb = pd.DataFrame(rows).sort_values("total_return", ascending=False).reset_index(drop=True)
    lb.index = lb.index + 1
    lb.index.name = "rank"
    return lb
