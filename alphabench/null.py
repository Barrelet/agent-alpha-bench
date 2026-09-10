"""Null distribution: what does luck alone produce under the same rules?

Runs many seeded RandomAgents over exactly the same window and rules as the
configurations, and places every configuration, control and benchmark inside
that distribution. A percentile of 95 means "only 5% of random traders did
better"; anything between ~20 and ~80 is indistinguishable from luck.

Results are cached to parquet per (window, seeds, side), so the 1,000-seed run
is paid once.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from .agents.rules import RandomAgent
from .market import MarketData
from .metrics import concentration, equity_metrics, trade_metrics
from .replay import run_replay

COLS = ["seed", "long_only", "from_universe", "pyramid", "total_return", "sharpe", "max_drawdown", "n_trades", "win_rate", "max_position_weight"]

#: the null variants, in display order: (long_only, from_universe, pyramid) -> (column suffix, label)
VARIANTS = {
    (False, False, False): ("ls", "long/short"),
    (True, False, False): ("long", "long-only"),
    (False, True, False): ("ls_univ", "long/short, whole universe"),
    (True, True, False): ("long_univ", "long-only, whole universe"),
    (False, True, True): ("ls_univ_pyr", "long/short, whole universe, pyramiding"),
    (True, True, True): ("long_univ_pyr", "long-only, whole universe, pyramiding"),
    (False, False, True): ("ls_pyr", "long/short, pyramiding"),
    (True, False, True): ("long_pyr", "long-only, pyramiding"),
}


def _cache_path(cache_dir: Path, start, end, n_seeds: int, long_only: bool, from_universe: bool = False, pyramid: bool = False,
                max_position_weight: float | None = None) -> Path:
    tag = ("long" if long_only else "ls") + ("_univ" if from_universe else "") + ("_pyr" if pyramid else "") \
          + ("" if max_position_weight is None else f"_cap{int(round(max_position_weight * 100))}")
    return Path(cache_dir) / f"random_{pd.Timestamp(start).date()}_{pd.Timestamp(end).date()}_n{n_seeds}_{tag}.parquet"


def run_null(md: MarketData, start, end, n_seeds: int = 500, long_only: bool = False, from_universe: bool = False, pyramid: bool = False,
             cache_dir: Path | None = None, batch: int = 100, on_batch=None, payload_kwargs: dict | None = None,
             max_position_weight: float | None = None, **agent_kw) -> pd.DataFrame:
    """One row per seed: return, Sharpe, drawdown, trades, win rate — end-of-window
    liquidation included, like every other run. Agents step in lockstep in batches
    so the per-date work is shared; `on_batch(done, total, seconds)` reports progress.
    from_universe=True, or payload_kwargs={'candidates': 'universe'}, makes the seeds pick from
    all names; both are cached under the same `_univ` tag since they draw from the same pool.
    pyramid=True lets the seeds add to winners (RandomAgent p_add=0.5) — the null for an unconstrained prompt."""
    payload_kwargs = payload_kwargs or {}
    from_universe = from_universe or payload_kwargs.get("candidates") == "universe"
    if pyramid:
        agent_kw.setdefault("p_add", 0.5)
    if cache_dir is not None:
        p = _cache_path(cache_dir, start, end, n_seeds, long_only, from_universe, pyramid, max_position_weight)
        if p.exists():
            df = pd.read_parquet(p)
            for col, val in [("from_universe", from_universe), ("pyramid", pyramid), ("max_position_weight", np.nan)]:
                if col not in df.columns:              # caches written before these columns existed
                    df[col] = val
            return df[COLS]
    rows, t0 = [], time.time()
    for b0 in range(0, n_seeds, batch):
        seeds = range(b0, min(b0 + batch, n_seeds))
        agents = [RandomAgent(seed=s, long_only=long_only, from_universe=from_universe, **agent_kw) for s in seeds]
        res = run_replay(md, agents, start, end, progress=False, payload_kwargs=payload_kwargs, max_position_weight=max_position_weight)
        for s, a in zip(seeds, agents):
            r = res[a.name]
            em, tm, cc = equity_metrics(r["equity"]), trade_metrics(r["trades"]), concentration(r["fills"], r["trades"], r["equity"])
            rows.append({"seed": s, "long_only": long_only, "from_universe": from_universe, "pyramid": pyramid, "total_return": em["total_return"],
                         "sharpe": em["sharpe"], "max_drawdown": em["max_drawdown"], "n_trades": tm["n_trades"], "win_rate": tm["win_rate"],
                         "max_position_weight": cc["max_position_weight"]})
        if on_batch is not None:
            on_batch(len(rows), n_seeds, time.time() - t0)
    df = pd.DataFrame(rows, columns=COLS)
    if cache_dir is not None:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        df.to_parquet(p, index=False)
    return df


def percentile(null_values: pd.Series, value: float) -> float:
    """Share of null runs at or below `value`, in percent (a 'better than X% of random traders')."""
    v = null_values.dropna().to_numpy()
    if len(v) == 0 or value != value:
        return np.nan
    return float(100.0 * (v <= value).mean())


def null_table(observed: pd.DataFrame, null: pd.DataFrame, metrics=("total_return", "sharpe")) -> pd.DataFrame:
    """observed: index = name, columns include the metrics. Adds, per metric, the
    percentile within each null present: long/short and long-only, from the screened
    list (`_ls`, `_long`) and — when those seeds were run — from the whole universe
    (`_ls_univ`, `_long_univ`)."""
    out = observed[list(metrics)].copy()
    for m in metrics:
        for key, (tag, _) in VARIANTS.items():
            sub = _variant(null, key)[m]
            if sub.empty:
                continue
            out[f"{m}_pct_{tag}"] = [percentile(sub, x) for x in observed[m]]
    return out


def _variant(null: pd.DataFrame, key: tuple[bool, bool, bool]) -> pd.DataFrame:
    lo, fu, py = key
    univ = null["from_universe"] if "from_universe" in null.columns else pd.Series(False, index=null.index)
    pyr = null["pyramid"] if "pyramid" in null.columns else pd.Series(False, index=null.index)
    return null[(null["long_only"] == lo) & (univ == fu) & (pyr == py)]


def null_summary(null: pd.DataFrame, metrics=("total_return", "sharpe", "max_drawdown", "n_trades", "win_rate", "max_position_weight")) -> pd.DataFrame:
    """Median / 5th / 95th percentile / share positive, per null variant present."""
    rows = []
    for key, (_, label) in VARIANTS.items():
        sub = _variant(null, key)
        if sub.empty:
            continue
        row = {"side": label, "n_seeds": len(sub), "share_positive_return": float((sub["total_return"] > 0).mean())}
        for m in metrics:
            row[f"{m}_p05"], row[f"{m}_median"], row[f"{m}_p95"] = (float(sub[m].quantile(q)) for q in (0.05, 0.5, 0.95))
        rows.append(row)
    return pd.DataFrame(rows).set_index("side")
