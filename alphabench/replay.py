"""Walk the calendar and run every agent through the same cycles.

All agents step in lockstep per decision date so the date-level work (slice,
screen, summary table) is done once. Each agent has its own Portfolio.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import pandas as pd

from .agents.base import Agent
from .engine import Portfolio
from .market import MarketData
from .prompt import build_base, build_payload
from .schema import Decision
from .universe import TICKERS

log = logging.getLogger(__name__)


def decision_dates(md: MarketData, start, end=None, warmup: int = 60) -> pd.DatetimeIndex:
    """Market days on which decisions are taken. Each needs a *next* market day to
    fill on, so the last date in the data is never a decision date."""
    dates = md.dates
    start_i = max(dates.searchsorted(pd.Timestamp(start)), warmup)
    end_i = len(dates) - 1 if end is None else min(dates.searchsorted(pd.Timestamp(end), side="right"), len(dates) - 1)
    return dates[start_i:end_i]


def run_replay(md: MarketData, agents: list[Agent], start, end=None, warmup: int = 60,
               initial_capital: float = 10_000.0, tickers: list[str] = TICKERS,
               payload_kwargs: dict | None = None, log_dir: Path | None = None,
               progress: bool = True, on_cycle=None, close_at_end: bool = True,
               max_position_weight: float | None = None) -> dict[str, dict]:
    """on_cycle(i, n, decision_date, {agent_name: equity}, seconds_elapsed) is called after every cycle.
    close_at_end: force-close every open position at the last close (fee charged) so trade
    statistics include positions still open when the window ends.
    max_position_weight: per-name cap as a share of equity, enforced on every agent (None = leaderboard rules, no cap)."""
    payload_kwargs = payload_kwargs or {}
    dds = decision_dates(md, start, end, warmup)
    if len(dds) == 0:
        raise ValueError("no decision dates — check start/end/warmup")
    all_dates = md.dates
    books = {a.name: Portfolio(initial_capital=initial_capital, max_position_weight=max_position_weight) for a in agents}
    logs = {a.name: [] for a in agents}
    for a in agents:
        a.reset()
        books[a.name].mark(dds[0], md.close.loc[dds[0]])  # starting equity

    t0 = time.time()
    for i, t in enumerate(dds):
        t1 = all_dates[all_dates.searchsorted(t) + 1]
        md_t = md.asof(t)
        base = build_base(md_t, t, tickers=tickers, **payload_kwargs)
        close_t, open_1, low_1, high_1, close_1 = (md.close.loc[t], md.open.loc[t1], md.low.loc[t1], md.high.loc[t1], md.close.loc[t1])
        for a in agents:
            book = books[a.name]
            snap = book.snapshot(close_t)
            payload = build_payload(md_t, snap, t, base=base, held_detail=getattr(a, "needs_detail", True))
            try:
                decision = a.decide(payload)
                if not isinstance(decision, Decision):
                    raise TypeError("agent must return a Decision")
            except Exception as e:  # an invalid decision is a hold, and is logged as such
                log.warning("%s on %s failed: %s", a.name, t.date(), e)
                decision = Decision.hold(f"invalid output: {e}")
            tradeable = a.tradeable if a.tradeable is not None else set(tickers)
            fills = book.execute(decision, t1, open_1, ref_equity=snap["equity"], tradeable=tradeable)
            inv = book.check_invalidations(t1, open_1, low_1, high_1)
            eq = book.mark(t1, close_1)
            logs[a.name].append({
                "decision_date": str(t.date()), "fill_date": str(t1.date()), "equity": round(eq, 2),
                "decision": decision.model_dump(), "n_fills": len(fills), "n_invalidations": len(inv),
                "rejections": [r for r in book.rejections if r["date"] == t1],
            })
        if progress and (i % 50 == 0 or i == len(dds) - 1):
            log.info("cycle %d/%d (%s) %.1fs", i + 1, len(dds), t.date(), time.time() - t0)
        if on_cycle is not None:
            on_cycle(i + 1, len(dds), t, {a.name: books[a.name].equity_curve[-1][1] for a in agents}, time.time() - t0)

    results = {}
    for a in agents:
        book = books[a.name]
        if close_at_end and book.positions:
            last = book.equity_curve[-1][0]
            book.close_all(last, md.close.loc[last])
            book.equity_curve[-1] = (last, book.equity(md.close.loc[last]))   # final mark net of liquidation fees
        results[a.name] = {"equity": book.equity_series(), "trades": book.trades_df(), "fills": book.fills_df(),
                           "rejections": pd.DataFrame(book.rejections), "scaled": book.scaled_df(),
                           "log": logs[a.name], "portfolio": book}
    if log_dir:
        save_results(results, Path(log_dir))
    return results


def save_results(results: dict[str, dict], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    eq = pd.DataFrame({k: v["equity"] for k, v in results.items()})
    eq.to_parquet(out / "equity.parquet")
    parts = [v["trades"].assign(agent=k) for k, v in results.items() if not v["trades"].empty]
    trades = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["agent"])
    trades.to_parquet(out / "trades.parquet", index=False)
    with open(out / "decisions.jsonl", "w") as f:
        for k, v in results.items():
            for row in v["log"]:
                f.write(json.dumps({"agent": k, **row}, default=str) + "\n")
