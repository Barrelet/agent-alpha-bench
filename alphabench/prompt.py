"""Cycle payload: everything an agent is allowed to see for one decision.

`build_payload()` returns a plain dict — the contract between market data and
agents. Rule agents read the numbers; the LLM agent renders it with
`render_text()`. Because it is built from an `.asof()` slice, the payload can
only contain information up to the decision date.
"""

from __future__ import annotations

import json
import pandas as pd

from .market import MarketData, rsi
from .screener import screen
from .universe import BENCHMARK, TICKERS, sector_of

N_DAILY = 20   # was 30; trimmed so the full prompt stays well under 8k tokens
N_WEEKLY = 12  # was 26
TOP_N = 5

INVESTOR_MANDATE = (
    "You manage a $10,000 paper portfolio of US large-cap equities as a medium-term investor. "
    "Build positions you would be comfortable holding for weeks. You may go long or short. "
    "Fees are 0.1% per side; there is no leverage. Every open or add must state a thesis, an "
    "invalidation condition and an explicit invalidation_price on the losing side of the entry; "
    "positions are closed automatically if that price trades. Only open a position when your "
    "confidence is at least 0.80. At most one new position per cycle and ten positions in total. "
    "Adding is allowed only to positions in profit. If nothing meets the bar, hold."
)


CANDLE_FORMAT = ("rows are [open, high, low, close, volume_millions], oldest first, one per trading day (daily) "
                 "or per week (weekly); first_date/last_date give the span, the last row ends on decision_date")


INVESTOR_MANDATE_V2 = INVESTOR_MANDATE + (
    " Risk discipline, in order of importance: (1) at most ONE open_long/open_short per cycle — if several look "
    "attractive, take the best and mention the others in reasoning; (2) set invalidation_price at least two typical "
    "daily moves away from the current price, where the typical daily move is vol_30d_ann/16 (vol_30d_ann 0.32 means "
    "~2% per day, so the stop goes at least 4% away), and never closer than 3%; (3) add only to positions whose "
    "unrealized_pct is positive; (4) size 10-20% of equity per position and go larger only with confidence >= 0.9; "
    "(5) when nothing is compelling, return an empty decisions list — every trade costs 0.2% round trip; "
    "(6) percent_of_equity is a share of TOTAL equity and the shares of all open positions cannot add up to more than 100% — "
    "check portfolio.gross_exposure_pct before opening: if it is above 80, do not open anything."
)


INVESTOR_MANDATE_V3 = """You are the portfolio manager of a $10,000 paper account in US large-cap equities, judged over months on net-of-cost return, Sharpe ratio, drawdown, and on whether your stated confidence matches your realised hit rate. You are a medium-term investor: positions are meant to be held for weeks. You may go long or short; there is no leverage; each fill costs 0.1%.

{how_to_read}

DECISION PROCEDURE, every cycle, in this order:
1. Review each held position, but remember the horizon is weeks: a position opened less than 7 trading days ago is left alone unless its invalidation_price is about to trade. Beyond that, close it only if the invalidation reason in your thesis has actually happened, or if it is underwater AND both sma20_pct and ret_10d have turned against it. Day-to-day wobbles in RSI or a single red candle are not reasons. Add only if unrealized_pct is positive and the trend has strengthened. The default action on a held position is to do nothing.
2. Score each candidate ({candidate_set}) on four points: trend (sign and size of sma20_pct, sma50_pct, ret_30d), momentum quality (RSI 45-70 healthy for longs, above 78 stretched, below 30 washed out; ret_5d against ret_30d signals a pullback or a reversal), participation (vol_ratio above 1.2 confirms a move, below 0.8 says it lacks sponsorship), and location (hi20_pct near 0 is a breakout or a ceiling; lo20_pct near 0 is a floor or a breakdown).
3. Base rates you must respect: over weeks, large caps mostly move with the market — check SPY's trend before going against it; shorting a large cap in an uptrend loses more often than it wins; a stop within one daily move is noise, not risk control; most days the right answer is no new position.
4. Open at most ONE new position, and only when it clearly beats holding cash after 0.2% round-trip cost. Size 10-20% of total equity; 25-30% only with confidence >= 0.9. The open positions together cannot exceed 100% of equity: if portfolio.gross_exposure_pct is above 80, do not open anything.
5. Stop placement: invalidation_price must be at least max(3%, 2 x atr14_pct) away from the current price, below it for a long and above it for a short, ideally just beyond a level that would prove the thesis wrong (recent swing low/high, the 20-day average).
6. Confidence is a probability: 0.80 means you expect four such trades in five to be profitable. It is scored, so be honest — a marginal setup is a hold, not a 0.80.

RULES THE ENGINE ENFORCES (violations are refused and logged): confidence >= 0.80 to open; max 10 positions; one new position per cycle; no add to a losing position; no re-entry the same cycle a symbol was closed; stop on the losing side of the entry.

Write reasoning as a short audit trail: what you checked, what you rejected and why, what you chose. Thesis and invalidation must be specific to the numbers you saw."""

HOW_TO_READ = {
    "screened": ("HOW TO READ THE DATA. portfolio = your current book (cash, positions with entry, stop, unrealized_pct and your own earlier thesis). "
                 "screened = today's five most liquid, most active names; detail = candles and features for screened and held names only; "
                 "universe = a one-line summary of all 50 names. Returns and distances are decimals in the universe table (0.052 = +5.2%) and "
                 "percentages in features (5.2 = +5.2%). vol_30d_ann/16 is the typical daily move in decimals; atr14_pct is the same thing in percent."),
    "universe": ("HOW TO READ THE DATA. portfolio = your current book (cash, positions with entry, stop, unrealized_pct and your own earlier thesis). "
                 "universe = one row per name for all 50 names, every one of them a candidate: price, returns (decimals: 0.052 = +5.2%), "
                 "vol_30d_ann, rsi_1d, adv_20d_bn, sector, then the features in percent (atr14_pct, sma20_pct, sma50_pct, hi20_pct, lo20_pct, "
                 "vol_ratio, ret_5d; 5.2 = +5.2%). detail = candles for held names only; benchmark = SPY candles. "
                 "vol_30d_ann/16 is the typical daily move in decimals; atr14_pct is the same thing in percent."),
}
CANDIDATE_SET = {"screened": "the screened names", "universe": "every row of the universe table"}


def cap_rule(max_position_weight: float | None) -> str:
    """The sentence every prompt gets when the engine enforces a per-name cap."""
    if max_position_weight is None:
        return ""
    return (f" Risk limit enforced by the engine: no single position may exceed {max_position_weight:.0%} of equity, "
            f"counting adds; a request that would exceed it is cut down to the limit, and an add to a name already at the limit is refused.")


def mandate(version: str = "v3", candidates: str = "screened", max_position_weight: float | None = None) -> str:
    """System prompt for a prompt version and a candidate mode ('screened' = top-5 screener
    with candles; 'universe' = all 50 names are candidates, features in the table, no candles).
    max_position_weight adds the engine's per-name cap to the stated rules."""
    if version == "v1":
        text = INVESTOR_MANDATE
    elif version == "v2":
        text = INVESTOR_MANDATE_V2
    else:
        text = INVESTOR_MANDATE_V3.format(how_to_read=HOW_TO_READ[candidates], candidate_set=CANDIDATE_SET[candidates])
    return text + cap_rule(max_position_weight)


def _px(x: float) -> float:
    """Price precision scaled to magnitude: 2 dp below 100, 1 dp below 1000, else 0 dp.
    Keeps ~0.05% precision while shaving a digit (= a token) per price."""
    x = float(x)
    return round(x, 2) if x < 100 else round(x, 1) if x < 1000 else round(x)


def _candles(df: pd.DataFrame, n: int) -> dict:
    """Token-frugal candles. Most tokenizers split numbers into single digits, so
    every digit costs a token: dates are stated once (first/last) instead of per
    row, volume is in millions with one decimal, prices keep two decimals."""
    d = df.tail(n)
    if d.empty:
        return {"first_date": None, "last_date": None, "rows": []}
    return {
        "first_date": str(d.index[0].date()), "last_date": str(d.index[-1].date()),
        "rows": [[_px(r.open), _px(r.high), _px(r.low), _px(r.close), round(float(r.volume) / 1e6, 1)] for r in d.itertuples()],
    }


def last_close(block: dict) -> float | None:
    """Close of the last daily candle in a detail block (helper for agents)."""
    rows = (block or {}).get("daily", {}).get("rows") or []
    return float(rows[-1][3]) if rows else None


FEATURE_FORMAT = ("features: atr14_pct = average true range over 14 days as % of price (typical daily move); "
                  "sma20_pct / sma50_pct = price vs 20/50-day average in %; hi20_pct / lo20_pct = distance to 20-day "
                  "high / low in %; vol_ratio = last 5 days' volume vs 20-day average; ret_5d in %")


def _features(daily: pd.DataFrame) -> dict:
    """Derived numbers a model cannot reliably compute from candles by reading them."""
    c, h, l, v = daily["close"], daily["high"], daily["low"], daily["volume"]
    px = float(c.iloc[-1])
    prev = c.shift(1)
    tr = pd.concat([h - l, (h - prev).abs(), (l - prev).abs()], axis=1).max(axis=1)
    f = {
        "atr14_pct": round(float(tr.tail(14).mean() / px * 100), 2),
        "sma20_pct": round(float(px / c.tail(20).mean() * 100 - 100), 2),
        "sma50_pct": round(float(px / c.tail(50).mean() * 100 - 100), 2) if len(c) >= 50 else None,
        "hi20_pct": round(float(px / h.tail(20).max() * 100 - 100), 2),
        "lo20_pct": round(float(px / l.tail(20).min() * 100 - 100), 2),
        "vol_ratio": round(float(v.tail(5).mean() / v.tail(20).mean()), 2) if v.tail(20).mean() > 0 else None,
        "ret_5d": round(float(px / c.iloc[-6] * 100 - 100), 2) if len(c) > 5 else None,
    }
    return f


def _detail_block(md: MarketData, sym: str, n_daily: int, n_weekly: int, include_rsi: bool, include_features: bool = True) -> dict:
    daily = md.symbol_bars(sym)
    weekly = md.weekly(sym)
    block = {"daily": _candles(daily, n_daily), "weekly": _candles(weekly, n_weekly)}
    if include_rsi:
        block["rsi_1d"] = int(round(float(rsi(daily["close"]).iloc[-1])))
        block["rsi_1w"] = int(round(float(rsi(weekly["close"]).iloc[-1]))) if len(weekly) > 15 else None
    if include_features:
        block["features"] = _features(daily)
    return block


def _compact_table(table: pd.DataFrame) -> pd.DataFrame:
    """Round the universe table with digits in mind (see _candles)."""
    out = pd.DataFrame(index=table.index)
    out["price"] = table["price"].round(2)   # full precision here: this is the reference for invalidation levels
    for c in ("ret_1d", "ret_10d", "ret_30d", "pct_from_30d_high"):
        out[c] = table[c].round(3)
    out["vol_30d_ann"] = table["vol_30d_ann"].round(2)
    out["rsi_1d"] = table["rsi_1d"].round().astype("Int64")
    out["adv_20d_bn"] = (table["adv_20d_usd"] / 1e9).round(2)
    return out


FEATURE_COLS = ["atr14_pct", "sma20_pct", "sma50_pct", "hi20_pct", "lo20_pct", "vol_ratio", "ret_5d"]


def build_base(md: MarketData, decision_date=None, tickers: list[str] = TICKERS, top_n: int = TOP_N,
               n_daily: int = N_DAILY, n_weekly: int = N_WEEKLY, include_rsi: bool = True,
               include_summary: bool = True, include_features: bool = True, earnings: dict[str, str] | None = None,
               candidates: str = "screened") -> dict:
    """The agent-independent part of a cycle payload (computed once per date).
    candidates='screened': the top-`top_n` screener names get candles + features and are the
    only candidates the rule agents consider. candidates='universe': all names are candidates,
    the universe table carries the features, and candles are given for held names and SPY only."""
    decision_date = pd.Timestamp(decision_date or md.last_date)
    assert md.last_date <= decision_date, "MarketData must be sliced .asof(decision_date)"
    if candidates not in ("screened", "universe"):
        raise ValueError("candidates must be 'screened' or 'universe'")
    if candidates == "screened":
        screened = screen(md, tickers, top_n)
        cand_list = [{"symbol": s, "rank": int(r["rank"]), "adv_20d_bn": round(r["adv_20d_usd"] / 1e9, 2), "mom_10d": round(r["mom_10d"], 3)}
                     for s, r in screened.iterrows()]
        detail = {s: _detail_block(md, s, n_daily, n_weekly, include_rsi, include_features) for s in screened.index}
    else:
        cand_list = [{"symbol": s} for s in tickers]      # unranked: no funnel
        detail = {}
    base = {
        "decision_date": str(decision_date.date()),
        "mandate": INVESTOR_MANDATE,
        "candidates": candidates,
        "screened": cand_list,
        "detail": detail,
        "benchmark": {BENCHMARK: {"daily": _candles(md.symbol_bars(BENCHMARK), n_daily)}} if BENCHMARK in md.symbols else {},
        "candle_format": CANDLE_FORMAT + ("; " + FEATURE_FORMAT if include_features else ""),
        "_opts": {"n_daily": n_daily, "n_weekly": n_weekly, "include_rsi": include_rsi, "include_features": include_features},
    }
    if include_summary or candidates == "universe":
        table = _compact_table(md.summary_table(tickers))
        table["sector"] = [sector_of(s) for s in table.index]
        if candidates == "universe":
            feats = pd.DataFrame({s: _features(md.symbol_bars(s)) for s in table.index}).T
            for c in FEATURE_COLS:
                table[c] = feats[c].astype(float).round(2)
        if earnings:
            table["next_earnings"] = [earnings.get(s) for s in table.index]
        # columnar: one header + one array per row. Repeating 10 key names for 50
        # rows would cost ~1.5k tokens of pure JSON overhead.
        t = table.reset_index()
        base["universe"] = {"columns": list(t.columns), "rows": t.astype(object).where(t.notna(), None).values.tolist()}
    return base


def universe_rows(payload: dict) -> list[dict]:
    """Universe table as a list of dicts (helper for agents; the payload keeps it columnar)."""
    u = payload.get("universe")
    if not u:
        return []
    if isinstance(u, list):
        return u
    return [dict(zip(u["columns"], r)) for r in u["rows"]]


def build_payload(md: MarketData, portfolio_snapshot: dict, decision_date=None, base: dict | None = None,
                  held_detail: bool = True, **kwargs) -> dict:
    """Assemble the cycle payload from an already-sliced MarketData. Pass `base`
    (from `build_base`) to reuse the date-level work across agents. held_detail=False
    skips candle blocks for held names (agents that never read candles)."""
    base = base or build_base(md, decision_date, **kwargs)
    opts = base["_opts"]
    detail = dict(base["detail"])
    for p in (portfolio_snapshot.get("positions", []) if held_detail else []):
        s = p["symbol"]
        if s not in detail and s in md.symbols:
            detail[s] = _detail_block(md, s, opts["n_daily"], opts["n_weekly"], opts["include_rsi"], opts.get("include_features", True))
    # Key order matters for LLMs: the state the model must not lose (portfolio,
    # screener, universe table) comes first; the bulky candles come last, so any
    # truncation by a backend eats candles rather than the portfolio.
    payload = {"decision_date": base["decision_date"], "mandate": base["mandate"], "portfolio": portfolio_snapshot}
    if base.get("candidates", "screened") == "screened":
        payload["screened"] = base["screened"]
    else:
        payload["candidates"] = "all names in the universe table"
        payload["_screened"] = base["screened"]           # rule agents' candidate list; stripped before rendering
    if "universe" in base:
        payload["universe"] = base["universe"]
    payload["candle_format"] = base["candle_format"]
    payload["benchmark"] = base["benchmark"]
    payload["detail"] = detail
    return payload


def render_text(payload: dict) -> str:
    """Render the payload as the prompt text an LLM sees. Compact JSON keeps
    tokens down; the mandate and output contract are plain prose."""
    body = {k: v for k, v in payload.items() if k != "mandate" and not k.startswith("_")}
    return (
        f"{payload['mandate']}\n\n"
        f"DATA (as of {payload['decision_date']} close):\n"
        f"{json.dumps(body, separators=(',', ':'), default=str)}\n\n"
        "Respond with strict JSON only: {\"reasoning\": str, \"market_context\": str, "
        "\"decisions\": [{\"symbol\", \"action\": open_long|open_short|add|close|hold, "
        "\"percent_of_equity\": 10-100, \"confidence\": 0-1, \"thesis\", \"invalidation\", \"invalidation_price\"}]}"
    )


def estimate_tokens(text: str) -> int:
    """Rough token estimate. Numbers-heavy JSON tokenises at ~2.5 chars/token on
    digit-splitting tokenizers (Qwen, Llama 3, GPT-4o…); prose at ~4."""
    return int(len(text) / 2.5)
