"""Rule-based control agents.

They play by exactly the same rules as the LLMs (one new position per cycle,
confidence gate, invalidation price, fees). Any LLM that cannot beat these has
nothing to say about markets. Each control reads only the payload dict.
"""

from __future__ import annotations

import numpy as np

from ..prompt import last_close, universe_rows
from ..schema import MAX_DECISIONS, Decision, DecisionItem
from ..universe import BENCHMARK
from .base import Agent


def _price(payload: dict, symbol: str) -> float | None:
    for row in universe_rows(payload):
        if row["symbol"] == symbol:
            return float(row["price"])
    det = payload["detail"].get(symbol) or payload.get("benchmark", {}).get(symbol)
    return last_close(det)


def _candidates(payload: dict) -> list[dict]:
    """The names an agent may open: the screener's list, or every name when the payload was
    built with candidates='universe'."""
    return payload.get("screened") or payload.get("_screened") or [{"symbol": r["symbol"]} for r in universe_rows(payload)]


def _held(payload: dict) -> dict[str, dict]:
    return {p["symbol"]: p for p in payload["portfolio"].get("positions", [])}


class BuyAndHoldBenchmark(Agent):
    """Buys SPY with 100% of equity on the first cycle and never trades again.
    The invalidation price is set effectively unreachable."""
    name = "control_buy_and_hold_spy"
    tradeable = {BENCHMARK}

    def decide(self, payload: dict) -> Decision:
        if BENCHMARK in _held(payload):
            return Decision.hold("holding benchmark")
        px = _price(payload, BENCHMARK)
        if px is None:
            return Decision.hold("no benchmark price")
        return Decision(reasoning="buy and hold the benchmark", decisions=[DecisionItem(
            symbol=BENCHMARK, action="open_long", percent_of_equity=100.0, confidence=1.0,
            thesis="Passive exposure to the US equity market.",
            invalidation="Never — this is a buy-and-hold control.", invalidation_price=round(px * 0.01, 2),
        )])


class Momentum10(Agent):
    """10-day momentum on the candidate names: open the strongest signal, long or short,
    and exit when momentum flips. Stop 8% against the entry.

    The engine allows one *new* position per cycle for every agent, so longs and shorts
    compete in a single pool ranked by |10-day momentum| rather than shorts being
    considered only when no long qualifies. With the earlier long-first ordering the
    short branch was unreachable on a 50-name universe, which made the `_ls` and
    `_long` controls the same agent under two names.

    Gates are deliberately asymmetric: a long needs momentum above `long_threshold`,
    a short needs it below `short_threshold` (shorts are the harder trade).
    """
    name = "control_momentum_10d"

    def __init__(self, allow_short: bool = True, stop_pct: float = 0.08, size_pct: float = 10.0,
                 long_threshold: float = 0.02, short_threshold: float = -0.05):
        self.allow_short, self.stop_pct, self.size_pct = allow_short, stop_pct, size_pct
        self.long_threshold, self.short_threshold = long_threshold, short_threshold
        self.name = "control_momentum_10d" + ("_ls" if allow_short else "_long")

    def decide(self, payload: dict) -> Decision:
        held = _held(payload)
        mom = {r["symbol"]: float(r["ret_10d"]) for r in universe_rows(payload) if r["ret_10d"] is not None and r["ret_10d"] == r["ret_10d"]}
        items: list[DecisionItem] = []
        # exits: momentum flipped against the position
        for sym, p in held.items():
            m = mom.get(sym)
            if m is None:
                continue
            if (p["side"] == "long" and m < 0) or (p["side"] == "short" and m > 0):
                items.append(DecisionItem(symbol=sym, action="close"))
        closing = {i.symbol for i in items}
        # one entry per cycle: longs and shorts compete in one pool, ranked by |momentum|
        if len(held) - len(closing) < 10:
            cands = [(s["symbol"], mom.get(s["symbol"], 0.0)) for s in _candidates(payload)
                     if s["symbol"] not in held and s["symbol"] not in closing]
            pool = [(sym_, m_, "open_long") for sym_, m_ in cands if m_ > self.long_threshold]
            if self.allow_short:
                pool += [(sym_, m_, "open_short") for sym_, m_ in cands if m_ < self.short_threshold]
            best = max(pool, key=lambda c: abs(c[1]), default=None)
            if best:
                sym, _m, action = best
                px = _price(payload, sym)
                if px:
                    inv = px * (1 - self.stop_pct) if action == "open_long" else px * (1 + self.stop_pct)
                    items.append(DecisionItem(
                        symbol=sym, action=action, percent_of_equity=self.size_pct, confidence=0.85,
                        thesis=f"10-day momentum {mom.get(sym, 0):+.1%}; trend continuation.",
                        invalidation=f"{self.stop_pct:.0%} move against entry or momentum flips.",
                        invalidation_price=round(inv, 2),
                    ))
        return Decision(reasoning="rule: 10d momentum", decisions=items) if items else Decision.hold("no signal")


class RandomAgent(Agent):
    """Random long/short picks under the same constraints. Seeded, so it is
    reproducible — a null distribution rather than one lucky path (run several seeds)."""
    name = "control_random"
    needs_detail = False

    def __init__(self, seed: int = 0, p_open: float = 0.5, p_close: float = 0.1, stop_pct: float = 0.05, long_only: bool = False,
                 from_universe: bool = False, p_add: float = 0.0):
        """from_universe=True picks from all 50 names instead of the five the screener hands out —
        the diagnostic that separates the screener's contribution from everything else.
        p_add > 0 lets the coin also *add* to a held position that is in profit (10/20/30% of equity),
        the pyramiding an unconstrained prompt does; the engine still caps gross exposure at 100%."""
        self.seed, self.p_open, self.p_close, self.stop_pct, self.long_only, self.from_universe, self.p_add = \
            seed, p_open, p_close, stop_pct, long_only, from_universe, p_add
        self.name = f"control_random{'_long' if long_only else ''}{'_univ' if from_universe else ''}{'_pyr' if p_add else ''}_s{seed}"
        self.reset()

    def reset(self) -> None:
        self.rng = np.random.default_rng(self.seed)

    def decide(self, payload: dict) -> Decision:
        held = _held(payload)
        items: list[DecisionItem] = []
        for sym, p in held.items():
            if self.rng.random() < self.p_close:
                items.append(DecisionItem(symbol=sym, action="close"))
            elif self.p_add and p.get("unrealized_pct", 0) > 0 and self.rng.random() < self.p_add:
                items.append(DecisionItem(symbol=sym, action="add", percent_of_equity=float(self.rng.choice([10, 20, 30])),
                                          confidence=float(self.rng.uniform(0.80, 1.0)), thesis="Random add (null control).",
                                          invalidation="Random stop.", invalidation_price=float(p["invalidation_price"]) if p.get("invalidation_price") else None))
        closing = {i.symbol for i in items if i.action == "close"}
        if len(held) - len(closing) < 10 and self.rng.random() < self.p_open:
            pool = universe_rows(payload) if self.from_universe else _candidates(payload)
            cands = [s["symbol"] for s in pool if s["symbol"] not in held and s["symbol"] not in closing]
            if cands:
                sym = str(self.rng.choice(cands))
                px = _price(payload, sym)
                if px:
                    side = "open_long" if (self.long_only or self.rng.random() < 0.5) else "open_short"
                    inv = px * (1 - self.stop_pct) if side == "open_long" else px * (1 + self.stop_pct)
                    items.append(DecisionItem(
                        symbol=sym, action=side, percent_of_equity=float(self.rng.choice([10, 20, 30])),
                        confidence=float(self.rng.uniform(0.80, 1.0)),
                        thesis="Random pick (null control).", invalidation="Random stop.", invalidation_price=round(inv, 2),
                    ))
        items = items[:MAX_DECISIONS]     # the schema caps a decision at MAX_DECISIONS items; a coin can exceed it
        return Decision(reasoning="random control", decisions=items) if items else Decision.hold("coin said hold")
