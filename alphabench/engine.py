"""Paper-trading engine: portfolio state, rule enforcement, fills, invalidation, marks.

Cycle timing (the anti-look-ahead contract)
-------------------------------------------
  decision date t  : agent sees data <= close(t), returns a Decision
  next market day  : 1) orders fill at open(t+1) less fees
                     2) invalidation levels checked against low/high(t+1)
                     3) portfolio marked at close(t+1)

Rules enforced here, identically for every agent:
  max 10 positions, max 1 *new* position per cycle, confidence >= 0.80 to open,
  add only to positions in profit, no re-entry in the cycle a symbol was closed,
  invalidation_price must sit on the losing side of the fill, no leverage
  (gross exposure <= equity), 0.1% fee per side, no slippage.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd

from .schema import Decision, DecisionItem, MIN_CONFIDENCE_TO_OPEN

FEE_RATE = 0.001
MAX_POSITIONS = 10
MAX_NEW_PER_CYCLE = 1
MIN_TRADE_FRACTION = 0.01  # reject fills smaller than 1% of equity after cash capping
MAX_POSITION_WEIGHT = None  # optional per-name cap as a share of equity (e.g. 0.25); None = no cap (leaderboard rules)


@dataclass
class Position:
    symbol: str
    side: int                 # +1 long, -1 short
    qty: float                # shares, always positive
    entry_price: float        # average fill price
    entry_date: pd.Timestamp
    invalidation_price: float
    thesis: str = ""
    invalidation: str = ""
    confidence: float = float("nan")
    fees_paid: float = 0.0
    last_price: float = float("nan")

    def __post_init__(self):
        if np.isnan(self.last_price):
            self.last_price = self.entry_price

    def market_value(self, price: float) -> float:
        return self.side * self.qty * price

    def unrealized_pnl(self, price: float) -> float:
        return self.side * self.qty * (price - self.entry_price)

    def unrealized_pct(self, price: float) -> float:
        return self.side * (price / self.entry_price - 1.0)


@dataclass
class Fill:
    date: pd.Timestamp
    symbol: str
    action: str
    side: int
    qty: float
    price: float
    notional: float
    fee: float
    reason: str = ""


@dataclass
class Trade:
    """A closed (or partially closed) position leg."""
    symbol: str
    side: int
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    exit_price: float
    qty: float
    pnl: float           # net of fees on this leg
    pnl_pct: float       # gross price return in the direction of the trade
    fees: float
    reason: str          # 'close' | 'invalidation'
    confidence: float
    holding_days: int


@dataclass
class Portfolio:
    initial_capital: float = 10_000.0
    fee_rate: float = FEE_RATE
    #: largest single position as a share of equity, enforced on opens and adds; None = unlimited
    max_position_weight: float | None = MAX_POSITION_WEIGHT
    cash: float = field(init=False)
    positions: dict[str, Position] = field(default_factory=dict)
    trades: list[Trade] = field(default_factory=list)
    fills: list[Fill] = field(default_factory=list)
    rejections: list[dict] = field(default_factory=list)
    scaled: list[dict] = field(default_factory=list)      # fills cut down by the cash / exposure cap (soft violations)
    equity_curve: list[tuple[pd.Timestamp, float]] = field(default_factory=list)
    total_fees: float = 0.0

    def __post_init__(self):
        self.cash = float(self.initial_capital)

    # ---- valuation ----------------------------------------------------------
    def _px(self, prices: pd.Series, pos: Position) -> float:
        """Price for valuation; falls back to the last mark if a symbol is missing today."""
        px = float(prices.get(pos.symbol, np.nan))
        return pos.last_price if np.isnan(px) else px

    def equity(self, prices: pd.Series) -> float:
        return self.cash + sum(p.market_value(self._px(prices, p)) for p in self.positions.values())

    def gross_exposure(self, prices: pd.Series) -> float:
        return sum(p.qty * self._px(prices, p) for p in self.positions.values())

    def snapshot(self, prices: pd.Series) -> dict:
        eq = self.equity(prices)
        return {
            "cash": round(self.cash, 2),
            "equity": round(eq, 2),
            "n_positions": len(self.positions),
            "gross_exposure_pct": round(100.0 * self.gross_exposure(prices) / eq, 1) if eq else 0.0,
            "positions": [
                {
                    "symbol": s, "side": "long" if p.side > 0 else "short", "qty": round(p.qty, 4),
                    "entry_price": round(p.entry_price, 2), "entry_date": str(p.entry_date.date()),
                    "last_price": round(self._px(prices, p), 2),
                    "unrealized_pnl": round(p.unrealized_pnl(self._px(prices, p)), 2),
                    "unrealized_pct": round(100.0 * p.unrealized_pct(self._px(prices, p)), 2),
                    "invalidation_price": p.invalidation_price, "thesis": p.thesis,
                    "invalidation": p.invalidation, "confidence": p.confidence,
                }
                for s, p in self.positions.items()
            ],
        }

    # ---- helpers ------------------------------------------------------------
    def _reject(self, date, item: DecisionItem, why: str):
        self.rejections.append({"date": date, "symbol": item.symbol, "action": item.action, "reason": why})

    def _fill(self, date, symbol, action, side, qty, price, reason="") -> Fill:
        notional = qty * price
        fee = notional * self.fee_rate
        self.cash += -side * notional - fee   # long: pay; short: receive proceeds
        self.total_fees += fee
        f = Fill(date, symbol, action, side, qty, price, notional, fee, reason)
        self.fills.append(f)
        return f

    def _close_leg(self, date, pos: Position, qty: float, price: float, reason: str) -> Trade:
        # closing a long = sell (side -1 cash effect); closing a short = buy back
        f = self._fill(date, pos.symbol, "close", -pos.side, qty, price, reason)
        gross = pos.side * qty * (price - pos.entry_price)
        entry_fee_share = pos.fees_paid * (qty / pos.qty) if pos.qty else 0.0
        t = Trade(
            symbol=pos.symbol, side=pos.side, entry_date=pos.entry_date, exit_date=date,
            entry_price=pos.entry_price, exit_price=price, qty=qty,
            pnl=gross - f.fee - entry_fee_share, pnl_pct=pos.side * (price / pos.entry_price - 1.0),
            fees=f.fee + entry_fee_share, reason=reason, confidence=pos.confidence,
            holding_days=int((date - pos.entry_date).days),
        )
        self.trades.append(t)
        pos.fees_paid -= entry_fee_share
        pos.qty -= qty
        if pos.qty <= 1e-9:
            del self.positions[pos.symbol]
        return t

    # ---- 1) execute a decision at next open -----------------------------------
    def execute(self, decision: Decision, date: pd.Timestamp, open_prices: pd.Series,
                ref_equity: float, tradeable: set[str]) -> list[Fill]:
        """Apply a Decision at `open_prices` (the open of `date`). `ref_equity` is the
        equity at the decision-time close, used to resolve percent_of_equity."""
        fills: list[Fill] = []
        closed_this_cycle: set[str] = set()
        new_opened = 0
        order = {"close": 0, "add": 1, "open_long": 2, "open_short": 2, "hold": 3}
        for item in sorted(decision.decisions, key=lambda d: order[d.action]):
            if item.action == "hold":
                continue
            sym = item.symbol
            px = float(open_prices.get(sym, np.nan))
            if np.isnan(px):
                self._reject(date, item, "no price"); continue

            if item.action == "close":
                pos = self.positions.get(sym)
                if pos is None:
                    self._reject(date, item, "not held"); continue
                frac = (item.percent_of_position or 100.0) / 100.0
                qty = pos.qty if frac >= 0.999 else pos.qty * frac
                self._close_leg(date, pos, qty, px, "close")
                closed_this_cycle.add(sym)
                fills.append(self.fills[-1])
                continue

            if item.action == "add":
                pos = self.positions.get(sym)
                if pos is None:
                    self._reject(date, item, "add: not held"); continue
                if pos.unrealized_pnl(px) <= 0:
                    self._reject(date, item, "add: position not in profit (no averaging down)"); continue
                dollars = item.percent_of_equity / 100.0 * ref_equity
                qty = self._size(dollars, px, pos.side, open_prices, date, sym, "add")
                if qty is None:
                    self._reject(date, item, "add: position cap" if self._at_cap(sym, px, open_prices) else "add: insufficient cash / exposure cap"); continue
                f = self._fill(date, sym, "add", pos.side, qty, px)
                pos.entry_price = (pos.entry_price * pos.qty + px * qty) / (pos.qty + qty)
                pos.qty += qty
                pos.fees_paid += f.fee
                if item.invalidation_price:
                    pos.invalidation_price = item.invalidation_price
                fills.append(f)
                continue

            # open_long / open_short
            side = 1 if item.action == "open_long" else -1
            if sym not in tradeable:
                self._reject(date, item, "symbol not tradeable"); continue
            if sym in self.positions:
                self._reject(date, item, "already held (use add)"); continue
            if sym in closed_this_cycle:
                self._reject(date, item, "no re-entry in the cycle it was closed"); continue
            if item.confidence < MIN_CONFIDENCE_TO_OPEN:
                self._reject(date, item, f"confidence {item.confidence:.2f} < {MIN_CONFIDENCE_TO_OPEN}"); continue
            if new_opened >= MAX_NEW_PER_CYCLE:
                self._reject(date, item, "max one new position per cycle"); continue
            if len(self.positions) >= MAX_POSITIONS:
                self._reject(date, item, f"max {MAX_POSITIONS} positions"); continue
            if (side > 0 and item.invalidation_price >= px) or (side < 0 and item.invalidation_price <= px):
                self._reject(date, item, "invalidation_price not on the losing side of the fill"); continue
            dollars = item.percent_of_equity / 100.0 * ref_equity
            qty = self._size(dollars, px, side, open_prices, date, sym, item.action)
            if qty is None:
                self._reject(date, item, "insufficient cash / exposure cap"); continue
            f = self._fill(date, sym, item.action, side, qty, px)
            self.positions[sym] = Position(
                symbol=sym, side=side, qty=qty, entry_price=px, entry_date=date,
                invalidation_price=item.invalidation_price, thesis=item.thesis or "",
                invalidation=item.invalidation or "", confidence=item.confidence, fees_paid=f.fee,
            )
            new_opened += 1
            fills.append(f)
        return fills

    def _at_cap(self, symbol: str, px: float, prices: pd.Series) -> bool:
        """True when the name already sits at (or above) the per-position cap."""
        if self.max_position_weight is None or symbol not in self.positions:
            return False
        return abs(self.positions[symbol].qty) * px >= self.max_position_weight * self.equity(prices) * (1 - MIN_TRADE_FRACTION)

    def _size(self, dollars: float, px: float, side: int, prices: pd.Series, date=None, symbol: str = "", action: str = "") -> float | None:
        """Resolve a dollar target to shares under the cash and no-leverage caps.
        A request larger than what is available is scaled down (and logged in
        `scaled`); only a request with < 1% of equity available is refused."""
        eq = self.equity(prices)
        headroom = max(0.0, eq - self.gross_exposure(prices)) / (1.0 + self.fee_rate)  # gross exposure <= equity, fee included
        if side > 0:
            headroom = min(headroom, self.cash / (1.0 + self.fee_rate))  # can't spend cash you don't have
        cap_hit = False
        if self.max_position_weight is not None:
            pos = self.positions.get(symbol)
            current = abs(pos.qty) * px if pos is not None else 0.0
            room = max(0.0, self.max_position_weight * eq - current)   # the position cap binds on the name, not the book
            if room < headroom:
                headroom, cap_hit = room, True
        filled = min(dollars, headroom)
        if filled < MIN_TRADE_FRACTION * eq or px <= 0:
            return None
        if filled < dollars * 0.999:
            self.scaled.append({"date": date, "symbol": symbol, "action": action, "requested_usd": round(dollars, 2),
                                "filled_usd": round(filled, 2), "shortfall_pct": round(100 * (1 - filled / dollars), 1),
                                "reason": "position_cap" if cap_hit and filled >= room * 0.999 else "cash_or_exposure"})
        return filled / px

    # ---- 2) invalidation monitor (daily, conservative) --------------------------
    def check_invalidations(self, date: pd.Timestamp, open_: pd.Series, low: pd.Series, high: pd.Series) -> list[Trade]:
        out = []
        for sym in list(self.positions):
            pos = self.positions[sym]
            o, lo, hi = (float(x.get(sym, np.nan)) for x in (open_, low, high))
            if np.isnan(lo) or np.isnan(hi):
                continue
            if pos.side > 0 and lo <= pos.invalidation_price:
                px = min(pos.invalidation_price, o)   # gap-through fills at the open
                out.append(self._close_leg(date, pos, pos.qty, px, "invalidation"))
            elif pos.side < 0 and hi >= pos.invalidation_price:
                px = max(pos.invalidation_price, o)
                out.append(self._close_leg(date, pos, pos.qty, px, "invalidation"))
        return out

    # ---- 2b) end-of-window liquidation --------------------------------------------
    def close_all(self, date: pd.Timestamp, prices: pd.Series, reason: str = "end_of_window") -> list[Trade]:
        """Force-close every open position at `prices` (fee charged), so trade-level
        statistics include positions still open when the window ends. Without this,
        win rate and calibration count only stopped-out losers and are biased down."""
        out = []
        for sym in list(self.positions):
            pos = self.positions[sym]
            px = float(prices.get(sym, np.nan))
            if np.isnan(px):
                px = pos.last_price
            out.append(self._close_leg(date, pos, pos.qty, px, reason))
        return out

    # ---- 3) mark ------------------------------------------------------------
    def mark(self, date: pd.Timestamp, close: pd.Series) -> float:
        for p in self.positions.values():
            p.last_price = self._px(close, p)
        eq = self.equity(close)
        self.equity_curve.append((pd.Timestamp(date), eq))
        return eq

    # ---- export -------------------------------------------------------------
    def equity_series(self) -> pd.Series:
        if not self.equity_curve:
            return pd.Series(dtype=float)
        d, v = zip(*self.equity_curve)
        return pd.Series(v, index=pd.DatetimeIndex(d), name="equity")

    def trades_df(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(t) for t in self.trades])

    def fills_df(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(f) for f in self.fills])

    def scaled_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.scaled)
