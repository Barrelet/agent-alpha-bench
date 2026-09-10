import numpy as np
import pandas as pd
import pytest

from alphabench.engine import Portfolio, MAX_POSITIONS
from alphabench.schema import Decision, DecisionItem, parse_decision

D = pd.Timestamp("2024-01-02")
S = lambda **kw: pd.Series(kw, dtype=float)


def _open(sym, action, pct=50, conf=0.9, inv=None):
    return DecisionItem(symbol=sym, action=action, percent_of_equity=pct, confidence=conf,
                        thesis="t", invalidation="i", invalidation_price=inv)


def test_long_roundtrip_pnl_and_fees():
    p = Portfolio(10_000)
    p.execute(Decision(decisions=[_open("A", "open_long", 50, inv=90)]), D, S(A=100), 10_000, {"A"})
    assert p.positions["A"].qty == pytest.approx(5000 / 100)
    assert p.cash == pytest.approx(10_000 - 5000 - 5)          # 0.1% fee
    p.execute(Decision(decisions=[DecisionItem(symbol="A", action="close")]), D, S(A=110), 10_000, {"A"})
    t = p.trades[0]
    assert t.pnl == pytest.approx(50 * 10 - 5 - 5.5)           # gross - entry fee - exit fee
    assert p.cash == pytest.approx(10_000 + 500 - 5 - 5.5)
    assert not p.positions


def test_short_roundtrip_pnl():
    p = Portfolio(10_000)
    p.execute(Decision(decisions=[_open("A", "open_short", 50, inv=110)]), D, S(A=100), 10_000, {"A"})
    assert p.cash == pytest.approx(10_000 + 5000 - 5)
    assert p.equity(S(A=100)) == pytest.approx(10_000 - 5)
    p.execute(Decision(decisions=[DecisionItem(symbol="A", action="close")]), D, S(A=90), 10_000, {"A"})
    assert p.trades[0].pnl == pytest.approx(50 * 10 - 5 - 4.5)
    assert p.trades[0].pnl_pct == pytest.approx(0.10)


def test_rules_confidence_gate_one_new_per_cycle_and_max_positions():
    p = Portfolio(10_000)
    d = Decision(decisions=[_open("A", "open_long", 10, conf=0.5, inv=1), _open("B", "open_long", 10, inv=1), _open("C", "open_long", 10, inv=1)])
    p.execute(d, D, S(A=10, B=10, C=10), 10_000, {"A", "B", "C"})
    assert set(p.positions) == {"B"}
    reasons = [r["reason"] for r in p.rejections]
    assert any("confidence" in r for r in reasons) and any("one new" in r for r in reasons)
    for i in range(MAX_POSITIONS + 2):
        sym = f"S{i}"
        p.execute(Decision(decisions=[_open(sym, "open_long", 5 if False else 10, inv=1)]), D, S(**{sym: 10}), 10_000, {sym})
    assert len(p.positions) == MAX_POSITIONS


def test_invalidation_side_and_no_averaging_down():
    p = Portfolio(10_000)
    p.execute(Decision(decisions=[_open("A", "open_long", 10, inv=120)]), D, S(A=100), 10_000, {"A"})
    assert not p.positions and "losing side" in p.rejections[-1]["reason"]
    p.execute(Decision(decisions=[_open("A", "open_long", 10, inv=90)]), D, S(A=100), 10_000, {"A"})
    p.execute(Decision(decisions=[_open("A", "add", 10, inv=85)]), D, S(A=95), 10_000, {"A"})
    assert "averaging down" in p.rejections[-1]["reason"]
    p.execute(Decision(decisions=[_open("A", "add", 10, inv=95)]), D, S(A=110), 10_000, {"A"})
    assert p.positions["A"].qty == pytest.approx(10 + 1000 / 110)


def test_invalidation_monitor_fills_at_level_or_gap_open():
    p = Portfolio(10_000)
    p.execute(Decision(decisions=[_open("A", "open_long", 10, inv=90), ]), D, S(A=100), 10_000, {"A"})
    p.check_invalidations(D, S(A=95), S(A=89), S(A=96))           # traded through 90 intraday
    assert p.trades[-1].exit_price == 90 and p.trades[-1].reason == "invalidation"
    p.execute(Decision(decisions=[_open("B", "open_short", 10, inv=110)]), D, S(B=100), 10_000, {"B"})
    p.check_invalidations(D, S(B=115), S(B=112), S(B=120))        # gapped above the level
    assert p.trades[-1].exit_price == 115


def test_no_leverage_cap_on_shorts():
    p = Portfolio(10_000)
    p.execute(Decision(decisions=[_open("A", "open_short", 100, inv=200)]), D, S(A=100), 10_000, {"A"})
    p.execute(Decision(decisions=[_open("B", "open_short", 100, inv=200)]), D, S(A=100, B=100), 10_000, {"A", "B"})
    assert "B" not in p.positions
    assert p.gross_exposure(S(A=100)) <= p.equity(S(A=100)) + 1e-6


def test_schema_rejects_open_without_invalidation():
    d, err = parse_decision({"decisions": [{"symbol": "A", "action": "open_long", "percent_of_equity": 10, "confidence": 0.9}]})
    assert d is None and "missing" in err
    d, err = parse_decision('{"decisions":[{"symbol":"*","action":"hold"}]}')
    assert err is None


def test_close_all_charges_fee_and_tags_reason():
    p = Portfolio(10_000)
    p.execute(Decision(decisions=[_open("A", "open_long", 50, inv=90)]), D, S(A=100), 10_000, {"A"})
    p.close_all(D, S(A=110))
    assert not p.positions
    assert p.trades[0].reason == "end_of_window"
    assert p.trades[0].pnl == pytest.approx(50 * 10 - 5 - 5.5)


def test_scaled_fill_is_logged():
    p = Portfolio(10_000)
    p.execute(Decision(decisions=[_open("A", "open_long", 60, inv=90)]), D, S(A=100), 10_000, {"A"})   # 6,000 in, ~4,000 headroom left
    p.execute(Decision(decisions=[_open("B", "open_long", 50, inv=90)]), D, S(A=100, B=100), 10_000, {"A", "B"})  # asks 5,000, gets ~4,000
    df = p.scaled_df()
    assert len(df) >= 1 and set(df.columns) >= {"symbol", "requested_usd", "filled_usd", "shortfall_pct"}
    assert (df["filled_usd"] < df["requested_usd"]).all()


def test_position_cap_binds_on_open_and_add():
    p = Portfolio(10_000, max_position_weight=0.25)
    p.execute(Decision(decisions=[_open("A", "open_long", 60, inv=90)]), D, S(A=100), 10_000, {"A"})
    assert p.positions["A"].qty * 100 == pytest.approx(2500, rel=0.01)          # cut to the 25% cap
    assert p.scaled[-1]["reason"] == "position_cap"
    p.execute(Decision(decisions=[DecisionItem(symbol="A", action="add", percent_of_equity=10, confidence=0.9, thesis="t",
                                               invalidation="i", invalidation_price=90)]), D, S(A=110), 10_000, {"A"})
    assert p.rejections[-1]["reason"] == "add: position cap"                    # already at the cap (price rose, so above it)
    q = Portfolio(10_000)                                                       # default: no cap, leaderboard rules
    q.execute(Decision(decisions=[_open("A", "open_long", 60, inv=90)]), D, S(A=100), 10_000, {"A"})
    assert q.positions["A"].qty * 100 == pytest.approx(6000, rel=0.01)
