"""The rule-based controls must actually differ from one another.

Regression test for a bug where Momentum10 considered shorts only when no long
qualified: on a 50-name universe some name is almost always up more than 2%, so
the short branch never fired and `control_momentum_10d_ls` and
`control_momentum_10d_long` produced identical results under two names.
"""

import pandas as pd

from alphabench.agents.rules import Momentum10
from alphabench.market import synthetic_prices
from alphabench.replay import decision_dates, run_replay
from alphabench.universe import ALL_SYMBOLS

COLS = ["symbol", "price", "ret_10d"]


def _payload(rows: list[tuple]) -> dict:
    """Minimal cycle payload: a universe table and an empty book."""
    return {
        "universe": {"columns": COLS, "rows": [list(r) for r in rows]},
        "portfolio": {"positions": [], "cash": 10_000.0, "equity": 10_000.0},
        "detail": {},
    }


def test_short_fires_when_it_is_the_strongest_signal():
    """A long qualifies (+3%) but a short is the bigger absolute move (-12%)."""
    payload = _payload([("AAA", 100.0, 0.03), ("BBB", 50.0, -0.12)])
    item = Momentum10(allow_short=True).decide(payload).decisions[0]
    assert item.action == "open_short" and item.symbol == "BBB"
    assert item.invalidation_price > 50.0            # stop sits above a short entry


def test_long_only_variant_takes_the_long_instead():
    payload = _payload([("AAA", 100.0, 0.03), ("BBB", 50.0, -0.12)])
    item = Momentum10(allow_short=False).decide(payload).decisions[0]
    assert item.action == "open_long" and item.symbol == "AAA"
    assert item.invalidation_price < 100.0


def test_long_wins_when_it_is_the_bigger_move():
    payload = _payload([("AAA", 100.0, 0.20), ("BBB", 50.0, -0.06)])
    assert Momentum10(allow_short=True).decide(payload).decisions[0].action == "open_long"


def test_short_gate_is_stricter_than_the_long_gate():
    """-3% clears no gate; +3% clears the long gate."""
    quiet = Momentum10(allow_short=True).decide(_payload([("BBB", 50.0, -0.03)]))
    assert [i.action for i in quiet.decisions] == ["hold"]
    assert Momentum10(allow_short=True).decide(_payload([("AAA", 100.0, 0.03)])).decisions[0].action == "open_long"


def test_the_two_controls_are_not_the_same_agent_over_a_replay():
    md = synthetic_prices(ALL_SYMBOLS, "2023-09-01", "2024-06-30")
    dds = decision_dates(md, "2024-03-01")[:40]
    ls, lo = Momentum10(allow_short=True), Momentum10(allow_short=False)
    res = run_replay(md, [ls, lo], dds[0], dds[-1], progress=False)
    assert ls.name != lo.name
    a, b = res[ls.name]["equity"], res[lo.name]["equity"]
    assert not a.equals(b), "long/short and long-only controls produced identical equity curves"
    assert (res[ls.name]["trades"]["side"] == -1).any(), "the long/short control never opened a short"
