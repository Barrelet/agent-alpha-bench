import json

import pandas as pd

from alphabench.agents.llm import LLMAgent, MockBackend, _strip_fences
from alphabench.agents.rules import Momentum10
from alphabench.market import synthetic_prices
from alphabench.replay import run_replay
from alphabench.universe import ALL_SYMBOLS, UNIVERSE

GOOD = json.dumps({"reasoning": "r", "market_context": "m", "decisions": [
    {"symbol": "AAPL", "action": "open_long", "percent_of_equity": 20, "confidence": 0.9,
     "thesis": "t", "invalidation": "i", "invalidation_price": 1.0}]})
BAD = '{"decisions": [{"symbol": "AAPL", "action": "open_long"}]}'
PAYLOAD = {"decision_date": "2024-01-02", "mandate": "M", "screened": [], "detail": {}, "portfolio": {"positions": []}}


def test_valid_response_and_cache(tmp_path):
    be = MockBackend(lambda s, u: GOOD)
    a = LLMAgent("t", be, "m", cache_dir=tmp_path)
    d1 = a.decide(PAYLOAD); d2 = a.decide(PAYLOAD)
    assert d1.decisions[0].symbol == "AAPL" and d2 == d1
    assert be.calls == 1 and a.records[-1].cached is True
    assert a.stats()["calls"] == 2 and a.stats()["cached"] == 1


def test_repair_then_ok(tmp_path):
    be = MockBackend(lambda s, u: GOOD if "previous output was invalid" in u else BAD)
    a = LLMAgent("t", be, "m", cache_dir=tmp_path)
    d = a.decide(PAYLOAD)
    assert d.decisions[0].action == "open_long" and be.calls == 2
    assert [r.attempt for r in a.records] == [1, 2] and a.stats()["repairs"] == 1


def test_double_failure_is_hold(tmp_path):
    a = LLMAgent("t", MockBackend(lambda s, u: BAD), "m", cache_dir=tmp_path)
    d = a.decide(PAYLOAD)
    assert d.decisions[0].action == "hold" and a.stats()["invalid_final"] == 1


def test_llm_schema_is_strict():
    from alphabench.schema import json_schema, parse_llm_decision, MAX_DECISIONS
    sch = json_schema(for_llm=True)
    assert sch["properties"]["decisions"]["maxItems"] == MAX_DECISIONS
    assert set(sch["$defs"]["LLMTradeItem"]["required"]) >= {"symbol", "action", "percent_of_equity", "confidence", "thesis", "invalidation", "invalidation_price"}
    assert "hold" not in str(sch["$defs"]["LLMTradeItem"]["properties"]["action"])
    d, err = parse_llm_decision({"reasoning": "nothing", "market_context": "", "decisions": []})
    assert err is None and d.decisions == []
    d, err = parse_llm_decision({"reasoning": "x", "market_context": "", "decisions": [
        {"symbol": "AAPL", "action": "add", "percent_of_equity": 0.05, "invalidation": "i", "invalidation_price": 330.0}]})
    assert d is None and "confidence" in err and "thesis" in err
    d, err = parse_llm_decision({"reasoning": "x", "market_context": "", "decisions": [{"symbol": "AAPL", "action": "close"}]})
    assert err is None and d.decisions[0].action == "close"


def test_strip_fences():
    assert _strip_fences('```json\n{"a": 1}\n```') == '{"a": 1}'
    assert _strip_fences('Sure! {"a": 1} done') == '{"a": 1}'


def test_llm_agent_in_replay(tmp_path):
    md = synthetic_prices(ALL_SYMBOLS, "2023-10-01", "2024-02-15", sectors=UNIVERSE)

    def responder(system, user):  # buy the top screened name, 10%, stop 5% below
        p = json.loads(user.split("\n", 1)[1])
        sym = p["screened"][0]["symbol"]; px = p["detail"][sym]["daily"]["rows"][-1][3]
        held = {x["symbol"] for x in p["portfolio"]["positions"]}
        if sym in held:
            return json.dumps({"reasoning": "hold", "market_context": "", "decisions": []})
        return json.dumps({"reasoning": "buy", "market_context": "", "decisions": [{"symbol": sym, "action": "open_long", "percent_of_equity": 10,
                           "confidence": 0.9, "thesis": "t", "invalidation": "i", "invalidation_price": round(px * 0.95, 2)}]})

    a = LLMAgent("mock", MockBackend(responder), "m", cache_dir=tmp_path)
    res = run_replay(md, [a, Momentum10()], "2024-01-01", "2024-02-10", progress=False)
    assert len(res["mock"]["equity"]) > 5 and a.stats()["invalid_final"] == 0
    assert not res["mock"]["fills"].empty


def test_fraction_percent_is_normalised_and_wrong_side_repaired(tmp_path):
    from alphabench.agents.llm import truncation_warning
    payload = {"decision_date": "2024-01-02", "mandate": "M", "screened": [], "detail": {},
               "universe": [{"symbol": "AAPL", "price": 300.0}], "portfolio": {"positions": []}}
    def responder(system, user):
        inv = 250.0 if "must be BELOW" in user else 380.0   # wrong side first, fixed on repair
        return json.dumps({"reasoning": "r", "market_context": "m", "decisions": [
            {"symbol": "AAPL", "action": "open_long", "percent_of_equity": 0.15, "confidence": 0.9,
             "thesis": "t", "invalidation": "i", "invalidation_price": inv}]})
    a = LLMAgent("t", MockBackend(responder), "m", cache_dir=tmp_path)
    d = a.decide(payload)
    assert d.decisions[0].percent_of_equity == 15.0 and d.decisions[0].invalidation_price == 250.0
    assert a.stats()["repairs"] == 1 and a.stats()["normalised"] == 2
    assert "must be BELOW" in a.records[0].error
    assert truncation_warning(8194, 16384) and truncation_warning(7031) is None
