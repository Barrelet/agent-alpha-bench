import numpy as np
import pandas as pd

from alphabench.agents.rules import RandomAgent
from alphabench.market import synthetic_prices
from alphabench.null import run_null, percentile, null_table, null_summary
from alphabench.prompt import build_payload
from alphabench.engine import Portfolio
from alphabench.replay import decision_dates, run_replay
from alphabench.universe import ALL_SYMBOLS

md = synthetic_prices(ALL_SYMBOLS, "2023-09-01", "2024-06-30")
dds = decision_dates(md, "2024-03-01")[:15]


def test_random_long_only_never_shorts():
    a = RandomAgent(seed=3, long_only=True, p_open=1.0)
    assert a.name == "control_random_long_s3" and a.needs_detail is False
    res = run_replay(md, [a], dds[0], dds[-1], progress=False)
    trades = res[a.name]["trades"]
    assert not trades.empty and (trades["side"] == 1).all()


def test_run_null_shape_cache_and_percentiles(tmp_path):
    df = run_null(md, dds[0], dds[-1], n_seeds=6, long_only=False, cache_dir=tmp_path, batch=4)
    assert list(df["seed"]) == list(range(6)) and (df["long_only"] == False).all()
    assert df["total_return"].notna().all()
    cached = run_null(md, dds[0], dds[-1], n_seeds=6, long_only=False, cache_dir=tmp_path)
    pd.testing.assert_frame_equal(df, cached)
    assert percentile(df["total_return"], df["total_return"].max()) == 100.0
    assert percentile(df["total_return"], -1.0) == 0.0
    obs = pd.DataFrame({"total_return": [df["total_return"].median()], "sharpe": [0.0]}, index=["x"])
    tab = null_table(obs, df)
    assert 0 < tab.loc["x", "total_return_pct_ls"] <= 100 and "total_return_pct_long" not in tab.columns
    assert null_summary(df).loc["long/short", "n_seeds"] == 6


def test_run_null_is_reproducible():
    a = run_null(md, dds[0], dds[-1], n_seeds=3, batch=3)
    b = run_null(md, dds[0], dds[-1], n_seeds=3, batch=1)     # batch size must not change results
    pd.testing.assert_frame_equal(a, b)


def test_random_from_universe_picks_outside_screened():
    a = RandomAgent(seed=5, from_universe=True, p_open=1.0)
    assert a.name == "control_random_univ_s5"
    res = run_replay(md, [a], dds[0], dds[-1], progress=False)
    assert len(set(res[a.name]["fills"]["symbol"])) >= 3
    df = run_null(md, dds[0], dds[-1], n_seeds=3, from_universe=True, batch=3)
    assert df["from_universe"].all() and null_summary(df).index.tolist() == ["long/short, whole universe"]


def test_universe_candidate_mode():
    from alphabench.prompt import build_base, build_payload, render_text, mandate, FEATURE_COLS
    from alphabench.engine import Portfolio
    t = dds[3]; md_t = md.asof(t)
    base = build_base(md_t, t, candidates="universe")
    pl = build_payload(md_t, Portfolio(10_000).snapshot(md_t.close.iloc[-1]), t, base=base)
    assert "screened" not in pl and pl["detail"] == {} and len(pl["_screened"]) == 50
    assert set(FEATURE_COLS) <= set(pl["universe"]["columns"])
    assert "_screened" not in render_text(pl)
    from alphabench.agents.llm import LLMAgent, MockBackend
    _, user = LLMAgent(name="t", backend=MockBackend(lambda s, u: "{}"), model="m", cache_dir="/tmp/ttp_test_cache").messages(pl)
    assert "_screened" not in user and '"candidates":"all names' in user
    assert "every one of them a candidate" in mandate("v3", "universe") and "screened = today" in mandate("v3", "screened")
    # rule agents draw from all 50 names
    res = run_replay(md, [RandomAgent(seed=1, p_open=1.0)], dds[0], dds[-1], progress=False, payload_kwargs={"candidates": "universe"})
    assert len(set(res["control_random_s1"]["fills"]["symbol"])) >= 3
    # and the null runner tags such seeds as whole-universe
    df = run_null(md, dds[0], dds[-1], n_seeds=2, batch=2, payload_kwargs={"candidates": "universe"})
    assert df["from_universe"].all()


def test_pyramiding_random_and_concentration():
    from alphabench.metrics import concentration
    a = RandomAgent(seed=2, p_add=1.0, p_open=1.0, p_close=0.0)
    assert a.name == "control_random_pyr_s2"
    res = run_replay(md, [a], dds[0], dds[-1], progress=False)
    r = res[a.name]
    assert (r["fills"]["action"] == "add").sum() >= 1
    cc = concentration(r["fills"], r["trades"], r["equity"])
    assert 0 < cc["max_position_weight"] <= 1.2 and cc["n_adds"] >= 1
    df = run_null(md, dds[0], dds[-1], n_seeds=2, batch=2, pyramid=True)
    assert df["pyramid"].all() and "max_position_weight" in df.columns
    assert null_summary(df).index.tolist() == ["long/short, pyramiding"]


def test_cap_flows_through_replay_null_and_prompt():
    from alphabench.prompt import mandate
    from alphabench.metrics import concentration
    assert "25%" in mandate("v1", max_position_weight=0.25) and "25%" not in mandate("v1")
    a = RandomAgent(seed=2, p_add=1.0, p_open=1.0, p_close=0.0)
    res = run_replay(md, [a], dds[0], dds[-1], progress=False, max_position_weight=0.25)
    cc = concentration(res[a.name]["fills"], res[a.name]["trades"], res[a.name]["equity"])
    assert cc["max_position_weight"] <= 0.30
    df = run_null(md, dds[0], dds[-1], n_seeds=2, batch=2, pyramid=True, max_position_weight=0.25)
    assert (df["max_position_weight"] <= 0.30).all()
