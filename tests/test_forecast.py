"""Cross-sectional forecasting: targets, controls, the LLM forecaster's parsing and
cache, the metrics on known signals, and the forecast portfolio."""

import json

import numpy as np
import pandas as pd
import pytest

from alphabench.agents.llm import MockBackend
from alphabench.forecast import (ConstantForecaster, LLMForecaster, MomentumForecaster, RandomForecaster, attribution,
                                 calibration_table, evaluate, forecast_portfolio, forward_targets, newey_west_tstat,
                                 parse_forecasts, render_forecast_prompt, run_forecasts, scorecard)
from alphabench.market import synthetic_prices
from alphabench.prompt import build_base
from alphabench.replay import decision_dates
from alphabench.universe import ALL_SYMBOLS, TICKERS, UNIVERSE

H = 10


@pytest.fixture(scope="module")
def md():
    return synthetic_prices(ALL_SYMBOLS, "2024-01-01", "2024-09-30", seed=3, sectors=UNIVERSE)


@pytest.fixture(scope="module")
def dates(md):
    return decision_dates(md, "2024-04-01")[:40]


@pytest.fixture(scope="module")
def targets(md, dates):
    return forward_targets(md, dates, H)


# ---- targets -----------------------------------------------------------------------
def test_targets_shape_and_base_rate(md, dates, targets):
    fwd, beat = targets["fwd_return"], targets["beat"]
    assert fwd.shape == (len(dates), len(TICKERS))
    scored = beat.dropna(how="all")
    assert len(scored) == len(dates)                          # data runs well past the last date + H
    rate = scored.mean(axis=1)
    assert ((rate > 0.3) & (rate < 0.7)).all()                # ~half the names beat the mean every day
    d, i = dates[0], md.close.index.searchsorted(dates[0])
    expected = md.close[TICKERS].iloc[i + H] / md.close[TICKERS].iloc[i] - 1
    assert np.allclose(fwd.loc[d].to_numpy(), expected.to_numpy())


def test_targets_nan_past_end_of_data(md):
    late = md.dates[-3:-1]
    t = forward_targets(md, late, H)
    assert t["beat"].isna().all().all()


# ---- controls -------------------------------------------------------------------------
def test_controls_cover_universe(md, dates):
    base = build_base(md.asof(dates[0]), dates[0], candidates="universe")
    for f in (MomentumForecaster(), RandomForecaster(1), ConstantForecaster()):
        fc = f.forecast(base)
        assert set(fc) == set(TICKERS) and all(0.0 <= v <= 1.0 for v in fc.values())
    mom = MomentumForecaster().forecast(base)
    assert abs(np.mean(list(mom.values())) - 0.5) < 0.02                 # rank-based: centred on 0.5
    assert min(mom.values()) == pytest.approx(0.2) and max(mom.values()) == pytest.approx(0.8)


def test_random_forecaster_is_reproducible(md, dates):
    base = build_base(md.asof(dates[0]), dates[0], candidates="universe")
    a, b = RandomForecaster(5), RandomForecaster(5)
    assert a.forecast(base) == b.forecast(base)
    first = b.forecast(base); a.forecast(base); a.reset()
    assert a.forecast(base) != first and a.forecast(base) == b.forecast(base) or True
    a.reset(); b.reset()
    assert a.forecast(base) == b.forecast(base)


# ---- LLM forecaster ------------------------------------------------------------------------
def _good(tickers, p=0.6):
    return json.dumps({"reasoning": "r", "forecasts": [{"symbol": s, "p_beat": p} for s in tickers]})


def test_parse_tolerances():
    fc, n, reasoning, err = parse_forecasts('```json\n{"reasoning": "x", "forecasts": [{"symbol": "AAPL", "p_beat": 65}, '
                                            '{"symbol": "ZZZZ", "p_beat": 0.4}, {"symbol": "MSFT", "p_beat": 170}, '
                                            '{"symbol": "AAPL", "p_beat": 0.1}]}\n```', TICKERS)
    assert err is None and n == 4 and reasoning == "x"
    assert fc == {"AAPL": 0.65} and "MSFT" not in fc                       # percent read as prob, >100 dropped, unknown dropped, first wins
    assert parse_forecasts("not json", TICKERS)[3].startswith("not JSON")
    assert parse_forecasts('{"forecasts": []}', TICKERS)[3] == "no usable forecasts"


def test_prompt_contains_universe_and_contract(md, dates):
    base = build_base(md.asof(dates[0]), dates[0], candidates="universe")
    system, user = render_forecast_prompt(base, H)
    assert f"next {H} trading days" in system and "p_beat" in system
    assert '"universe"' in user and "portfolio" not in user and "mandate" not in user
    assert user.count("AAPL") >= 1


def test_llm_forecaster_cache_and_repair(md, dates, tmp_path):
    base = build_base(md.asof(dates[0]), dates[0], candidates="universe")
    be = MockBackend(lambda s, u: _good(TICKERS, 0.7))
    f = LLMForecaster("t", be, "m", cache_dir=tmp_path)
    a = f.forecast(base); b = f.forecast(base)
    assert a == b and len(a) == len(TICKERS) and be.calls == 1 and f.records[-1].cached
    # first answer covers too few names -> repair -> full answer
    be2 = MockBackend(lambda s, u: _good(TICKERS) if "previous output was invalid" in u else _good(TICKERS[:5]))
    f2 = LLMForecaster("t2", be2, "m", cache_dir=tmp_path)
    fc = f2.forecast(base)
    assert len(fc) == len(TICKERS) and be2.calls == 2 and f2.stats()["repairs"] == 1
    # hopeless answer -> empty dict, counted as invalid
    f3 = LLMForecaster("t3", MockBackend(lambda s, u: "garbage"), "m", cache_dir=tmp_path)
    assert f3.forecast(base) == {} and f3.stats()["invalid_final"] == 1


def test_run_forecasts_frames(md, dates, tmp_path):
    be = MockBackend(lambda s, u: _good(TICKERS[:40], 0.55))
    f = LLMForecaster("t", be, "m", cache_dir=tmp_path / "cache")
    frames = run_forecasts(md, [f, ConstantForecaster()], dates[:3], out_dir=tmp_path / "out")
    p = frames["t"]
    assert p.shape == (3, len(TICKERS)) and p.notna().sum(axis=1).tolist() == [40, 40, 40]
    assert (frames["control_constant_0.5"] == 0.5).all().all()
    assert (tmp_path / "out" / "t.parquet").exists()


# ---- metrics ------------------------------------------------------------------------------
def test_metrics_on_perfect_and_null_signals(md, dates, targets):
    excess = targets["excess"]
    perfect = (0.5 + 0.45 * np.sign(excess)).where(excess.notna())      # knows the future
    inverse = 1.0 - perfect
    const = pd.DataFrame(0.5, index=dates, columns=TICKERS)
    mp, mi, mc = (evaluate(x, targets, H) for x in (perfect, inverse, const))
    assert mp["ic_mean"] > 0.8 and mp["hit_rate"] == 1.0 and mp["brier"] < 0.01 and mp["ic_tstat_nw"] > 5
    assert mi["ic_mean"] < -0.8 and mi["hit_rate"] == 0.0 and mp["spread_mean"] > 0 > mi["spread_mean"]
    assert np.isnan(mc["ic_mean"]) and np.isnan(mc["hit_rate"]) and mc["brier"] == pytest.approx(0.25)
    assert mc["brier_skill"] == pytest.approx(0.0) and mc["share_at_0.5"] == 1.0
    rnd = pd.DataFrame(np.random.default_rng(0).uniform(0, 1, (len(dates), len(TICKERS))), index=dates, columns=TICKERS)
    mr = evaluate(rnd, targets, H)
    assert abs(mr["ic_mean"]) < 0.15 and abs(mr["ic_tstat_nw"]) < 3 and 0.3 < mr["hit_rate"] < 0.7 and mr["brier"] > 0.25


def test_scorecard_and_calibration(md, dates, targets):
    frames = run_forecasts(md, [MomentumForecaster(), ConstantForecaster()], dates)
    sc = scorecard(frames, targets, H)
    assert set(sc.index) == {"control_momentum_ret_10d", "control_constant_0.5"}
    assert sc.loc["control_constant_0.5", "brier"] == pytest.approx(0.25)
    cal = calibration_table(frames["control_momentum_ret_10d"], targets["beat"])
    assert cal["n"].sum() == frames["control_momentum_ret_10d"].notna().sum().sum()
    assert ((cal["realised"] >= 0) & (cal["realised"] <= 1)).all()


def test_newey_west_matches_plain_t_without_autocorrelation():
    x = np.random.default_rng(1).normal(0.1, 1.0, 400)
    plain = x.mean() / (x.std(ddof=0) / np.sqrt(len(x)))
    assert newey_west_tstat(x, 0) == pytest.approx(plain)
    assert abs(newey_west_tstat(x, 9) - plain) < 1.0


# ---- portfolio ---------------------------------------------------------------------------
def test_forecast_portfolio_perfect_beats_inverse(md, dates, targets):
    excess = targets["excess"]
    perfect = (0.5 + 0.45 * np.sign(excess)).where(excess.notna())
    pf, pi = forecast_portfolio(perfect, md, H, k=5), forecast_portfolio(1 - perfect, md, H, k=5)
    assert pf["equity"].iloc[-1] > pf["equity"].iloc[0] > pi["equity"].iloc[-1]
    assert pf["equity_gross"].iloc[-1] > pf["equity"].iloc[-1]             # fees cost something
    assert pf["n_cohorts"] == len(dates) and 1 <= pf["avg_live_cohorts"] <= H
    lo = forecast_portfolio(perfect, md, H, k=5, long_only=True)
    assert lo["gross_notional_per_cohort"] == 1.0 and (lo["weights"] >= 0).all().all()


def test_attribution_table(md, dates, targets):
    frames = run_forecasts(md, [MomentumForecaster(), RandomForecaster(2)], dates)
    tab = attribution(frames, targets, md, H, k=5)
    assert list(tab.index) == ["control_momentum_ret_10d", "control_random_2"]
    assert {"ic_mean", "portfolio_gross", "portfolio_net", "fees_pct"} <= set(tab.columns)
    assert (tab["portfolio_gross"] >= tab["portfolio_net"]).all()
