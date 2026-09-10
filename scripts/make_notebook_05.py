"""Generates notebook 05 — cross-sectional forecasts, scored before any portfolio.

    python scripts/make_notebook_05.py      -> notebooks/05_forecast_first.ipynb

Writes a notebook with NO outputs; regenerate first, then Run All. Every model call is
cached under data/llm_cache/<forecaster>/, so a re-run is free and an interrupted run resumes.
"""
import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

md("""# 05 — Forecast first: every name, every day, scored before any trade

Notebooks 03 and 04 judged each agent by one trade a day: over 60 days that is 4 to 17 trades per agent, which cannot separate skill from luck. This notebook changes the question. Every day the forecaster is shown the same universe table as before and must state, **for each of the 50 names, the probability that it beats the equal-weight average of the 50 over the next 10 trading days**. Fifty forecasts a day, ~3,000 per window, scored directly:

- **rank IC** — Spearman correlation between the forecast and the realised excess return, per day; its mean and a Newey-West t-statistic (forecasts on overlapping horizons are autocorrelated, so the plain t-stat would flatter everyone)
- **hit rate** — of the names given more than 0.5, how many beat the average
- **Brier score** — mean squared error of the probability; 0.25 is what "0.5 for everything" scores, so anything above 0.25 is worse than saying nothing
- **calibration** — among names given ~0.7, did ~70% beat the average?
- **quintile spread** — realised excess return of the top fifth of names by forecast minus the bottom fifth, in return units

Only then is a portfolio built from the forecasts — mechanically, top-5 long and bottom-5 short, held 10 days, fees charged — so that signal quality and implementation can be told apart: *was the forecast bad, or was the trading of it bad?* This is how a desk evaluates any signal source.

Controls: 10-day momentum (rank of `ret_10d`), its reverse, random probabilities, and the constant 0.5. The last one matters: it is perfectly calibrated and has no information, and it sets the Brier floor.""")

code("""import sys, json, time, logging
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))
logging.basicConfig(level=logging.WARNING, format="%(message)s")
pd.set_option("display.width", 180, "display.max_columns", 40)

from alphabench.universe import TICKERS, ALL_SYMBOLS, BENCHMARK
from alphabench.market import load_prices
from alphabench.agents import OllamaBackend
from alphabench.replay import decision_dates
from alphabench.metrics import equal_weight_index
from alphabench.forecast import (HORIZON, LLMForecaster, MomentumForecaster, RandomForecaster, ConstantForecaster,
                                 forward_targets, run_forecasts, load_forecasts, evaluate, scorecard, calibration_table,
                                 daily_ic, quintile_spread, forecast_portfolio, attribution, newey_west_tstat)

CACHE, LLM_CACHE = ROOT / "data" / "prices.parquet", ROOT / "data" / "llm_cache"
RESULTS = ROOT / "data" / "results" / "forecast_h10"
md = load_prices(ALL_SYMBOLS, "2023-09-01", None, CACHE)

# ---- the same fixed window as notebooks 03/04, so the two experiments describe the same summer
WINDOW_START, N_CYCLES = "2026-06-01", 60
dates = decision_dates(md, WINDOW_START)[:N_CYCLES]
targets = forward_targets(md, dates, HORIZON)
n_scored = int(targets["beat"].notna().any(axis=1).sum())
print(f"{len(dates)} forecast dates: {dates[0].date()} → {dates[-1].date()}  |  horizon {HORIZON} trading days  |  "
      f"{n_scored} dates have a complete target (prices end {md.last_date.date()})")
if n_scored < len(dates):
    print(f"→ {len(dates) - n_scored} dates cannot be scored yet: refresh prices with load_prices(..., refresh=True) once "
          f"{HORIZON} trading days have passed since {dates[-1].date()}.")""")

md("""## 1. Controls (seconds)

Momentum and reversal turn a cross-sectional rank into probabilities between 0.2 and 0.8 — an ordering with modest confidence. Three random forecasters give a first feel for noise; section 5 draws the full null distribution.""")

code("""controls = [MomentumForecaster("ret_10d"), MomentumForecaster("ret_10d", reverse=True),
            RandomForecaster(0), RandomForecaster(1), RandomForecaster(2), ConstantForecaster()]
fc_controls = run_forecasts(md, controls, dates, out_dir=RESULTS / "controls")
scorecard(fc_controls, targets).round(3)""")

md("""## 2. The model (long — leave it running)

One call per day: the universe table for all 50 names plus SPY candles in (~8-10k tokens), a JSON list of 50 probabilities out. Nothing is traded at this stage, so there is no equity to print; instead each line shows how many names came back, the mean probability (should sit near 0.5), that day's rank IC against the already-known outcome, and the running mean IC — the number section 3 will report. Ollama's structured output restricts symbols to the universe. A call that returns fewer than half the names is sent back once with the error; whatever comes back is kept and the missing names stay NaN (they are simply not scored). Cached after the first run.""")

code("""backend = OllamaBackend(think=False)
MODEL = "qwen3:8b"
assert any(m.startswith(MODEL) for m in backend.list_models()), f"{MODEL} not pulled: ollama pull {MODEL}"

llm = LLMForecaster(f"{MODEL.replace(':', '-')}_fc_h{HORIZON}", backend, MODEL, horizon=HORIZON, cache_dir=LLM_CACHE)

def on_date(i, n, t, elapsed, frames):
    r = llm.records[-1] if llm.records else None
    tag = "cached" if (r and r.cached) else f"{(r.latency_s if r else 0):.0f}s"
    p = frames[llm.name].loc[:t]
    ic = daily_ic(p, targets["excess"])                     # the window is in the past, so each day's target is already known
    today = f"{ic.iloc[-1]:+.2f}" if ic.notna().any() and ic.index[-1] == t and ic.iloc[-1] == ic.iloc[-1] else "  n/a"
    print(f"  {i:3d}/{n} {t.date()}  {tag:>7}  names {r.n_valid if r else 0:2d}  mean p {p.loc[t].mean():.2f}  "
          f"IC today {today}  running mean IC {ic.mean():+.3f}  elapsed {elapsed/60:4.0f} min  eta {(elapsed/i)*(n-i)/60:4.0f} min", flush=True)

fc_llm = run_forecasts(md, [llm], dates, out_dir=RESULTS / "llm", on_date=on_date)
forecasts = {**fc_llm, **fc_controls}
pd.Series(llm.stats()).to_frame("qwen3-8b").T""")

md("""## 3. Scorecard

`ic_tstat_nw` is the number to read first: above ~2 in absolute value and the ordering is unlikely to be noise; the sign tells you whether the model ranks names the right way round or backwards. `brier_skill` is 1 − Brier/0.25: positive means better than saying 0.5 for everything, negative means worse. `share_at_0.5` and `share_confident` describe how the probabilities are used; a model that never leaves 0.5 has abstained, a model that is always confident is the notebook-04 finding again.""")

code("""sc = scorecard(forecasts, targets)
sc.to_csv(RESULTS / "scorecard.csv")
cols = ["n_forecasts", "n_dates_scored", "coverage", "ic_mean", "ic_tstat_nw", "ic_share_positive", "hit_rate", "brier", "brier_skill",
        "spread_mean", "spread_tstat_nw", "p_mean", "p_std", "share_at_0.5", "share_confident"]
sc[cols].style.format({"coverage": "{:.0%}", "ic_mean": "{:+.3f}", "ic_tstat_nw": "{:+.2f}", "ic_share_positive": "{:.0%}", "hit_rate": "{:.1%}",
                       "brier": "{:.4f}", "brier_skill": "{:+.3f}", "spread_mean": "{:+.2%}", "spread_tstat_nw": "{:+.2f}",
                       "p_mean": "{:.3f}", "p_std": "{:.3f}", "share_at_0.5": "{:.0%}", "share_confident": "{:.0%}"}, na_rep="—")""")

code("""PALETTE = {"llm": "#2a78d6", "momentum": "#eb6834", "reversal": "#1baf7a", "random": "#b0b0b0", "constant": "#7a7a7a"}
def colour(name):
    for k, c in PALETTE.items():
        if k in name: return c
    return PALETTE["llm"]

fig, ax = plt.subplots(figsize=(12, 4.5))
for name, p in forecasts.items():
    if "random_1" in name or "random_2" in name or "constant" in name: continue
    ic = daily_ic(p, targets["excess"]).dropna()
    ax.plot(ic.index, ic.rolling(5, min_periods=1).mean(), lw=2 if "fc_" in name else 1.3, color=colour(name),
            ls="-" if "fc_" in name else "--", label=f"{name}  (mean {ic.mean():+.3f})")
ax.axhline(0, color="#ccc", lw=0.8)
ax.set_title(f"Daily rank IC, 5-day rolling mean — {HORIZON}-day horizon", loc="left"); ax.set_ylabel("Spearman IC")
ax.legend(frameon=False, fontsize=8); ax.grid(axis="y", color="#e5e5e5"); ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.show()""")

md("""## 4. Calibration and how the probabilities are used

Left: stated probability against the realised beat rate, per bucket (the diagonal is perfect calibration; the constant control sits on it at 0.5 by definition). Right: the distribution of the probabilities the model wrote down.""")

code("""fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5))
a1.plot([0, 1], [0, 1], color="#ccc", lw=1)
for name, p in forecasts.items():
    if "random" in name or "constant" in name: continue
    cal = calibration_table(p, targets["beat"])
    a1.plot(cal["stated"], cal["realised"], "o-", color=colour(name), label=name, lw=2 if "fc_" in name else 1.2)
    for s, r, n in zip(cal["stated"], cal["realised"], cal["n"]):
        if "fc_" in name: a1.annotate(f"n={n}", (s, r), textcoords="offset points", xytext=(4, -10), fontsize=7, color="#666")
a1.set_xlabel("stated p_beat (bucket mean)"); a1.set_ylabel("realised share that beat the universe"); a1.set_xlim(0, 1); a1.set_ylim(0, 1)
a1.set_title("Calibration", loc="left"); a1.legend(frameon=False, fontsize=8); a1.spines[["top", "right"]].set_visible(False)
for name, p in fc_llm.items():
    a2.hist(p.stack(future_stack=True).dropna(), bins=np.linspace(0, 1, 21), color=colour(name), alpha=0.8, label=name)
a2.axvline(0.5, color="#666", lw=1, ls="--"); a2.set_xlabel("p_beat"); a2.set_ylabel("count"); a2.set_title("What the model wrote down", loc="left")
a2.legend(frameon=False, fontsize=8); a2.spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.show()
calibration_table(next(iter(fc_llm.values())), targets["beat"]).round(3)""")

md("""## 5. Against a thousand random forecasters

The same 60 days, the same 50 names, probabilities drawn at random. Where does the model's mean IC, and its quintile spread, sit in that distribution? "Beat 95 in 100" is the bar for taking the ordering seriously.""")

code("""N_NULL = 1000
null_ic, null_spread = [], []
rng = np.random.default_rng(123)
for _ in range(N_NULL):
    r = pd.DataFrame(rng.uniform(0, 1, (len(dates), len(TICKERS))), index=dates, columns=TICKERS)
    null_ic.append(daily_ic(r, targets["excess"]).mean()); null_spread.append(quintile_spread(r, targets["excess"]).mean())
null_ic, null_spread = pd.Series(null_ic), pd.Series(null_spread)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
for ax, null, key, label in [(axes[0], null_ic, "ic_mean", "mean daily rank IC"), (axes[1], null_spread, "spread_mean", "mean quintile spread")]:
    ax.hist(null, bins=40, color="#d8d8d8")
    for name in forecasts:
        if "random" in name or "constant" in name: continue
        v = sc.loc[name, key]; pct = (null < v).mean() * 100
        ax.axvline(v, color=colour(name), lw=2 if "fc_" in name else 1.2, ls="-" if "fc_" in name else "--")
        ax.text(v, ax.get_ylim()[1] * (0.92 if "fc_" in name else 0.8), f" {name}: beat {pct:.0f} in 100", color=colour(name), fontsize=8)
    ax.set_title(f"{N_NULL} random forecasters — {label}", loc="left"); ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.show()""")

md("""## 6. From forecast to portfolio — where does the P&L go?

The same forecasts, traded mechanically: each day the top 5 names by probability go long and the bottom 5 short, equal-weighted, entered at the next close and held 10 days; ten such cohorts run at once, each with a tenth of the capital; 0.1% per side on every entry and exit. No discretion, no sizing decisions, no stops — the only input is the ordering. Read the table left to right: forecast quality, then the gross return the ordering produced, then what fees left of it. A good spread with a poor net return is an implementation problem; a poor spread is a signal problem, and no trading rule will fix it.""")

code("""first, last = dates[0], md.dates[min(md.dates.searchsorted(dates[-1]) + 1 + HORIZON, len(md.dates) - 1)]
ew = equal_weight_index(md.close[TICKERS], start=first).loc[:last]
attr = attribution(forecasts, targets, md, HORIZON, k=5, long_only=False, benchmark=ew)
attr.to_csv(RESULTS / "attribution.csv")
attr.style.format({"ic_mean": "{:+.3f}", "ic_tstat_nw": "{:+.2f}", "hit_rate": "{:.1%}", "brier": "{:.4f}", "spread_mean": "{:+.2%}", "spread_tstat_nw": "{:+.2f}",
                   "portfolio_gross": "{:+.1%}", "portfolio_net": "{:+.1%}", "sharpe_net": "{:.2f}", "max_drawdown_net": "{:+.1%}", "fees_pct": "{:.2%}",
                   "excess_vs_benchmark": "{:+.1%}"}, na_rep="—")""")

code("""fig, ax = plt.subplots(figsize=(12, 5))
for name, p in forecasts.items():
    if "random_1" in name or "random_2" in name or "constant" in name: continue
    pf = forecast_portfolio(p, md, HORIZON, k=5)
    eq = pf["equity"] / pf["equity"].iloc[0] * 100
    ax.plot(eq.index, eq, color=colour(name), lw=2.2 if "fc_" in name else 1.3, ls="-" if "fc_" in name else "--", label=f"{name}  {eq.iloc[-1]-100:+.1f}%")
ewi = ew / ew.iloc[0] * 100
ax.plot(ewi.index, ewi, color="#333", lw=1.2, ls=":", label=f"equal-weight universe (long only)  {ewi.iloc[-1]-100:+.1f}%")
ax.axhline(100, color="#ccc", lw=0.8)
ax.set_title("Long top-5 / short bottom-5 by forecast, 10-day cohorts, net of fees — indexed to 100", loc="left")
ax.legend(frameon=False, fontsize=8); ax.grid(axis="y", color="#e5e5e5"); ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.show()""")

md("""## 7. What this shows, and what it does not

Fill in after the run. The questions this notebook can now answer, which notebook 03 could not:

1. **Does the model order names better than chance?** — the IC t-stat and the percentile against 1,000 random forecasters (sections 3 and 5).
2. **Is its confidence information?** — calibration by bucket, Brier against the 0.25 floor, and the shape of the probability distribution (section 4). Three thousand forecasts make this quantitative; notebook 04 had fifty trades.
3. **If it lost money in notebook 03, was it the signal or the trading?** — the spread against the portfolio return (section 6). A model with a positive, significant spread that lost money in the one-trade-a-day game had a sizing and discipline problem, not a forecasting one; a model with no spread never had anything to trade.

What it still does not show: one summer, one 8B model, one prompt. Sixty overlapping 10-day windows in a calm market are about six independent observations of the horizon; the Newey-West correction is honest about that, and a year of data is the fix. The forecast contract itself (probability of beating the equal-weight mean) is one of several possible; a model that is good at direction but poor at relative ranking would look bad here and might not elsewhere.""")

nb["cells"] = cells
out = Path(__file__).resolve().parent.parent / "notebooks" / "05_forecast_first.ipynb"
nbf.write(nb, out)
print("wrote", out)
