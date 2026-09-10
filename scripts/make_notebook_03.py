"""Generates the model / prompt comparison notebook.

    python scripts/make_notebook_03.py                 -> notebooks/03_model_comparison.ipynb
    python scripts/make_notebook_03.py --cap 0.25      -> notebooks/03_new_rule_model_comparison.ipynb

The flag adds one rule to the game: no single position above that share of equity,
enforced by the engine on every agent and stated in every prompt. The two notebooks
are otherwise the same experiment, which is why they come from one generator.

NOTE: this writes a notebook with NO outputs. Regenerating overwrites the executed
results in the .ipynb — regenerate first, then Run All. Every LLM call and every null
run is cached, so a re-run takes minutes rather than hours.
"""
import argparse
import nbformat as nbf
from pathlib import Path

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--cap", type=float, default=None,
                help="per-name position limit as a share of equity, e.g. 0.25 (default: none, the leaderboard rules)")
ap.add_argument("--out", type=Path, default=None, help="output path (default: derived from --cap)")
args = ap.parse_args()
CAP = args.cap
CAPPED = CAP is not None

#: placeholders substituted into every cell, so both variants come from one source.
#: the engine, the configs and the null all take max_position_weight=None to mean
#: "no cap", so the two notebooks differ only in this value and in the prose.
REPL = {
    "__CAP_VALUE__": "None" if not CAPPED else repr(CAP),
    "__CAP_FLAG__":  "" if not CAPPED else f" --cap {CAP}",
    "__CAP_TITLE__": "" if not CAPPED else f", {CAP:.0%} position limit",
}

def sub(s: str) -> str:
    for k, v in REPL.items():
        s = s.replace(k, v)
    return s

nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(sub(s)))
code = lambda s: cells.append(nbf.v4.new_code_cell(sub(s)))

HEADER_CAPPED = """# 03 (new rule) — Model and prompt comparison with a position limit

Same experiment as notebook 03, with one rule added to the game: **no single position may exceed 25% of equity**, enforced by the engine on every agent — the three prompts, the momentum controls and all the random traders — and stated in every prompt.

Why. In notebook 03, without a limit, the loosest prompt (v1, the leaderboard-style mandate) made +11% by adding to one name until it was 98% of the account; the disciplined prompt, which sized at 10-20%, made nothing. That comparison was between one prompt that could bet the account and two that could not, so it said nothing about the prompts themselves. A per-name limit is what any real account has; with it in place, the only thing that differs between the three runs is the prompt.

The original question stands: **does a better model or a better prompt help more?**"""

HEADER_PLAIN = """# 03 — Model and prompt comparison on a fixed window

The first real question of the harness: **does a better model or a better prompt help more?**

There is no per-name position limit here — these are the leaderboard's rules, under which a prompt with no sizing discipline may put the whole account into a single name. `03_new_rule_model_comparison` runs the identical experiment with a 25% cap; the two are meant to be read together."""

md((HEADER_CAPPED if CAPPED else HEADER_PLAIN) + """

Every configuration below faces the same decision cycles (the first cell prints how many and the exact window), the same rules, the same controls and the same benchmarks. Positions still open when the window ends are closed at the last close, fee charged, so trade statistics include them.

**Candidate mode.** `CANDIDATES = "universe"` (default) makes every one of the 50 names a candidate: the model gets one feature row per name and candles only for what it holds. `"screened"` is the original design — a daily top-5 funnel by dollar volume × recent move, with candles for those five. The first comparison in this project used the funnel, and 1,000 random traders under each mode (section 6) showed the funnel itself moved the median outcome by more than any model or prompt did; the whole-universe mode removes that confound. Results and caches of the two modes are kept apart. Configurations run one after another (leave it overnight); every call is cached, so an interrupted run resumes where it stopped and re-running the notebook is free.

What is compared, per configuration:
- **validity** — usable answers, repairs needed
- **rule-violation rate** — decisions the engine had to refuse (second open in a cycle, add to a loser, …)
- **stop-out rate** — trades closed by the invalidation monitor rather than by choice
- **win rate, return, Sharpe, max drawdown** — against the momentum and random controls and the SPY / equal-weight benchmarks
- **seconds per decision** — the cost of the experiment
- **percentile against 1,000 random traders** — the same rules, the same window, no information: skill or luck?

A few dozen cycles is the minimum for differences to mean anything; treat the result as a first reading, not a verdict.

**Reproducing this.** The decision window is pinned in the first cell, so the result does not move when the price cache is refreshed. Two things sit outside that guarantee: prices come from yfinance and are split- and dividend-adjusted retroactively, so a clone run months from now can see slightly different history for the same dates; and the cache in `data/llm_cache/` is what makes a re-run free — delete it and the models are queried again, and a local model is not bit-identical across versions. Cached, this notebook replays exactly; uncached, treat it as a re-run.""")

code("""import sys, json, time, logging
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))
logging.basicConfig(level=logging.WARNING, format="%(message)s")
pd.set_option("display.width", 180, "display.max_columns", 40, "display.max_colwidth", 100)

from alphabench.universe import TICKERS, ALL_SYMBOLS, UNIVERSE, BENCHMARK
from alphabench.market import load_prices
from alphabench.agents import LLMAgent, OllamaBackend, Momentum10, RandomAgent
from alphabench.compare import default_configs, mark_runnable, run_configs, PROMPTS
from alphabench.replay import run_replay, decision_dates
from alphabench.metrics import leaderboard, equal_weight_index, calibration, concentration

CACHE     = ROOT / "data" / "prices.parquet"
LLM_CACHE = ROOT / "data" / "llm_cache"
md = load_prices(ALL_SYMBOLS, "2023-09-01", None, CACHE)

CANDIDATES = "universe"              # "universe" = all 50 names are candidates (no funnel); "screened" = daily top-5 screener
MAX_POSITION_WEIGHT = __CAP_VALUE__   # per-name cap as a share of equity, enforced on every agent. None = the leaderboard rules
PAYLOAD = {"candidates": CANDIDATES}
OTHER = "screened" if CANDIDATES == "universe" else "universe"
CAP_TAG = "" if MAX_POSITION_WEIGHT is None else f"_cap{int(round(MAX_POSITION_WEIGHT * 100))}"
RESULTS = ROOT / "data" / "results" / (("compare" if CANDIDATES == "screened" else "compare_universe") + CAP_TAG)

# ---- the fixed window: same cycles for everyone, independent of when prices were last refreshed
WINDOW_START, N_CYCLES = "2026-06-01", 60          # N_CYCLES is a cap; the printed count is what actually ran
dds = decision_dates(md, WINDOW_START)[:N_CYCLES]
START, END = dds[0], dds[-1]
print(f"{len(dds)} cycles: {START.date()} → {END.date()}")""")

md("""## 1. Configurations

The list lives in `alphabench/compare.py` (`default_configs`) so the notebook and `scripts/run_compare.py` run exactly the same thing. `prompt` v1 = TradeRank-style mandate, v2 = v1 + risk rules, v3 = full decision procedure; `think` turns on Qwen3's reasoning mode (slower, may decide better). Models that are not pulled are skipped — `ollama pull <name>`.

**Tip:** run the long part from a terminal instead — `python scripts/run_compare.py --candidates universe__CAP_FLAG__` — and come back to this notebook for the tables; the cache makes section 3 instant afterwards.""")

code("""backends = {False: OllamaBackend(think=False), True: OllamaBackend(think=True)}
RUN_THINKING = False          # True also runs qwen3-8b_v3_thinking (~2-3x slower per call)
ONLY = None                   # e.g. ["qwen3-8b_v3"] to run a single configuration

CONFIGS = mark_runnable(default_configs(think_enabled=RUN_THINKING, candidates=CANDIDATES, max_position_weight=MAX_POSITION_WEIGHT), set(backends[False].list_models()), ONLY)
for c in CONFIGS:
    print(f"{'▶' if c['runnable'] else '–'} {c['name']:24s} {c['model']:12s} prompt {c['prompt']}  {c['skip_reason'] or ''}")""")

md("""## 2. Controls and benchmarks (seconds)

Two momentum rules as controls: the same 10-day signal, one allowed to short and one not. Longs and shorts compete for the one new position per cycle the engine allows, ranked together by absolute momentum, so the long/short control really does short (an earlier version considered a short only when no long qualified, which on 50 names never happens, and the two controls were then the same agent under two names).

Random traders are not two seeds here but a whole distribution — section 6 runs 1,000 of them.""")

code("""controls = run_replay(md, [Momentum10(allow_short=True), Momentum10(allow_short=False)],
                      START, END, log_dir=RESULTS / "controls", progress=False, payload_kwargs=PAYLOAD, max_position_weight=MAX_POSITION_WEIGHT)
first, last = controls["control_momentum_10d_ls"]["equity"].index[[0, -1]]   # benchmarks must cover exactly the replay window
benchmarks = {"spy": (md.close[BENCHMARK].loc[first:last] / md.close[BENCHMARK].loc[first] * 10_000),
              "ew_universe": equal_weight_index(md.close[TICKERS], start=first).loc[:last]}
print("controls done")""")

md("""## 3. Run the configurations (long — leave it running)

One line per cycle (latency or *cached*, running equity, elapsed, ETA). Safe to interrupt; re-running resumes from the cache. For a quick look set `N_CYCLES = 10` in the first cell.""")

code("""llm_results, agents, timings = run_configs(md, CONFIGS, START, END, RESULTS, LLM_CACHE, backends, payload_kwargs=PAYLOAD)
results = {**controls, **llm_results}""")

md("## 4. Comparison")

code("""def violation_rate(res):
    n_items = sum(len(r["decision"]["decisions"]) for r in res["log"])
    n_rej = 0 if res["rejections"].empty else len(res["rejections"])
    return n_rej / n_items if n_items else np.nan

lb = leaderboard(results, benchmarks)
lb["rule_violation_rate"] = [violation_rate(results[a]) for a in lb["agent"]]
lb["no_action_share"] = [sum(1 for r in results[a]["log"] if not r["decision"]["decisions"]) / len(results[a]["log"]) for a in lb["agent"]]
for col, key in [("validity", None), ("repairs", "repairs"), ("s_per_call", "avg_latency_s"), ("prompt_tok_per_s", "prompt_tokens_per_s")]:
    lb[col] = [ (1 - agents[a].stats()["invalid_final"] / agents[a].stats()["calls"]) if col == "validity" and a in agents
                else agents[a].stats().get(key) if a in agents else np.nan for a in lb["agent"]]
bench = pd.DataFrame([{"agent": f"benchmark_{k}", "total_return": v.iloc[-1] / v.iloc[0] - 1} for k, v in benchmarks.items()])
cols = ["agent", "total_return", "sharpe", "max_drawdown", "excess_vs_ew_universe", "n_trades", "n_open_at_end", "win_rate", "avg_holding_days", "invalidation_rate",
        "rule_violation_rate", "no_action_share", "validity", "repairs", "brier", "s_per_call"]
lb.to_csv(RESULTS / "comparison.csv")
display(bench.style.format({"total_return": "{:+.1%}"}).set_caption("Benchmarks over the window"))
lb[cols].style.format({"total_return": "{:+.1%}", "sharpe": "{:.2f}", "max_drawdown": "{:+.1%}", "excess_vs_ew_universe": "{:+.1%}",
                       "win_rate": "{:.0%}", "avg_holding_days": "{:.1f}", "invalidation_rate": "{:.0%}", "rule_violation_rate": "{:.0%}", "no_action_share": "{:.0%}",
                       "validity": "{:.0%}", "brier": "{:.3f}", "s_per_call": "{:.0f}"}, na_rep="—")""")

md("""`n_trades` counts every trade including those force-closed at the window end (`n_open_at_end`); `win_rate` and `brier` are therefore not biased towards closed losers. `no_action_share` is the share of cycles with an empty decision list. `s_per_call` is the mean over *live* calls only (0 or blank on a fully cached re-run) and depends on machine load at the time — a configuration run while the laptop was busy with something else shows a higher number that says nothing about the model.""")

code("""PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#7a7a7a", "#8e6bbf"]
eq = pd.DataFrame({k: v["equity"] for k, v in results.items()}); eq_idx = eq / eq.iloc[0] * 100
bm = pd.DataFrame(benchmarks).reindex(eq.index).ffill(); bm = bm / bm.iloc[0] * 100
fig, ax = plt.subplots(figsize=(12, 5.5))
for col, c in zip(bm.columns, ["#6b6b6b", "#9a9a9a"]):
    ax.plot(bm.index, bm[col], "--", lw=1.5, color=c, label=f"benchmark {col}")
llm_cols = [c for c in eq_idx.columns if not c.startswith("control_")]; ctl_cols = [c for c in eq_idx.columns if c.startswith("control_")]
for i, col in enumerate(llm_cols):
    ax.plot(eq_idx.index, eq_idx[col], lw=2.2, color=PALETTE[i % len(PALETTE)], label=col)
for col in ctl_cols:
    ax.plot(eq_idx.index, eq_idx[col], ":", lw=1.4, color="#b0b0b0", label=col.replace("control_", "control "))
ax.axhline(100, color="#ccc", lw=0.8)
ax.set_title(f"{len(dds)}-cycle comparison__CAP_TITLE__ — equity indexed to 100 (LLMs in colour, controls dotted grey)", loc="left")
ax.legend(frameon=False, fontsize=8, ncol=3); ax.grid(axis="y", color="#e5e5e5"); ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.show()""")

md("""## 5. Where the P&L comes from — concentration, stops, rule violations, and what each model actually did

`max_position_weight` is the largest single position as a share of equity, measured right after each fill (approximate: fill price against that day's closing equity); `best_trade_share_of_pnl` is the biggest trade's P&L over the total. A configuration whose return is one pyramided name shows a weight near 1 and a share above 1.

`best_trade_share_of_pnl` is blank when total P&L is zero or negative: the ratio has no meaning when the denominator is a loss, so it is left empty rather than printed as a misleading number. A blank there means the configuration lost money overall.""")

code("""rows = []
for name in agents:
    res = results[name]; tr = res["trades"]
    rej = res["rejections"]["reason"].value_counts().to_dict() if not res["rejections"].empty else {}
    actions = pd.Series([d["action"] for r in res["log"] for d in r["decision"]["decisions"]]).value_counts().to_dict()
    holds = sum(1 for r in res["log"] if not r["decision"]["decisions"])
    cc = concentration(res["fills"], tr, res["equity"])
    rows.append({"config": name, "cycles_no_action": holds, "avg_holding_days": np.nan if tr.empty else float(tr["holding_days"].mean()),
                 "max_position_weight": cc["max_position_weight"], "max_gross_exposure": cc["max_gross_exposure"], "n_adds": cc["n_adds"],
                 "best_trade_pnl": cc["best_trade_pnl"], "best_trade_share_of_pnl": cc["best_trade_share"],
                 "total_fees": np.nan if tr.empty else float(tr["fees"].sum()), **{f"act_{k}": v for k, v in actions.items()},
                 "stopped_out": int((tr["reason"] == "invalidation").sum()) if not tr.empty else 0,
                 "closed_by_choice": int((tr["reason"] == "close").sum()) if not tr.empty else 0,
                 "closed_at_window_end": int((tr["reason"] == "end_of_window").sum()) if not tr.empty else 0,
                 "avg_stop_distance_pct": np.nan if tr.empty else float((abs(tr["entry_price"] - tr["exit_price"]) / tr["entry_price"])[tr["reason"] == "invalidation"].mean() * 100),
                 **{f"rej_{k[:28]}": v for k, v in rej.items()}})
pd.DataFrame(rows).set_index("config").T""")

md("""### Scaled fills

The engine never leverages: when a model asks for more than the free cash allows, the fill is scaled down to what fits and the shortfall is logged. A high count here means the model's sizing ignores its own portfolio state.""")

code("""rows = []
for name in agents:
    sc = results[name]["scaled"]
    rows.append({"config": name, "n_scaled_fills": len(sc), "n_fills": len(results[name]["fills"]),
                 "avg_shortfall_pct": np.nan if sc.empty else float(sc["shortfall_pct"].mean()),
                 "worst_shortfall_pct": np.nan if sc.empty else float(sc["shortfall_pct"].max())})
scaled = pd.DataFrame(rows).set_index("config")
display(scaled.style.format({"avg_shortfall_pct": "{:.0f}%", "worst_shortfall_pct": "{:.0f}%"}, na_rep="—"))
for name in agents:
    if not results[name]["scaled"].empty:
        print(f"\\n{name} — first scaled fills"); display(results[name]["scaled"].head(5))""")

md("""### Calibration — stated confidence against realised wins

Every open carries a confidence and the engine refuses to open below 0.80. This asks whether the number means anything: a reliability table per configuration and a Brier score (lower is better; a constant 0.5 forecast scores 0.25). Read it against two things — the trade count, since a Brier over four or five trades is a description rather than an estimate, and the spread of the buckets, since confidence that never varies cannot be calibrated, only compared with the realised rate.""")

code("""# calibration: stated confidence vs realised win, per configuration (needs trades to be meaningful)
for name in agents:
    table, brier = calibration(results[name]["trades"])
    print(f"\\n{name}: Brier {brier:.3f}" if brier == brier else f"\\n{name}: no trades with confidence"); display(table)""")

md("""## 6. Skill or luck? 1,000 random traders

Every random trader plays by the same rules on the same cycles, with no information at all: a coin decides whether to trade, which candidate name (the screened five or all 50, per the candidate mode), which side (500 seeds long/short, 500 long-only), and a size of 10/20/30% of equity. The result is the distribution of outcomes that luck alone produces on this window. A configuration at the 90th percentile beat nine random traders out of ten; anything between roughly the 20th and 80th is indistinguishable from a coin flip. Momentum controls and the two benchmarks get the same treatment.

About a minute on a laptop; cached to parquet afterwards.""")

code("""from alphabench.null import run_null, null_table, null_summary

N_SEEDS = 500                                   # per side → 1,000 random traders
report = lambda done, n, s: print(f"\\r  {done}/{n} seeds  {s:.0f}s", end="")
null = pd.concat([run_null(md, START, END, n_seeds=N_SEEDS, long_only=lo, cache_dir=ROOT / "data" / "results" / "compare" / "null", on_batch=report, payload_kwargs=PAYLOAD, max_position_weight=MAX_POSITION_WEIGHT)
                  for lo in (False, True)], ignore_index=True)
print(); display(null_summary(null).T.style.format("{:.3f}"))""")

PYR_CAPPED = """With the position limit in force the adds can no longer pile one name up to most of the account — compare the `max_position_weight` rows with notebook 03, where the pyramiding seeds' median was about 0.9."""
PYR_PLAIN = """A prompt with no sizing rules can keep adding to a winner until one name is most of the account — a different game with a fatter tail, and the right null to judge such a prompt against. The engine still refuses leverage."""

md("""### A null that is allowed to pyramid

The random traders above never add to a position. These seeds may add to any position in profit (coin flip, 10-30% of equity each time), in this run's candidate mode. """ + (PYR_CAPPED if CAPPED else PYR_PLAIN) + """ Compare `max_position_weight` here with the configurations' in section 5, and read the `_pyr` percentile columns for the prompts that pyramided. Slower: a few minutes.""")

code("""null_pyr = pd.concat([run_null(md, START, END, n_seeds=N_SEEDS, long_only=lo, pyramid=True, cache_dir=ROOT / "data" / "results" / "compare" / "null", on_batch=report, payload_kwargs=PAYLOAD, max_position_weight=MAX_POSITION_WEIGHT)
                      for lo in (False, True)], ignore_index=True)
print(); display(null_summary(null_pyr).T.style.format("{:.3f}"))""")

md("""### The other candidate mode

The same 1,000 random traders again, in the other candidate mode (`OTHER`): if the notebook runs on the whole universe, this is the top-5 screener funnel, and vice versa. The difference between the two distributions is what the candidate list alone contributes — under the same rules, fees and stops. Run before the percentile table so it can show both.""")

code("""null_other = pd.concat([run_null(md, START, END, n_seeds=N_SEEDS, long_only=lo, cache_dir=ROOT / "data" / "results" / "compare" / "null", on_batch=report, payload_kwargs={"candidates": OTHER}, max_position_weight=MAX_POSITION_WEIGHT)
                        for lo in (False, True)], ignore_index=True)
print(); display(null_summary(pd.concat([null, null_other], ignore_index=True)).T.style.format("{:.3f}"))
fig, ax = plt.subplots(figsize=(8, 4))
for df_, c, lab in [(null, "#2a78d6", f"this run: {CANDIDATES}"), (null_other, "#9a9a9a", f"other mode: {OTHER}")]:
    ax.hist(df_.loc[~df_["long_only"], "total_return"], bins=40, color=c, alpha=0.6, label=f"random long/short — {lab}")
for k, v in benchmarks.items():
    ax.axvline(v.iloc[-1] / v.iloc[0] - 1, color="#6b6b6b", ls="--", lw=1.2); ax.text(v.iloc[-1] / v.iloc[0] - 1, ax.get_ylim()[1] * 0.95, " bm " + k, fontsize=8, color="#6b6b6b", va="top")
ax.set_title("Return of 500 random long/short traders: screened five vs whole universe", loc="left", fontsize=10)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:+.0%}")); ax.legend(frameon=False, fontsize=8)
ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", color="#e5e5e5"); plt.tight_layout(); plt.show()""")

code("""observed = lb.set_index("agent")[["total_return", "sharpe"]].copy()
for k, v in benchmarks.items():
    r = v.pct_change().dropna()
    observed.loc[f"benchmark_{k}"] = [v.iloc[-1] / v.iloc[0] - 1, float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else 0.0]
nt = null_table(observed, pd.concat([null, null_other, null_pyr], ignore_index=True)).sort_values("total_return", ascending=False)
nt.to_csv(RESULTS / "null_percentiles.csv")
nt.style.format({"total_return": "{:+.1%}", "sharpe": "{:.2f}", **{c: "{:.0f}" for c in nt.columns if "_pct_" in c}}).set_caption(
    f"Percentile within the random-trader distributions — ls = long/short, long = long-only; no suffix = picking from the screened five, _univ = all 50 names, _pyr = allowed to add to winners (this run: {CANDIDATES})")""")

code("""fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
for ax, m, fmt in zip(axes, ["total_return", "sharpe"], ["{:+.0%}", "{:.1f}"]):
    for df_, lo, c, lab in [(null, False, "#9a9a9a", "random long/short"), (null, True, "#cfcfcf", "random long-only"), (null_pyr, False, "#e8c48a", "random long/short, pyramiding")]:
        ax.hist(df_.loc[df_["long_only"] == lo, m], bins=40, color=c, alpha=0.6, label=lab)
    ymax = ax.get_ylim()[1]
    for i, name in enumerate([n for n in observed.index if not n.startswith("control_")]):
        x = observed.loc[name, m]
        col = "#6b6b6b" if name.startswith("benchmark_") else PALETTE[i % len(PALETTE)]
        ax.axvline(x, color=col, lw=2 if not name.startswith("benchmark_") else 1.5, ls="-" if not name.startswith("benchmark_") else "--")
        ax.text(x, ymax * (0.95 - 0.07 * (i % 6)), " " + name.replace("benchmark_", "bm "), color=col, fontsize=8, va="top")
    ax.set_title(f"{m.replace('_', ' ')} — 1,000 random traders vs configurations", loc="left", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", color="#e5e5e5")
    if m == "total_return": ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:+.0%}"))
axes[0].legend(frameon=False, fontsize=8); plt.tight_layout(); plt.show()""")

FINDINGS_CAPPED = """## 7. What this run found

*From the run of 9 September 2026: 60 cycles, 1 June to 25 August 2026, qwen3:8b on a laptop, whole-universe candidates, 25% position limit. The tables above are the evidence; this section states the reading so that nobody has to derive it.*

**Nothing beat holding the universe.** The equal-weight index of the same 50 names returned +5.4% at a Sharpe of 1.98 — better than 91% of the random long/short traders. The best configuration, v2, returned +0.3% at a Sharpe of 0.16. v3 returned -0.8% and v1 -4.4%. Both benchmarks beat all three prompts; the momentum controls (-4.7% long-only, -8.2% long/short) did no better than the worst of them.

**All three sit inside the band this notebook calls luck.** Against 1,000 random traders drawn from the same universe under the same rules, fees and stops: v2 at the 65th percentile, v3 at the 57th, v1 at the 29th. Section 6 sets roughly the 20th to the 80th as indistinguishable from a coin flip, and all three are inside it. Measured against the long-only null — the fairer comparison on a window that rose — they fall to the 40th, 33rd and 12th. On this window, with this information, the models added nothing that luck does not supply.

**Stated confidence carries no information.** All three state about 0.85 and realise 7% (v1), 25% (v2) and 20% (v3). Brier scores of 0.669, 0.547 and 0.582 against 0.434 for the long-only momentum control and 0.468 for the long/short one: the models are worse calibrated than a one-line rule, which at least sometimes loses when it says 0.85. Two caveats keep this honest. Confidence barely varies — every trade lands in a single bucket just above the 0.80 gate the engine requires to open — so this is not a calibration curve, it is the finding that confidence is close to a constant. And n is 14, 4 and 5 trades. What can be said is that the number is uniformly too high and uniformly uninformative; how these models would rank a real spread of confidences is untested.

**Adding risk rules to the prompt made rule-following worse.** v2 is v1 plus risk rules, and the engine had to refuse 86% of its decision items against v1's 68% — mostly adds, to positions not in profit or already at the cap. The rules changed what the model asked for without changing what it understood about its own book. v3, the full decision procedure, violated nothing at all, which is the one clear win for prompt engineering in this table.

**v3 barely played.** 55 of 60 cycles ended with an empty decision list; it opened five positions (four long, one short) and traded five times. Its -0.8% is close to what not trading at all would have returned, and every v3 number here rests on n = 5. Read it as a model that declined to act rather than one that acted well.

**What the position limit changed.** Notebook 03, without the limit, put v1 at +11.0% and the 98th percentile of the plain null — a return produced by adding to one name until it was most of the account, and one that sits at the 90th percentile of the *pyramiding* null, which is the comparison it actually deserves. With the cap in force v1 is the worst of the three. The cap did not make the prompts better; it removed one prompt's licence to place a bet large enough to dominate the comparison. That is the whole reason both notebooks exist."""

FINDINGS_PLAIN = """## 7. What this run found

*From the run of 9 September 2026: 60 cycles, 1 June to 25 August 2026, qwen3:8b on a laptop, whole-universe candidates, no position limit — the leaderboard's rules. The tables above are the evidence; this section states the reading so that nobody has to derive it.*

**v1 made money, and it made it with one bet.** +11.0% at a Sharpe of 1.87, the 98th percentile of the plain random long/short distribution. Section 5 shows where it came from: adds into a single name until it was most of the account. Against the null that is allowed to pyramid — the right comparison for a prompt with no sizing rules — it falls to the 90th percentile. The result is real, and it is a result about position sizing rather than about stock picking.

**Only v1 beat the universe.** The equal-weight index of the same 50 names returned +5.4% at a Sharpe of 1.98. v2 returned +5.0% but at a Sharpe of 0.71, so it took considerably more risk to get less. v3 returned -0.9%.

**Stated confidence carries no information.** v1 states about 0.85 and wins 12% of 17 trades (Brier 0.640); v2 and v3 state 0.85 and win 20% of 5 trades each (Brier 0.583). The momentum control scores 0.434. Two caveats: confidence barely varies, every trade landing just above the 0.80 gate the engine requires to open, so this is not a calibration curve but the finding that confidence is close to a constant; and the trade counts are small.

**Rule violations are a prompt property, not a model property.** The engine refused 54% of v1's decision items and 59% of v2's, against 0% for v3 — the same model in all three cases. The full decision procedure is what stops the model asking for things the rules forbid.

**v3 barely played**, opening four positions and trading five times in 60 cycles. Every v3 number here rests on n = 5.

**Read this notebook next to `03_new_rule_model_comparison`**, which runs the same experiment with a 25% per-name cap. Under the cap v1 is the worst of the three rather than the best, which is the clearest evidence in the project that this comparison was measuring sizing licence rather than judgement."""

md((FINDINGS_CAPPED if CAPPED else FINDINGS_PLAIN) + """

### What would change the conclusion

- **A larger model.** Only qwen3:8b ran here. If the 14B or 12B beats it on return *and* Sharpe against the same controls, model size is the lever, and the next rung is a hosted frontier model via `OpenAICompatibleBackend`.
- **More trades, not more cycles.** Sixty cycles is enough to measure rule-following and validity. It is not enough to separate returns when the configurations trade five to seventeen times. Trade count is the binding constraint on everything in section 4.
- **Other windows.** Everything above is one rising quarter. A finding that survives 2024 and 2025 is a finding; this one is a reading.
- **A better null, not a better story.** The honest summary of any configuration is its percentile, not its return. Anything inside the middle of the random-trader distribution has shown nothing, however green the equity curve looks.

Next: the same table on 2024 and 2025 windows (regime dependence), then the candle ablation (`include_summary` / `n_daily`) on the best configuration.""")

nb["cells"] = cells
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
default_name = "03_new_rule_model_comparison.ipynb" if CAPPED else "03_model_comparison.ipynb"
out = args.out or Path(__file__).resolve().parents[1] / "notebooks" / default_name
nbf.write(nb, out)
print("wrote", out)
