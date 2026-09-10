"""Generates notebooks/01_data_and_engine.ipynb (kept as code so it is diffable)."""
import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

md("""# 01 — Data and engine (no LLM calls)

**agent-alpha-bench** is an evaluation harness for LLMs acting as medium-term equity investors, inspired by TradeRank's AI trading leaderboard. This first notebook verifies the plumbing before any money is spent on inference:

1. load and cache adjusted daily OHLCV for the 50-name universe plus SPY
2. build a **point-in-time** payload for a decision date (candles, RSI, screener, portfolio snapshot)
3. validate decisions against the JSON schema every agent must satisfy
4. run the **paper-trading engine** (fills at next open, fees, invalidation monitor, no leverage)
5. replay the **rule-based controls** over 2024-01-01 → today and produce the leaderboard
6. run explicit **look-ahead checks**

Everything an LLM will later see and do flows through the same code paths exercised here. The controls are the bar every model has to clear.

> If Yahoo Finance is unreachable the notebook falls back to synthetic prices so the mechanics can still be checked. Results on synthetic data are meaningless as findings — the banner below tells you which mode you are in.""")

code("""import sys, logging, json, time
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))
logging.basicConfig(level=logging.INFO, format="%(message)s")
pd.set_option("display.width", 160, "display.max_columns", 30, "display.float_format", "{:,.4f}".format)

from alphabench.universe import TICKERS, ALL_SYMBOLS, UNIVERSE, BENCHMARK, UNIVERSE_ASOF
from alphabench.market import load_prices, synthetic_prices, MarketData, rsi
from alphabench.screener import screen
from alphabench.prompt import build_payload, render_text, estimate_tokens
from alphabench.schema import Decision, DecisionItem, parse_decision, json_schema
from alphabench.engine import Portfolio
from alphabench.agents import BuyAndHoldBenchmark, Momentum10, RandomAgent
from alphabench.replay import run_replay, decision_dates
from alphabench.metrics import leaderboard, equity_metrics, calibration, equal_weight_index

# ---- configuration ----------------------------------------------------------
DATA_START   = "2023-09-01"   # extra history so RSI / screener / weekly candles are warm on day one
REPLAY_START = "2024-01-01"
REPLAY_END   = None           # None = latest bar in the data
CACHE        = ROOT / "data" / "prices.parquet"
RESULTS_DIR  = ROOT / "data" / "results" / "controls"
FORCE_SYNTHETIC = False       # set True to test the mechanics offline on purpose
print(ROOT)""")

md("## 1. Universe\n\nThe 50 largest US-domiciled companies by market cap as of 2024-12-31 (approximate snapshot — verify against a point-in-time source before the headline run; see the survivorship note in `alphabench/universe.py`). SPY is the benchmark and is only tradeable by the buy-and-hold control.")

code("""pd.Series(UNIVERSE).value_counts().rename("tickers per sector").to_frame().T""")

md("## 2. Prices — fetch once, cache to parquet\n\nAdjusted OHLCV from Yahoo Finance. The cache means the fetch happens once; delete `data/prices.parquet` or pass `refresh=True` to refetch.")

code("""SYNTHETIC = FORCE_SYNTHETIC
if not SYNTHETIC:
    try:
        md = load_prices(ALL_SYMBOLS, DATA_START, None, CACHE)
    except Exception as e:
        print(f"⚠️  live data unavailable ({type(e).__name__}: {e}) — falling back to synthetic prices")
        SYNTHETIC = True
if SYNTHETIC:
    md = synthetic_prices(ALL_SYMBOLS, DATA_START, pd.Timestamp.today().strftime("%Y-%m-%d"), sectors=UNIVERSE)

print("MODE:", "SYNTHETIC (mechanics only — not a result)" if SYNTHETIC else "LIVE Yahoo Finance data")
print(f"{md.close.shape[0]} bars x {md.close.shape[1]} symbols, {md.dates[0].date()} → {md.dates[-1].date()}")
missing = md.close.isna().sum().sort_values(ascending=False)
print("symbols with missing closes:", missing[missing > 0].to_dict() or "none")
md.close[["SPY", "AAPL", "XOM"]].tail(3)""")

md("""## 3. Point-in-time slicing — the single look-ahead guard

Every payload is built from `md.asof(t)`, which contains only bars with `index <= t`. Fills use `open[t+1]` explicitly in the engine. If you only ever build payloads and screens from an `.asof()` slice, you cannot peek.""")

code("""t = md.dates[-40]
md_t = md.asof(t)
assert md_t.last_date == t and (md_t.dates <= t).all()
print("decision date:", t.date(), "| bars visible:", len(md_t.dates), "| last visible bar:", md_t.last_date.date())
md_t.weekly("AAPL").tail(3)  # last row may be a partial week — correct point-in-time behaviour""")

md("## 4. Indicators, universe table and screener\n\nThe screener is TradeRank's rule: `20-day average dollar volume × (1 + |10-day return|)`, top 5 shown to the agent alongside anything it holds.")

code("""table = md_t.summary_table(TICKERS)
display(table.sort_values("adv_20d_usd", ascending=False).head(8))
screen(md_t, TICKERS, top_n=5)""")

md("""## 5. The cycle payload and the rendered prompt

`build_payload()` returns a dict — the contract between market data and agents. Rule agents read the numbers; the LLM agent will render it with `render_text()`. Token count is the main cost driver for the LLM layer, so we measure it now, including the ablation variants.""")

code("""empty_book = Portfolio(10_000).snapshot(md_t.close.iloc[-1])
payload = build_payload(md_t, empty_book, t)
print("keys:", list(payload))
print("screened:", [s["symbol"] for s in payload["screened"]])
first = payload["detail"][payload["screened"][0]["symbol"]]["daily"]
print("candles shown for", list(payload["detail"]), "| daily span:", first["first_date"], "→", first["last_date"], "| last row:", first["rows"][-1])
assert all(d["daily"]["last_date"] <= str(t.date()) for d in payload["detail"].values()), "candle after decision date!"

text = render_text(payload)
print(f"\\nprompt: {len(text):,} chars ≈ {estimate_tokens(text):,} tokens (rough; digit-splitting tokenizers count ~2.5 chars/token on this JSON)")
print(text[:600] + " ...")""")

code("""variants = {
    "default (20d + 12w candles, RSI, universe table)": {},
    "30d + 26w candles (TradeRank-like)": dict(n_daily=30, n_weekly=26),
    "no universe table": dict(include_summary=False),
    "no RSI": dict(include_rsi=False),
    "10 daily candles only": dict(n_daily=10, n_weekly=0),
}
pd.DataFrame({k: {"chars": len(render_text(build_payload(md_t, empty_book, t, **v))),
                  "≈tokens": estimate_tokens(render_text(build_payload(md_t, empty_book, t, **v)))}
              for k, v in variants.items()}).T""")

md("""## 6. Decision schema

Structural validation lives in `alphabench/schema.py`; rule validation (max positions, one new position per cycle, no averaging down, invalidation on the losing side…) lives in the engine so it is applied identically to every agent.""")

code("""good = {"reasoning": "example", "decisions": [
    {"symbol": "AAPL", "action": "open_long", "percent_of_equity": 20, "confidence": 0.85,
     "thesis": "…", "invalidation": "…", "invalidation_price": 150.0}]}
bad  = {"decisions": [{"symbol": "AAPL", "action": "open_long", "percent_of_equity": 20, "confidence": 0.85}]}
for name, obj in [("good", good), ("bad", bad)]:
    d, err = parse_decision(obj)
    print(f"{name}: {'OK' if d else 'REJECTED → ' + next(l.strip() for l in err.splitlines() if 'missing' in l)}")
print(json.dumps(json_schema()["properties"]["decisions"], indent=1)[:400], "…")""")

md("""## 7. Engine walkthrough — one cycle by hand

Timing contract: decide at close *t* → fill at open *t+1* less 0.1% fee → invalidation check against low/high *t+1* → mark at close *t+1*.""")

code("""book = Portfolio(10_000)
t0, t1 = md.dates[-40], md.dates[-39]
snap = book.snapshot(md.close.loc[t0])
px = float(md.close.loc[t0, "AAPL"])
dec = Decision(reasoning="demo", decisions=[DecisionItem(
    symbol="AAPL", action="open_long", percent_of_equity=30, confidence=0.9,
    thesis="demo", invalidation="demo", invalidation_price=round(px * 0.95, 2))])
fills = book.execute(dec, t1, md.open.loc[t1], ref_equity=snap["equity"], tradeable=set(TICKERS))
inv = book.check_invalidations(t1, md.open.loc[t1], md.low.loc[t1], md.high.loc[t1])
eq = book.mark(t1, md.close.loc[t1])
print(f"decided {t0.date()} at close {px:.2f} → filled {t1.date()} at open {fills[0].price:.2f}, fee {fills[0].fee:.2f}")
print(f"invalidations: {len(inv)} | equity at close {t1.date()}: {eq:,.2f} | cash {book.cash:,.2f}")
pd.DataFrame(book.snapshot(md.close.loc[t1])["positions"])""")

md("""## 8. Replay the rule-based controls

Same rules as the LLMs will face: one new position per cycle, confidence gate, invalidation price, fees. These are the null hypotheses — an LLM that cannot beat them has nothing to say.

Two **benchmarks** sit alongside the controls but are *not* run through the engine (so they are not bound by the 10-position cap): SPY buy-and-hold, and an equal-weight, daily-rebalanced index of the whole 50-name universe — "what did the universe itself do?".""")

code("""agents = [
    BuyAndHoldBenchmark(),           # SPY, 100%, never trades again (the engine-run version of the SPY benchmark)
    Momentum10(allow_short=True),    # long strongest / short weakest 10d momentum, 8% stop
    Momentum10(allow_short=False),
    RandomAgent(seed=0), RandomAgent(seed=1), RandomAgent(seed=2),   # null distribution
]
dds = decision_dates(md, REPLAY_START, REPLAY_END)
print(f"{len(dds)} decision cycles: {dds[0].date()} → {dds[-1].date()}")
t_start = time.time()
results = run_replay(md, agents, REPLAY_START, REPLAY_END, log_dir=RESULTS_DIR, progress=False)
print(f"replay took {time.time() - t_start:.0f}s; results saved to {RESULTS_DIR}")

# benchmarks (not engine-run): rebased to the starting capital on the first decision date
first, last = results[agents[0].name]["equity"].index[[0, -1]]   # exactly the replay window
benchmarks = {
    "spy": (md.close[BENCHMARK].loc[first:last] / md.close[BENCHMARK].loc[first] * 10_000).rename("spy"),
    "ew_universe": equal_weight_index(md.close[TICKERS], start=first).loc[:last],
}
pd.DataFrame(benchmarks).to_parquet(RESULTS_DIR / "benchmarks.parquet")""")

md("## 9. Leaderboard")

code("""lb = leaderboard(results, benchmarks)
bench_rows = pd.DataFrame([{"agent": f"benchmark_{k}", **equity_metrics(v, benchmarks)} for k, v in benchmarks.items()])
lb.to_csv(RESULTS_DIR / "leaderboard.csv"); bench_rows.to_csv(RESULTS_DIR / "benchmarks_metrics.csv", index=False)
display(bench_rows[["agent", "total_return", "cagr", "sharpe", "max_drawdown"]].style.format({"total_return": "{:+.1%}", "cagr": "{:+.1%}", "max_drawdown": "{:+.1%}", "sharpe": "{:.2f}"}).set_caption("Benchmarks (not engine-run)"))
cols = ["agent", "total_return", "cagr", "sharpe", "max_drawdown", "excess_vs_spy", "excess_vs_ew_universe", "n_trades", "win_rate", "avg_holding_days", "invalidation_rate", "total_fees", "brier"]
lb[cols].style.format({c: "{:+.1%}" for c in ["total_return", "cagr", "max_drawdown", "excess_vs_spy", "excess_vs_ew_universe"]} |
                      {"sharpe": "{:.2f}", "win_rate": "{:.0%}", "invalidation_rate": "{:.0%}", "avg_holding_days": "{:.0f}", "total_fees": "${:,.0f}", "brier": "{:.3f}"}, na_rep="—")""")

md("## 10. Equity curves and drawdowns")

code("""PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#7a7a7a"]
eq = pd.DataFrame({k: v["equity"] for k, v in results.items()})
eq_idx = eq / eq.iloc[0] * 100
bm_idx = pd.DataFrame(benchmarks).reindex(eq.index).ffill(); bm_idx = bm_idx / bm_idx.iloc[0] * 100

def spread(ys, min_gap):
    \"\"\"Push label y-positions apart so end labels never overlap.\"\"\"
    order = np.argsort(ys); out = np.array(ys, dtype=float)
    for a, b in zip(order[:-1], order[1:]):
        if out[b] - out[a] < min_gap: out[b] = out[a] + min_gap
    return out

fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1.4]})
all_idx = pd.concat([eq_idx, bm_idx], axis=1)
label_y = spread(all_idx.iloc[-1].values, min_gap=(all_idx.values.max() - all_idx.values.min()) * 0.035)
for j, col in enumerate(bm_idx.columns):
    axes[0].plot(bm_idx.index, bm_idx[col], "--", lw=1.6, color="#6b6b6b", label=f"benchmark {col}")
    axes[0].annotate(f"benchmark {col}", (bm_idx.index[-1], label_y[len(eq_idx.columns) + j]),
                     xytext=(8, 0), textcoords="offset points", va="center", fontsize=9, color="#6b6b6b")
for i, col in enumerate(eq_idx.columns):
    c = PALETTE[i % len(PALETTE)]
    style = ":" if col.startswith("control_random") else "-"
    axes[0].plot(eq_idx.index, eq_idx[col], style, lw=2, color=c, label=col.replace("control_", ""))
    axes[0].annotate(col.replace("control_", ""), (eq_idx.index[-1], label_y[i]),
                     xytext=(8, 0), textcoords="offset points", va="center", fontsize=9, color="#333")
    dd = eq[col] / eq[col].cummax() - 1
    axes[1].plot(dd.index, dd * 100, style, lw=1.5, color=c)
axes[0].set_title(f"Controls and benchmarks — equity indexed to 100 {'(SYNTHETIC DATA)' if SYNTHETIC else ''}", loc="left", fontsize=12)
axes[0].set_ylabel("equity (start = 100)"); axes[1].set_ylabel("drawdown %")
axes[0].legend(loc="upper left", frameon=False, fontsize=9, ncol=2)
for ax in axes:
    ax.grid(axis="y", color="#e5e5e5", lw=0.8); ax.spines[["top", "right"]].set_visible(False)
axes[0].axhline(100, color="#bbb", lw=0.8)
plt.tight_layout(); plt.show()""")

md("## 11. Trades, rejections and the calibration machinery\n\nThe rejection log is where you see the rules biting. Calibration compares each trade's *stated* confidence to whether it won — for the random control it is noise by construction; for LLMs it becomes the most interesting chart in the project.")

code("""tr = pd.concat([v["trades"].assign(agent=k) for k, v in results.items() if not v["trades"].empty])
display(tr.sort_values("pnl").tail(5)[["agent", "symbol", "side", "entry_date", "exit_date", "entry_price", "exit_price", "pnl", "pnl_pct", "reason", "confidence"]])
rej = pd.concat([v["rejections"].assign(agent=k) for k, v in results.items() if not v["rejections"].empty])
display(rej.groupby(["agent", "reason"]).size().rename("n").to_frame() if not rej.empty else "no rejections")
table, brier = calibration(results["control_random_s0"]["trades"])
print(f"random control Brier score: {brier:.3f}"); table""")

md("""## 12. Look-ahead checks

Three explicit tests, kept in the notebook so they run every time:

1. every fill date is strictly after its decision date
2. no candle in any payload is dated after its decision date (checked on a sample of cycles)
3. **truncation test** — replay with the data cut off at date *T* must reproduce exactly the same equity path up to *T* as the full replay. If anything downstream peeked at the future, the two runs would diverge.""")

code("""# 1. fills after decisions
for k, v in results.items():
    for row in v["log"]:
        assert row["fill_date"] > row["decision_date"], (k, row)
print("✓ every fill is after its decision date")

# 2. payload dates on a sample of cycles
for t_chk in dds[::max(1, len(dds) // 12)]:
    p = build_payload(md.asof(t_chk), Portfolio().snapshot(md.close.loc[t_chk]), t_chk)
    assert p["decision_date"] == str(t_chk.date())
    for d in p["detail"].values():
        assert d["daily"]["last_date"] <= p["decision_date"] and (not d["weekly"]["rows"] or d["weekly"]["last_date"] <= p["decision_date"])
    assert p["universe"]["rows"]  # table exists
print("✓ no payload contains bars after its decision date")

# 3. truncation test
T = dds[len(dds) // 3]
md_cut = md.between(md.dates[0], md.dates[md.dates.searchsorted(T) + 1])   # keep T+1 so the last decision can fill
res_cut = run_replay(md_cut, [Momentum10(True), RandomAgent(seed=0)], REPLAY_START, None, progress=False)
for k in res_cut:
    full, cut = results[k]["equity"], res_cut[k]["equity"]
    pd.testing.assert_series_equal(full.loc[:cut.index[-1]], cut, check_names=False)
print(f"✓ truncated replay (data cut at {T.date()}) reproduces the full replay exactly up to that date")""")

md("""## 13. What this notebook established, and what's next

- The data path, screener, payload, schema, engine and metrics all work end to end under the rules the LLMs will face.
- Prompt size is ~8–10k tokens for the full payload; the ablation variants show where to trim.
- The controls define the bar; the two benchmarks (SPY, equal-weight universe) say what the market and the universe did. Everything is saved under `data/results/controls/` (equity, trades, decisions, benchmarks, leaderboard.csv).

**Next (notebook 02):** `alphabench/agents/llm.py` — a provider-agnostic adapter (`complete(messages, model, temperature, json_schema)`) with on-disk response caching, the repair loop, and an Ollama backend first so iteration is free and offline. Then the provider cost evaluation on 20 real cycles.""")

nb["cells"] = cells
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = Path(__file__).resolve().parents[1] / "notebooks" / "01_data_and_engine.ipynb"
nbf.write(nb, out)
print("wrote", out)
