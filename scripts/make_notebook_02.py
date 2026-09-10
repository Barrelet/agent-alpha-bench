"""Generates notebooks/02_first_llm_cycles.ipynb."""
import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

md("""# 02 — First LLM cycles (local, via Ollama)

Notebook 01 proved the harness with rule-based controls. This one puts a real model in the seat for a handful of cycles and measures the four things that decide the provider question: **validity** (does it return usable JSON?), **latency / tokens per second** on your machine, **prompt size** as the model actually counts it, and **behaviour** (what it does with the rules).

Everything is cached under `data/llm_cache/<agent>/`, keyed on model + options + prompt, so re-running a cell is free and a replay is reproducible byte-for-byte. Delete that folder to force fresh calls.

> Requires the Ollama app running (llama icon in the menu bar) and at least one pulled model: `ollama pull qwen3:8b`.""")

code("""import sys, json, time, logging
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))
logging.basicConfig(level=logging.WARNING, format="%(message)s")
pd.set_option("display.width", 160, "display.max_columns", 30, "display.max_colwidth", 120)

from alphabench.universe import TICKERS, ALL_SYMBOLS, UNIVERSE, BENCHMARK
from alphabench.market import load_prices
from alphabench.prompt import build_payload
from alphabench.engine import Portfolio
from alphabench.agents import LLMAgent, OllamaBackend, Momentum10, RandomAgent
from alphabench.agents.llm import truncation_warning
from alphabench.replay import run_replay, decision_dates
from alphabench.metrics import leaderboard, equal_weight_index, calibration

CACHE      = ROOT / "data" / "prices.parquet"
LLM_CACHE  = ROOT / "data" / "llm_cache"
RESULTS    = ROOT / "data" / "results" / "llm_smoke"
md = load_prices(ALL_SYMBOLS, "2023-09-01", None, CACHE)
print(f"{md.close.shape[0]} bars → {md.dates[-1].date()}")""")

md("## 1. Is Ollama up, and what is pulled?")

code("""backend = OllamaBackend()            # http://localhost:11434, num_ctx=32k, thinking off
try:
    models = backend.list_models()
    print("Ollama is running. Models:", models)
except Exception as e:
    raise SystemExit(f"Cannot reach Ollama at {backend.host}: {e}\\nStart the Ollama app, then re-run this cell.")

MODEL = next((m for m in models if m.startswith("qwen3:8b")), models[0])
print("using:", MODEL)""")

md("""## 2. One cycle, by hand

Build the payload for a recent date, send it, and look at the raw response before any replay. The first call loads the model into memory, so it is slower than the rest.""")

code("""t = md.dates[-30]
md_t = md.asof(t)
book = Portfolio(10_000).snapshot(md_t.close.iloc[-1])
payload = build_payload(md_t, book, t)

agent = LLMAgent(name=f"ollama_{MODEL.replace(':', '-')}", backend=backend, model=MODEL, temperature=0.0, seed=7, cache_dir=LLM_CACHE)
system, user = agent.messages(payload)
print(f"system prompt: {len(system):,} chars | user prompt: {len(user):,} chars (≈{int(len(user)/2.5):,} tokens at ~2.5 chars/token)")

t0 = time.time()
decision = agent.decide(payload)
rec = agent.records[-1]
print(f"\\n{'cached' if rec.cached else 'live'} call: {rec.latency_s:.1f}s | prompt tokens as counted by the model: {rec.prompt_tokens} | completion: {rec.completion_tokens} | valid: {rec.ok}")
if rec.notes: print("parser notes:", rec.notes)
if agent.records[0].error: print("first attempt error →", agent.records[0].error[:400])
warn = truncation_warning(rec.prompt_tokens, backend.num_ctx)
print("⚠️ " + warn if warn else f"prompt fits: {rec.prompt_tokens} tokens for {len(user):,} chars ({len(user)/rec.prompt_tokens:.2f} chars/token)")
print("\\nRAW RESPONSE:\\n", rec.text[:2500])""")

code("""print("REASONING:", decision.reasoning[:800], "\\n")
pd.DataFrame([d.model_dump() for d in decision.decisions])""")

md("""## 2b. Timing on a fresh call

The cache makes repeats free, so timing needs a date that has not been sent before. This finds one, sends it, and splits the latency into **prompt reading** and **answer generation** using Ollama's own timers. Prompt reading is the bottleneck on a laptop; it scales with prompt tokens, so this is the number the compaction and the Ollama settings (flash attention, 8-bit KV cache, context length) act on. Run `ollama ps` in a terminal afterwards to see the effective context window.""")

code("""timing_agent = LLMAgent(name=f"ollama_{MODEL.replace(':', '-')}", backend=backend, model=MODEL, temperature=0.0, seed=7, cache_dir=LLM_CACHE)
for back in range(31, 80):
    t_fresh = md.dates[-back]; md_f = md.asof(t_fresh)
    pl = build_payload(md_f, Portfolio(10_000).snapshot(md_f.close.iloc[-1]), t_fresh)
    sy, us = timing_agent.messages(pl)
    key = timing_agent.cache.key(backend=backend.name, model=MODEL, temperature=0.0, seed=7, system=sy, user=us, schema=timing_agent.schema, max_tokens=timing_agent.max_tokens)
    if timing_agent.cache.get(key) is None:
        break
timing_agent.decide(pl)
r = timing_agent.records[-1]
print(f"fresh call on {t_fresh.date()}: {r.latency_s:.1f}s total | prompt {r.prompt_tokens} tokens read in {r.prompt_eval_s or float('nan'):.1f}s "
      f"({(r.prompt_tokens or 0)/(r.prompt_eval_s or float('nan')):.0f} tok/s) | answer {r.completion_tokens} tokens in {r.eval_s or float('nan'):.1f}s "
      f"({(r.completion_tokens or 0)/(r.eval_s or float('nan')):.0f} tok/s) | valid: {r.ok} | num_ctx requested: {backend.num_ctx}")
print("→ a full 670-cycle replay at this speed ≈", f"{r.latency_s * 670 / 3600:.1f} h")""")

md("""## 3. A short replay: the model vs the controls

Twenty cycles is enough to measure throughput and see the rules bite (rejections), not enough to say anything about skill. Under the output contract an empty `decisions` list means hold. Cached calls are skipped, so re-running costs nothing.""")

code("""N_CYCLES = 20
dds_all = decision_dates(md, "2024-01-01")
start, end = dds_all[-N_CYCLES], dds_all[-1]
print(f"{N_CYCLES} cycles: {start.date()} → {end.date()}")

agent = LLMAgent(name=f"ollama_{MODEL.replace(':', '-')}", backend=backend, model=MODEL, temperature=0.0, seed=7, cache_dir=LLM_CACHE)
agents = [agent, Momentum10(allow_short=True), RandomAgent(seed=0)]
t0 = time.time()
results = run_replay(md, agents, start, end, log_dir=RESULTS, progress=False)
print(f"replay took {time.time() - t0:.0f}s")
stats = agent.stats(); stats""")

code("""per_cycle = stats["avg_latency_s"]
print(f"≈ {per_cycle:.0f}s per live decision → full 2024→today replay (~670 cycles) ≈ {per_cycle * 670 / 3600:.1f} h for this model on this machine")
print(f"validity: {stats['calls'] - stats['invalid_final']}/{stats['calls']} calls usable, {stats['repairs']} needed a repair attempt")""")

md("## 4. What did it do?")

code("""log = pd.DataFrame([{"date": r["decision_date"], "equity": r["equity"], "fills": r["n_fills"],
                     "actions": ", ".join(f"{d['action']}:{d['symbol']}" for d in r["decision"]["decisions"]) or "hold (empty list)",
                     "reasoning": r["decision"]["reasoning"][:140]} for r in results[agent.name]["log"]])
display(log)
rej = results[agent.name]["rejections"]
display(rej.groupby("reason").size().rename("n").to_frame() if not rej.empty else "no rejections — every decision passed the rules")""")

code("""first, last = results[agent.name]["equity"].index[[0, -1]]
benchmarks = {"spy": (md.close[BENCHMARK].loc[first:last] / md.close[BENCHMARK].loc[first] * 10_000),
              "ew_universe": equal_weight_index(md.close[TICKERS], start=first).loc[:last]}
lb = leaderboard(results, benchmarks)
lb[["agent", "total_return", "sharpe", "max_drawdown", "excess_vs_ew_universe", "n_trades", "win_rate", "invalidation_rate", "brier"]]""")

code("""eq = pd.DataFrame({k: v["equity"] for k, v in results.items()})
eq_idx = eq / eq.iloc[0] * 100
bm = pd.DataFrame(benchmarks).reindex(eq.index).ffill(); bm = bm / bm.iloc[0] * 100
fig, ax = plt.subplots(figsize=(11, 4.5))
for col, c in zip(bm.columns, ["#6b6b6b", "#9a9a9a"]):
    ax.plot(bm.index, bm[col], "--", lw=1.5, color=c, label=f"benchmark {col}")
for col, c in zip(eq_idx.columns, ["#2a78d6", "#eb6834", "#1baf7a"]):
    ax.plot(eq_idx.index, eq_idx[col], lw=2, color=c, label=col)
ax.set_title(f"{N_CYCLES}-cycle smoke test — equity indexed to 100", loc="left"); ax.legend(frameon=False, fontsize=9)
ax.grid(axis="y", color="#e5e5e5"); ax.spines[["top", "right"]].set_visible(False); plt.tight_layout(); plt.show()""")

md("""## 5. Prompt-size ablation, as the model counts it

The rough 4-chars-per-token rule under-counts numbers-heavy JSON. This measures real prompt tokens for the payload variants from notebook 01, using one call each (cached afterwards).""")

code("""variants = {"default (20d + 12w candles)": {}, "30d + 26w candles (TradeRank-like)": dict(n_daily=30, n_weekly=26),
            "no universe table": dict(include_summary=False), "no RSI": dict(include_rsi=False),
            "10 daily candles only": dict(n_daily=10, n_weekly=0)}
rows = []
for name, kw in variants.items():
    a = LLMAgent(name=f"ablation_{MODEL.replace(':', '-')}", backend=backend, model=MODEL, cache_dir=LLM_CACHE, max_tokens=400)
    a.decide(build_payload(md_t, book, t, **kw))
    r = a.records[-1]
    rows.append({"variant": name, "prompt_tokens": r.prompt_tokens, "latency_s": round(r.latency_s, 1), "valid": r.ok})
pd.DataFrame(rows).set_index("variant")""")

md("""## 6. What to take from this

- **Validity** and **repairs** tell you whether this model can be trusted to follow the contract; a model that needs repairs on more than a few percent of cycles is not worth running for 670 cycles.
- **Seconds per decision × 670** is the real cost of a local backtest; for the headline run this is where a paid provider may earn its keep.
- **Prompt tokens as counted** feed the cost model in `BLUEPRINT.md` §7.

Next: pull `llama3.1:8b`, `gemma3:12b` and `qwen3:14b`, run this notebook once per model (just change `MODEL`), and put the four stats tables side by side. That is the local half of the provider evaluation.""")

nb["cells"] = cells
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = Path(__file__).resolve().parents[1] / "notebooks" / "02_first_llm_cycles.ipynb"
nbf.write(nb, out)
print("wrote", out)
