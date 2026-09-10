"""Run the model / prompt comparison from a terminal (no Jupyter needed; safe to interrupt and resume).

    conda activate alphabench
    python scripts/run_compare.py                      # all enabled configs, 60 cycles from 2026-06-01
    python scripts/run_compare.py --cycles 10          # quick look
    python scripts/run_compare.py --only qwen3-8b_v3   # one config
    python scripts/run_compare.py --think              # also run the thinking-mode config
    python scripts/run_compare.py --candidates screened # the original top-5 screener funnel (default: universe = all 50 names)
    python scripts/run_compare.py --cap 0.25           # add the 25% per-name position limit

Then open the matching notebook — 03_model_comparison without --cap, 03_new_rule_model_comparison
with it. Every call is cached, so its section 3 completes instantly.
"""
import argparse, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from alphabench.universe import ALL_SYMBOLS
from alphabench.market import load_prices
from alphabench.agents import OllamaBackend
from alphabench.replay import decision_dates
from alphabench.compare import default_configs, mark_runnable, run_configs

ap = argparse.ArgumentParser()
ap.add_argument("--start", default="2026-06-01"); ap.add_argument("--cycles", type=int, default=60)
ap.add_argument("--only", nargs="*", default=None); ap.add_argument("--think", action="store_true")
ap.add_argument("--candidates", choices=["screened", "universe"], default="universe")
ap.add_argument("--cap", type=float, default=None, help="per-name position limit as a share of equity, e.g. 0.25 (default: none)")
args = ap.parse_args()

md = load_prices(ALL_SYMBOLS, "2023-09-01", None, ROOT / "data" / "prices.parquet")
dds = decision_dates(md, args.start)[:args.cycles]
print(f"{len(dds)} cycles: {dds[0].date()} → {dds[-1].date()}")

backends = {False: OllamaBackend(think=False), True: OllamaBackend(think=True)}
configs = mark_runnable(default_configs(think_enabled=args.think, candidates=args.candidates, max_position_weight=args.cap), set(backends[False].list_models()), args.only)
for c in configs:
    print(f"{'▶' if c['runnable'] else '–'} {c['name']:24s} {c['model']:12s} {c['skip_reason'] or ''}")
results_dir = ROOT / "data" / "results" / (("compare" if args.candidates == "screened" else "compare_universe") + ("" if args.cap is None else f"_cap{int(round(args.cap * 100))}"))
run_configs(md, configs, dds[0], dds[-1], results_dir, ROOT / "data" / "llm_cache", backends)
notebook = "03_new_rule_model_comparison" if args.cap is not None else "03_model_comparison"
print(f"\nall done — open notebooks/{notebook}.ipynb for the comparison tables")
