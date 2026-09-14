"""Run the model / prompt comparison from a terminal (no Jupyter needed; safe to interrupt and resume).

    conda activate alphabench
    python scripts/run_compare.py                      # all enabled configs, 60 cycles from 2026-06-01
    python scripts/run_compare.py --cycles 10          # quick look
    python scripts/run_compare.py --only qwen3-8b_v3   # one config
    python scripts/run_compare.py --think              # also run the thinking-mode config
    python scripts/run_compare.py --candidates screened # the original top-5 screener funnel (default: universe = all 50 names)
    python scripts/run_compare.py --cap 0.25           # add the 25% per-name position limit
    python scripts/run_compare.py --cap 0.25 --frontier gpt-5.1 --frontier-only   # the same three prompts on a hosted model
    python scripts/run_compare.py --list-models        # what the hosted API offers (needs the key)

The hosted backend reads OPENAI_API_KEY from the environment, from local_settings.py in the repo root
(copy local_settings.example.py, paste the key) or from a .env file — all gitignored. --base-url /
--api-key-env switch provider.

Then open the matching notebook — 03_model_comparison without --cap, 03_new_rule_model_comparison
with it. Every call is cached, so its section 3 completes instantly.
"""
import argparse, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from alphabench.universe import ALL_SYMBOLS
from alphabench.market import load_prices
from alphabench.agents import OllamaBackend, OpenAICompatibleBackend, load_env_file
from alphabench.replay import decision_dates
from alphabench.compare import default_configs, mark_runnable, run_configs

ap = argparse.ArgumentParser()
ap.add_argument("--start", default="2026-06-01"); ap.add_argument("--cycles", type=int, default=60)
ap.add_argument("--only", nargs="*", default=None); ap.add_argument("--think", action="store_true")
ap.add_argument("--candidates", choices=["screened", "universe"], default="universe")
ap.add_argument("--cap", type=float, default=None, help="per-name position limit as a share of equity, e.g. 0.25 (default: none)")
ap.add_argument("--frontier", nargs="?", const="gpt-5.1", default=None, help="hosted model id to run the same prompts on (bare flag = gpt-5.1)")
ap.add_argument("--frontier-only", action="store_true", help="skip the local Ollama configs")
ap.add_argument("--frontier-prompts", nargs="*", default=["v1", "v2", "v3"])
ap.add_argument("--reasoning", default="low", choices=["minimal", "low", "medium", "high"], help="reasoning_effort for reasoning models")
ap.add_argument("--base-url", default="https://api.openai.com/v1"); ap.add_argument("--api-key-env", default="OPENAI_API_KEY")
ap.add_argument("--list-models", action="store_true", help="print the hosted API's model ids and exit")
args = ap.parse_args()

load_env_file(ROOT)          # local_settings.py or .env in the repo root
hosted = OpenAICompatibleBackend(base_url=args.base_url, api_key_env=args.api_key_env, reasoning_effort=args.reasoning)
if args.list_models:
    print("\n".join(hosted.list_models())); sys.exit(0)

md = load_prices(ALL_SYMBOLS, "2023-09-01", None, ROOT / "data" / "prices.parquet")
dds = decision_dates(md, args.start)[:args.cycles]
print(f"{len(dds)} cycles: {dds[0].date()} → {dds[-1].date()}")

backends = {False: OllamaBackend(think=False), True: OllamaBackend(think=True), "openai": hosted}
local_models = set() if args.frontier_only else set(backends[False].list_models())
remote_models = set(hosted.list_models()) if args.frontier else None
configs = mark_runnable(default_configs(think_enabled=args.think, candidates=args.candidates, max_position_weight=args.cap,
                                        frontier_model=args.frontier, frontier_prompts=tuple(args.frontier_prompts), local=not args.frontier_only),
                        local_models, args.only, available_remote=remote_models)
for c in configs:
    print(f"{'▶' if c['runnable'] else '–'} {c['name']:28s} {c['model']:14s} {c['skip_reason'] or ''}")
if not any(c["runnable"] for c in configs):
    sys.exit("nothing to run")
results_dir = ROOT / "data" / "results" / (("compare" if args.candidates == "screened" else "compare_universe") + ("" if args.cap is None else f"_cap{int(round(args.cap * 100))}"))
run_configs(md, configs, dds[0], dds[-1], results_dir, ROOT / "data" / "llm_cache", backends)
notebook = "03_new_rule_model_comparison" if args.cap is not None else "03_model_comparison"
print(f"\nall done — open notebooks/{notebook}.ipynb for the comparison tables")
