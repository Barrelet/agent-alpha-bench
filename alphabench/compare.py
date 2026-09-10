"""Shared runner for the model / prompt comparison (notebook 03 and scripts/run_compare.py)."""

from __future__ import annotations

import sys
import time
from pathlib import Path

from .agents import LLMAgent, OllamaBackend
from .prompt import INVESTOR_MANDATE, INVESTOR_MANDATE_V2, mandate
from .replay import run_replay

PROMPTS = {"v1": INVESTOR_MANDATE, "v2": INVESTOR_MANDATE_V2, "v3": mandate("v3", "screened")}


def default_configs(think_enabled: bool = False, candidates: str = "screened", max_position_weight: float | None = None) -> list[dict]:
    """prompt effect: same model, three prompts; model effect: strongest prompt on three models.
    candidates='universe' names the configs *_all50, a position cap adds _cap<pct>, so results and cache stay apart."""
    sfx = ("" if candidates == "screened" else "_all50") + ("" if max_position_weight is None else f"_cap{int(round(max_position_weight * 100))}")
    return [
        dict(name=f"qwen3-8b_v1{sfx}",          model="qwen3:8b",   prompt="v1", think=False, enabled=True,          candidates=candidates, max_position_weight=max_position_weight),
        dict(name=f"qwen3-8b_v2{sfx}",          model="qwen3:8b",   prompt="v2", think=False, enabled=True,          candidates=candidates, max_position_weight=max_position_weight),
        dict(name=f"qwen3-8b_v3{sfx}",          model="qwen3:8b",   prompt="v3", think=False, enabled=True,          candidates=candidates, max_position_weight=max_position_weight),
        dict(name=f"qwen3-14b_v3{sfx}",         model="qwen3:14b",  prompt="v3", think=False, enabled=True,          candidates=candidates, max_position_weight=max_position_weight),
        dict(name=f"gemma3-12b_v3{sfx}",        model="gemma3:12b", prompt="v3", think=False, enabled=True,          candidates=candidates, max_position_weight=max_position_weight),
        dict(name=f"qwen3-8b_v3_thinking{sfx}", model="qwen3:8b",   prompt="v3", think=True,  enabled=think_enabled, candidates=candidates, max_position_weight=max_position_weight),
    ]


def mark_runnable(configs: list[dict], available: set[str], only: list[str] | None = None) -> list[dict]:
    for c in configs:
        pulled = any(m.startswith(c["model"]) for m in available)
        wanted = c["enabled"] and (not only or c["name"] in only)
        c["runnable"] = wanted and pulled
        c["skip_reason"] = None if c["runnable"] else ("model not pulled" if wanted else "disabled" if not c["enabled"] else "not selected")
    return configs


def make_agent(c: dict, cache_dir: Path, backends: dict[bool, OllamaBackend]) -> LLMAgent:
    return LLMAgent(name=c["name"], backend=backends[c["think"]], model=c["model"], temperature=0.0, seed=7,
                    system_prompt=mandate(c["prompt"], c.get("candidates", "screened"), c.get("max_position_weight")), cache_dir=cache_dir)


def run_configs(md, configs: list[dict], start, end, results_dir: Path, cache_dir: Path,
                backends: dict[bool, OllamaBackend], out=sys.stdout, payload_kwargs: dict | None = None) -> tuple[dict, dict, dict]:
    """Run each runnable config over the same window with per-cycle progress. Returns (results, agents, timings).
    Each config's `candidates` mode is passed to the payload builder (overrides payload_kwargs)."""
    results, agents, timings = {}, {}, {}
    for c in configs:
        if not c.get("runnable"):
            continue
        agent = make_agent(c, cache_dir, backends)
        t0 = time.time()
        print(f"\n→ {c['name']}  ({c['model']}, prompt {c['prompt']}{', thinking' if c['think'] else ''})", file=out, flush=True)

        def on_cycle(i, n, t, eq, elapsed, _name=c["name"], _agent=agent):
            r = _agent.records[-1] if _agent.records else None
            tag = "cached" if (r and r.cached) else f"{(r.latency_s if r else 0):.0f}s"
            eta = (elapsed / i) * (n - i) / 60
            print(f"  cycle {i:3d}/{n} {t.date()}  {tag:>7}  equity {eq[_name]:>9,.0f}  elapsed {elapsed/60:4.0f} min  eta {eta:4.0f} min", file=out, flush=True)

        pk = {**(payload_kwargs or {}), "candidates": c.get("candidates", "screened")}
        res = run_replay(md, [agent], start, end, log_dir=results_dir / c["name"], progress=False, on_cycle=on_cycle, payload_kwargs=pk,
                         max_position_weight=c.get("max_position_weight"))
        timings[c["name"]] = time.time() - t0
        results[c["name"]] = res[c["name"]]; agents[c["name"]] = agent
        st = agent.stats()
        print(f"  done in {timings[c['name']]/60:.0f} min | {st['calls']} calls ({st['cached']} cached) | repairs {st['repairs']} | "
              f"invalid {st['invalid_final']} | {st['avg_latency_s']:.0f}s/live call | final equity {res[c['name']]['equity'].iloc[-1]:,.0f}", file=out, flush=True)
    return results, agents, timings
