"""Shared runner for the model / prompt comparison (notebook 03 and scripts/run_compare.py)."""

from __future__ import annotations

import sys
import time
from pathlib import Path

from .agents import LLMAgent, OllamaBackend
from .prompt import INVESTOR_MANDATE, INVESTOR_MANDATE_V2, mandate
from .replay import run_replay

PROMPTS = {"v1": INVESTOR_MANDATE, "v2": INVESTOR_MANDATE_V2, "v3": mandate("v3", "screened")}

#: $ per million tokens (input, output) for cost accounting; 0 for local models. Check the
#: provider's price list before trusting a total — these are entered by hand and go stale.
PRICES_PER_MTOK = {   # list prices seen on a third-party tracker, 11 Sep 2026 — verify against platform.openai.com/pricing
    "gpt-5.6-sol": (4.0, 20.0), "gpt-5.6-terra": (2.0, 12.0), "gpt-5.6-luna": (0.2, 1.2),
    "gpt-5.5-pro": (30.0, 180.0), "gpt-5.5": (5.0, 30.0),
    "gpt-5.4-pro": (30.0, 180.0), "gpt-5.4-mini": (0.75, 4.5), "gpt-5.4-nano": (0.2, 1.25), "gpt-5.4": (2.5, 15.0),
    "gpt-5.2": (1.75, 14.0), "gpt-5.1": (1.25, 10.0), "gpt-5-mini": (0.25, 2.0), "gpt-5-nano": (0.05, 0.40), "gpt-5": (1.25, 10.0),
    "gpt-4.1-mini": (0.4, 1.6), "gpt-4.1-nano": (0.1, 0.4), "gpt-4.1": (2.0, 8.0), "gpt-4o-mini": (0.15, 0.6), "gpt-4o": (2.5, 10.0),
    "o4-mini": (1.1, 4.4), "o3-mini": (1.1, 4.4), "o3": (2.0, 8.0),
}


def slug(model: str) -> str:
    """'gpt-5' -> 'gpt-5', 'openai/gpt-5.1' -> 'gpt-5-1', 'qwen3:8b' -> 'qwen3-8b' — a config-name-safe model id."""
    return model.split("/")[-1].replace(":", "-").replace(".", "-")


def price_for(model: str) -> tuple[float, float]:
    m = model.split("/")[-1]
    for k in sorted(PRICES_PER_MTOK, key=len, reverse=True):     # longest prefix wins (gpt-5-mini before gpt-5)
        if m.startswith(k):
            return PRICES_PER_MTOK[k]
    return (0.0, 0.0)


def default_configs(think_enabled: bool = False, candidates: str = "screened", max_position_weight: float | None = None,
                    frontier_model: str | None = None, frontier_prompts: tuple[str, ...] = ("v1", "v2", "v3"),
                    local: bool = True) -> list[dict]:
    """prompt effect: same model, three prompts; model effect: strongest prompt on three models.
    candidates='universe' names the configs *_all50, a position cap adds _cap<pct>, so results and cache stay apart.
    frontier_model adds the same prompts on a hosted model through the OpenAI-compatible backend
    (backend='openai'); local=False drops the Ollama configs so a run is the frontier model alone."""
    sfx = ("" if candidates == "screened" else "_all50") + ("" if max_position_weight is None else f"_cap{int(round(max_position_weight * 100))}")
    frontier = [dict(name=f"{slug(frontier_model)}_{v}{sfx}", model=frontier_model, prompt=v, think=False, enabled=True, backend="openai",
                     candidates=candidates, max_position_weight=max_position_weight, price_per_mtok=price_for(frontier_model))
                for v in frontier_prompts] if frontier_model else []
    if not local:
        return frontier
    return [
        dict(name=f"qwen3-8b_v1{sfx}",          model="qwen3:8b",   prompt="v1", think=False, enabled=True,          candidates=candidates, max_position_weight=max_position_weight),
        dict(name=f"qwen3-8b_v2{sfx}",          model="qwen3:8b",   prompt="v2", think=False, enabled=True,          candidates=candidates, max_position_weight=max_position_weight),
        dict(name=f"qwen3-8b_v3{sfx}",          model="qwen3:8b",   prompt="v3", think=False, enabled=True,          candidates=candidates, max_position_weight=max_position_weight),
        dict(name=f"qwen3-14b_v3{sfx}",         model="qwen3:14b",  prompt="v3", think=False, enabled=True,          candidates=candidates, max_position_weight=max_position_weight),
        dict(name=f"gemma3-12b_v3{sfx}",        model="gemma3:12b", prompt="v3", think=False, enabled=True,          candidates=candidates, max_position_weight=max_position_weight),
        dict(name=f"qwen3-8b_v3_thinking{sfx}", model="qwen3:8b",   prompt="v3", think=True,  enabled=think_enabled, candidates=candidates, max_position_weight=max_position_weight),
    ] + frontier


def mark_runnable(configs: list[dict], available: set[str], only: list[str] | None = None,
                  available_remote: set[str] | None = None) -> list[dict]:
    """available = models Ollama has pulled; available_remote = model ids the hosted API lists
    (None = not checked: a remote config is runnable whenever it is wanted)."""
    for c in configs:
        if c.get("backend") == "openai":
            pulled = True if available_remote is None else any(m == c["model"] or m.startswith(c["model"]) for m in available_remote)
            why = "model not offered by the API"
        else:
            pulled = any(m.startswith(c["model"]) for m in available)
            why = "model not pulled"
        wanted = c["enabled"] and (not only or c["name"] in only)
        c["runnable"] = wanted and pulled
        c["skip_reason"] = None if c["runnable"] else (why if wanted else "disabled" if not c["enabled"] else "not selected")
    return configs


def backend_key(c: dict):
    """Which entry of the `backends` dict serves this config: 'openai' for hosted, else the Ollama think flag."""
    return "openai" if c.get("backend") == "openai" else c["think"]


def make_agent(c: dict, cache_dir: Path, backends: dict) -> LLMAgent:
    key = backend_key(c)
    if key not in backends:
        raise KeyError(f"config {c['name']} needs backend {key!r}; backends has {list(backends)}")
    return LLMAgent(name=c["name"], backend=backends[key], model=c["model"], temperature=0.0, seed=7,
                    system_prompt=mandate(c["prompt"], c.get("candidates", "screened"), c.get("max_position_weight")), cache_dir=cache_dir,
                    price_per_mtok=tuple(c.get("price_per_mtok", (0.0, 0.0))))


def run_configs(md, configs: list[dict], start, end, results_dir: Path, cache_dir: Path,
                backends: dict, out=sys.stdout, payload_kwargs: dict | None = None) -> tuple[dict, dict, dict]:
    """Run each runnable config over the same window with per-cycle progress. Returns (results, agents, timings).
    Each config's `candidates` mode is passed to the payload builder (overrides payload_kwargs)."""
    results, agents, timings = {}, {}, {}
    for c in configs:
        if not c.get("runnable"):
            continue
        agent = make_agent(c, cache_dir, backends)
        t0 = time.time()
        print(f"\n→ {c['name']}  ({c['model']}, prompt {c['prompt']}{', thinking' if c['think'] else ''}{', hosted' if c.get('backend') == 'openai' else ''})", file=out, flush=True)

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
        cost = f" | spent ${st['cost_live_usd']:.2f} (run value ${st['cost_usd']:.2f})" if st.get("cost_usd") else ""
        print(f"  done in {timings[c['name']]/60:.0f} min | {st['calls']} calls ({st['cached']} cached) | repairs {st['repairs']} | "
              f"invalid {st['invalid_final']} | {st['avg_latency_s']:.0f}s/live call | final equity {res[c['name']]['equity'].iloc[-1]:,.0f}{cost}", file=out, flush=True)
    return results, agents, timings
