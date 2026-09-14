"""LLM agent: provider-agnostic backends, on-disk response cache, structured
output, one repair attempt, and per-call usage accounting.

    backend = OllamaBackend()                          # local, free, offline
    agent   = LLMAgent("qwen3-8b", backend, model="qwen3:8b")
    run_replay(md, [agent, Momentum10()], start, end)

Reproducibility contract: temperature 0, a fixed seed where the backend
supports one, and every (model, options, system, user, schema) tuple hashed to
a cache key. Re-running a season replays from cache byte-for-byte; delete
`data/llm_cache/<agent>` to force fresh calls.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

from ..prompt import last_close, render_text, universe_rows
from ..schema import Decision, json_schema, parse_llm_decision
from .base import Agent

log = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path("data") / "llm_cache"
NUM_CTX = 32_768          # digit-splitting tokenizers make the prompt ~7-8k tokens; leave room for held names + repair

OUTPUT_CONTRACT = (
    "Respond with a single JSON object and nothing else: "
    '{"reasoning": string (<= 100 words), "market_context": string (<= 40 words), "decisions": [...]}. '
    "`decisions` lists ONLY the trades to execute this cycle, at most 3 items. "
    'A trade item is {"symbol", "action": "open_long"|"open_short"|"add", "percent_of_equity", "confidence", '
    '"thesis" (<= 30 words), "invalidation" (<= 20 words), "invalidation_price"} with EVERY field present. '
    "percent_of_equity is a PERCENTAGE between 10 and 100 (20 means 20% of equity, never 0.2). "
    "confidence is 0-1 and must be >= 0.80 to open. invalidation_price must be below the current price for "
    "open_long and above it for open_short. Use open_long/open_short for symbols you do NOT hold; use add "
    "ONLY for symbols listed under portfolio.positions and only if they show a profit. "
    'A close item is {"symbol", "action": "close"} for a symbol you hold. '
    'If there is nothing to do, return "decisions": []. Never list symbols you are not trading.'
)


@dataclass
class LLMResponse:
    text: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    latency_s: float = 0.0
    cached: bool = False
    raw: dict = field(default_factory=dict)


# ---- backends ----------------------------------------------------------------
class Backend:
    """Interface: complete(system, user, model, temperature, schema) -> LLMResponse."""
    name = "backend"

    def complete(self, system: str, user: str, model: str, temperature: float, schema: dict | None,
                 seed: int | None = None, max_tokens: int = 900) -> LLMResponse:  # pragma: no cover
        raise NotImplementedError

    def list_models(self) -> list[str]:  # pragma: no cover
        return []


class OllamaBackend(Backend):
    """Local Ollama server (https://ollama.com). Uses /api/chat with structured
    output (`format` = JSON schema) so the model is constrained to valid JSON."""
    name = "ollama"

    def __init__(self, host: str = "http://localhost:11434", num_ctx: int = NUM_CTX, think: bool | None = False, timeout: int = 600):
        self.host, self.num_ctx, self.think, self.timeout = host.rstrip("/"), num_ctx, think, timeout

    @property
    def signature(self) -> dict:
        """Backend options that change the answer (part of the cache key). num_ctx is
        deliberately excluded: it only matters when it truncates, which we now avoid."""
        return {"think": self.think}

    def _post(self, path: str, payload: dict) -> dict:
        import requests
        r = requests.post(f"{self.host}{path}", json=payload, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def list_models(self) -> list[str]:
        import requests
        r = requests.get(f"{self.host}/api/tags", timeout=10)
        r.raise_for_status()
        return sorted(m["name"] for m in r.json().get("models", []))

    def complete(self, system, user, model, temperature, schema, seed=None, max_tokens=900) -> LLMResponse:
        body = {
            "model": model, "stream": False,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "options": {"temperature": temperature, "num_ctx": self.num_ctx, "num_predict": max_tokens,
                        **({"seed": seed} if seed is not None else {})},
        }
        if schema:
            body["format"] = schema
        if self.think is not None:
            body["think"] = self.think    # qwen3 / deepseek-r1 style thinking; False = fast, deterministic-ish
        t0 = time.time()
        try:
            out = self._post("/api/chat", body)
        except Exception as e:  # models without a thinking mode reject the `think` field
            if "think" in body and "400" in str(e):
                body.pop("think"); out = self._post("/api/chat", body)
            else:
                raise
        return LLMResponse(
            text=out.get("message", {}).get("content", ""),
            prompt_tokens=out.get("prompt_eval_count"), completion_tokens=out.get("eval_count"),
            latency_s=time.time() - t0, raw={k: v for k, v in out.items() if k != "message"},
        )


def load_env_file(path: Path | str = ".env") -> dict[str, str]:
    """Read secrets into os.environ (existing variables win). Accepts a `.env` file of
    KEY=VALUE lines, or a Python file such as `local_settings.py` with `OPENAI_API_KEY = "sk-..."`
    at module level — both are gitignored, so keys never reach a notebook or a commit.
    Given a directory, tries `local_settings.py` then `.env` inside it."""
    import os
    p = Path(path)
    if p.is_dir():
        loaded = load_env_file(p / "local_settings.py")
        return loaded or load_env_file(p / ".env")
    loaded = {}
    if not p.exists():
        return loaded
    if p.suffix == ".py":
        ns: dict = {}
        exec(compile(p.read_text(), str(p), "exec"), ns)
        for k, v in ns.items():
            if k.isupper() and isinstance(v, str) and k not in os.environ:
                os.environ[k] = v
                loaded[k] = v
        return loaded
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v
            loaded[k] = v
    return loaded


#: models whose chat-completions API takes `max_completion_tokens`, ignores `temperature`
#: (fixed at 1) and accepts `reasoning_effort` — OpenAI's reasoning families.
REASONING_PREFIXES = ("gpt-5", "o1", "o3", "o4")
REASONING_TOKEN_BUDGET = 6_000   # extra completion tokens reserved for hidden reasoning


class OpenAICompatibleBackend(Backend):
    """Any OpenAI-style /chat/completions endpoint: OpenAI, OpenRouter, Groq,
    Together, Mistral, a local vLLM… `api_key_env` names the environment variable.

    Reasoning models (gpt-5*, o-series) are handled: `max_completion_tokens` instead of
    `max_tokens`, no `temperature`, and `reasoning_effort` (part of the cache signature).
    Transient failures (429, 5xx, timeouts) are retried with backoff."""
    name = "openai_compatible"

    def __init__(self, base_url: str = "https://openrouter.ai/api/v1", api_key_env: str = "OPENROUTER_API_KEY",
                 timeout: int = 300, extra_headers: dict | None = None, use_json_schema: bool = True,
                 reasoning_effort: str | None = None, max_retries: int = 4):
        import os
        self.base_url, self.timeout = base_url.rstrip("/"), timeout
        self.api_key_env = api_key_env
        self.api_key = os.environ.get(api_key_env, "")
        self.extra_headers, self.use_json_schema = extra_headers or {}, use_json_schema
        self.reasoning_effort, self.max_retries = reasoning_effort, max_retries
        self.name = "openrouter" if "openrouter" in base_url else "openai" if "api.openai.com" in base_url else "openai_compatible"

    @property
    def signature(self) -> dict:
        """Options that change the answer, hashed into the cache key."""
        sig = {"base_url": self.base_url}
        if self.reasoning_effort:
            sig["reasoning_effort"] = self.reasoning_effort
        return sig

    @staticmethod
    def is_reasoning_model(model: str) -> bool:
        m = model.lower().split("/")[-1]
        return m.startswith(REASONING_PREFIXES)

    def _headers(self) -> dict:
        if not self.api_key:
            raise RuntimeError(f"no API key: set {self.api_key_env} in the environment or in .env")
        return {"Authorization": f"Bearer {self.api_key}", **self.extra_headers}

    def list_models(self) -> list[str]:
        import requests
        r = requests.get(f"{self.base_url}/models", headers=self._headers(), timeout=60)
        r.raise_for_status()
        return sorted(m["id"] for m in r.json().get("data", []))

    def build_body(self, system, user, model, temperature, schema, seed=None, max_tokens=900) -> dict:
        body = {"model": model, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        if self.is_reasoning_model(model):
            body["max_completion_tokens"] = max_tokens + REASONING_TOKEN_BUDGET
            if self.reasoning_effort:
                body["reasoning_effort"] = self.reasoning_effort
        else:
            body["temperature"] = temperature
            body["max_tokens"] = max_tokens
        if seed is not None:
            body["seed"] = seed
        if schema and self.use_json_schema:
            body["response_format"] = {"type": "json_schema", "json_schema": {"name": "decision", "schema": schema, "strict": False}}
        elif schema:
            body["response_format"] = {"type": "json_object"}
        return body

    def complete(self, system, user, model, temperature, schema, seed=None, max_tokens=900) -> LLMResponse:
        import requests
        body = self.build_body(system, user, model, temperature, schema, seed, max_tokens)
        t0 = time.time()
        last_err: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                r = requests.post(f"{self.base_url}/chat/completions", json=body, timeout=self.timeout, headers=self._headers())
                if r.status_code in (429, 500, 502, 503, 504):
                    raise requests.HTTPError(f"{r.status_code}: {r.text[:300]}", response=r)
                if r.status_code >= 400:
                    raise RuntimeError(f"{self.name} {r.status_code}: {r.text[:500]}")   # a 4xx other than 429 will not fix itself
                break
            except (requests.HTTPError, requests.ConnectionError, requests.Timeout) as e:
                last_err = e
                if attempt == self.max_retries:
                    raise
                wait = min(60, 2 ** attempt * 2)
                log.warning("%s transient error (%s); retry %d/%d in %ds", self.name, str(e)[:120], attempt + 1, self.max_retries, wait)
                time.sleep(wait)
        out = r.json()
        usage = out.get("usage", {})
        choice = out["choices"][0]
        return LLMResponse(text=choice["message"]["content"] or "", prompt_tokens=usage.get("prompt_tokens"),
                           completion_tokens=usage.get("completion_tokens"), latency_s=time.time() - t0,
                           raw={"id": out.get("id"), "model": out.get("model"), "usage": usage, "finish_reason": choice.get("finish_reason")})


class MockBackend(Backend):
    """Deterministic stand-in for tests and offline notebooks. `responder(system, user) -> str`."""
    name = "mock"

    def __init__(self, responder):
        self.responder, self.calls = responder, 0

    def complete(self, system, user, model, temperature, schema, seed=None, max_tokens=900) -> LLMResponse:
        self.calls += 1
        text = self.responder(system, user)
        return LLMResponse(text=text, prompt_tokens=len(system + user) // 4, completion_tokens=len(text) // 4, latency_s=0.0)


# ---- cache ---------------------------------------------------------------------
class ResponseCache:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def key(**parts) -> str:
        return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:24]

    def get(self, key: str) -> LLMResponse | None:
        p = self.root / f"{key}.json"
        if not p.exists():
            return None
        d = json.loads(p.read_text())
        return LLMResponse(cached=True, **{k: v for k, v in d["response"].items() if k != "cached"})

    def put(self, key: str, system: str, user: str, model: str, response: LLMResponse, meta: dict) -> None:
        (self.root / f"{key}.json").write_text(json.dumps(
            {"model": model, "meta": meta, "system": system, "user": user, "response": asdict(response)}, default=str))


# ---- the agent -------------------------------------------------------------------
@dataclass
class CallRecord:
    decision_date: str
    attempt: int
    ok: bool
    error: str | None
    prompt_tokens: int | None
    completion_tokens: int | None
    latency_s: float
    cached: bool
    text: str
    notes: list[str] = field(default_factory=list)
    prompt_eval_s: float | None = None   # time reading the prompt (Ollama reports it)
    eval_s: float | None = None          # time generating the answer


class LLMAgent(Agent):
    """Renders the payload, calls the backend, validates, repairs once, else holds."""

    def __init__(self, name: str, backend: Backend, model: str, temperature: float = 0.0, seed: int | None = 7,
                 system_prompt: str | None = None, use_schema: bool = True, cache_dir: Path | None = None,
                 max_tokens: int = 900, price_per_mtok: tuple[float, float] = (0.0, 0.0)):
        self.name, self.backend, self.model, self.temperature, self.seed = name, backend, model, temperature, seed
        self.system_prompt, self.use_schema, self.max_tokens = system_prompt, use_schema, max_tokens
        self.price_in, self.price_out = price_per_mtok           # $ per million tokens (input, output)
        self.cache = ResponseCache(Path(cache_dir or DEFAULT_CACHE_DIR) / name)
        self.schema = json_schema(for_llm=True) if use_schema else None
        self.records: list[CallRecord] = []
        self.tradeable = None

    # -- prompt assembly
    def messages(self, payload: dict) -> tuple[str, str]:
        system = (self.system_prompt or payload["mandate"]) + "\n\n" + OUTPUT_CONTRACT
        body = {k: v for k, v in payload.items() if k != "mandate" and not k.startswith("_")}   # "_"-keys are agent-side helpers
        user = f"DATA (as of {payload['decision_date']} close):\n" + json.dumps(body, separators=(",", ":"), default=str)
        return system, user

    def _call(self, system: str, user: str, decision_date: str, attempt: int, payload: dict | None = None) -> tuple[Decision | None, str | None]:
        key = self.cache.key(backend=self.backend.name, model=self.model, temperature=self.temperature, seed=self.seed,
                             system=system, user=user, schema=self.schema, max_tokens=self.max_tokens,
                             **({"backend_opts": self.backend.signature} if getattr(self.backend, "signature", None) else {}))
        resp = self.cache.get(key)
        if resp is None:
            resp = self.backend.complete(system, user, self.model, self.temperature, self.schema, self.seed, self.max_tokens)
            self.cache.put(key, system, user, self.model, resp, {"agent": self.name, "decision_date": decision_date, "attempt": attempt})
        notes: list[str] = []
        try:
            obj = _normalise(json.loads(_strip_fences(resp.text)), notes)
        except json.JSONDecodeError as e:
            obj = resp.text
            notes.append(f"not JSON: {e}")
        decision, err = parse_llm_decision(obj)
        if decision is not None and payload is not None:
            problems = _precheck(decision, payload)
            if problems:
                decision, err = None, "Rule problems:\n- " + "\n- ".join(problems)
        raw = resp.raw or {}
        self.records.append(CallRecord(decision_date, attempt, decision is not None, err, resp.prompt_tokens,
                                       resp.completion_tokens, resp.latency_s, resp.cached, resp.text, notes,
                                       prompt_eval_s=(raw.get("prompt_eval_duration") or 0) / 1e9 or None,
                                       eval_s=(raw.get("eval_duration") or 0) / 1e9 or None))
        return decision, err

    def decide(self, payload: dict) -> Decision:
        system, user = self.messages(payload)
        d = payload["decision_date"]
        decision, err = self._call(system, user, d, attempt=1, payload=payload)
        if decision is None:
            repair_user = (user + "\n\nYour previous output was invalid:\n" + (err or "")[:1500]
                           + "\n\nReturn a corrected JSON object only.")
            decision, err = self._call(system, repair_user, d, attempt=2, payload=payload)
        if decision is None:
            log.warning("%s %s: invalid after repair → hold", self.name, d)
            return Decision.hold(f"invalid output after repair: {(err or '')[:200]}")
        return decision

    # -- accounting
    def stats(self) -> dict:
        r = self.records
        if not r:
            return {}
        pin = sum(x.prompt_tokens or 0 for x in r); pout = sum(x.completion_tokens or 0 for x in r)
        live = [x for x in r if not x.cached]
        return {
            "calls": len(r), "cached": sum(x.cached for x in r), "repairs": sum(x.attempt == 2 for x in r),
            "invalid_final": sum(1 for x in r if x.attempt == 2 and not x.ok),
            "normalised": sum(1 for x in r if any("read as" in n for n in x.notes)),
            "prompt_tokens": pin, "completion_tokens": pout,
            "avg_prompt_tokens": pin / len(r), "avg_completion_tokens": pout / len(r),
            "avg_latency_s": (sum(x.latency_s for x in live) / len(live)) if live else 0.0,
            "tokens_per_s": (sum((x.completion_tokens or 0) for x in live) / max(sum(x.latency_s for x in live), 1e-9)) if live else 0.0,
            "prompt_tokens_per_s": (sum((x.prompt_tokens or 0) for x in live if x.prompt_eval_s) / max(sum(x.prompt_eval_s or 0 for x in live), 1e-9)) if any(x.prompt_eval_s for x in live) else None,
            "gen_tokens_per_s": (sum((x.completion_tokens or 0) for x in live if x.eval_s) / max(sum(x.eval_s or 0 for x in live), 1e-9)) if any(x.eval_s for x in live) else None,
            "cost_usd": pin / 1e6 * self.price_in + pout / 1e6 * self.price_out,          # what the whole run would cost live
            "cost_live_usd": (sum(x.prompt_tokens or 0 for x in live) / 1e6 * self.price_in
                              + sum(x.completion_tokens or 0 for x in live) / 1e6 * self.price_out),   # what this pass actually spent
        }


def _normalise(obj: dict, notes: list[str]) -> dict:
    """Tolerances for common small-model slips, each logged in `notes`:
    percent_of_equity given as a fraction (0.15 -> 15). Values are only touched
    when they are unambiguous (a valid percentage is never below 10)."""
    for d in obj.get("decisions", []) if isinstance(obj, dict) else []:
        p = d.get("percent_of_equity") if isinstance(d, dict) else None
        if isinstance(p, (int, float)) and 0 < p < 1:
            d["percent_of_equity"] = round(p * 100, 2)
            notes.append(f"{d.get('symbol')}: percent_of_equity {p} read as {d['percent_of_equity']}%")
    return obj


def _price_of(payload: dict, symbol: str) -> float | None:
    for row in universe_rows(payload):
        if row["symbol"] == symbol:
            return float(row["price"])
    return last_close(payload.get("detail", {}).get(symbol))


def _precheck(decision: Decision, payload: dict) -> list[str]:
    """Rule problems the model can fix if told: invalidation on the wrong side,
    opening a symbol already held. Returns human-readable error strings."""
    held = {p["symbol"] for p in payload.get("portfolio", {}).get("positions", [])}
    errs = []
    for d in decision.decisions:
        px = _price_of(payload, d.symbol)
        if d.action in ("open_long", "open_short", "add") and px and d.invalidation_price:
            if d.action != "open_short" and d.invalidation_price >= px:
                errs.append(f"{d.symbol}: invalidation_price {d.invalidation_price} must be BELOW the current price {px:.2f} for a long")
            if d.action == "open_short" and d.invalidation_price <= px:
                errs.append(f"{d.symbol}: invalidation_price {d.invalidation_price} must be ABOVE the current price {px:.2f} for a short")
        if d.action in ("open_long", "open_short") and d.symbol in held:
            errs.append(f"{d.symbol} is already held — use 'add' or 'close'")
        if d.action in ("add", "close") and d.symbol not in held:
            errs.append(f"{d.symbol} is not held (portfolio.positions is {sorted(held) or 'empty'}) — use open_long/open_short or drop it")
    return errs


SUSPICIOUS_COUNTS = {2048, 4096, 8192, 16384, 32768}


def truncation_warning(prompt_tokens: int | None, num_ctx: int | None = None) -> str | None:
    """Ollama silently truncates prompts to its context window; the symptom is a
    prompt_eval_count sitting exactly at a power of two (+/- a few tokens)."""
    if prompt_tokens is None:
        return None
    for n in SUSPICIOUS_COUNTS:
        if abs(prompt_tokens - n) <= 4:
            return (f"prompt_tokens={prompt_tokens} is suspiciously close to {n}: the backend probably TRUNCATED the prompt. "
                    f"Raise the context length in the Ollama app settings (requested num_ctx={num_ctx}).")
    return None


def _strip_fences(text: str) -> str:
    """Tolerate ```json fences and leading prose before the first '{'."""
    t = text.strip()
    if t.startswith("```"):
        t = t.split("```", 2)[1]
        t = t[4:] if t.startswith("json") else t
    i, j = t.find("{"), t.rfind("}")
    return t[i:j + 1] if i >= 0 and j > i else t
