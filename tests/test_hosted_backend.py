"""The hosted (OpenAI-compatible) backend and the frontier configs — no network."""
import json
import os

import pytest

from alphabench.agents.llm import OpenAICompatibleBackend, load_env_file, LLMAgent
from alphabench.compare import default_configs, mark_runnable, make_agent, backend_key, slug, price_for


class _Resp:
    def __init__(self, status, payload):
        self.status_code, self._payload, self.text = status, payload, json.dumps(payload)

    def json(self):
        return self._payload


def _ok(content='{"reasoning":"r","market_context":"m","decisions":[]}'):
    return _Resp(200, {"id": "x", "model": "gpt-5", "usage": {"prompt_tokens": 100, "completion_tokens": 20},
                       "choices": [{"finish_reason": "stop", "message": {"content": content}}]})


def test_reasoning_models_get_the_right_parameters(monkeypatch):
    monkeypatch.setenv("TEST_KEY", "sk-test")
    b = OpenAICompatibleBackend(base_url="https://api.openai.com/v1", api_key_env="TEST_KEY", reasoning_effort="low")
    body = b.build_body("s", "u", "gpt-5", 0.0, {"type": "object"}, seed=7, max_tokens=900)
    assert "temperature" not in body and body["max_completion_tokens"] == 6900 and body["reasoning_effort"] == "low"
    assert body["response_format"]["type"] == "json_schema" and body["seed"] == 7
    body = b.build_body("s", "u", "gpt-4o", 0.0, None, seed=7, max_tokens=900)
    assert body["temperature"] == 0.0 and body["max_tokens"] == 900 and "reasoning_effort" not in body and "response_format" not in body
    assert b.name == "openai" and b.signature == {"base_url": "https://api.openai.com/v1", "reasoning_effort": "low"}


def test_complete_retries_transient_errors_then_succeeds(monkeypatch):
    monkeypatch.setenv("TEST_KEY", "sk-test")
    import requests
    calls = []

    def fake_post(url, json=None, timeout=None, headers=None):
        calls.append(url)
        return _Resp(429, {"error": "slow down"}) if len(calls) < 3 else _ok()

    monkeypatch.setattr(requests, "post", fake_post)
    monkeypatch.setattr("time.sleep", lambda s: None)
    b = OpenAICompatibleBackend(base_url="https://api.openai.com/v1", api_key_env="TEST_KEY", max_retries=4)
    r = b.complete("s", "u", "gpt-5", 0.0, None)
    assert len(calls) == 3 and r.prompt_tokens == 100 and r.raw["finish_reason"] == "stop"


def test_complete_does_not_retry_a_bad_request(monkeypatch):
    monkeypatch.setenv("TEST_KEY", "sk-test")
    import requests
    monkeypatch.setattr(requests, "post", lambda *a, **k: _Resp(400, {"error": "unsupported parameter"}))
    b = OpenAICompatibleBackend(base_url="https://api.openai.com/v1", api_key_env="TEST_KEY")
    with pytest.raises(RuntimeError, match="400"):
        b.complete("s", "u", "gpt-5", 0.0, None)


def test_missing_key_is_a_clear_error(monkeypatch):
    monkeypatch.delenv("NOPE_KEY", raising=False)
    b = OpenAICompatibleBackend(api_key_env="NOPE_KEY")
    with pytest.raises(RuntimeError, match="NOPE_KEY"):
        b.list_models()


def test_env_file_loader(tmp_path, monkeypatch):
    monkeypatch.delenv("ALPHABENCH_TEST_A", raising=False); monkeypatch.setenv("ALPHABENCH_TEST_B", "keep")
    (tmp_path / ".env").write_text('# comment\nALPHABENCH_TEST_A="sk-abc"\nALPHABENCH_TEST_B=override\n\nbroken line\n')
    loaded = load_env_file(tmp_path / ".env")
    assert os.environ["ALPHABENCH_TEST_A"] == "sk-abc" and os.environ["ALPHABENCH_TEST_B"] == "keep" and loaded == {"ALPHABENCH_TEST_A": "sk-abc"}
    assert load_env_file(tmp_path / "missing.env") == {}


def test_frontier_configs_and_agent_wiring(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_KEY", "sk-test")
    cs = default_configs(candidates="universe", max_position_weight=0.25, frontier_model="gpt-5", local=False)
    assert [c["name"] for c in cs] == ["gpt-5_v1_all50_cap25", "gpt-5_v2_all50_cap25", "gpt-5_v3_all50_cap25"]
    assert all(c["backend"] == "openai" and backend_key(c) == "openai" for c in cs)
    assert slug("openai/gpt-5.1") == "gpt-5-1" and price_for("gpt-5-mini") == (0.25, 2.0) and price_for("qwen3:8b") == (0.0, 0.0)
    both = default_configs(candidates="universe", max_position_weight=0.25, frontier_model="gpt-5")
    assert len(both) == 9 and both[0]["name"].startswith("qwen3-8b") and both[-1]["name"].startswith("gpt-5")
    mark_runnable(cs, set(), None, available_remote={"gpt-5", "gpt-4o"}); assert all(c["runnable"] for c in cs)
    mark_runnable(cs, set(), None, available_remote={"gpt-4o"}); assert {c["skip_reason"] for c in cs} == {"model not offered by the API"}
    mark_runnable(cs, set(), None); assert all(c["runnable"] for c in cs)          # not checked → runnable
    hosted = OpenAICompatibleBackend(base_url="https://api.openai.com/v1", api_key_env="TEST_KEY")
    a = make_agent(cs[2], tmp_path, {False: None, True: None, "openai": hosted})
    assert a.backend is hosted and a.price_in == 1.25 and "25%" in a.system_prompt and "every one of them a candidate" in a.system_prompt
    with pytest.raises(KeyError):
        make_agent(cs[2], tmp_path, {False: None, True: None})


def test_local_settings_py_is_loaded(tmp_path, monkeypatch):
    monkeypatch.delenv("ALPHABENCH_TEST_C", raising=False)
    (tmp_path / "local_settings.py").write_text('"""doc"""\nALPHABENCH_TEST_C = "sk-from-py"\nlower = "ignored"\nNUM = 3\n')
    assert load_env_file(tmp_path) == {"ALPHABENCH_TEST_C": "sk-from-py"} and os.environ["ALPHABENCH_TEST_C"] == "sk-from-py"
    assert load_env_file(tmp_path / "nothing") == {}
