"""Tests for the hosted-OpenAI backend (offline, faked transport)."""

import types

from service.backends import base
from service.backends import openai_cloud_backend

from typing import Any


class _FakeCompletions:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        message = types.SimpleNamespace(content="hi", tool_calls=None)
        return types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=message)], usage=None
        )


class _FakeClient:
    def __init__(self) -> None:
        self.completions = _FakeCompletions()

    @property
    def chat(self) -> Any:
        return types.SimpleNamespace(completions=self.completions)

    def models(self) -> Any:
        raise AssertionError("unused")


def _backend() -> tuple[Any, _FakeClient]:
    """Builds a cloud backend with faked transport."""
    backend = openai_cloud_backend.OpenAICloudBackend.__new__(
        openai_cloud_backend.OpenAICloudBackend
    )
    from service.tools import registry

    backend._model = "gpt-5.6-terra"
    backend._system_prompt = "sys"
    backend._base_url = "https://api.openai.com/v1"
    backend._max_tokens = 32
    backend._context_window = None
    backend._compact_model = "gpt-5.6-terra"
    backend._compact_summary_max_tokens = 32
    backend._compact_keep = 20
    tools = registry.ToolRegistry()
    backend._tools = tools
    import hashlib

    from service import cache_policy

    backend._cache_gate = cache_policy.CacheGate(
        freshness_window_s=30 * 60, idle_disable_after=3
    )
    digest = hashlib.sha256(b"sys").hexdigest()[:16]
    backend._cache_key = f"peer-agent:{digest}"
    client = _FakeClient()
    backend._client = client  # type: ignore[assignment]
    return backend, client


def test_explicit_cache_headers_sent() -> None:
    """First request carries explicit mode, ttl, key, and breakpoint."""
    backend, client = _backend()
    messages: list[dict[str, Any]] = [{"role": "user", "content": "hi"}]
    backend._request_turn(messages, base.ToolContext(workdir=None))
    call = client.completions.calls[0]
    assert call["max_completion_tokens"] == 32
    assert "max_tokens" not in call
    assert call["extra_body"] == {
        "prompt_cache_options": {"mode": "explicit", "ttl": "30m"},
        "prompt_cache_key": backend._cache_key,
    }
    system = call["messages"][0]
    assert system["content"][0]["prompt_cache_breakpoint"] == {"mode": "explicit"}


def test_earlier_models_send_plain() -> None:
    """Pre-5.6 models get no breakpoint or cache options (they 400 them)."""
    backend, client = _backend()
    backend._model = "gpt-5.4-mini"
    messages: list[dict[str, Any]] = [{"role": "user", "content": "hi"}]
    backend._request_turn(messages, base.ToolContext(workdir=None))
    call = client.completions.calls[0]
    assert call["extra_body"] is None
    assert call["messages"][0] == {"role": "system", "content": "sys"}


def test_explicit_cache_generation_gate() -> None:
    """Only 5.6+ generations are marked cache-capable."""
    supports = openai_cloud_backend.OpenAICloudBackend._supports_explicit_cache
    assert supports("gpt-5.6-terra") is True
    assert supports("gpt-6-sol") is True
    assert supports("gpt-5.4-mini") is False
    assert supports("gpt-5") is False
    assert supports("gpt-4o") is False
    assert supports("o3") is False


def test_gate_suspends_explicit_caching() -> None:
    """After sustained idleness the cache options drop off."""
    backend, client = _backend()
    # Force three over-window gaps through the real gate clock.
    import time

    now = time.monotonic()
    backend._cache_gate.should_cache(now=now - 3 * 31 * 60)
    backend._cache_gate.should_cache(now=now - 2 * 31 * 60)
    backend._cache_gate.should_cache(now=now - 31 * 60)
    messages: list[dict[str, Any]] = [{"role": "user", "content": "hi"}]
    backend._request_turn(messages, base.ToolContext(workdir=None))
    assert client.completions.calls[0]["extra_body"] is None


def test_write_tokens_mapped() -> None:
    """OpenAI cache_write_tokens land in cache_creation_input_tokens."""
    usage = types.SimpleNamespace(
        prompt_tokens=100,
        completion_tokens=10,
        total_tokens=110,
        prompt_tokens_details=types.SimpleNamespace(cached_tokens=40),
        input_tokens_details=types.SimpleNamespace(cache_write_tokens=60),
    )
    mapped = openai_cloud_backend.OpenAICloudBackend._to_token_usage(usage)
    assert mapped.cached_prompt_tokens == 40
    assert mapped.cache_creation_input_tokens == 60


def test_describe_names_openai() -> None:
    """Identity distinguishes cloud from compat in logs."""
    backend, _ = _backend()
    assert backend.describe() == {"model": "gpt-5.6-terra", "backend": "openai"}
