"""Hosted-OpenAI backend: explicit prompt caching over the compat loop.

GPT-5.6+ charges 1.25x for cache writes, so the default implicit mode
loses money on agentic turns that never repeat. This subclass pins
explicit mode with one breakpoint after the stable system/tools prefix
(30-minute floor, the only TTL) and gates it on the shared recency
policy, mirroring the Anthropic backend. Pre-5.6 models and local
servers stay on OpenAICompatBackend, where plain sends are optimal.
"""

import hashlib
import re

from service import cache_policy
from service import compaction
from service import log_events
from service.backends import base
from service.backends import openai_compat_backend

from typing import Any, cast, override

# Freshness window tracking the 30-minute cache floor: gaps inside it
# guarantee a live cache. Breakeven h = 0.25/1.15 ~= 0.217.
_CACHE_FRESHNESS_WINDOW_S = 30 * 60
# Consecutive over-window gaps that suspend explicit caching. Same
# judgment call as Anthropic's gate: sustained idleness implies
# P(next request hits) < breakeven.
_CACHE_IDLE_DISABLE_AFTER = 3


class OpenAICloudBackend(openai_compat_backend.OpenAICompatBackend):
    """OpenAI-cloud variant with explicit caching and completion tokens."""

    # Models with explicit-cache support (5.6+): validated live that
    # earlier models reject prompt_cache_options with a 400.
    _EXPLICIT_CACHE_RE = re.compile(r"gpt-(\d+)(?:\.(\d+))?")

    def __init__(
        self,
        *args: Any,
        api_key_env: str = "OPENAI_API_KEY",
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, api_key_env=api_key_env, **kwargs)
        self._cache_gate = cache_policy.CacheGate(
            freshness_window_s=_CACHE_FRESHNESS_WINDOW_S,
            idle_disable_after=_CACHE_IDLE_DISABLE_AFTER,
        )
        digest = hashlib.sha256(self._system_prompt.encode()).hexdigest()[:16]
        self._cache_key = f"peer-agent:{digest}"

    @classmethod
    def _supports_explicit_cache(cls, model: str) -> bool:
        """Reports whether the model accepts explicit cache options."""
        match = cls._EXPLICIT_CACHE_RE.search(model)
        if not match:
            return False
        major = int(match.group(1))
        if major != 5:
            return major > 5
        minor = match.group(2)
        return minor is not None and int(minor) >= 6

    def _should_cache(self) -> bool:
        """Delegates to the shared recency gate (one decision per request)."""
        return self._supports_explicit_cache(
            self._model
        ) and self._cache_gate.should_cache()

    def _chat_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Adds the explicit cache breakpoint after the system prefix."""
        params = super()._chat_messages(messages)
        if not self._supports_explicit_cache(self._model):
            return params
        if params and params[0].get("role") == "system":
            system = dict(params[0])
            content = system.get("content")
            if isinstance(content, str):
                system["content"] = [
                    {
                        "type": "text",
                        "text": content,
                        "prompt_cache_breakpoint": {"mode": "explicit"},
                    }
                ]
            elif isinstance(content, list) and content:
                parts = [dict(p) if isinstance(p, dict) else p for p in content]
                if isinstance(parts[-1], dict):
                    parts[-1]["prompt_cache_breakpoint"] = {"mode": "explicit"}
                system["content"] = parts
            params[0] = system
        return params

    def _cache_options(self) -> dict[str, Any]:
        """Returns explicit-mode cache opts, or {} when gated off."""
        if not self._should_cache():
            return {}
        return {
            "prompt_cache_options": {"mode": "explicit", "ttl": "30m"},
            "prompt_cache_key": self._cache_key,
        }

    def _request_turn(
        self, messages: list[dict[str, Any]], context: base.ToolContext
    ) -> Any:
        """Makes one chat-completions call with explicit caching."""
        return self._client.chat.completions.create(
            model=self._model,
            max_completion_tokens=self._max_tokens,
            messages=cast(Any, self._chat_messages(messages)),
            extra_body=self._cache_options() or None,
            **self._extra_params(context),
        )

    @staticmethod
    def _to_token_usage(usage: Any) -> log_events.TokenUsage:
        base_usage = (
            openai_compat_backend.OpenAICompatBackend._to_token_usage(usage)
        )
        written: int | None = None
        details = getattr(usage, "input_tokens_details", None)
        if details is not None:
            tokens = getattr(details, "cache_write_tokens", None)
            if isinstance(tokens, int):
                written = tokens
        return log_events.TokenUsage(
            prompt_tokens=base_usage.prompt_tokens,
            completion_tokens=base_usage.completion_tokens,
            total_tokens=base_usage.total_tokens,
            cached_prompt_tokens=base_usage.cached_prompt_tokens,
            cache_creation_input_tokens=written,
        )

    @override
    def describe(self) -> dict[str, str]:
        """Returns backend identity for logging (no secrets)."""
        return {"model": self._model, "backend": "openai"}

    @override
    def summarize(self, messages: list[dict[str, Any]]) -> str:
        """Distills history into a handoff summary with tools disabled.

        Raises compaction.SummaryTruncatedError when the reply exhausts
        its output budget: callers must refuse compaction rather than
        persist a cut-off handoff.
        """
        trimmed = compaction.truncate_for_summary(messages)
        if not trimmed:
            return ""
        if isinstance(trimmed[-1].get("content"), str):
            summarizer_messages: list[dict[str, Any]] = [
                *trimmed[:-1],
                {
                    "role": "user",
                    "content": trimmed[-1]["content"]
                    + "\n\nWrite the handoff summary now.",
                },
            ]
        else:
            summarizer_messages = [
                *trimmed,
                {"role": "user", "content": "Write the handoff summary now."},
            ]
        params = cast(Any, self._chat_messages(summarizer_messages))
        params[0] = {"role": "system", "content": compaction.SUMMARY_SYSTEM_PROMPT}
        response = self._client.chat.completions.create(
            model=self._compact_model,
            max_completion_tokens=self._compact_summary_max_tokens,
            messages=params,
        )
        if self._is_truncated(response):
            raise compaction.SummaryTruncatedError(
                f"Summary truncated at the {self._compact_summary_max_tokens}-token "
                f"budget (model {self._compact_model}); compaction refused. Raise "
                "--compact-summary-max-tokens and retry."
            )
        return (getattr(response.choices[0].message, "content", None) or "").strip()
