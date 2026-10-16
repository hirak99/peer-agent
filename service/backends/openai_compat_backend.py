"""OpenAI-compatible backend with a server-side tool loop.

Speaks the OpenAI Chat Completions dialect, which covers hosted OpenAI
as well as local servers (llama.cpp, vLLM, Ollama). No prompt caching
and no model-metadata API: the context window is configured, not
probed. Thinking models that return a separate reasoning field are
supported by ignoring that field: it is never logged as reply text and
never fed back into history.
"""

import json
import os
from collections.abc import Callable

import openai

from service import compaction
from service import log_events
from service.backends import base
from service.backends import tool_loop
from service.tools import registry

from typing import Any, cast, override

# Environment variable holding the API key by default (never logged).
# The router overrides this per provider (e.g. OPENAI_API_KEY for
# hosted OpenAI); --api-key-env overrides both.
DEFAULT_KEY_ENV = "LOCAL_MODEL_API_KEY"
# Default host-controlled cap on tool turns per reply; yields with
# continuation_required instead of crashing. Per-request host_max_tool_turns
# overrides; None (default) means unlimited.
DEFAULT_HOST_MAX_TOOL_TURNS: int | None = None
# Default per-turn output budget for visible tokens. Thinking models spend
# hidden reasoning tokens against their own accounting, so keep this sized
# for the reply text, not the thinking.
DEFAULT_MAX_TOKENS = 8192
# Default token budget for the compaction summary reply.
DEFAULT_COMPACT_SUMMARY_MAX_TOKENS = 2000
# Substrings marking a rejected call as context overflow (not other 400s).
_OVERFLOW_MARKERS = (
    "too long",
    "too large",
    "exceed",
    "max_input",
    "context length",
    "maximum context",
    "context window",
    "prompt is too",
)


class OpenAICompatBackend(base.Backend):
    """Runs replies via an OpenAI-compatible chat-completions endpoint."""

    def __init__(
        self,
        model: str,
        system_prompt: str,
        tools: registry.ToolRegistry,
        base_url: str,
        api_key_env: str = DEFAULT_KEY_ENV,
        api_key: str | None = None,
        host_max_tool_turns: int | None = DEFAULT_HOST_MAX_TOOL_TURNS,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        context_window: int | None = None,
        compact_model: str | None = None,
        compact_summary_max_tokens: int = DEFAULT_COMPACT_SUMMARY_MAX_TOKENS,
        compact_keep_messages: int = compaction.DEFAULT_KEEP_MESSAGES,
    ) -> None:
        key = api_key if api_key is not None else os.environ.get(api_key_env)
        if not key:
            raise RuntimeError(f"Set {api_key_env} to use this backend.")
        self._model = model
        self._system_prompt = system_prompt
        self._tools = tools
        self._base_url = base_url
        self._default_host_max_tool_turns = host_max_tool_turns
        self._max_tokens = max_tokens
        self._context_window = context_window
        self._compact_model = compact_model or model
        self._compact_summary_max_tokens = compact_summary_max_tokens
        self._compact_keep = compact_keep_messages
        self._client = openai.OpenAI(api_key=key, base_url=base_url)
        self._validate_endpoint()

    def _validate_endpoint(self) -> None:
        """Fails fast on unreachable endpoints or bad keys via /v1/models."""
        try:
            self._client.models.list()
        except openai.AuthenticationError as exc:
            raise RuntimeError("API key rejected by the endpoint.") from exc
        except Exception as exc:
            raise RuntimeError(f"Cannot reach models endpoint: {exc}") from exc

    @override
    def describe(self) -> dict[str, str]:
        """Returns backend identity for logging (no secrets)."""
        return {
            "model": self._model,
            "backend": "openai-compat",
            "base_url": self._base_url,
        }

    @override
    def get_context_window(self) -> int | None:
        """Returns the configured input budget, or None when unknown."""
        window = getattr(self, "_context_window", None)
        return window if isinstance(window, int) else None

    @override
    def get_max_output_tokens(self) -> int | None:
        """Returns the per-turn output reserve for margin checks."""
        reserve = getattr(self, "_max_tokens", None)
        return reserve if isinstance(reserve, int) else None

    @staticmethod
    def is_context_overflow(exc: BaseException) -> bool:
        """Reports whether an API error means the history no longer fits."""
        status = getattr(exc, "status_code", None)
        if status not in (400, 401, 413):
            status = getattr(getattr(exc, "response", None), "status_code", None)
        if status not in (400, 413):
            return False
        text = str(exc).lower()
        return any(marker in text for marker in _OVERFLOW_MARKERS)

    def _chat_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Translates stored history into Chat Completions messages."""
        params: list[dict[str, Any]] = [
            {"role": "system", "content": self._system_prompt}
        ]
        for message in messages:
            role = message.get("role")
            content = message.get("content")
            if isinstance(content, str):
                params.append({"role": role, "content": content})
                continue
            if not isinstance(content, list):
                params.append({"role": role, "content": str(content)})
                continue
            text_parts: list[str] = []
            tool_calls: list[dict[str, Any]] = []
            tool_results: list[dict[str, Any]] = []
            for block in content:
                if not isinstance(block, dict):
                    text_parts.append(str(block))
                    continue
                kind = block.get("type")
                if kind == "tool_use":
                    tool_calls.append(
                        {
                            "id": block.get("id", ""),
                            "type": "function",
                            "function": {
                                "name": block.get("name", ""),
                                "arguments": json.dumps(
                                    block.get("input", {}), default=str
                                ),
                            },
                        }
                    )
                elif kind == "tool_result":
                    tool_results.append(
                        {
                            "role": "tool",
                            "tool_call_id": block.get("tool_use_id", ""),
                            "content": str(block.get("content", "")),
                        }
                    )
                elif kind == "text":
                    text_parts.append(str(block.get("text", "")))
                else:
                    text_parts.append(str(block))
            if tool_results and role == "user":
                params.extend(tool_results)
                if text_parts:
                    params.append({"role": role, "content": "".join(text_parts)})
            elif tool_calls:
                params.append(
                    {
                        "role": "assistant",
                        "content": "".join(text_parts) or None,
                        "tool_calls": tool_calls,
                    }
                )
            else:
                params.append({"role": role, "content": "".join(text_parts)})
        return params

    def _extra_params(self, context: base.ToolContext) -> dict[str, Any]:
        schemas = self._tools.openai_schemas(
            include_workdir_tools=context.workdir is not None
        )
        if not schemas:
            return {}
        return {"tools": schemas}

    @staticmethod
    def _parse_response(response: Any) -> tuple[str, list[tuple[str, str, dict[str, Any]]], str]:
        """Splits a chat completion into (text, [(id, name, input)], reasoning).

        Captures reasoning content for logging: it is billed but is neither
        reply text nor history.
        """
        choices = getattr(response, "choices", None) or []
        if not choices:
            return "", [], ""
        message = getattr(choices[0], "message", None)
        text = getattr(message, "content", None) or ""
        reasoning = getattr(message, "reasoning_content", None) or ""
        calls: list[tuple[str, str, dict[str, Any]]] = []
        for call in getattr(message, "tool_calls", None) or []:
            function = getattr(call, "function", None)
            name = getattr(function, "name", None) or ""
            raw = getattr(function, "arguments", None) or "{}"
            try:
                parsed = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                parsed = {}
            tool_input = parsed if isinstance(parsed, dict) else {}
            calls.append((getattr(call, "id", None) or "", name, tool_input))
        return text, calls, reasoning

    @staticmethod
    def _to_token_usage(usage: Any) -> log_events.TokenUsage:
        prompt_tokens: int | None = getattr(usage, "prompt_tokens", None)
        completion_tokens: int | None = getattr(usage, "completion_tokens", None)
        total_tokens: int | None = getattr(usage, "total_tokens", None)
        if total_tokens is None and (
            prompt_tokens is not None or completion_tokens is not None
        ):
            total_tokens = (prompt_tokens or 0) + (completion_tokens or 0)
        cached: int | None = None
        details = getattr(usage, "prompt_tokens_details", None)
        if details is not None:
            cached_tokens = getattr(details, "cached_tokens", None)
            if isinstance(cached_tokens, int):
                cached = cached_tokens
        return log_events.TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cached_prompt_tokens=cached,
            cache_creation_input_tokens=None,
        )

    def _request_turn(
        self, messages: list[dict[str, Any]], context: base.ToolContext
    ) -> Any:
        """Makes one chat-completions call for the current history."""
        return self._client.chat.completions.create(
            model=self._model,
            max_tokens=self._max_tokens,
            messages=cast(Any, self._chat_messages(messages)),
            **self._extra_params(context),
        )

    @override
    def summarize(self, messages: list[dict[str, Any]]) -> str:
        """Distills history into a handoff summary with tools disabled."""
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
            max_tokens=self._compact_summary_max_tokens,
            messages=params,
        )
        return (getattr(response.choices[0].message, "content", None) or "").strip()

    def _compact_for_recovery(
        self, messages: list[dict[str, Any]]
    ) -> base.CompactionRecord | None:
        """Replaces old history with a summary in place; None when too short."""
        pre_est = compaction.estimate_tokens(messages)
        keep = getattr(self, "_compact_keep", compaction.DEFAULT_KEEP_MESSAGES)
        old, recent = compaction.split_for_compaction(messages, keep)
        if not old:
            return None
        summary = self.summarize(compaction.truncate_for_summary(old))
        compacted = compaction.build_compacted_history(summary, recent, len(old))
        messages[:] = compacted
        return base.CompactionRecord(
            pre_messages=len(old) + len(recent),
            post_messages=len(compacted),
            pre_tokens_est=pre_est,
            post_tokens_est=compaction.estimate_tokens(compacted),
            reason="error_recovery",
        )

    def _yield_completion(
        self,
        messages: list[dict[str, Any]],
        text_prefix: str,
        tool_events: list[base.ToolEvent],
        usage: log_events.TokenUsage | None,
        tool_turns: int,
        cap: int,
    ) -> base.Completion:
        """Forces a summary text turn when the tool budget is spent."""
        messages.append({"role": "user", "content": tool_loop.budget_notice(cap)})
        response = self._request_turn(messages, base.ToolContext(workdir=None))
        if response.usage is not None:
            usage = tool_loop.accumulate_usage(
                usage, self._to_token_usage(response.usage)
            )
        text, _, turn_reasoning = self._parse_response(response)
        full = (text_prefix + "\n\n" + text).strip() if text_prefix else text
        messages.append({"role": "assistant", "content": full})
        hint = (
            f"Host tool budget reached ({tool_turns}/{cap}). "
            "Re-call with prompt='continue' and the same conversation_id to resume."
        )
        return base.Completion(
            text=full,
            usage=usage,
            tool_events=tool_events,
            status="continuation_required",
            hint=hint,
            tool_turns=tool_turns,
            reasoning=turn_reasoning,
        )

    def _store_assistant_turn(
        self, messages: list[dict[str, Any]], text: str, calls: list[Any]
    ) -> None:
        """Appends history in the shared Anthropic-flavored block shape."""
        if not calls:
            messages.append({"role": "assistant", "content": text})
            return
        blocks: list[dict[str, Any]] = []
        if text:
            blocks.append({"type": "text", "text": text})
        for call_id, name, tool_input in calls:
            blocks.append(
                {
                    "type": "tool_use",
                    "id": call_id,
                    "name": name,
                    "input": tool_input,
                }
            )
        messages.append({"role": "assistant", "content": blocks})

    def _store_tool_results(
        self,
        messages: list[dict[str, Any]],
        executed: list[tuple[str, str, str, bool]],
    ) -> None:
        """Appends tool results as a user message of tool_result blocks."""
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": call_id,
                        "content": output,
                        "is_error": is_error,
                    }
                    for call_id, _name, output, is_error in executed
                ],
            }
        )

    @override
    def complete(
        self,
        messages: list[dict[str, Any]],
        context: base.ToolContext,
        host_max_tool_turns: int | None = DEFAULT_HOST_MAX_TOOL_TURNS,
        on_tool_event: Callable[[base.ToolEvent], None] | None = None,
    ) -> base.Completion:
        """Runs model turns until final text, executing tools between turns."""
        cap = host_max_tool_turns
        tool_events: list[base.ToolEvent] = []
        seen: list[tuple[str, bool]] = []
        usage: log_events.TokenUsage | None = None
        tool_turns = 0
        text_prefix = ""
        reasoning_parts: list[str] = []
        recovered: base.CompactionRecord | None = None
        while True:
            if cap is not None and tool_turns >= cap:
                completion = self._yield_completion(
                    messages, text_prefix, tool_events, usage, tool_turns, cap
                )
                completion.compaction = recovered
                if reasoning_parts or completion.reasoning:
                    combined = [*reasoning_parts, completion.reasoning]
                    completion.reasoning = "\n\n".join(
                        part for part in combined if part
                    ).strip()
                return completion
            try:
                response = self._request_turn(messages, context)
            except Exception as exc:
                if recovered is not None or not self.is_context_overflow(exc):
                    raise
                record = self._compact_for_recovery(messages)
                if record is None:
                    raise
                recovered = record
                continue
            if response.usage is not None:
                usage = tool_loop.accumulate_usage(
                    usage, self._to_token_usage(response.usage)
                )
            text, calls, turn_reasoning = self._parse_response(response)
            if turn_reasoning:
                reasoning_parts.append(turn_reasoning)
            if not calls:
                full = (text_prefix + "\n\n" + text).strip() if text_prefix else text
                messages.append({"role": "assistant", "content": full})
                return base.Completion(
                    text=full,
                    usage=usage,
                    tool_events=tool_events,
                    status="done",
                    hint=None,
                    tool_turns=tool_turns,
                    compaction=recovered,
                    reasoning="\n\n".join(reasoning_parts).strip(),
                )
            if text:
                text_prefix = (
                    (text_prefix + "\n\n" + text).strip() if text_prefix else text
                )
            self._store_assistant_turn(messages, text, calls)
            executed = tool_loop.execute_calls(
                self._tools,
                [(call_id, name, tool_input) for call_id, name, tool_input in calls],
                tool_events,
                context,
                seen,
                on_tool_event,
            )
            self._store_tool_results(messages, executed)
            tool_turns += 1
