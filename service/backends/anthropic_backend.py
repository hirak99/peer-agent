"""Anthropic backend with a server-side tool loop."""

import os
from collections.abc import Callable, Iterator

import anthropic

from service import cache_policy
from service import compaction
from service import log_events
from service.backends import base
from service.backends import tool_loop
from service.tools import registry

from typing import Any, cast, override

# Environment variable holding the Anthropic API key (never logged).
_ANTHROPIC_KEY_ENV = "ANTHROPIC_API_KEY"
# Default host-controlled cap on tool turns per reply; yields with
# continuation_required instead of crashing. Per-request host_max_tool_turns
# overrides; None (default) means unlimited.
DEFAULT_HOST_MAX_TOOL_TURNS: int | None = None
# Default per-turn output budget; override with --max-tokens.
DEFAULT_MAX_TOKENS = 8192
# Default token budget for the compaction summary reply.
DEFAULT_COMPACT_SUMMARY_MAX_TOKENS = 8000
# Substrings marking a rejected call as context overflow (not other 400s).
_OVERFLOW_MARKERS = (
    "too long",
    "too large",
    "exceed",
    "max_input",
    "context length",
    "maximum context",
    "prompt is too",
)
# After this many consecutive identical (name+input) *failing* calls, hold
# the next one (do not execute) and ask the model to justify retrying.
# Shared policy; see service/backends/tool_loop.py.
_HOLD_AFTER_IDENTICAL_FAILURES = tool_loop.HOLD_AFTER_IDENTICAL_FAILURES
# Cap on remembered call history; trailing streaks only need the tail.
_SEEN_HISTORY_CAP = tool_loop.SEEN_HISTORY_CAP
# Prompt-cache breakpoint: prefix content is reused across calls (5-min TTL).
# Attached to the system prompt, the last tool definition, and the history
# tail on every request; per-request copies keep stored history clean.
_CACHE_BREAKPOINT: dict[str, str] = {"type": "ephemeral"}
# NOTE: both constants below mirror Anthropic's current ephemeral-cache
# terms (5-min TTL; breakeven 0.2083 from 1.25x writes / 0.05x reads).
# Revisit them if Anthropic changes its TTL or cache pricing multipliers.
# See docs/prompt-caching.md for pricing and the breakeven derivation,
# which applies generally beyond these Anthropic-specific numbers.
# Freshness window tracking the ephemeral-cache TTL: gaps inside it
# guarantee a live cache.
_CACHE_FRESHNESS_WINDOW_S = 5 * 60
# Consecutive over-window gaps that suspend breakpoints. Tune so that after
# N consecutive misses P(next request hits) < breakeven (0.2083 here);
# deeper slow bursts than N stay attached, fully dead sessions waste one
# extra write per level.
_CACHE_IDLE_DISABLE_AFTER = 3


class AnthropicBackend(base.Backend):
    """Runs replies via the Anthropic messages API, executing tools inline."""

    def __init__(
        self,
        model: str,
        system_prompt: str,
        tools: registry.ToolRegistry,
        host_max_tool_turns: int | None = DEFAULT_HOST_MAX_TOOL_TURNS,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        compact_model: str | None = None,
        compact_summary_max_tokens: int = DEFAULT_COMPACT_SUMMARY_MAX_TOKENS,
        compact_keep_messages: int = compaction.DEFAULT_KEEP_MESSAGES,
    ) -> None:
        api_key = os.environ.get(_ANTHROPIC_KEY_ENV)
        if not api_key:
            raise RuntimeError(
                f"Set {_ANTHROPIC_KEY_ENV} to use the Anthropic backend."
            )
        self._model = model
        self._system_prompt = system_prompt
        self._tools = tools
        self._default_host_max_tool_turns = host_max_tool_turns
        self._max_tokens = max_tokens
        self._compact_model = compact_model or model
        self._compact_summary_max_tokens = compact_summary_max_tokens
        self._compact_keep = compact_keep_messages
        self._max_input_tokens: int | None = None
        self._client = anthropic.Anthropic(api_key=api_key)
        # Recency gate for cache breakpoints, shared across conversations.
        self._cache_gate = cache_policy.CacheGate(
            freshness_window_s=_CACHE_FRESHNESS_WINDOW_S,
            idle_disable_after=_CACHE_IDLE_DISABLE_AFTER,
        )
        self._validate_model()

    def _validate_model(self) -> None:
        """Pre-flights model id and key via the free Models API (no tokens)."""
        try:
            info = self._client.models.retrieve(self._model)
        except anthropic.NotFoundError as exc:
            raise RuntimeError(f"Unknown model: {self._model}.") from exc
        except anthropic.AuthenticationError as exc:
            raise RuntimeError(
                f"Invalid {_ANTHROPIC_KEY_ENV}: authentication failed."
            ) from exc
        window = getattr(info, "max_input_tokens", None)
        self._max_input_tokens = window if isinstance(window, int) else None

    @override
    def describe(self) -> dict[str, str]:
        """Returns Anthropic backend identity for logging."""
        return {"model": self._model, "backend": "anthropic"}

    @override
    def get_context_window(self) -> int | None:
        """Returns the model input budget stashed at startup."""
        window = getattr(self, "_max_input_tokens", None)
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
        if status not in (400, 413):
            return False
        text = str(exc).lower()
        return any(marker in text for marker in _OVERFLOW_MARKERS)

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
        compact_model = getattr(self, "_compact_model", getattr(self, "_model", ""))
        summary_budget = getattr(
            self, "_compact_summary_max_tokens", DEFAULT_COMPACT_SUMMARY_MAX_TOKENS
        )
        use_cache = self._should_cache()
        response = self._client.messages.create(
            model=compact_model,
            max_tokens=summary_budget,
            system=self._system_param(compaction.SUMMARY_SYSTEM_PROMPT, use_cache),
            messages=self._message_params(summarizer_messages, use_cache),
        )
        if getattr(response, "stop_reason", None) == "max_tokens":
            raise compaction.SummaryTruncatedError(
                f"Summary truncated at the {summary_budget}-token budget "
                f"(model {compact_model}); compaction refused. Raise "
                "--compact-summary-max-tokens and retry."
            )
        return "".join(
            block.text for block in response.content if block.type == "text"
        ).strip()

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

    @staticmethod
    def _to_token_usage(usage: Any) -> log_events.TokenUsage:
        prompt_tokens: int | None = getattr(usage, "input_tokens", None)
        completion_tokens: int | None = getattr(usage, "output_tokens", None)
        total_tokens: int | None = None
        if prompt_tokens is not None or completion_tokens is not None:
            total_tokens = (prompt_tokens or 0) + (completion_tokens or 0)
        return log_events.TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cached_prompt_tokens=getattr(usage, "cache_read_input_tokens", None),
            cache_creation_input_tokens=getattr(
                usage, "cache_creation_input_tokens", None
            ),
        )

    @staticmethod
    def _accumulate_usage(
        total: log_events.TokenUsage | None, latest: log_events.TokenUsage | None
    ) -> log_events.TokenUsage | None:
        """Merges per-turn usage: last prompt size, summed generation/cost."""
        return tool_loop.accumulate_usage(total, latest)

    def _should_cache(self) -> bool:
        """Delegates to the shared recency gate (one decision per request)."""
        gate = getattr(self, "_cache_gate", None)
        if gate is None:
            gate = cache_policy.CacheGate(
                freshness_window_s=_CACHE_FRESHNESS_WINDOW_S,
                idle_disable_after=_CACHE_IDLE_DISABLE_AFTER,
            )
            self._cache_gate = gate
        return gate.should_cache()

    def _system_param(self, text: str | None = None, use_cache: bool = True) -> Any:
        """Returns a system prompt, cache-marked unless gated off."""
        block: dict[str, Any] = {
            "type": "text",
            "text": text if text is not None else self._system_prompt,
        }
        if use_cache:
            block["cache_control"] = dict(_CACHE_BREAKPOINT)
        return [block]

    def _message_params(
        self, messages: list[dict[str, Any]], use_cache: bool = True
    ) -> list[anthropic.types.MessageParam]:
        """Copies history with a cache breakpoint on the tail, unless gated."""
        params = [dict(message) for message in messages]
        if not params or not use_cache:
            return cast(list[anthropic.types.MessageParam], params)
        last = params[-1]
        content = last.get("content")
        if isinstance(content, str):
            last["content"] = [
                {
                    "type": "text",
                    "text": content,
                    "cache_control": dict(_CACHE_BREAKPOINT),
                }
            ]
        elif isinstance(content, list) and content:
            blocks = [
                dict(block) if isinstance(block, dict) else block for block in content
            ]
            if isinstance(blocks[-1], dict):
                blocks[-1]["cache_control"] = dict(_CACHE_BREAKPOINT)
            last["content"] = blocks
        return cast(list[anthropic.types.MessageParam], params)

    def _extra_params(
        self,
        context: base.ToolContext,
        *,
        with_tools: bool = True,
        use_cache: bool = True,
    ) -> dict[str, Any]:
        if not with_tools:
            return {}
        schemas = self._tools.anthropic_schemas(
            include_workdir_tools=context.workdir is not None
        )
        if not schemas:
            return {}
        copied = [dict(schema) for schema in schemas]
        if use_cache:
            copied[-1]["cache_control"] = dict(_CACHE_BREAKPOINT)
        return {"tools": cast(list[anthropic.types.ToolUnionParam], copied)}

    @staticmethod
    def _call_key(name: str, tool_input: dict[str, Any]) -> str:
        return tool_loop.call_key(name, tool_input)

    @staticmethod
    def _extract_reasoning(content: list[Any]) -> str:
        """Collects thinking text from response content blocks."""
        parts: list[str] = []
        for block in content:
            if getattr(block, "type", None) == "thinking":
                text = getattr(block, "thinking", None)
                if isinstance(text, str) and text:
                    parts.append(text)
        return "\n\n".join(parts)

    def _run_calls(
        self,
        calls: list[Any],
        tool_events: list[base.ToolEvent],
        context: base.ToolContext,
        seen: list[tuple[str, bool]],
        on_tool_event: Callable[[base.ToolEvent], None] | None = None,
    ) -> list[dict[str, Any]]:
        """Executes tool_use blocks into Anthropic tool_result blocks."""
        triples = [
            (
                call.id,
                call.name,
                dict(call.input) if isinstance(call.input, dict) else {},
            )
            for call in calls
        ]
        return [
            {
                "type": "tool_result",
                "tool_use_id": call_id,
                "content": output,
                "is_error": is_error,
            }
            for call_id, _name, output, is_error in tool_loop.execute_calls(
                self._tools, triples, tool_events, context, seen, on_tool_event
            )
        ]

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
        messages.append(
            {
                "role": "user",
                "content": (
                    f"Host tool budget of {cap} reached. Summarize progress so far, "
                    "state exactly what remains, then stop. The host will send "
                    "'continue' to resume if needed."
                ),
            }
        )
        use_cache = self._should_cache()
        response = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=self._system_param(use_cache=use_cache),
            messages=self._message_params(messages, use_cache),
            **self._extra_params(
                base.ToolContext(workdir=None),
                with_tools=False,
                use_cache=use_cache,
            ),
        )
        if response.usage is not None:
            usage = self._accumulate_usage(usage, self._to_token_usage(response.usage))
        summary = "".join(
            block.text for block in response.content if block.type == "text"
        )
        reasoning = self._extract_reasoning(response.content)
        text = (text_prefix + "\n\n" + summary).strip() if text_prefix else summary
        messages.append({"role": "assistant", "content": text})
        hint = (
            f"Host tool budget reached ({tool_turns}/{cap}). "
            "Re-call with prompt='continue' and the same conversation_id to resume."
        )
        return base.Completion(
            text=text,
            usage=usage,
            tool_events=tool_events,
            status="continuation_required",
            hint=hint,
            tool_turns=tool_turns,
            reasoning=reasoning,
            truncated=getattr(response, "stop_reason", None) == "max_tokens",
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
        turn = 0
        recovered: base.CompactionRecord | None = None
        while True:
            if cap is not None and tool_turns >= cap:
                completion = self._yield_completion(
                    messages, text_prefix, tool_events, usage, tool_turns, cap
                )
                completion.compaction = recovered
                if reasoning_parts or completion.reasoning:
                    completion.reasoning = "\n\n".join(
                        reasoning_parts + ([completion.reasoning] if completion.reasoning else [])
                    ).strip()
                return completion
            try:
                use_cache = self._should_cache()
                response = self._client.messages.create(
                    model=self._model,
                    max_tokens=self._max_tokens,
                    system=self._system_param(use_cache=use_cache),
                    messages=self._message_params(messages, use_cache),
                    **self._extra_params(context, use_cache=use_cache),
                )
            except Exception as exc:
                if recovered is not None or not self.is_context_overflow(exc):
                    raise
                record = self._compact_for_recovery(messages)
                if record is None:
                    raise
                recovered = record
                continue
            if response.usage is not None:
                usage = self._accumulate_usage(
                    usage, self._to_token_usage(response.usage)
                )
            turn_reasoning = self._extract_reasoning(response.content)
            if turn_reasoning:
                reasoning_parts.append(turn_reasoning)
            calls = [block for block in response.content if block.type == "tool_use"]
            if not calls:
                text = "".join(
                    block.text for block in response.content if block.type == "text"
                )
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
                    truncated=getattr(response, "stop_reason", None) == "max_tokens",
                )
            # Collect any text accompanying tool_use for the yield summary.
            partial = "".join(
                block.text for block in response.content if block.type == "text"
            )
            if partial:
                text_prefix = (
                    (text_prefix + "\n\n" + partial).strip() if text_prefix else partial
                )
            messages.append(
                {
                    "role": "assistant",
                    "content": [block.model_dump() for block in response.content],
                }
            )
            messages.append(
                {
                    "role": "user",
                    "content": self._run_calls(calls, tool_events, context, seen, on_tool_event),
                }
            )
            tool_turns += 1
            turn += 1

    @override
    def stream_text(
        self,
        messages: list[dict[str, Any]],
        tool_events: list[base.ToolEvent],
        context: base.ToolContext,
        host_max_tool_turns: int | None = DEFAULT_HOST_MAX_TOOL_TURNS,
        meta: dict[str, Any] | None = None,
        on_tool_event: Callable[[base.ToolEvent], None] | None = None,
    ) -> Iterator[str]:
        """Streams reply tokens live, running tool turns between streams."""
        cap = host_max_tool_turns
        seen: list[tuple[str, bool]] = []
        usage: log_events.TokenUsage | None = None
        tool_turns = 0
        recovered: base.CompactionRecord | None = None
        yielded_any = False
        while True:
            if cap is not None and tool_turns >= cap:
                completion = self._yield_completion(
                    messages, "", tool_events, usage, tool_turns, cap
                )
                if meta is not None:
                    meta["usage"] = completion.usage
                    meta["status"] = completion.status
                    meta["hint"] = completion.hint
                    meta["tool_turns"] = completion.tool_turns
                    meta["compaction"] = recovered
                    meta["truncated"] = completion.truncated
                yield completion.text
                return
            try:
                use_cache = self._should_cache()
                stream_manager = self._client.messages.stream(
                    model=self._model,
                    max_tokens=self._max_tokens,
                    system=self._system_param(use_cache=use_cache),
                    messages=self._message_params(messages, use_cache),
                    **self._extra_params(context, use_cache=use_cache),
                )
            except Exception as exc:
                if (
                    recovered is not None
                    or yielded_any
                    or not self.is_context_overflow(exc)
                ):
                    raise
                record = self._compact_for_recovery(messages)
                if record is None:
                    raise
                recovered = record
                continue
            with stream_manager as stream:
                for text in stream.text_stream:
                    yielded_any = True
                    yield text
                final = stream.get_final_message()
            final_usage = getattr(final, "usage", None)
            if final_usage is not None:
                usage = self._accumulate_usage(usage, self._to_token_usage(final_usage))
            calls = [block for block in final.content if block.type == "tool_use"]
            if not calls:
                text = "".join(
                    block.text for block in final.content if block.type == "text"
                )
                messages.append({"role": "assistant", "content": text})
                if meta is not None:
                    meta["usage"] = usage
                    meta["status"] = "done"
                    meta["hint"] = None
                    meta["tool_turns"] = tool_turns
                    meta["compaction"] = recovered
                    meta["truncated"] = getattr(final, "stop_reason", None) == "max_tokens"
                return
            messages.append(
                {
                    "role": "assistant",
                    "content": [block.model_dump() for block in final.content],
                }
            )
            messages.append(
                {
                    "role": "user",
                    "content": self._run_calls(calls, tool_events, context, seen, on_tool_event),
                }
            )
            tool_turns += 1
