"""Anthropic backend with a server-side tool loop."""

import hashlib
import json
import os
from collections.abc import Iterator

import anthropic

from service import log_events
from service.backends import base
from service.tools import registry

from typing import Any, cast

# Environment variable holding the Anthropic API key (never logged).
_ANTHROPIC_KEY_ENV = "ANTHROPIC_API_KEY"
# Default host-controlled cap on tool turns per reply; yields with
# continuation_required instead of crashing. Per-request host_max_tool_turns
# overrides; None (default) means unlimited.
DEFAULT_HOST_MAX_TOOL_TURNS: int | None = None
# Default per-turn output budget; override with --max-tokens.
DEFAULT_MAX_TOKENS = 8192
# After this many consecutive identical (name+input) *failing* calls, hold
# the next one (do not execute) and ask the model to justify retrying.
# The hold consumes the streak, so the retry always runs. Successes and
# different calls reset the streak. Volume alone never holds.
_HOLD_AFTER_IDENTICAL_FAILURES = 5
# Cap on remembered call history; trailing streaks only need the tail.
_SEEN_HISTORY_CAP = 512


class AnthropicBackend(base.Backend):
    """Runs replies via the Anthropic messages API, executing tools inline."""

    def __init__(
        self,
        model: str,
        system_prompt: str,
        tools: registry.ToolRegistry,
        host_max_tool_turns: int | None = DEFAULT_HOST_MAX_TOOL_TURNS,
        max_tokens: int = DEFAULT_MAX_TOKENS,
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
        self._client = anthropic.Anthropic(api_key=api_key)
        self._validate_model()

    def _validate_model(self) -> None:
        """Pre-flights model id and key via the free Models API (no tokens)."""
        try:
            self._client.models.retrieve(self._model)
        except anthropic.NotFoundError as exc:
            raise RuntimeError(f"Unknown model: {self._model}.") from exc
        except anthropic.AuthenticationError as exc:
            raise RuntimeError(
                f"Invalid {_ANTHROPIC_KEY_ENV}: authentication failed."
            ) from exc

    def describe(self) -> dict[str, str]:
        """Returns Anthropic backend identity for logging."""
        return {"model": self._model, "backend": "anthropic"}

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
        if latest is None:
            return total
        if total is None:
            return latest
        prompt = (
            latest.prompt_tokens
            if latest.prompt_tokens is not None
            else total.prompt_tokens
        )
        completion = (total.completion_tokens or 0) + (latest.completion_tokens or 0)
        cached = None
        if (
            total.cached_prompt_tokens is not None
            or latest.cached_prompt_tokens is not None
        ):
            cached = (total.cached_prompt_tokens or 0) + (
                latest.cached_prompt_tokens or 0
            )
        created = None
        if (
            total.cache_creation_input_tokens is not None
            or latest.cache_creation_input_tokens is not None
        ):
            created = (total.cache_creation_input_tokens or 0) + (
                latest.cache_creation_input_tokens or 0
            )
        total_sum = None
        if total.total_tokens is not None or latest.total_tokens is not None:
            total_sum = (total.total_tokens or 0) + (latest.total_tokens or 0)
        return log_events.TokenUsage(
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=total_sum,
            cached_prompt_tokens=cached,
            cache_creation_input_tokens=created,
        )

    def _message_params(
        self, messages: list[dict[str, Any]]
    ) -> list[anthropic.types.MessageParam]:
        return cast(list[anthropic.types.MessageParam], messages)

    def _extra_params(
        self, context: base.ToolContext, *, with_tools: bool = True
    ) -> dict[str, Any]:
        if not with_tools or context.workdir is None:
            return {}
        schemas = self._tools.anthropic_schemas()
        if not schemas:
            return {}
        return {"tools": cast(list[anthropic.types.ToolUnionParam], schemas)}

    @staticmethod
    def _call_key(name: str, tool_input: dict[str, Any]) -> str:
        return f"{name}:{json.dumps(tool_input, sort_keys=True, default=str)}"

    def _run_calls(
        self,
        calls: list[Any],
        tool_events: list[base.ToolEvent],
        context: base.ToolContext,
        seen: list[tuple[str, bool]],
    ) -> list[dict[str, Any]]:
        """Executes tool_use blocks into Anthropic tool_result blocks."""
        results: list[dict[str, Any]] = []
        for call in calls:
            tool = self._tools.get(call.name)
            tool_input = dict(call.input) if isinstance(call.input, dict) else {}
            digest = hashlib.sha256(
                self._call_key(call.name, tool_input).encode()
            ).hexdigest()
            trailing = 0
            for prior_digest, prior_failed in reversed(seen):
                if prior_digest != digest or not prior_failed:
                    break
                trailing += 1
            if trailing >= _HOLD_AFTER_IDENTICAL_FAILURES:
                reason = f"held: identical call failed {trailing} times in a row"
                output = (
                    f"Held: this exact `{call.name}` call failed {trailing} "
                    "times in a row, so it was not run. Write a sentence to "
                    "yourself about why retrying could help, then retry if "
                    "you still want to."
                )
                del seen[len(seen) - trailing :]
                seen.append((digest, False))
                tool_events.append(
                    base.ToolEvent(
                        name=call.name,
                        tool_input=tool_input,
                        output=output,
                        failure_reason=reason,
                    )
                )
                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": call.id,
                        "content": output,
                        "is_error": True,
                    }
                )
                continue
            if tool is None:
                result = registry.ToolResult(
                    output=f"<unknown tool: {call.name}>",
                    failure_reason="unknown tool",
                )
            else:
                result = tool.handler(tool_input, context)
            seen.append((digest, result.failure_reason is not None))
            if len(seen) > _SEEN_HISTORY_CAP:
                del seen[:-_SEEN_HISTORY_CAP]
            tool_events.append(
                base.ToolEvent(
                    name=call.name,
                    tool_input=tool_input,
                    output=result.output,
                    failure_reason=result.failure_reason,
                )
            )
            results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": call.id,
                    "content": result.output,
                    "is_error": result.failure_reason is not None,
                }
            )
        return results

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
        response = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=self._system_prompt,
            messages=self._message_params(messages),
            **self._extra_params(base.ToolContext(workdir=None), with_tools=False),
        )
        if response.usage is not None:
            usage = self._accumulate_usage(usage, self._to_token_usage(response.usage))
        summary = "".join(
            block.text for block in response.content if block.type == "text"
        )
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
        )

    def complete(
        self,
        messages: list[dict[str, Any]],
        context: base.ToolContext,
        host_max_tool_turns: int | None = DEFAULT_HOST_MAX_TOOL_TURNS,
    ) -> base.Completion:
        """Runs model turns until final text, executing tools between turns."""
        cap = host_max_tool_turns
        tool_events: list[base.ToolEvent] = []
        seen: list[tuple[str, bool]] = []
        usage: log_events.TokenUsage | None = None
        tool_turns = 0
        text_prefix = ""
        turn = 0
        while True:
            if cap is not None and tool_turns >= cap:
                return self._yield_completion(
                    messages, text_prefix, tool_events, usage, tool_turns, cap
                )
            response = self._client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                system=self._system_prompt,
                messages=self._message_params(messages),
                **self._extra_params(context),
            )
            if response.usage is not None:
                usage = self._accumulate_usage(
                    usage, self._to_token_usage(response.usage)
                )
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
                    "content": self._run_calls(calls, tool_events, context, seen),
                }
            )
            tool_turns += 1
            turn += 1

    def stream_text(
        self,
        messages: list[dict[str, Any]],
        tool_events: list[base.ToolEvent],
        context: base.ToolContext,
        host_max_tool_turns: int | None = DEFAULT_HOST_MAX_TOOL_TURNS,
        meta: dict[str, Any] | None = None,
    ) -> Iterator[str]:
        """Streams reply tokens live, running tool turns between streams."""
        cap = host_max_tool_turns
        seen: list[tuple[str, bool]] = []
        usage: log_events.TokenUsage | None = None
        tool_turns = 0
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
                yield completion.text
                return
            with self._client.messages.stream(
                model=self._model,
                max_tokens=self._max_tokens,
                system=self._system_prompt,
                messages=self._message_params(messages),
                **self._extra_params(context),
            ) as stream:
                for text in stream.text_stream:
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
                    "content": self._run_calls(calls, tool_events, context, seen),
                }
            )
            tool_turns += 1
