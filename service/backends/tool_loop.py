"""Shared agentic tool-loop helpers (provider-agnostic).

Every tool-capable backend runs the same loop: collect model-requested
calls, hold back exact repeats of failing calls, execute the rest via
the registry, and feed results back until final text. Only the wire
shapes differ per provider, so that translation stays in each backend
while the hold policy, usage merge, and budget notice live here.
"""

import hashlib
import json

from service import log_events
from service.backends import base
from service.tools import registry

from typing import Any

# After this many consecutive identical (name+input) *failing* calls, hold
# the next one (do not execute) and ask the model to justify retrying.
# The hold consumes the streak, so the retry always runs. Successes and
# different calls reset the streak. Volume alone never holds.
HOLD_AFTER_IDENTICAL_FAILURES = 5
# Cap on remembered call history; trailing streaks only need the tail.
SEEN_HISTORY_CAP = 512


def call_key(name: str, tool_input: dict[str, Any]) -> str:
    """Returns the identity string for hold tracking (name + sorted input)."""
    return f"{name}:{json.dumps(tool_input, sort_keys=True, default=str)}"


def check_hold(
    seen: list[tuple[str, bool]], name: str, tool_input: dict[str, Any]
) -> tuple[int, str | None]:
    """Reports the trailing identical-failure streak and hold text, if held.

    Returns (streak, output): output is None unless the streak reached
    the hold threshold, in which case it holds the call and the caller
    should feed output back to the model instead of executing.
    """
    digest = hashlib.sha256(call_key(name, tool_input).encode()).hexdigest()
    trailing = 0
    for prior_digest, prior_failed in reversed(seen):
        if prior_digest != digest or not prior_failed:
            break
        trailing += 1
    if trailing < HOLD_AFTER_IDENTICAL_FAILURES:
        return trailing, None
    output = (
        f"Held: this exact `{name}` call failed {trailing} "
        "times in a row, so it was not run. Write a sentence to "
        "yourself about why retrying could help, then retry if "
        "you still want to."
    )
    del seen[len(seen) - trailing :]
    return trailing, output


def note_call(
    seen: list[tuple[str, bool]], name: str, tool_input: dict[str, Any], failed: bool
) -> None:
    """Records one executed call outcome for future hold checks."""
    digest = hashlib.sha256(call_key(name, tool_input).encode()).hexdigest()
    seen.append((digest, failed))
    if len(seen) > SEEN_HISTORY_CAP:
        del seen[:-SEEN_HISTORY_CAP]


def execute_calls(
    tools: registry.ToolRegistry,
    calls: list[tuple[str, str, dict[str, Any]]],
    tool_events: list[base.ToolEvent],
    context: base.ToolContext,
    seen: list[tuple[str, bool]],
) -> list[tuple[str, str, str, bool]]:
    """Executes (call_id, name, input) triples via the registry.

    Returns (call_id, name, output, is_error) tuples; the caller wraps
    them in provider-shaped result blocks. Held and unknown-tool calls
    never execute.
    """
    executed: list[tuple[str, str, str, bool]] = []
    for call_id, name, tool_input in calls:
        _trailing, held_output = check_hold(seen, name, tool_input)
        if held_output is not None:
            note_call(seen, name, tool_input, False)
            tool_events.append(
                base.ToolEvent(
                    name=name,
                    tool_input=tool_input,
                    output=held_output,
                    failure_reason="held: identical call failed repeatedly",
                )
            )
            executed.append((call_id, name, held_output, True))
            continue
        tool = tools.get(name)
        if tool is None:
            result = registry.ToolResult(
                output=f"<unknown tool: {name}>",
                failure_reason="unknown tool",
            )
        else:
            result = tool.handler(tool_input, context)
        failed = result.failure_reason is not None
        note_call(seen, name, tool_input, failed)
        tool_events.append(
            base.ToolEvent(
                name=name,
                tool_input=tool_input,
                output=result.output,
                failure_reason=result.failure_reason,
            )
        )
        executed.append((call_id, name, result.output, failed))
    return executed


def accumulate_usage(
    total: log_events.TokenUsage | None, latest: log_events.TokenUsage | None
) -> log_events.TokenUsage | None:
    """Merges per-turn usage: last prompt size, summed generation/cost."""
    if latest is None:
        return total
    if total is None:
        return latest
    prompt = (
        latest.prompt_tokens if latest.prompt_tokens is not None else total.prompt_tokens
    )
    completion = (total.completion_tokens or 0) + (latest.completion_tokens or 0)
    cached = None
    if (
        total.cached_prompt_tokens is not None
        or latest.cached_prompt_tokens is not None
    ):
        cached = (total.cached_prompt_tokens or 0) + (latest.cached_prompt_tokens or 0)
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


def budget_notice(cap: int) -> str:
    """Returns the user-side notice appended when the tool budget is spent."""
    return (
        f"Host tool budget of {cap} reached. Summarize progress so far, "
        "state exactly what remains, then stop. The host will send "
        "'continue' to resume if needed."
    )
