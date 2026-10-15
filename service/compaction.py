"""Margin-triggered history compaction helpers (pure, no provider calls)."""

import dataclasses
import json

from typing import Any

# Fallback headroom when no margin is configured explicitly.
DEFAULT_MARGIN_TOKENS = 20000
# Floor on effective headroom as a fraction of the window, so a fixed
# margin isn't vanishingly thin on large-context models.
MIN_MARGIN_FRACTION = 0.05
# Recent tail preserved verbatim across a compaction.
DEFAULT_KEEP_MESSAGES = 20
# Per-message cap applied before sending history to the summarizer.
SUMMARY_INPUT_PER_MESSAGE_CHARS = 6000
# Instruction for the summarizer call; backend supplies model and budget.
SUMMARY_SYSTEM_PROMPT = (
    "Distill the prior conversation into a compact handoff for a fresh agent "
    "resuming the work. Cover: task goal, key decisions, files touched and "
    "outcomes, distilled tool findings (not raw logs), errors and retries, "
    "open todos and what remains. Drop chatter. Stay under the token budget."
)


@dataclasses.dataclass
class CompactionConfig:
    """Tuning for proactive compaction in begin_turn."""

    # Headroom kept free: compact when est + reserve + margin > window.
    margin_tokens: int | None = DEFAULT_MARGIN_TOKENS
    # Recent tail preserved verbatim.
    keep_messages: int | None = DEFAULT_KEEP_MESSAGES
    # Manual window override (tests, unknown models); else backend value.
    context_window_override: int | None = None


def estimate_tokens(messages: list[dict[str, Any]]) -> int:
    """Heuristic input size (~chars/4) when provider usage is unavailable."""
    return len(json.dumps(messages, default=str)) // 4


def resolve_window(backend: Any, config: CompactionConfig | None) -> int | None:
    """Returns the effective input budget, or None when unknown."""
    if config is not None and config.context_window_override is not None:
        return config.context_window_override
    get_window = getattr(backend, "get_context_window", None)
    if callable(get_window):
        window = get_window()
        return window if isinstance(window, int) else None
    return None


def resolve_keep(config: CompactionConfig | None) -> int:
    """Returns the recent tail to preserve."""
    if config is not None and config.keep_messages is not None:
        return config.keep_messages
    return DEFAULT_KEEP_MESSAGES


def resolve_margin(config: CompactionConfig | None) -> int:
    """Returns the headroom to keep free."""
    if config is not None and config.margin_tokens is not None:
        return config.margin_tokens
    return DEFAULT_MARGIN_TOKENS


def resolve_effective_margin(
    config: CompactionConfig | None, context_window: int | None
) -> int:
    """Returns max(configured margin, 5% of window)."""
    margin = resolve_margin(config)
    if context_window is None:
        return margin
    floor = int(context_window * MIN_MARGIN_FRACTION)
    return max(margin, floor)


def needs_compaction(
    estimated_tokens: int,
    reserve_tokens: int | None,
    margin_tokens: int,
    context_window: int | None,
) -> bool:
    """Reports whether est + reserve + margin exceeds the window."""
    if context_window is None:
        return False
    return estimated_tokens + (reserve_tokens or 0) + margin_tokens > context_window


def _is_plain_user_message(message: dict[str, Any]) -> bool:
    """Reports whether a message is a plain user text turn."""
    return message.get("role") == "user" and isinstance(message.get("content"), str)


def split_for_compaction(
    messages: list[dict[str, Any]], keep: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Splits history into (old, recent) without breaking tool pairs."""
    if keep <= 0:
        keep_cast: int = 0
    else:
        keep_cast = keep
    if len(messages) <= keep_cast:
        return [], list(messages)
    index = len(messages) - keep_cast
    while index < len(messages) and not _is_plain_user_message(messages[index]):
        index += 1
    if index >= len(messages):
        return [], list(messages)
    return list(messages[:index]), list(messages[index:])


def _truncate_text(text: str, limit: int) -> str:
    """Keeps head+tail of an oversized string with a marker."""
    if len(text) <= limit:
        return text
    head = limit * 3 // 4
    tail = limit - head
    return (
        text[:head] + f"\n...[truncated {len(text) - limit} chars]...\n" + text[-tail:]
    )


def truncate_for_summary(
    messages: list[dict[str, Any]],
    per_message_chars: int = SUMMARY_INPUT_PER_MESSAGE_CHARS,
) -> list[dict[str, Any]]:
    """Copies history with oversized contents shortened for the summarizer."""
    truncated: list[dict[str, Any]] = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            truncated.append(
                {**message, "content": _truncate_text(content, per_message_chars)}
            )
        elif isinstance(content, list):
            blocks: list[Any] = []
            for block in content:
                if isinstance(block, dict):
                    block_content = block.get("content")
                    if isinstance(block_content, str):
                        blocks.append(
                            {
                                **block,
                                "content": _truncate_text(
                                    block_content, per_message_chars
                                ),
                            }
                        )
                    else:
                        blocks.append(dict(block))
                else:
                    blocks.append(block)
            truncated.append({**message, "content": blocks})
        else:
            truncated.append(dict(message))
    return truncated


def build_compacted_history(
    summary_text: str, recent: list[dict[str, Any]], compacted_turns: int
) -> list[dict[str, Any]]:
    """Builds the replacement history: summary preamble plus recent tail."""
    preamble = (
        f"[Conversation summary — {compacted_turns} earlier messages compacted]:\n"
        + summary_text
    )
    return [
        {"role": "user", "content": preamble},
        {"role": "assistant", "content": "Understood, continuing with that context."},
        *recent,
    ]
