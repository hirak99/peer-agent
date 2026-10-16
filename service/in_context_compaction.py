"""In-context history compaction (warm-cache path for slow-prefill models).

Unlike a separate summarize() call (cold prefill of the truncated prefix),
this appends a transient summary request to the live history, runs one
tool-less turn against the already-warm KV cache, then removes the exchange.
The caller splices the summary via compaction.build_compacted_history.
compaction.py itself is untouched: this module only reuses its pure helpers.
"""

from service import compaction
from service.backends import base

from typing import Any

# Transient instruction; popped from history before splicing.
SUMMARY_REQUEST = (
    compaction.SUMMARY_SYSTEM_PROMPT + " Reply with only the summary, no tool calls."
)


def attempt_summary(
    backend: base.Backend,
    history: list[dict[str, Any]],
    keep: int,
) -> str | None:
    """Summarizes the compactable prefix via a transient in-context turn.

    Returns the summary text, or None when there is no safe split point,
    the turn fails, or the reply is empty (caller falls back to a separate
    summarize() call). History is always restored: the transient exchange
    never persists and is never logged.
    """
    old, _recent = compaction.split_for_compaction(history, keep)
    if not old:
        return None
    pre_len = len(history)
    history.append({"role": "user", "content": SUMMARY_REQUEST})
    try:
        completion = backend.complete(
            history, base.ToolContext(workdir=None), host_max_tool_turns=None
        )
    except Exception:
        return None
    finally:
        del history[pre_len:]
    if getattr(completion, "truncated", False):
        return None
    summary = completion.text.strip()
    return summary or None
