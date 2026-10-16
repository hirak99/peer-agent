"""Single-turn pipeline shared by /chat and /chat/stream.

Locking stays with the caller: resolve the id, acquire the id's lock,
then run begin/finish inside it.
"""

import dataclasses
import pathlib
import threading

import fastapi

from service import compaction
from service import event_log
from service import in_context_compaction
from service import log_events
from service import models
from service import store
from service.backends import base

from typing import Any


class ConversationLocks:
    """Per-conversation mutexes for a single-process server."""

    def __init__(self) -> None:
        self._guard = threading.Lock()
        self._locks: dict[str, threading.Lock] = {}

    def for_id(self, conversation_id: str) -> threading.Lock:
        """Returns the lock guarding one conversation, creating it on demand."""
        with self._guard:
            lock = self._locks.get(conversation_id)
            if lock is None:
                lock = threading.Lock()
                self._locks[conversation_id] = lock
            return lock


@dataclasses.dataclass
class TurnState:
    """Live state for one turn; caller must hold the id's lock."""

    # Conversation being continued or started.
    conversation_id: str
    # Mutable stored history with the new user turn already appended.
    history: list[dict[str, Any]]
    # Tool execution context for this turn.
    context: base.ToolContext
    # Host-controlled tool-turn budget for this reply.
    cap: int | None
    # Proactive compaction applied in begin_turn, if any.
    compaction: base.CompactionRecord | None = None


def resolve_id(
    conversations: store.ConversationStore, request: models.ChatRequest
) -> str:
    """Returns the request id, or a fresh one for a new conversation."""
    return request.conversation_id or conversations.new_id()


def resolve_host_cap(
    request: models.ChatRequest, server_default: int | None
) -> int | None:
    """Resolves the host-controlled tool budget: request wins, else server."""
    if request.host_max_tool_turns is not None:
        return request.host_max_tool_turns
    return server_default


def resolve_context(request: models.ChatRequest) -> base.ToolContext:
    """Validates cwd into a tool context, or an empty one for null cwd."""
    if request.cwd is None:
        return base.ToolContext(workdir=None)
    workdir = pathlib.Path(request.cwd)
    if not workdir.is_absolute() or not workdir.is_dir():
        raise fastapi.HTTPException(
            status_code=400, detail="cwd must be an existing absolute directory."
        )
    return base.ToolContext(workdir=workdir)


def maybe_emit_system_prompt(
    conversations: store.ConversationStore,
    events: event_log.EventLog,
    conversation_id: str,
    system_prompt: str,
) -> None:
    """Logs conversation_start/changed when the system prompt is new."""
    previous = conversations.load_system_prompt(conversation_id)
    if previous is None:
        events.record(
            conversation_id,
            log_events.SystemPromptEvent(
                system_prompt=system_prompt, reason="conversation_start"
            ),
        )
        conversations.save_system_prompt(conversation_id, system_prompt)
    elif previous != system_prompt:
        events.record(
            conversation_id,
            log_events.SystemPromptEvent(system_prompt=system_prompt, reason="changed"),
        )
        conversations.save_system_prompt(conversation_id, system_prompt)


def log_compaction(
    events: event_log.EventLog,
    conversation_id: str,
    model: str,
    record: base.CompactionRecord,
) -> None:
    """Records a compaction event for later inspection."""
    reason = record.reason
    if reason not in ("margin", "forced", "error_recovery", "manual"):
        reason = "margin"
    events.record(
        conversation_id,
        log_events.CompactionEvent(
            pre_messages=record.pre_messages,
            post_messages=record.post_messages,
            pre_tokens_est=record.pre_tokens_est,
            post_tokens_est=record.post_tokens_est,
            model=model,
            reason=reason,  # type: ignore[arg-type]
        ),
    )


def log_completion(
    backend: base.Backend,
    events: event_log.EventLog,
    conversation_id: str,
    completion: base.Completion,
) -> None:
    """Records tool traffic plus the final response event."""
    for tool_event in completion.tool_events:
        events.record(
            conversation_id,
            log_events.ToolCallEvent(
                name=tool_event.name, tool_input=tool_event.tool_input
            ),
        )
        events.record(
            conversation_id,
            log_events.ToolResultEvent(
                name=tool_event.name,
                output=tool_event.output,
                failure_reason=tool_event.failure_reason,
            ),
        )
    identity = backend.describe()
    events.record(
        conversation_id,
        log_events.ResponseEvent(
            text=completion.text,
            usage=completion.usage,
            model=identity["model"],
            backend=identity["backend"],
            status=completion.status,  # type: ignore[arg-type]
            tool_turns=completion.tool_turns,
            reasoning=completion.reasoning or None,
        ),
    )


def _maybe_compact_proactive(
    events: event_log.EventLog,
    backend: base.Backend | None,
    config: compaction.CompactionConfig | None,
    history: list[dict[str, Any]],
    forced: bool,
) -> base.CompactionRecord | None:
    """Compacts history in place when forced or over the margin."""
    if backend is None or (not forced and config is None):
        return None
    active = config or compaction.CompactionConfig()
    keep = compaction.resolve_keep(active)
    old, recent = compaction.split_for_compaction(history, keep)
    if not old:
        return None
    if not forced:
        window = compaction.resolve_window(backend, active)
        if window is None:
            return None
        reserve = backend.get_max_output_tokens()
        if not compaction.needs_compaction(
            compaction.estimate_tokens(history),
            reserve,
            compaction.resolve_effective_margin(active, window),
            window,
        ):
            return None
    pre_messages = len(history)
    pre_est = compaction.estimate_tokens(history)
    summary: str | None = None
    if active.in_context:
        summary = in_context_compaction.attempt_summary(backend, history, keep)
    if summary is None:
        try:
            summary = backend.summarize(compaction.truncate_for_summary(old))
        except Exception:
            return None
    compacted = compaction.build_compacted_history(summary, recent, len(old))
    history[:] = compacted
    record = base.CompactionRecord(
        pre_messages=pre_messages,
        post_messages=len(compacted),
        pre_tokens_est=pre_est,
        post_tokens_est=compaction.estimate_tokens(compacted),
        reason="forced" if forced else "margin",
    )
    return record


def begin_turn(
    conversations: store.ConversationStore,
    events: event_log.EventLog,
    system_prompt: str,
    default_host_max_tool_turns: int | None,
    request: models.ChatRequest,
    conversation_id: str,
    backend: base.Backend | None = None,
    compaction_config: compaction.CompactionConfig | None = None,
) -> TurnState:
    """Loads history and records the new prompt; call inside the id's lock."""
    if request.conversation_id is not None and not conversations.exists(
        conversation_id
    ):
        raise fastapi.HTTPException(status_code=404, detail="Unknown conversation_id.")
    history = conversations.load(conversation_id) if request.conversation_id else []
    maybe_emit_system_prompt(conversations, events, conversation_id, system_prompt)
    events.record(
        conversation_id,
        log_events.PromptEvent(prompt=request.prompt, cwd=request.cwd),
    )
    history.append({"role": "user", "content": request.prompt})
    record: base.CompactionRecord | None = None
    if request.compact is not False:
        record = _maybe_compact_proactive(
            events, backend, compaction_config, history, forced=request.compact is True
        )
        if record is not None and backend is not None:
            log_compaction(events, conversation_id, backend.describe()["model"], record)
    return TurnState(
        conversation_id=conversation_id,
        history=history,
        context=resolve_context(request),
        cap=resolve_host_cap(request, default_host_max_tool_turns),
        compaction=record,
    )


def compact_conversation(
    conversations: store.ConversationStore,
    events: event_log.EventLog,
    backend: base.Backend,
    config: compaction.CompactionConfig | None,
    conversation_id: str,
) -> dict[str, Any]:
    """Compacts stored history on demand for POST /compact."""
    if not conversations.exists(conversation_id):
        raise fastapi.HTTPException(status_code=404, detail="Unknown conversation_id.")
    history = conversations.load(conversation_id)
    active = config or compaction.CompactionConfig()
    old, recent = compaction.split_for_compaction(
        history, compaction.resolve_keep(active)
    )
    if not old:
        return {
            "conversation_id": conversation_id,
            "compacted": False,
            "pre_messages": len(history),
            "post_messages": len(history),
        }
    pre_est = compaction.estimate_tokens(history)
    keep = compaction.resolve_keep(active)
    summary: str | None = None
    if active.in_context:
        summary = in_context_compaction.attempt_summary(backend, history, keep)
    if summary is None:
        summary = backend.summarize(compaction.truncate_for_summary(old))
    compacted = compaction.build_compacted_history(summary, recent, len(old))
    conversations.save(conversation_id, compacted)
    record = base.CompactionRecord(
        pre_messages=len(history),
        post_messages=len(compacted),
        pre_tokens_est=pre_est,
        post_tokens_est=compaction.estimate_tokens(compacted),
        reason="manual",
    )
    log_compaction(events, conversation_id, backend.describe()["model"], record)
    return {
        "conversation_id": conversation_id,
        "compacted": True,
        "pre_messages": record.pre_messages,
        "post_messages": record.post_messages,
        "pre_tokens_est": record.pre_tokens_est,
        "post_tokens_est": record.post_tokens_est,
    }


def finish_turn(
    conversations: store.ConversationStore,
    events: event_log.EventLog,
    backend: base.Backend,
    state: TurnState,
) -> models.ChatResponse:
    """Runs the backend, persists history, and builds the /chat response."""
    completion = backend.complete(state.history, state.context, state.cap)
    if completion.compaction is not None:
        log_compaction(
            events,
            state.conversation_id,
            backend.describe()["model"],
            completion.compaction,
        )
    conversations.save(state.conversation_id, state.history)
    log_completion(backend, events, state.conversation_id, completion)
    return models.ChatResponse(
        text=completion.text,
        conversation_id=state.conversation_id,
        status=completion.status,  # type: ignore[arg-type]
        hint=completion.hint,
        tool_turns=completion.tool_turns,
        usage=completion.usage,
    )


def finish_stream_turn(
    conversations: store.ConversationStore,
    events: event_log.EventLog,
    backend: base.Backend,
    state: TurnState,
    pieces: list[str],
    tool_events: list[base.ToolEvent],
    meta: dict[str, Any],
) -> dict[str, Any]:
    """Persists streamed history and returns the done-event payload."""
    usage = meta.get("usage")
    status = str(meta.get("status", "done"))
    hint = meta.get("hint")
    tool_turns = int(meta.get("tool_turns", len(tool_events)))
    record = meta.get("compaction")
    if isinstance(record, base.CompactionRecord):
        log_compaction(
            events, state.conversation_id, backend.describe()["model"], record
        )
    conversations.save(state.conversation_id, state.history)
    log_completion(
        backend,
        events,
        state.conversation_id,
        base.Completion(
            text="".join(pieces),
            usage=usage,
            tool_events=tool_events,
            status=status,
            hint=hint,
            tool_turns=tool_turns,
        ),
    )
    return {
        "conversation_id": state.conversation_id,
        "status": status,
        "hint": hint,
        "tool_turns": tool_turns,
    }
