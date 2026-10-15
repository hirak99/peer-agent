"""FastAPI peer-agent service: stateful chat with server-side tools."""

import argparse
import json
import logging
import pathlib
import threading
from collections.abc import Iterator

import dotenv
import fastapi
import uvicorn

from service import event_log
from service import log_events
from service import models
from service import store
from service.backends import anthropic_backend
from service.backends import base
from service.backends import stub_backend
from service.tools import bash_tool
from service.tools import edit_tool
from service.tools import read_tool
from service.tools import registry

from typing import Any

# Hardcoded loopback port; override with --port.
DEFAULT_PORT = 8471
# Default Anthropic model; override with --model.
_DEFAULT_MODEL = "claude-opus-5-5"
# Default peer system prompt; override with --system-prompt.
_DEFAULT_SYSTEM_PROMPT = (
    "You are a capable peer coding agent. Answer directly and use the bash "
    "tool when running commands helps. Keep replies concise. Only invoke "
    "tools actually offered to you; when no tools are offered, say what you "
    "cannot do instead of emitting fake tool calls or guessed outputs."
)
# Project-root-relative state directories.
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
_CONVERSATIONS_DIR = _PROJECT_ROOT / "conversations"
_LOGS_DIR = _PROJECT_ROOT / "logs"


def build_default_registry(
    timeout_s: int = bash_tool.DEFAULT_BASH_TIMEOUT_S,
    max_bytes: int = bash_tool.DEFAULT_BASH_MAX_BYTES,
) -> registry.ToolRegistry:
    """Builds the tool registry served to the peer agent."""
    tools = registry.ToolRegistry()
    tools.register(bash_tool.build_bash_tool(timeout_s, max_bytes))
    tools.register(read_tool.build_read_tool(max_bytes=max_bytes))
    tools.register(edit_tool.build_edit_tool())
    return tools


def build_backend(
    kind: str,
    model: str,
    system_prompt: str,
    host_max_tool_turns: int | None = anthropic_backend.DEFAULT_HOST_MAX_TOOL_TURNS,
    max_tokens: int = anthropic_backend.DEFAULT_MAX_TOKENS,
    bash_timeout_s: int = bash_tool.DEFAULT_BASH_TIMEOUT_S,
    bash_max_bytes: int = bash_tool.DEFAULT_BASH_MAX_BYTES,
) -> base.Backend:
    """Builds a backend by name: anthropic or stub."""
    if kind == "stub":
        return stub_backend.StubBackend()
    if kind == "anthropic":
        return anthropic_backend.AnthropicBackend(
            model=model,
            system_prompt=system_prompt,
            tools=build_default_registry(bash_timeout_s, bash_max_bytes),
            host_max_tool_turns=host_max_tool_turns,
            max_tokens=max_tokens,
        )
    raise ValueError(f"Unknown backend: {kind}")


def _resolve_host_cap(
    request: models.ChatRequest, server_default: int | None
) -> int | None:
    """Resolves the host-controlled tool budget: request wins, else server."""
    if request.host_max_tool_turns is not None:
        return request.host_max_tool_turns
    return server_default


def _resolve_context(request: models.ChatRequest) -> base.ToolContext:
    if request.cwd is None:
        return base.ToolContext(workdir=None)
    workdir = pathlib.Path(request.cwd)
    if not workdir.is_absolute() or not workdir.is_dir():
        raise fastapi.HTTPException(
            status_code=400, detail="cwd must be an existing absolute directory."
        )
    return base.ToolContext(workdir=workdir)


def _maybe_emit_system_prompt(
    conversations: store.ConversationStore,
    events: event_log.EventLog,
    conversation_id: str,
    system_prompt: str,
) -> None:
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


def _log_completion(
    backend: base.Backend,
    events: event_log.EventLog,
    conversation_id: str,
    completion: base.Completion,
) -> None:
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
        ),
    )


def build_app(
    backend: base.Backend,
    conversations: store.ConversationStore,
    events: event_log.EventLog,
    system_prompt: str,
    default_host_max_tool_turns: int
    | None = anthropic_backend.DEFAULT_HOST_MAX_TOOL_TURNS,
) -> fastapi.FastAPI:
    """Builds the FastAPI app over the given backend and storage."""
    app = fastapi.FastAPI(title="peer-agent")
    # Single-process guard: one server owns one port, so per-id threads
    # locks are enough to prevent lost updates on concurrent turns.
    _guard = threading.Lock()
    _locks: dict[str, threading.Lock] = {}

    def _lock_for(conversation_id: str) -> threading.Lock:
        with _guard:
            lock = _locks.get(conversation_id)
            if lock is None:
                lock = threading.Lock()
                _locks[conversation_id] = lock
            return lock

    @app.get("/health")
    def _health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/chat")
    def _chat(request: models.ChatRequest) -> models.ChatResponse:
        context = _resolve_context(request)
        conversation_id = request.conversation_id or conversations.new_id()
        with _lock_for(conversation_id):
            if request.conversation_id is not None and not conversations.exists(
                conversation_id
            ):
                raise fastapi.HTTPException(
                    status_code=404, detail="Unknown conversation_id."
                )
            history = (
                conversations.load(conversation_id) if request.conversation_id else []
            )
            _maybe_emit_system_prompt(
                conversations, events, conversation_id, system_prompt
            )
            events.record(
                conversation_id,
                log_events.PromptEvent(prompt=request.prompt, cwd=request.cwd),
            )
            history.append({"role": "user", "content": request.prompt})
            cap = _resolve_host_cap(request, default_host_max_tool_turns)
            completion = backend.complete(history, context, cap)
            conversations.save(conversation_id, history)
            _log_completion(backend, events, conversation_id, completion)
            return models.ChatResponse(
                text=completion.text,
                conversation_id=conversation_id,
                status=completion.status,  # type: ignore[arg-type]
                hint=completion.hint,
                tool_turns=completion.tool_turns,
                usage=completion.usage,
            )

    @app.post("/chat/stream")
    def _chat_stream(
        request: models.ChatRequest,
    ) -> fastapi.responses.StreamingResponse:
        def _generate() -> Iterator[str]:
            try:
                context = _resolve_context(request)
                conversation_id = request.conversation_id or conversations.new_id()
                with _lock_for(conversation_id):
                    if request.conversation_id is not None and not conversations.exists(
                        conversation_id
                    ):
                        raise fastapi.HTTPException(
                            status_code=404, detail="Unknown conversation_id."
                        )
                    history = (
                        conversations.load(conversation_id)
                        if request.conversation_id
                        else []
                    )
                    _maybe_emit_system_prompt(
                        conversations, events, conversation_id, system_prompt
                    )
                    events.record(
                        conversation_id,
                        log_events.PromptEvent(prompt=request.prompt, cwd=request.cwd),
                    )
                    history.append({"role": "user", "content": request.prompt})
                    cap = _resolve_host_cap(request, default_host_max_tool_turns)
                    tool_events: list[base.ToolEvent] = []
                    meta: dict[str, Any] = {}
                    pieces: list[str] = []
                    for token in backend.stream_text(
                        history, tool_events, context, cap, meta
                    ):
                        pieces.append(token)
                        yield (
                            "data: "
                            + json.dumps({"type": "token", "text": token})
                            + "\n\n"
                        )
                    usage = meta.get("usage")
                    status = str(meta.get("status", "done"))
                    hint = meta.get("hint")
                    tool_turns = int(meta.get("tool_turns", len(tool_events)))
                    conversations.save(conversation_id, history)
                    _log_completion(
                        backend,
                        events,
                        conversation_id,
                        base.Completion(
                            text="".join(pieces),
                            usage=usage,
                            tool_events=tool_events,
                            status=status,
                            hint=hint,
                            tool_turns=tool_turns,
                        ),
                    )
                    for tool_event in tool_events:
                        yield (
                            "data: "
                            + json.dumps(
                                {
                                    "type": "tool_call",
                                    "name": tool_event.name,
                                    "input": tool_event.tool_input,
                                    "failure_reason": tool_event.failure_reason,
                                }
                            )
                            + "\n\n"
                        )
                    yield (
                        "data: "
                        + json.dumps(
                            {
                                "type": "done",
                                "conversation_id": conversation_id,
                                "status": status,
                                "hint": hint,
                                "tool_turns": tool_turns,
                            }
                        )
                        + "\n\n"
                    )
            except fastapi.HTTPException as exc:
                yield (
                    "data: "
                    + json.dumps(
                        {
                            "type": "error",
                            "status": exc.status_code,
                            "detail": exc.detail,
                        }
                    )
                    + "\n\n"
                )

        return fastapi.responses.StreamingResponse(
            _generate(), media_type="text/event-stream"
        )

    return app


def main() -> None:
    """Serves the peer agent over loopback HTTP."""
    parser = argparse.ArgumentParser(description="Peer-agent chat service.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--backend", default="anthropic", choices=["anthropic", "stub"])
    parser.add_argument("--model", default=_DEFAULT_MODEL)
    parser.add_argument("--system-prompt", default=_DEFAULT_SYSTEM_PROMPT)
    parser.add_argument(
        "--host-max-tool-turns",
        type=int,
        default=anthropic_backend.DEFAULT_HOST_MAX_TOOL_TURNS,
    )
    parser.add_argument(
        "--max-tokens", type=int, default=anthropic_backend.DEFAULT_MAX_TOKENS
    )
    parser.add_argument(
        "--bash-timeout", type=int, default=bash_tool.DEFAULT_BASH_TIMEOUT_S
    )
    parser.add_argument(
        "--bash-max-bytes", type=int, default=bash_tool.DEFAULT_BASH_MAX_BYTES
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print each log event (the exact JSONL line) to stderr live.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    dotenv.load_dotenv(_PROJECT_ROOT / ".env")
    app = build_app(
        build_backend(
            args.backend,
            args.model,
            args.system_prompt,
            host_max_tool_turns=args.host_max_tool_turns,
            max_tokens=args.max_tokens,
            bash_timeout_s=args.bash_timeout,
            bash_max_bytes=args.bash_max_bytes,
        ),
        store.ConversationStore(_CONVERSATIONS_DIR),
        event_log.EventLog(_LOGS_DIR, verbose=args.verbose),
        args.system_prompt,
        default_host_max_tool_turns=args.host_max_tool_turns,
    )
    uvicorn.run(app, host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
