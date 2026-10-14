"""FastAPI peer-agent service: stateful chat with server-side tools."""

import argparse
import json
import logging
import pathlib
from collections.abc import Iterator

import dotenv
import fastapi
import uvicorn

from service import event_log
from service import models
from service import store
from service.backends import anthropic_backend
from service.backends import base
from service.backends import stub_backend
from service.tools import bash_tool
from service.tools import registry

from typing import Any

# Hardcoded loopback port; override with --port.
DEFAULT_PORT = 8471
# Default Anthropic model; override with --model.
_DEFAULT_MODEL = "claude-sonnet-4-5-20250929"
# Default peer system prompt; override with --system-prompt.
_DEFAULT_SYSTEM_PROMPT = (
    "You are a capable peer coding agent. Answer directly and use the bash "
    "tool when running commands helps. Keep replies concise."
)
# Project-root-relative state directories.
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
_CONVERSATIONS_DIR = _PROJECT_ROOT / "conversations"
_LOGS_DIR = _PROJECT_ROOT / "logs"


def build_default_registry() -> registry.ToolRegistry:
    """Builds the tool registry served to the peer agent."""
    tools = registry.ToolRegistry()
    tools.register(bash_tool.build_bash_tool())
    return tools


def build_backend(kind: str, model: str, system_prompt: str) -> base.Backend:
    """Builds a backend by name: anthropic or stub."""
    if kind == "stub":
        return stub_backend.StubBackend()
    if kind == "anthropic":
        return anthropic_backend.AnthropicBackend(
            model=model, system_prompt=system_prompt, tools=build_default_registry()
        )
    raise ValueError(f"Unknown backend: {kind}")


def _resolve_history(
    conversations: store.ConversationStore, request: models.ChatRequest
) -> tuple[str, list[dict[str, Any]]]:
    if request.conversation_id is None:
        conversation_id = conversations.new_id()
        return conversation_id, []
    if not conversations.exists(request.conversation_id):
        raise fastapi.HTTPException(status_code=404, detail="Unknown conversation_id.")
    return request.conversation_id, conversations.load(request.conversation_id)


def _log_completion(
    events: event_log.EventLog,
    conversation_id: str,
    completion: base.Completion,
) -> None:
    for tool_event in completion.tool_events:
        events.record(
            conversation_id,
            {
                "type": "tool_call",
                "name": tool_event.name,
                "input": tool_event.tool_input,
            },
        )
        events.record(
            conversation_id,
            {
                "type": "tool_result",
                "name": tool_event.name,
                "output": tool_event.output,
            },
        )
    events.record(
        conversation_id,
        {"type": "response", "text": completion.text, "usage": completion.usage},
    )


def build_app(
    backend: base.Backend,
    conversations: store.ConversationStore,
    events: event_log.EventLog,
) -> fastapi.FastAPI:
    """Builds the FastAPI app over the given backend and storage."""
    app = fastapi.FastAPI(title="peer-agent")

    @app.get("/health")
    def _health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/chat")
    def _chat(request: models.ChatRequest) -> models.ChatResponse:
        conversation_id, history = _resolve_history(conversations, request)
        events.record(conversation_id, {"type": "prompt", "prompt": request.prompt})
        history.append({"role": "user", "content": request.prompt})
        completion = backend.complete(history)
        conversations.save(conversation_id, history)
        _log_completion(events, conversation_id, completion)
        return models.ChatResponse(
            text=completion.text, conversation_id=conversation_id
        )

    @app.post("/chat/stream")
    def _chat_stream(
        request: models.ChatRequest,
    ) -> fastapi.responses.StreamingResponse:
        def _generate() -> Iterator[str]:
            conversation_id, history = _resolve_history(conversations, request)
            try:
                events.record(
                    conversation_id, {"type": "prompt", "prompt": request.prompt}
                )
                history.append({"role": "user", "content": request.prompt})
                tool_events: list[base.ToolEvent] = []
                pieces: list[str] = []
                for token in backend.stream_text(history, tool_events):
                    pieces.append(token)
                    yield (
                        "data: " + json.dumps({"type": "token", "text": token}) + "\n\n"
                    )
                conversations.save(conversation_id, history)
                _log_completion(
                    events,
                    conversation_id,
                    base.Completion(
                        text="".join(pieces), usage=None, tool_events=tool_events
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
                            }
                        )
                        + "\n\n"
                    )
                yield (
                    "data: "
                    + json.dumps({"type": "done", "conversation_id": conversation_id})
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
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    dotenv.load_dotenv(_PROJECT_ROOT / ".env")
    app = build_app(
        build_backend(args.backend, args.model, args.system_prompt),
        store.ConversationStore(_CONVERSATIONS_DIR),
        event_log.EventLog(_LOGS_DIR),
    )
    uvicorn.run(app, host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
