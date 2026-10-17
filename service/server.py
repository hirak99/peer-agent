"""FastAPI peer-agent service: stateful chat with server-side tools."""

import argparse
import json
import logging
import pathlib
from collections.abc import Iterator

import dotenv
import fastapi
import uvicorn

from service import chat_turn
from service import compaction
from service import event_log
from service import models
from service import store
from service.backends import anthropic_backend
from service.backends import base
from service.backends import openai_cloud_backend
from service.backends import openai_compat_backend
from service.backends import stub_backend
from service.tools import bash_tool
from service.tools import edit_tool
from service.tools import read_tool
from service.tools import registry
from service.tools import web_fetch_tool

from typing import Any

# Hardcoded loopback port; override with --port.
DEFAULT_PORT = 8471
# Default Anthropic model; override with --model (provider/model spec).
_DEFAULT_MODEL = "anthropic/claude-haiku-5-5"
# Providers selectable via the --model prefix.
_PROVIDERS = ("anthropic", "openai", "local", "stub")
# Hosted OpenAI endpoint for the openai provider.
_OPENAI_API_URL = "https://api.openai.com/v1"
# Default peer system prompt; override with --system-prompt.
_DEFAULT_SYSTEM_PROMPT = (
    "You are a capable peer coding agent. Answer directly and use the bash "
    "tool when running commands helps. Only invoke tools listed in the "
    "available tools; if none matches what you need, say so instead of "
    "emitting a plausible-looking call or inventing its result."
)
# Project-root-relative state directories.
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
_CONVERSATIONS_DIR = _PROJECT_ROOT / "conversations"
_LOGS_DIR = _PROJECT_ROOT / "logs"


def build_default_registry(
    timeout_s: int = bash_tool.DEFAULT_BASH_TIMEOUT_S,
    max_bytes: int = bash_tool.DEFAULT_BASH_MAX_BYTES,
    web_timeout_s: int = web_fetch_tool.DEFAULT_WEB_TIMEOUT_S,
    web_max_bytes: int = web_fetch_tool.DEFAULT_WEB_MAX_BYTES,
) -> registry.ToolRegistry:
    """Builds the tool registry served to the peer agent."""
    tools = registry.ToolRegistry()
    tools.register(bash_tool.build_bash_tool(timeout_s, max_bytes))
    tools.register(read_tool.build_read_tool(max_bytes=max_bytes))
    tools.register(edit_tool.build_edit_tool())
    tools.register(web_fetch_tool.build_web_fetch_tool(web_timeout_s, web_max_bytes))
    return tools


def parse_model_spec(spec: str) -> tuple[str, str]:
    """Splits a provider/model spec, or raises listing known providers."""
    if spec == "stub":
        return "stub", "stub"
    if spec == "local":
        # Single-model servers (llama.cpp) ignore the model field, so a
        # bare provider is enough; "default" is only ever logged.
        return "local", "default"
    if "/" in spec:
        provider, name = spec.split("/", 1)
        if provider in _PROVIDERS and name:
            return provider, name
    raise ValueError(
        f"Unknown model spec: {spec!r}. Use provider/model with one of: "
        f"{', '.join(_PROVIDERS)} (e.g. anthropic/claude-opus-5-5, "
        "local --base-url http://localhost:9931/v1)."
    )


def build_backend(
    model_spec: str,
    system_prompt: str,
    host_max_tool_turns: int | None = anthropic_backend.DEFAULT_HOST_MAX_TOOL_TURNS,
    max_tokens: int = anthropic_backend.DEFAULT_MAX_TOKENS,
    bash_timeout_s: int = bash_tool.DEFAULT_BASH_TIMEOUT_S,
    bash_max_bytes: int = bash_tool.DEFAULT_BASH_MAX_BYTES,
    web_timeout_s: int = web_fetch_tool.DEFAULT_WEB_TIMEOUT_S,
    web_max_bytes: int = web_fetch_tool.DEFAULT_WEB_MAX_BYTES,
    compact_model: str | None = None,
    compact_summary_max_tokens: int = anthropic_backend.DEFAULT_COMPACT_SUMMARY_MAX_TOKENS,
    compact_keep_messages: int = compaction.DEFAULT_KEEP_MESSAGES,
    base_url: str | None = None,
    api_key_env: str | None = None,
    context_window: int | None = None,
) -> base.Backend:
    """Builds a backend from a provider/model spec (see parse_model_spec)."""
    provider, model = parse_model_spec(model_spec)
    if provider == "stub":
        return stub_backend.StubBackend()
    if compact_model is not None and "/" in compact_model:
        compact_provider, compact_name = parse_model_spec(compact_model)
        if compact_provider != provider:
            raise ValueError(
                f"Compact model provider {compact_provider!r} differs from "
                f"main provider {provider!r}."
            )
        compact_model = compact_name
    if provider == "anthropic":
        return anthropic_backend.AnthropicBackend(
            model=model,
            system_prompt=system_prompt,
            tools=build_default_registry(
                bash_timeout_s, bash_max_bytes, web_timeout_s, web_max_bytes
            ),
            host_max_tool_turns=host_max_tool_turns,
            max_tokens=max_tokens,
            compact_model=compact_model,
            compact_summary_max_tokens=compact_summary_max_tokens,
            compact_keep_messages=compact_keep_messages,
        )
    if provider == "openai":
        return openai_cloud_backend.OpenAICloudBackend(
            model=model,
            system_prompt=system_prompt,
            tools=build_default_registry(
                bash_timeout_s, bash_max_bytes, web_timeout_s, web_max_bytes
            ),
            base_url=_OPENAI_API_URL,
            api_key_env=api_key_env or "OPENAI_API_KEY",
            host_max_tool_turns=host_max_tool_turns,
            max_tokens=max_tokens,
            context_window=context_window,
            compact_model=compact_model,
            compact_summary_max_tokens=compact_summary_max_tokens,
            compact_keep_messages=compact_keep_messages,
        )
    if provider == "local":
        if not base_url:
            raise ValueError("The local provider needs --base-url (e.g. http://localhost:9931/v1).")
        return openai_compat_backend.OpenAICompatBackend(
            model=model,
            system_prompt=system_prompt,
            tools=build_default_registry(
                bash_timeout_s, bash_max_bytes, web_timeout_s, web_max_bytes
            ),
            base_url=base_url,
            api_key_env=api_key_env or openai_compat_backend.DEFAULT_KEY_ENV,
            host_max_tool_turns=host_max_tool_turns,
            max_tokens=max_tokens,
            context_window=context_window,
            compact_model=compact_model,
            compact_summary_max_tokens=compact_summary_max_tokens,
            compact_keep_messages=compact_keep_messages,
        )
    raise ValueError(f"Unknown backend: {provider}")


def build_app(
    backend: base.Backend,
    conversations: store.ConversationStore,
    events: event_log.EventLog,
    system_prompt: str,
    default_host_max_tool_turns: int
    | None = anthropic_backend.DEFAULT_HOST_MAX_TOOL_TURNS,
    compaction_config: compaction.CompactionConfig | None = None,
) -> fastapi.FastAPI:
    """Builds the FastAPI app over the given backend and storage."""
    app = fastapi.FastAPI(title="peer-agent")
    # Single-process guard: one server owns one port, so per-id thread
    # locks are enough to prevent lost updates on concurrent turns.
    locks = chat_turn.ConversationLocks()
    active_compaction = compaction_config or compaction.CompactionConfig()

    @app.get("/health")
    def _health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/chat", response_model_exclude_unset=True)
    def _chat(request: models.ChatRequest) -> models.ChatResponse:
        conversation_id = chat_turn.resolve_id(conversations, request)
        with locks.for_id(conversation_id):
            state = chat_turn.begin_turn(
                conversations,
                events,
                system_prompt,
                default_host_max_tool_turns,
                request,
                conversation_id,
                backend,
                active_compaction,
            )
            return chat_turn.finish_turn(conversations, events, backend, state)

    @app.post("/compact")
    def _compact(request: models.CompactRequest) -> models.CompactResponse:
        with locks.for_id(request.conversation_id):
            body = chat_turn.compact_conversation(
                conversations,
                events,
                backend,
                active_compaction,
                request.conversation_id,
            )
            return models.CompactResponse(**body)

    @app.post("/chat/stream")
    def _chat_stream(
        request: models.ChatRequest,
    ) -> fastapi.responses.StreamingResponse:
        def _generate() -> Iterator[str]:
            try:
                conversation_id = chat_turn.resolve_id(conversations, request)
                with locks.for_id(conversation_id):
                    state = chat_turn.begin_turn(
                        conversations,
                        events,
                        system_prompt,
                        default_host_max_tool_turns,
                        request,
                        conversation_id,
                        backend,
                        active_compaction,
                    )
                    tool_events: list[base.ToolEvent] = []
                    meta: dict[str, Any] = {}
                    pieces: list[str] = []

                    def _record_live(event: base.ToolEvent) -> None:
                        chat_turn.record_tool_event(events, conversation_id, event)

                    for token in backend.stream_text(
                        state.history,
                        tool_events,
                        state.context,
                        state.cap,
                        meta,
                        _record_live,
                    ):
                        pieces.append(token)
                        yield (
                            "data: "
                            + json.dumps({"type": "token", "text": token})
                            + "\n\n"
                        )
                    done = chat_turn.finish_stream_turn(
                        conversations, events, backend, state, pieces, tool_events, meta
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
                                "conversation_id": done["conversation_id"],
                                "status": done["status"],
                                "hint": done["hint"],
                                "tool_turns": done["tool_turns"],
                                **(
                                    {"compaction": done["compaction"]}
                                    if "compaction" in done
                                    else {}
                                ),
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
    parser.add_argument(
        "--model",
        default=_DEFAULT_MODEL,
        help="provider/model spec (anthropic, openai, local, stub).",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Endpoint for the local provider (e.g. http://localhost:9931/v1).",
    )
    parser.add_argument(
        "--api-key-env",
        default=None,
        help="Env var holding the model API key; overrides the provider default.",
    )
    parser.add_argument(
        "--context-window",
        type=int,
        default=None,
        help="Input budget for providers without a metadata API (local).",
    )
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
        "--web-timeout", type=int, default=web_fetch_tool.DEFAULT_WEB_TIMEOUT_S
    )
    parser.add_argument(
        "--web-max-bytes", type=int, default=web_fetch_tool.DEFAULT_WEB_MAX_BYTES
    )
    parser.add_argument(
        "--compact-margin-tokens",
        type=int,
        default=compaction.DEFAULT_MARGIN_TOKENS,
    )
    parser.add_argument(
        "--compact-keep-messages",
        type=int,
        default=compaction.DEFAULT_KEEP_MESSAGES,
    )
    parser.add_argument(
        "--compact-summary-max-tokens",
        type=int,
        default=anthropic_backend.DEFAULT_COMPACT_SUMMARY_MAX_TOKENS,
    )
    parser.add_argument("--compact-model", default=None)
    parser.add_argument("--compact-context-window", type=int, default=None)
    parser.add_argument(
        "--compact-in-context",
        action="store_true",
        help="Compact via a transient in-context turn (warm cache) first, "
        "falling back to a separate summarize call. For slow-prefill local models.",
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
            args.model,
            args.system_prompt,
            host_max_tool_turns=args.host_max_tool_turns,
            max_tokens=args.max_tokens,
            bash_timeout_s=args.bash_timeout,
            bash_max_bytes=args.bash_max_bytes,
            web_timeout_s=args.web_timeout,
            web_max_bytes=args.web_max_bytes,
            compact_model=args.compact_model,
            compact_summary_max_tokens=args.compact_summary_max_tokens,
            compact_keep_messages=args.compact_keep_messages,
            base_url=args.base_url,
            api_key_env=args.api_key_env,
            context_window=args.context_window,
        ),
        store.ConversationStore(_CONVERSATIONS_DIR),
        event_log.EventLog(_LOGS_DIR, verbose=args.verbose),
        args.system_prompt,
        default_host_max_tool_turns=args.host_max_tool_turns,
        compaction_config=compaction.CompactionConfig(
            margin_tokens=args.compact_margin_tokens,
            keep_messages=args.compact_keep_messages,
            context_window_override=args.compact_context_window,
            in_context=args.compact_in_context,
        ),
    )
    uvicorn.run(app, host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
