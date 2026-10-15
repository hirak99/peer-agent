"""Wire shapes for the peer-agent HTTP API."""

import pydantic

from service import log_events

from typing import Literal


class ChatRequest(pydantic.BaseModel):
    """Body of POST /chat and POST /chat/stream."""

    # The new user turn. History comes from the stored conversation.
    prompt: str
    # Existing conversation to continue. Omitted to start a new one.
    conversation_id: str | None = None
    # Host working directory for tool execution. Required; null withholds
    # filesystem tools (bash) for this turn.
    cwd: str | None
    # Soft tool-turn budget for this reply. Omitted uses the server default
    # (30); explicit null means unbounded. On budget spend the server yields
    # status=continuation_required instead of crashing; send "continue".
    max_tool_turns: int | None = None


class ChatResponse(pydantic.BaseModel):
    """Body returned by POST /chat."""

    # The peer agent's final reply text.
    text: str
    # Echoes the request id, or a fresh id when none was given.
    conversation_id: str
    # Yield gate outcome: done, or paused for host to send "continue".
    status: Literal["done", "continuation_required"] = "done"
    # Human-readable resume instruction when status != done.
    hint: str | None = None
    # Tool turns executed to produce this reply.
    tool_turns: int = 0
    # Accumulated provider token counts, or None when unreported.
    usage: log_events.TokenUsage | None = None
