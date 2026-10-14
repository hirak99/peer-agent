"""Wire shapes for the peer-agent HTTP API."""

import pydantic


class ChatRequest(pydantic.BaseModel):
    """Body of POST /chat and POST /chat/stream."""

    # The new user turn. History comes from the stored conversation.
    prompt: str
    # Existing conversation to continue. Omitted to start a new one.
    conversation_id: str | None = None
    # Host working directory for tool execution. Required; null withholds
    # filesystem tools (bash) for this turn.
    cwd: str | None


class ChatResponse(pydantic.BaseModel):
    """Body returned by POST /chat."""

    # The peer agent's final reply text.
    text: str
    # Echoes the request id, or a fresh id when none was given.
    conversation_id: str
