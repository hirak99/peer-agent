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
    # Host-controlled tool-turn budget for this reply. None (default) means
    # unlimited. Set an int to make the server yield with
    # status=continuation_required when the peer spends that many tool
    # turns; send "continue" with the same conversation_id to resume.
    host_max_tool_turns: int | None = None
    # Compaction override for this turn. True forces compaction, False
    # inhibits it, None (default) follows the server margin policy.
    compact: bool | None = None


class CompactRequest(pydantic.BaseModel):
    """Body of POST /compact."""

    # Existing conversation to compact.
    conversation_id: str


class CompactResponse(pydantic.BaseModel):
    """Body returned by POST /compact."""

    # Echoes the compacted conversation id.
    conversation_id: str
    # Whether old turns were replaced with a summary.
    compacted: bool = False
    # Messages before compaction.
    pre_messages: int = 0
    # Messages after compaction.
    post_messages: int = 0
    # Estimated input tokens before compaction.
    pre_tokens_est: int | None = None
    # Estimated input tokens after compaction.
    post_tokens_est: int | None = None


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
