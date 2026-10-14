"""Typed structures for per-conversation JSONL log lines."""

import time

import pydantic

from typing import Annotated, Any, Literal


class BaseEvent(pydantic.BaseModel):
    """One timestamped line in a conversation log."""

    # Client-observed seconds since the epoch.
    ts: float = pydantic.Field(default_factory=time.time)


class TokenUsage(pydantic.BaseModel):
    """Provider-reported token counts, None fields when unreported."""

    # Tokens sent (Anthropic input_tokens).
    prompt_tokens: int | None = None
    # Tokens generated (Anthropic output_tokens).
    completion_tokens: int | None = None
    # Sum, when the provider reports it (else None).
    total_tokens: int | None = None
    # Prompt tokens served from cache.
    cached_prompt_tokens: int | None = None


class PromptEvent(BaseEvent):
    """A new host turn entering the conversation."""

    # Discriminator for the LogEvent union.
    type: Literal["prompt"] = "prompt"
    # The new user turn text.
    prompt: str
    # Host workdir for the turn, or None for no filesystem.
    cwd: str | None


class ResponseEvent(BaseEvent):
    """The peer agent's final reply for a turn."""

    # Discriminator for the LogEvent union.
    type: Literal["response"] = "response"
    # Final reply text.
    text: str
    # Token counts, or None when the backend reports none.
    usage: TokenUsage | None
    # Model id that produced the reply (per turn, for mid-thread switches).
    model: str
    # Backend kind that produced the reply.
    backend: str


class ToolCallEvent(BaseEvent):
    """A model tool invocation."""

    # Discriminator for the LogEvent union.
    type: Literal["tool_call"] = "tool_call"
    # Tool name from the model tool_use block.
    name: str
    # Decoded JSON input passed to the handler.
    tool_input: dict[str, Any]


class ToolResultEvent(BaseEvent):
    """A tool handler result fed back to the model."""

    # Discriminator for the LogEvent union.
    type: Literal["tool_result"] = "tool_result"
    # Tool name, matching the preceding call.
    name: str
    # Truncated handler output.
    output: str


class SystemPromptEvent(BaseEvent):
    """The system prompt at conversation start or when it changes."""

    # Discriminator for the LogEvent union.
    type: Literal["system_prompt"] = "system_prompt"
    # Full active system prompt.
    system_prompt: str
    # Why the event fired.
    reason: Literal["conversation_start", "changed"]


LogEvent = Annotated[
    PromptEvent | ResponseEvent | ToolCallEvent | ToolResultEvent | SystemPromptEvent,
    pydantic.Field(discriminator="type"),
]
