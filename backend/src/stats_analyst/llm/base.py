"""Provider-neutral chat types so the agent loop does not depend on any vendor SDK."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol


class LLMError(RuntimeError):
    """The model provider failed or returned something unusable."""


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema of the arguments object


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ChatMessage:
    role: Literal["user", "assistant", "tool"]
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None  # set on role="tool"
    tool_name: str | None = None  # set on role="tool"
    is_error: bool = False  # set on role="tool"
    # Provider-native payload of an assistant turn, replayed verbatim when the same
    # provider is called again (e.g. Claude thinking blocks must be sent back unchanged).
    raw: Any = None


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass(frozen=True)
class LLMResponse:
    message: ChatMessage
    usage: Usage = field(default_factory=Usage)


class LLMClient(Protocol):
    provider: str
    model: str

    def chat(self, system: str, messages: list[ChatMessage], tools: list[ToolSpec]) -> LLMResponse:
        """Run one model turn. The returned message may request tool calls."""
        ...
