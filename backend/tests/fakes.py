from __future__ import annotations

from collections.abc import Iterable

from stats_analyst.llm import ChatMessage, LLMResponse, ToolCall, ToolSpec


class ScriptedLLM:
    """Replays a fixed sequence of assistant turns and records what it was sent."""

    provider = "fake"
    model = "scripted"

    def __init__(self, turns: Iterable[ChatMessage]) -> None:
        self._turns = list(turns)
        self.calls: list[list[ChatMessage]] = []

    def chat(self, system: str, messages: list[ChatMessage], tools: list[ToolSpec]) -> LLMResponse:
        self.calls.append(list(messages))
        return LLMResponse(message=self._turns.pop(0))


def tool_turn(name: str, **arguments: object) -> ChatMessage:
    return ChatMessage(
        role="assistant", tool_calls=[ToolCall(id=f"call-{name}", name=name, arguments=arguments)]
    )


def final_turn(text: str) -> ChatMessage:
    return ChatMessage(role="assistant", content=text)
