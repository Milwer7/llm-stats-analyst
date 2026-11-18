"""Paid models through the Claude API."""

from __future__ import annotations

from typing import Any, cast

import anthropic
from anthropic.types import MessageParam, OutputConfigParam, ToolParam

from stats_analyst.llm.base import ChatMessage, LLMError, LLMResponse, ToolCall, ToolSpec, Usage


class AnthropicClient:
    provider = "anthropic"

    def __init__(
        self, api_key: str | None, model: str, effort: str, max_tokens: int, timeout: float
    ) -> None:
        self.model = model
        self._effort = effort
        self._max_tokens = max_tokens
        self._client = anthropic.Anthropic(api_key=api_key, timeout=timeout)

    def chat(self, system: str, messages: list[ChatMessage], tools: list[ToolSpec]) -> LLMResponse:
        tool_defs: list[ToolParam] = [
            {"name": t.name, "description": t.description, "input_schema": t.parameters}
            for t in tools
        ]
        if tool_defs:
            # Cache the static prefix (tools + system prompt) across the agent's turns.
            tool_defs[-1]["cache_control"] = {"type": "ephemeral"}
        try:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=self._max_tokens,
                system=system,
                tools=tool_defs,
                messages=_to_anthropic(messages),
                output_config=cast(OutputConfigParam, {"effort": self._effort}),
            )
        except anthropic.AuthenticationError as exc:
            raise LLMError("Claude API rejected the key. Check ANTHROPIC_API_KEY.") from exc
        except anthropic.RateLimitError as exc:
            raise LLMError("Claude API rate limit reached; try again shortly.") from exc
        except anthropic.APIStatusError as exc:
            raise LLMError(f"Claude API error {exc.status_code}: {exc.message}") from exc
        except anthropic.APIConnectionError as exc:
            raise LLMError("Cannot reach the Claude API.") from exc

        if response.stop_reason == "refusal":
            raise LLMError("The model declined this request.")

        text = "".join(block.text for block in response.content if block.type == "text")
        calls = [
            ToolCall(id=block.id, name=block.name, arguments=dict(block.input))
            for block in response.content
            if block.type == "tool_use"
        ]
        return LLMResponse(
            message=ChatMessage(
                role="assistant",
                content=text,
                tool_calls=calls,
                raw=[block.model_dump(exclude_none=True) for block in response.content],
            ),
            usage=Usage(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
            ),
        )


def _to_anthropic(messages: list[ChatMessage]) -> list[MessageParam]:
    """Convert neutral history; consecutive tool results are merged into one user turn."""
    converted: list[dict[str, Any]] = []
    for message in messages:
        if message.role == "tool":
            result = {
                "type": "tool_result",
                "tool_use_id": message.tool_call_id,
                "content": message.content,
                "is_error": message.is_error,
            }
            if converted and converted[-1]["role"] == "user" and _is_tool_results(converted[-1]):
                converted[-1]["content"].append(result)
            else:
                converted.append({"role": "user", "content": [result]})
        elif message.role == "assistant":
            content = message.raw or _assistant_blocks(message)
            converted.append({"role": "assistant", "content": content})
        else:
            converted.append({"role": "user", "content": message.content})
    return cast(list[MessageParam], converted)


def _is_tool_results(turn: dict[str, Any]) -> bool:
    content = turn["content"]
    return isinstance(content, list) and all(b.get("type") == "tool_result" for b in content)


def _assistant_blocks(message: ChatMessage) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    if message.content:
        blocks.append({"type": "text", "text": message.content})
    blocks.extend(
        {"type": "tool_use", "id": c.id, "name": c.name, "input": c.arguments}
        for c in message.tool_calls
    )
    return blocks
