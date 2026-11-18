"""Local models through Ollama (free, runs on your machine)."""

from __future__ import annotations

import json
import uuid
from typing import Any

import ollama

from stats_analyst.llm.base import ChatMessage, LLMError, LLMResponse, ToolCall, ToolSpec, Usage


class OllamaClient:
    provider = "ollama"

    def __init__(self, host: str, model: str, max_tokens: int, timeout: float) -> None:
        self.model = model
        self._max_tokens = max_tokens
        self._client = ollama.Client(host=host, timeout=timeout)

    def chat(self, system: str, messages: list[ChatMessage], tools: list[ToolSpec]) -> LLMResponse:
        payload = [{"role": "system", "content": system}, *map(_to_ollama, messages)]
        try:
            response = self._client.chat(
                model=self.model,
                messages=payload,
                tools=[_tool_to_ollama(tool) for tool in tools],
                options={"temperature": 0, "num_predict": self._max_tokens},
            )
        except ollama.ResponseError as exc:
            raise LLMError(f"Ollama error: {exc.error}") from exc
        except ConnectionError as exc:
            raise LLMError("Cannot reach Ollama. Is `ollama serve` running?") from exc

        message = response.message
        calls = [
            ToolCall(
                id=uuid.uuid4().hex[:8],
                name=call.function.name,
                arguments=_ensure_dict(call.function.arguments),
            )
            for call in message.tool_calls or []
        ]
        return LLMResponse(
            message=ChatMessage(role="assistant", content=message.content or "", tool_calls=calls),
            usage=Usage(
                input_tokens=response.prompt_eval_count or 0,
                output_tokens=response.eval_count or 0,
            ),
        )


def _to_ollama(message: ChatMessage) -> dict[str, Any]:
    if message.role == "tool":
        return {"role": "tool", "content": message.content, "tool_name": message.tool_name}
    payload: dict[str, Any] = {"role": message.role, "content": message.content}
    if message.tool_calls:
        payload["tool_calls"] = [
            {"function": {"name": call.name, "arguments": call.arguments}}
            for call in message.tool_calls
        ]
    return payload


def _tool_to_ollama(tool: ToolSpec) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        },
    }


def _ensure_dict(arguments: Any) -> dict[str, Any]:
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return dict(arguments or {})
