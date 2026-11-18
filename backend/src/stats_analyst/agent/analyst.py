"""The agent loop: let the model plan, call statistical tools, and explain the result."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from stats_analyst.agent.prompts import SYSTEM_PROMPT
from stats_analyst.agent.tools import run_tool, tool_specs
from stats_analyst.llm import ChatMessage, LLMClient


@dataclass(frozen=True)
class AnalysisStep:
    tool: str
    arguments: dict[str, Any]
    output: dict[str, Any]
    is_error: bool


@dataclass
class AnalysisResult:
    answer: str
    steps: list[AnalysisStep] = field(default_factory=list)
    provider: str = ""
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    duration_seconds: float = 0.0
    completed: bool = True


class StatisticalAnalyst:
    def __init__(self, llm: LLMClient, max_steps: int = 8) -> None:
        self._llm = llm
        self._max_steps = max_steps

    def analyze(self, df: pd.DataFrame, question: str) -> AnalysisResult:
        started = time.perf_counter()
        result = AnalysisResult(answer="", provider=self._llm.provider, model=self._llm.model)
        messages = [ChatMessage(role="user", content=question)]
        specs = tool_specs()

        for _ in range(self._max_steps):
            response = self._llm.chat(SYSTEM_PROMPT, messages, specs)
            result.input_tokens += response.usage.input_tokens
            result.output_tokens += response.usage.output_tokens
            reply = response.message
            messages.append(reply)

            if not reply.tool_calls:
                result.answer = reply.content.strip()
                break

            for call in reply.tool_calls:
                outcome = run_tool(call.name, call.arguments, df)
                result.steps.append(
                    AnalysisStep(call.name, call.arguments, outcome.output, outcome.is_error)
                )
                messages.append(
                    ChatMessage(
                        role="tool",
                        content=outcome.to_json(),
                        tool_call_id=call.id,
                        tool_name=call.name,
                        is_error=outcome.is_error,
                    )
                )
        else:
            result.completed = False
            result.answer = (
                "The analysis hit the step limit before reaching a conclusion. "
                "The tool results below are still valid."
            )

        result.duration_seconds = round(time.perf_counter() - started, 3)
        return result
