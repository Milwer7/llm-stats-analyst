from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    provider: str
    model: str


class DatasetResponse(BaseModel):
    id: str
    name: str
    profile: dict[str, Any]
    preview: list[dict[str, Any]]


class AnalyzeRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)


class StepResponse(BaseModel):
    tool: str
    arguments: dict[str, Any]
    output: dict[str, Any]
    is_error: bool


class AnalyzeResponse(BaseModel):
    answer: str
    steps: list[StepResponse]
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    duration_seconds: float
    completed: bool
