"""Application settings, loaded from environment variables or a `.env` file."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

LLMProvider = Literal["ollama", "anthropic"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    llm_provider: LLMProvider = "ollama"

    # Local, free models served by Ollama. The model must support tool calling.
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:7b"

    # Paid models through the Claude API. Defaults favour the cheapest option.
    anthropic_api_key: SecretStr | None = None
    anthropic_model: str = "claude-haiku-5-5"
    anthropic_effort: Literal["low", "medium", "high"] = "low"

    llm_max_tokens: int = Field(default=4096, ge=256)
    llm_timeout_seconds: float = Field(default=120.0, gt=0)
    agent_max_steps: int = Field(default=8, ge=1, le=20)

    max_upload_mb: float = Field(default=10.0, gt=0)
    max_datasets: int = Field(default=20, ge=1)
    cors_origins: list[str] = ["http://localhost:5173"]


@lru_cache
def get_settings() -> Settings:
    return Settings()
