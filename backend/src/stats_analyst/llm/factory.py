"""Build the configured LLM client: the provider switch lives here."""

from __future__ import annotations

from stats_analyst.config import Settings
from stats_analyst.llm.base import LLMClient, LLMError


def create_llm_client(settings: Settings) -> LLMClient:
    if settings.llm_provider == "ollama":
        from stats_analyst.llm.ollama_client import OllamaClient

        return OllamaClient(
            host=settings.ollama_host,
            model=settings.ollama_model,
            max_tokens=settings.llm_max_tokens,
            timeout=settings.llm_timeout_seconds,
        )
    if settings.llm_provider == "anthropic":
        from stats_analyst.llm.anthropic_client import AnthropicClient

        key = settings.anthropic_api_key.get_secret_value() if settings.anthropic_api_key else None
        return AnthropicClient(
            api_key=key,
            model=settings.anthropic_model,
            effort=settings.anthropic_effort,
            max_tokens=settings.llm_max_tokens,
            timeout=settings.llm_timeout_seconds,
        )
    raise LLMError(f"Unknown LLM provider '{settings.llm_provider}'.")
