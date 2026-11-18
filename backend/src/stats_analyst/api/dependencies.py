from __future__ import annotations

from functools import lru_cache

from stats_analyst.config import get_settings
from stats_analyst.datasets import DatasetStore
from stats_analyst.llm import LLMClient, create_llm_client


@lru_cache
def get_store() -> DatasetStore:
    return DatasetStore(max_datasets=get_settings().max_datasets)


@lru_cache
def get_llm() -> LLMClient:
    return create_llm_client(get_settings())
