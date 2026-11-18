from __future__ import annotations

import json
from dataclasses import asdict
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from fastapi.concurrency import run_in_threadpool

from stats_analyst.agent import StatisticalAnalyst
from stats_analyst.api.dependencies import get_llm, get_store
from stats_analyst.api.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    DatasetResponse,
    HealthResponse,
)
from stats_analyst.config import Settings, get_settings
from stats_analyst.datasets import Dataset, DatasetNotFoundError, DatasetStore, profile_dataset
from stats_analyst.datasets.store import DatasetParseError
from stats_analyst.llm import LLMClient, LLMError

router = APIRouter(prefix="/api")

StoreDep = Annotated[DatasetStore, Depends(get_store)]
LLMDep = Annotated[LLMClient, Depends(get_llm)]
SettingsDep = Annotated[Settings, Depends(get_settings)]

_PREVIEW_ROWS = 10


@router.get("/health")
def health(llm: LLMDep) -> HealthResponse:
    return HealthResponse(status="ok", provider=llm.provider, model=llm.model)


@router.post("/datasets", status_code=status.HTTP_201_CREATED)
async def upload_dataset(
    store: StoreDep, settings: SettingsDep, file: Annotated[UploadFile, File()]
) -> DatasetResponse:
    content = await file.read()
    if len(content) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, "File is too large.")
    try:
        dataset = store.add_csv(file.filename or "dataset.csv", content)
    except DatasetParseError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc
    return _dataset_response(dataset)


@router.get("/datasets/{dataset_id}")
def get_dataset(dataset_id: str, store: StoreDep) -> DatasetResponse:
    return _dataset_response(_get_or_404(store, dataset_id))


@router.post("/datasets/{dataset_id}/analyze")
async def analyze(
    dataset_id: str,
    request: AnalyzeRequest,
    store: StoreDep,
    llm: LLMDep,
    settings: SettingsDep,
) -> AnalyzeResponse:
    dataset = _get_or_404(store, dataset_id)
    analyst = StatisticalAnalyst(llm, max_steps=settings.agent_max_steps)
    try:
        result = await run_in_threadpool(analyst.analyze, dataset.frame, request.question)
    except LLMError as exc:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, str(exc)) from exc
    return AnalyzeResponse.model_validate(asdict(result))


def _get_or_404(store: DatasetStore, dataset_id: str) -> Dataset:
    try:
        return store.get(dataset_id)
    except DatasetNotFoundError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Dataset not found.") from exc


def _dataset_response(dataset: Dataset) -> DatasetResponse:
    preview = json.loads(dataset.frame.head(_PREVIEW_ROWS).to_json(orient="records"))
    return DatasetResponse(
        id=dataset.id, name=dataset.name, profile=profile_dataset(dataset.frame), preview=preview
    )
