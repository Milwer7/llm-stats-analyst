"""FastAPI entry point: `uvicorn stats_analyst.main:app --reload`."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from stats_analyst import __version__
from stats_analyst.api.routes import router
from stats_analyst.config import get_settings


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="LLM Stats Analyst", version=__version__)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router)
    return app


app = create_app()
