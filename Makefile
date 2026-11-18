.PHONY: install dev-backend dev-frontend test lint check sample-data up up-ollama

install:
	cd backend && uv sync
	cd frontend && npm install

dev-backend:
	cd backend && uv run uvicorn stats_analyst.main:app --reload --port 8000

dev-frontend:
	cd frontend && npm run dev

test:
	cd backend && uv run pytest

lint:
	cd backend && uv run ruff check . && uv run ruff format --check . && uv run mypy src
	cd frontend && npm run lint

check: lint test
	cd frontend && npm run build

sample-data:
	cd backend && uv run python scripts/generate_sample_data.py

up:
	docker compose up --build

up-ollama:
	OLLAMA_HOST=http://ollama:11434 docker compose --profile ollama up --build
