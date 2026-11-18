from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from stats_analyst.api.dependencies import get_llm, get_store
from stats_analyst.datasets import DatasetStore
from stats_analyst.main import create_app
from tests.fakes import ScriptedLLM, final_turn, tool_turn

CSV = b"group,value\n" + b"".join(
    f"{'a' if i % 2 else 'b'},{i % 7 + (i % 2) * 3}\n".encode() for i in range(80)
)


@pytest.fixture
def llm() -> ScriptedLLM:
    return ScriptedLLM(
        [
            tool_turn(
                "compare_means",
                value_column="value",
                group_column="group",
                group_a="a",
                group_b="b",
            ),
            final_turn("Group a is higher."),
        ]
    )


@pytest.fixture
def client(llm: ScriptedLLM) -> Iterator[TestClient]:
    app = create_app()
    store = DatasetStore()
    app.dependency_overrides[get_store] = lambda: store
    app.dependency_overrides[get_llm] = lambda: llm
    with TestClient(app) as test_client:
        yield test_client


def test_health_reports_provider(client: TestClient) -> None:
    assert client.get("/api/health").json() == {
        "status": "ok",
        "provider": "fake",
        "model": "scripted",
    }


def test_upload_then_analyze(client: TestClient) -> None:
    upload = client.post("/api/datasets", files={"file": ("data.csv", CSV, "text/csv")})
    assert upload.status_code == 201
    body = upload.json()
    assert body["profile"]["n_rows"] == 80
    assert len(body["preview"]) == 10

    response = client.post(f"/api/datasets/{body['id']}/analyze", json={"question": "Is a > b?"})
    assert response.status_code == 200
    result = response.json()
    assert result["answer"] == "Group a is higher."
    assert result["steps"][0]["output"]["significant"] is True


def test_unknown_dataset_returns_404(client: TestClient) -> None:
    response = client.post("/api/datasets/nope/analyze", json={"question": "anything"})
    assert response.status_code == 404


def test_invalid_csv_returns_422(client: TestClient) -> None:
    response = client.post("/api/datasets", files={"file": ("x.csv", b"", "text/csv")})
    assert response.status_code == 422
