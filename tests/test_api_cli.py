"""HTTP surface and command line interface.

Both are optional extras: the suite skips the HTTP tests when FastAPI is not
installed, and the CLI tests only need the standard library.

Author: 晨星
"""

from __future__ import annotations

import pytest

from atlas import cli

fastapi = pytest.importorskip("fastapi", reason="FastAPI is an optional extra")
from fastapi.testclient import TestClient  # noqa: E402

from atlas.api.app import create_app  # noqa: E402
from atlas.settings import Settings  # noqa: E402


@pytest.fixture(scope="module")
def client() -> TestClient:
    app = create_app(Settings.from_env(llm_provider="mock", embed_provider="hash"))
    with TestClient(app) as test_client:
        yield test_client


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["chunks"] > 0
    assert body["build"]["llm"] == "mock"


def test_deep_health_reports_invariants(client: TestClient) -> None:
    response = client.get("/health/deep")
    assert response.status_code == 200
    assert response.json()["ok"] is True


def test_stats(client: TestClient) -> None:
    response = client.get("/v1/stats")
    assert response.status_code == 200
    assert response.json()["corpus"]["chunks"] > 0


def test_corpus_listing(client: TestClient) -> None:
    body = client.get("/v1/corpus").json()
    assert body["doc_ids"]
    assert body["dim"] > 0


def test_query_retrieval(client: TestClient) -> None:
    response = client.post("/v1/query", json={"question": "为什么不能对检索分数直接加权求和？"})
    assert response.status_code == 200
    body = response.json()
    assert body["route"] == "retrieve"
    assert body["answer"]
    assert body["citations"]


def test_query_arithmetic(client: TestClient) -> None:
    response = client.post("/v1/query", json={"question": "计算 10*10+1 等于多少？"})
    assert response.status_code == 200
    body = response.json()
    assert body["route"] == "arithmetic"
    assert "101" in body["answer"]


def test_query_rejects_empty_question(client: TestClient) -> None:
    assert client.post("/v1/query", json={"question": ""}).status_code == 422


def test_query_rejects_out_of_range_k(client: TestClient) -> None:
    response = client.post("/v1/query", json={"question": "检索", "k": 999})
    assert response.status_code == 422


def test_streaming_endpoint(client: TestClient) -> None:
    with client.stream(
        "POST", "/v1/query/stream", json={"question": "接地核验为什么按句核验？"}
    ) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        payload = "".join(response.iter_text())
    assert "data:" in payload
    assert "[DONE]" in payload


def test_ingest_endpoint(client: TestClient) -> None:
    response = client.post(
        "/v1/ingest",
        json={"documents": {"probe-doc": "# 探针文档\n\n这是一段用于验证摄入接口的正文内容。"}},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["chunks"] >= 1
    assert "probe-doc" in body["doc_ids"]


def test_guard_probe_endpoint(client: TestClient) -> None:
    body = client.post("/v1/guard/probe").json()
    assert body["guarded_preserves_top1"] is True
    assert body["naive_breaks_top1"] is True


def test_eval_endpoint(client: TestClient) -> None:
    response = client.post("/v1/eval", json={"include_cases": False})
    assert response.status_code == 200
    body = response.json()
    assert body["n"] >= 12
    assert body["route_accuracy"] == pytest.approx(1.0)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def test_cli_ingest_demo() -> None:
    assert cli.main(["ingest", "--demo"]) == 0


def test_cli_ask_arithmetic() -> None:
    assert cli.main(["ask", "计算 2+2 等于多少？"]) == 0


def test_cli_selfcheck() -> None:
    assert cli.main(["selfcheck"]) == 0


def test_cli_doctor() -> None:
    assert cli.main(["doctor"]) == 0


def test_cli_eval() -> None:
    assert cli.main(["eval"]) == 0


def test_cli_eval_min_recall_gate() -> None:
    assert cli.main(["eval", "--min-recall", "0.99"]) == cli.EXIT_FAIL


def test_cli_requires_a_subcommand() -> None:
    with pytest.raises(SystemExit):
        cli.main([])
