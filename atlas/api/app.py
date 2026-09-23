"""FastAPI application: thin transport over :class:`AtlasContainer`.

Two deliberate choices worth calling out:

* Endpoints that may return either a JSON body **or** a ``StreamingResponse``
  are declared with ``response_model=None`` and no union annotation. FastAPI
  otherwise raises ``Invalid args for response field`` while collecting routes.
* The container is built once in the lifespan and stored on ``app.state``, so
  importing this module has no side effects and the app can be constructed in
  tests without touching a port.

Author: 晨星
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse

from .. import __version__
from ..pipeline.container import AtlasContainer
from ..settings import Settings
from ..telemetry import configure
from .schemas import (
    EvalRequest,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
)


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    configure(app.state.settings.log_level)
    app.state.container = AtlasContainer(app.state.settings)
    app.state.container.load_demo_corpus()
    yield


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build a fully routed ASGI app.

    Route registration happens here, not in the caller. Returning an app whose
    routes are attached elsewhere produced a ``create_app()`` that answered 404
    on every path - the module-level ``app`` worked, tests did not.
    """
    app = FastAPI(
        title="ATLAS",
        version=__version__,
        description=(
            "Autonomous Task-oriented Layered Agentic System - hybrid retrieval, "
            "guarded reranking, deterministic routing, grounding verification."
        ),
        lifespan=_lifespan,
    )
    app.state.settings = settings or Settings.from_env()
    return register_routes(app)


def _container(request: Request) -> AtlasContainer:
    container = getattr(request.app.state, "container", None)
    if container is None:  # pragma: no cover - only when lifespan is bypassed
        raise HTTPException(status_code=503, detail="container not initialised")
    return container


def register_routes(app: FastAPI) -> FastAPI:
    @app.get("/health", response_model=HealthResponse, tags=["ops"])
    def health(request: Request) -> HealthResponse:
        container = _container(request)
        return HealthResponse(
            status="ok",
            version=__version__,
            chunks=container.store.count_chunks(),
            documents=len(container.store.documents()),
            build=container.build_info().as_dict(),
        )

    @app.get("/health/deep", tags=["ops"])
    def health_deep(request: Request) -> dict:
        return _container(request).selfcheck()

    @app.get("/v1/stats", tags=["ops"])
    def stats(request: Request) -> dict:
        return _container(request).stats()

    @app.get("/v1/corpus", tags=["data"])
    def corpus(request: Request) -> dict:
        container = _container(request)
        return {
            "doc_ids": container.store.documents(),
            "chunks": container.store.count_chunks(),
            "dim": container.embedder.dim,
        }

    @app.post("/v1/ingest", response_model=IngestResponse, tags=["data"])
    def ingest(payload: IngestRequest, request: Request) -> IngestResponse:
        container = _container(request)
        if payload.replace:
            container.reset()
        report = container.ingest(payload.documents)
        return IngestResponse(**report.as_dict())

    @app.post("/v1/query", response_model=QueryResponse, tags=["query"])
    def query(payload: QueryRequest, request: Request) -> QueryResponse:
        return QueryResponse(**_container(request).ask(payload.question, payload.k))

    @app.post("/v1/query/stream", response_model=None, tags=["query"])
    def query_stream(payload: QueryRequest, request: Request):
        container = _container(request)

        def events():
            for piece in container.querying.stream(payload.question, payload.k):
                yield f"data: {json.dumps({'delta': piece}, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")

    @app.post("/v1/eval", tags=["eval"])
    def run_eval(payload: EvalRequest = Body(default=EvalRequest())) -> dict:
        container = _container_from_body(payload)
        report = container.evaluate(payload.k)
        return report.as_dict(include_cases=payload.include_cases)

    @app.post("/v1/guard/probe", tags=["eval"])
    def guard_probe(request: Request) -> dict:
        return _container(request).guard_probe()

    return app


def _container_from_body(payload: EvalRequest) -> AtlasContainer:
    """Evaluation always runs on a fresh container.

    Reusing the running instance would evaluate against a corpus polluted by
    earlier ingest calls and make the reported metrics non-reproducible.
    """
    del payload
    container = AtlasContainer(Settings.from_env())
    container.load_demo_corpus()
    return container


app = create_app()
