"""Request/response models for the HTTP surface.

Author: 晨星
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    documents: dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of stable doc_id to raw text.",
        examples=[{"release-notes": "# 版本说明\n本次发布包含检索融合修复。"}],
    )
    replace: bool = Field(
        default=False,
        description="When true the corpus is cleared before ingesting.",
    )


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    k: int | None = Field(default=None, ge=1, le=50)


class EvalRequest(BaseModel):
    k: int | None = Field(default=None, ge=1, le=50)
    include_cases: bool = True


class IngestResponse(BaseModel):
    documents: int
    chunks: int
    dim: int
    elapsed_ms: float
    doc_ids: list[str]


class HealthResponse(BaseModel):
    status: str
    version: str
    chunks: int
    documents: int
    build: dict[str, Any]


class QueryResponse(BaseModel):
    question: str
    answer: str
    citations: list[str]
    route: str
    steps: int
    trace: list[str]
    grounding: dict[str, Any]
    latency_ms: float


class ErrorResponse(BaseModel):
    detail: str
