"""Shared fixtures. Keeps the suite runnable straight from a source checkout.

Author: 晨星
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from atlas.pipeline.container import AtlasContainer  # noqa: E402
from atlas.settings import Settings  # noqa: E402


@pytest.fixture(scope="module")
def offline_settings() -> Settings:
    """Fully offline settings: mock LLM, hash embeddings, lexical reranking."""
    return Settings.from_env(
        llm_provider="mock",
        embed_provider="hash",
        rerank_provider="lexical",
        index_backend="memory",
    )


@pytest.fixture(scope="module")
def loaded_container(offline_settings: Settings) -> AtlasContainer:
    """A container with the packaged corpus already indexed."""
    container = AtlasContainer(offline_settings)
    report = container.load_demo_corpus()
    assert report.chunks >= 6, "packaged corpus must produce multiple chunks"
    return container


@pytest.fixture()
def empty_container(offline_settings: Settings) -> AtlasContainer:
    return AtlasContainer(offline_settings)
