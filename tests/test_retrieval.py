from __future__ import annotations

from contextlib import contextmanager
import os
import warnings
from typing import Any, Iterator

import pytest

from config import settings
from src.ingestion.service import NormalizedDocument, ingest
from src.indexing.indexer import index_batch
from src.retrieval.citations import extract_citation
from src.retrieval.search import retrieve


SCIKIT_URL = "https://scikit-learn.org/stable/modules/ensemble.html#ensemble"
FACTUAL_URL = "https://docs.python.org/3/glossary.html"
UNRELATED_URL = os.getenv(
    "TEST_UNRELATED_URL",
    "https://simple.wikipedia.org/wiki/Mount_Everest",
)


def _source_url(doc: NormalizedDocument) -> str:
    return doc.source_url or doc.url


@contextmanager
def _override_settings(**overrides: Any) -> Iterator[None]:
    original: dict[str, Any] = {}
    for key, value in overrides.items():
        original[key] = getattr(settings, key)
        setattr(settings, key, value)
    try:
        yield
    finally:
        for key, value in original.items():
            setattr(settings, key, value)


@pytest.fixture(scope="module")
def indexed_corpus(reset_index_tables) -> dict[str, Any]:
    reset_index_tables()
    try:
        docs: list[NormalizedDocument] = [
            ingest(SCIKIT_URL),
            ingest(FACTUAL_URL),
            ingest(UNRELATED_URL),
        ]
    except RuntimeError as exc:
        if "FIRECRAWL_API_KEY" in str(exc):
            pytest.skip(f"Skipping retrieval integration tests: {exc}")
        raise

    depths = [0, 0, 2]
    index_batch(docs, depths)
    return {
        "docs": docs,
        "source_urls": [_source_url(doc) for doc in docs],
    }


def _token_span_sum(chunks: list[Any]) -> int:
    return sum(max(0, int(chunk.token_end) - int(chunk.token_start)) for chunk in chunks)


@pytest.mark.usefixtures("indexed_corpus")
class TestRetrievalIntegration:
    def test_full_context_mode_switch_and_score(self, db_conn):
        with _override_settings(
            retrieval_full_context_threshold=10_000_000,
            retrieval_context_budget=10_000_000,
        ):
            result = retrieve(db_conn, "what is gradient boosting?")

        assert result.mode == "full_context"
        assert result.corpus_stats.total_tokens > 0
        assert len(result.chunks) > 0
        assert all(chunk.score == 1.0 for chunk in result.chunks)
        assert all(chunk.raw_similarity is None for chunk in result.chunks)
        assert all(chunk.parent_id is None for chunk in result.chunks)

    def test_chunk_mode_switch(self, db_conn):
        with _override_settings(
            retrieval_full_context_threshold=1,
            retrieval_context_budget=10_000_000,
        ):
            result = retrieve(db_conn, "how does gradient boosting work?")
        assert result.mode == "chunk"
        assert len(result.chunks) > 0

    def test_source_url_filter_scopes_counts(self, db_conn, indexed_corpus):
        filtered_source = [indexed_corpus["source_urls"][0]]
        with _override_settings(
            retrieval_full_context_threshold=10_000_000,
            retrieval_context_budget=10_000_000,
        ):
            all_result = retrieve(db_conn, "ensemble methods")
            filtered_result = retrieve(
                db_conn,
                "ensemble methods",
                source_urls=filtered_source,
            )

        assert filtered_result.corpus_stats.total_documents <= all_result.corpus_stats.total_documents
        assert filtered_result.corpus_stats.total_parent_chunks <= all_result.corpus_stats.total_parent_chunks
        assert all(chunk.source_url == filtered_source[0] for chunk in filtered_result.chunks)

    def test_threshold_clamp_warns(self, db_conn):
        with _override_settings(
            retrieval_full_context_threshold=5000,
            retrieval_context_budget=1000,
        ):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                _ = retrieve(db_conn, "quick check query")
            assert any(
                "RETRIEVAL_FULL_CONTEXT_THRESHOLD" in str(w.message) for w in caught
            )

    def test_full_context_grouping_and_order(self, db_conn):
        with _override_settings(
            retrieval_full_context_threshold=10_000_000,
            retrieval_context_budget=10_000_000,
        ):
            result = retrieve(db_conn, "glossary terms")

        by_source: dict[str, list[int]] = {}
        for chunk in result.chunks:
            by_source.setdefault(chunk.source_url, []).append(chunk.chunk_index)
        for indexes in by_source.values():
            assert indexes == sorted(indexes)

    def test_surface_selection_prefers_html_when_flags_present(self, db_conn):
        with _override_settings(
            retrieval_full_context_threshold=10_000_000,
            retrieval_context_budget=10_000_000,
        ):
            result = retrieve(db_conn, "mathematical notation and formulas")

        flagged = [
            chunk
            for chunk in result.chunks
            if chunk.has_math
            or chunk.has_table
            or chunk.has_code
            or chunk.has_definition_list
            or chunk.has_admonition
        ]
        assert len(flagged) > 0
        assert any(chunk.surface == "html" for chunk in flagged)

    def test_chunk_mode_semantic_match(self, db_conn, indexed_corpus):
        scikit_source = indexed_corpus["source_urls"][0]
        with _override_settings(
            retrieval_full_context_threshold=1,
            retrieval_context_budget=10_000_000,
        ):
            result = retrieve(db_conn, "random forest and gradient boosting in scikit-learn")
        assert len(result.chunks) > 0
        assert result.chunks[0].source_url == scikit_source

    def test_chunk_mode_parent_dedup(self, db_conn):
        with _override_settings(
            retrieval_full_context_threshold=1,
            retrieval_context_budget=10_000_000,
        ):
            result = retrieve(db_conn, "ensemble methods")
        ids = [chunk.chunk_id for chunk in result.chunks]
        assert len(ids) == len(set(ids))

    def test_depth_multiplier_behavior(self):
        from src.retrieval.search import _depth_multiplier

        assert _depth_multiplier(0) == 1.0
        assert _depth_multiplier(1) <= _depth_multiplier(0)
        assert _depth_multiplier(3) <= _depth_multiplier(1)
        assert _depth_multiplier(100) >= settings.retrieval_depth_floor

    def test_chunk_mode_budget_respected(self, db_conn):
        with _override_settings(
            retrieval_full_context_threshold=1,
            retrieval_context_budget=200,
        ):
            result = retrieve(db_conn, "ensemble learning methods")
        assert len(result.chunks) >= 1
        total = _token_span_sum(result.chunks)
        assert total <= settings.retrieval_context_budget or len(result.chunks) == 1

    def test_similarity_floor_excludes_noise(self, db_conn):
        with _override_settings(
            retrieval_full_context_threshold=1,
            retrieval_context_budget=10_000_000,
            retrieval_similarity_floor=0.99,
        ):
            result = retrieve(db_conn, "zzzz qqqq nonsemantic nonsense phrase")
        assert result.mode == "chunk"
        assert result.chunks == []

    def test_chunk_mode_returns_at_least_one_when_hit_exists(self, db_conn):
        with _override_settings(
            retrieval_full_context_threshold=1,
            retrieval_context_budget=1,
        ):
            result = retrieve(db_conn, "random forest")
        assert len(result.chunks) >= 1

    def test_surface_selection_steps_do_not_force_html(self, db_conn):
        with _override_settings(
            retrieval_full_context_threshold=10_000_000,
            retrieval_context_budget=10_000_000,
        ):
            result = retrieve(db_conn, "step by step")

        step_chunks = [chunk for chunk in result.chunks if chunk.has_steps]
        if not step_chunks:
            pytest.skip("No has_steps chunks in this corpus.")
        assert any(chunk.surface == "markdown" for chunk in step_chunks)

    def test_citation_extract_valid_and_invalid_offsets(self, db_conn):
        with _override_settings(
            retrieval_full_context_threshold=10_000_000,
            retrieval_context_budget=10_000_000,
        ):
            result = retrieve(db_conn, "ensemble learning")

        markdown_chunk = next((chunk for chunk in result.chunks if chunk.surface == "markdown"), None)
        if markdown_chunk is None:
            pytest.skip("No markdown-surface chunk available for citation test.")

        quote_len = min(20, len(markdown_chunk.selected_text))
        good = extract_citation(
            markdown_chunk,
            markdown_chunk.char_start,
            markdown_chunk.char_start + quote_len,
        )
        assert good is not None
        assert good.source_url == markdown_chunk.source_url

        bad = extract_citation(
            markdown_chunk,
            markdown_chunk.char_end + 1,
            markdown_chunk.char_end + 5,
        )
        assert bad is None

    def test_return_contract_properties(self, db_conn):
        with _override_settings(
            retrieval_full_context_threshold=10_000_000,
            retrieval_context_budget=10_000_000,
        ):
            result = retrieve(db_conn, "ensemble learning")
        assert isinstance(result.is_empty, bool)
        assert result.top_score >= 0.0
        assert len(result.source_urls) == len(set(result.source_urls))
        assert result.timing.embed_ms >= 0.0
        assert result.timing.search_ms >= 0.0
        assert result.timing.total_ms >= 0.0


def test_empty_corpus_returns_empty_result(db_conn, reset_index_tables):
    reset_index_tables()
    result = retrieve(db_conn, "any query")
    assert result.mode == "full_context"
    assert result.chunks == []
    assert result.query_embedding == []
    assert result.corpus_stats.total_tokens == 0
    assert result.is_empty is True
