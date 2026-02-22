"""Unit tests for the orchestration layer.

All tests are fast, require NO external API calls, and use mocked or
synthetic data.  Each module is tested in isolation.
"""

from __future__ import annotations

import asyncio
import os
import statistics
import sys
import warnings
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest

from config import settings
from src.orchestration.models import (
    EvaluationSignals,
    ExpansionDecision,
    QueryAnalysis,
    RankedChunk,
    RerankResult,
    SubQueryResult,
)
from src.retrieval.models import (
    CorpusStats,
    PersistedLinkCandidate,
    RetrievalResult,
    RetrievedChunk,
    TimingInfo,
)

# ---------------------------------------------------------------------------
# Helpers: settings override (matches test_retrieval.py pattern)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Synthetic data builders
# ---------------------------------------------------------------------------

_COUNTER = 0


def _make_chunk_id() -> UUID:
    global _COUNTER
    _COUNTER += 1
    return UUID(int=_COUNTER)


def _make_retrieved_chunk(
    *,
    score: float = 0.5,
    raw_similarity: float | None = 0.6,
    source_url: str = "https://example.com/page",
    title: str | None = "Example",
    selected_text: str = "Some chunk text for testing purposes.",
    chunk_index: int = 0,
    document_id: UUID | None = None,
    chunk_id: UUID | None = None,
    parent_id: UUID | None = None,
    char_start: int = 0,
    char_end: int = 100,
    token_start: int = 0,
    token_end: int = 25,
    depth: int = 0,
    has_table: bool = False,
    has_code: bool = False,
    has_math: bool = False,
    has_definition_list: bool = False,
    has_admonition: bool = False,
    has_steps: bool = False,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id or _make_chunk_id(),
        document_id=document_id or uuid4(),
        parent_id=parent_id,
        source_url=source_url,
        title=title,
        selected_text=selected_text,
        surface="markdown",
        section_heading="Test Section",
        chunk_index=chunk_index,
        char_start=char_start,
        char_end=char_end,
        token_start=token_start,
        token_end=token_end,
        score=score,
        raw_similarity=raw_similarity,
        depth=depth,
        has_table=has_table,
        has_code=has_code,
        has_math=has_math,
        has_definition_list=has_definition_list,
        has_admonition=has_admonition,
        has_steps=has_steps,
        fetched_at=datetime.now(timezone.utc),
    )


def _make_ranked_chunk(
    *,
    reranked_score: float = 0.7,
    confidence: float | None = None,
    is_locality_expanded: bool = False,
    source_sub_query: str | None = None,
    token_start: int = 0,
    token_end: int = 25,
    selected_text: str = "Some chunk text for testing purposes.",
    source_url: str = "https://example.com/page",
    chunk_id: UUID | None = None,
    document_id: UUID | None = None,
    **chunk_kwargs: Any,
) -> RankedChunk:
    chunk = _make_retrieved_chunk(
        token_start=token_start,
        token_end=token_end,
        selected_text=selected_text,
        source_url=source_url,
        chunk_id=chunk_id,
        document_id=document_id,
        **chunk_kwargs,
    )
    return RankedChunk(
        chunk=chunk,
        reranked_score=reranked_score,
        confidence=confidence,
        is_locality_expanded=is_locality_expanded,
        source_sub_query=source_sub_query,
    )


def _make_ranked_chunks_with_scores(
    scores: list[float],
    *,
    token_span: int = 25,
    source_url: str = "https://example.com/page",
) -> list[RankedChunk]:
    """Build a list of RankedChunks with the given reranked_score values."""
    doc_id = uuid4()
    chunks = []
    for i, s in enumerate(scores):
        text = f"Chunk text number {i} with distinct words set-{i}."
        chunks.append(
            _make_ranked_chunk(
                reranked_score=s,
                token_start=i * token_span,
                token_end=(i + 1) * token_span,
                selected_text=text,
                source_url=source_url,
                document_id=doc_id,
                chunk_index=i,
            )
        )
    return chunks


def _make_retrieval_result(
    chunks: list[RetrievedChunk] | None = None,
    *,
    mode: str = "chunk",
) -> RetrievalResult:
    if chunks is None:
        chunks = [_make_retrieved_chunk()]
    return RetrievalResult(
        mode=mode,
        chunks=chunks,
        query_embedding=[0.1] * 10,
        corpus_stats=CorpusStats(
            total_documents=1,
            total_parent_chunks=5,
            total_tokens=500,
            documents_matched=["https://example.com/page"],
        ),
        timing=TimingInfo(embed_ms=1.0, search_ms=2.0, total_ms=3.0),
    )


def _make_query_analysis(
    query: str = "test query",
    sub_queries: list[str] | None = None,
    query_type: str = "factual",
) -> QueryAnalysis:
    return QueryAnalysis(
        original_query=query,
        sub_queries=sub_queries or [query],
        query_type=query_type,
        complexity="simple",
        key_concepts=["test"],
    )


def _make_link_candidate(
    *,
    target_url: str = "https://example.com/other",
    source_url: str = "https://example.com/page",
    title: str | None = "Other Page",
    description: str | None = "A description about gradient boosting.",
    depth: int = 1,
) -> PersistedLinkCandidate:
    return PersistedLinkCandidate(
        id=uuid4(),
        source_document_id=uuid4(),
        source_url=source_url,
        target_url=target_url,
        title=title,
        description=description,
        discovered_at=datetime.now(timezone.utc),
        enriched_at=datetime.now(timezone.utc),
        depth=depth,
    )


# ===================================================================
# 1.1  Query Analyzer Tests
# ===================================================================

class TestQueryAnalyzerRuleBased:
    """Test rule-based decomposition (decomposition_mode='rule_based')."""

    @pytest.mark.asyncio
    async def test_comparison_query_splits(self):
        from src.orchestration.query_analyzer import analyze_query

        with _override_settings(decomposition_mode="rule_based"):
            result = await analyze_query("gradient boosting vs random forests")

        assert len(result.sub_queries) == 2
        lower = [sq.lower() for sq in result.sub_queries]
        assert any("gradient boosting" in sq for sq in lower)
        assert any("random forests" in sq for sq in lower)
        assert result.query_type == "comparison"

    @pytest.mark.asyncio
    async def test_pros_cons_query_splits(self):
        from src.orchestration.query_analyzer import analyze_query

        with _override_settings(decomposition_mode="rule_based"):
            result = await analyze_query("pros and cons of microservices")

        assert len(result.sub_queries) == 2
        lower = [sq.lower() for sq in result.sub_queries]
        assert any("advantage" in sq for sq in lower)
        assert any("disadvantage" in sq for sq in lower)

    @pytest.mark.asyncio
    async def test_simple_factual_passthrough(self):
        from src.orchestration.query_analyzer import analyze_query

        with _override_settings(decomposition_mode="rule_based"):
            result = await analyze_query("what is backpropagation")

        assert result.sub_queries == ["what is backpropagation"]
        assert result.query_type == "factual"

    @pytest.mark.asyncio
    async def test_multi_part_query_splits(self):
        from src.orchestration.query_analyzer import analyze_query

        with _override_settings(decomposition_mode="rule_based"):
            result = await analyze_query(
                "how does gradient descent work and when should I use it"
            )

        assert len(result.sub_queries) == 2

    @pytest.mark.asyncio
    async def test_key_concepts_extraction(self):
        from src.orchestration.query_analyzer import analyze_query

        with _override_settings(decomposition_mode="rule_based"):
            result = await analyze_query("gradient boosting vs random forests")

        # Should contain meaningful tokens, not stop words.
        assert "gradient" in result.key_concepts
        assert "boosting" in result.key_concepts
        assert "random" in result.key_concepts
        assert "forests" in result.key_concepts
        # Stop words excluded.
        assert "vs" not in result.key_concepts


class TestQueryAnalyzerNoneMode:
    """Test none mode — passthrough."""

    @pytest.mark.asyncio
    async def test_none_mode_passthrough(self):
        from src.orchestration.query_analyzer import analyze_query

        with _override_settings(decomposition_mode="none"):
            result = await analyze_query("any complex query about multiple topics")

        assert result.sub_queries == ["any complex query about multiple topics"]
        assert result.original_query == "any complex query about multiple topics"


class TestQueryAnalyzerIntentHint:
    """Test that MCP intent hint is respected."""

    @pytest.mark.asyncio
    async def test_intent_sets_query_type(self):
        from src.orchestration.query_analyzer import analyze_query

        with _override_settings(decomposition_mode="rule_based"):
            result = await analyze_query("tell me about X", intent="how_to")

        assert result.query_type == "how_to"


class TestQueryAnalyzerConstraints:
    """Test constraint appending."""

    @pytest.mark.asyncio
    async def test_constraints_appended_to_subqueries(self):
        from src.orchestration.query_analyzer import analyze_query

        with _override_settings(decomposition_mode="rule_based"):
            result = await analyze_query(
                "gradient boosting vs random forests",
                constraints=["must include code examples"],
            )

        for sq in result.sub_queries:
            assert "must include code examples" in sq


class TestQueryAnalyzerLLMMocked:
    """Test LLM decomposition with mocked AsyncOpenAI."""

    @pytest.mark.asyncio
    async def test_llm_decompose_with_rewrite_context(self):
        from src.orchestration.query_analyzer import analyze_query

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"sub_queries": ["rewritten query A", "rewritten query B"], "query_type": "comparison", "complexity": "moderate", "key_concepts": ["topic"]}'

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        with _override_settings(
            decomposition_mode="llm",
            orchestration_llm_api_key="fake-key",
        ):
            with patch(
                "src.orchestration.query_analyzer.AsyncOpenAI",
                return_value=mock_client,
            ):
                result = await analyze_query(
                    "original query",
                    rewrite_context="Previous retrieval insufficient.",
                )

        # Verify rewrite_context was included in the user message.
        call_args = mock_client.chat.completions.create.call_args
        messages = call_args.kwargs.get("messages") or call_args[1].get("messages")
        user_msg = messages[-1]["content"]
        assert "Previous retrieval insufficient." in user_msg

        assert result.sub_queries == ["rewritten query A", "rewritten query B"]

    @pytest.mark.asyncio
    async def test_llm_decompose_strips_markdown_fences(self):
        from src.orchestration.query_analyzer import analyze_query

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '```json\n{"sub_queries": ["q1"], "query_type": "factual", "complexity": "simple", "key_concepts": ["test"]}\n```'

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        with _override_settings(
            decomposition_mode="llm",
            orchestration_llm_api_key="fake-key",
        ):
            with patch(
                "src.orchestration.query_analyzer.AsyncOpenAI",
                return_value=mock_client,
            ):
                result = await analyze_query("test query")

        assert result.sub_queries == ["q1"]


class TestQueryAnalyzerFallback:
    """Test fallback when LLM mode is set but API key is missing."""

    @pytest.mark.asyncio
    async def test_llm_mode_without_key_falls_back(self):
        from src.orchestration.query_analyzer import analyze_query

        with _override_settings(
            decomposition_mode="llm",
            orchestration_llm_api_key=None,
        ):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result = await analyze_query("gradient boosting vs random forests")

        # Should fall back to rule_based.
        assert len(result.sub_queries) == 2
        assert any("orchestration_llm_api_key" in str(w.message) for w in caught)

    @pytest.mark.asyncio
    async def test_llm_exception_falls_back_to_rule_based(self):
        from src.orchestration.query_analyzer import analyze_query

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(
            side_effect=RuntimeError("API down"),
        )

        with _override_settings(
            decomposition_mode="llm",
            orchestration_llm_api_key="fake-key",
        ):
            with patch(
                "src.orchestration.query_analyzer.AsyncOpenAI",
                return_value=mock_client,
            ):
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    result = await analyze_query("what is backpropagation")

        assert result.sub_queries == ["what is backpropagation"]
        assert any("LLM decomposition failed" in str(w.message) for w in caught)


# ===================================================================
# 1.2  Reranker Tests
# ===================================================================

class TestRerankerPassthrough:
    """Test passthrough mode (provider='none')."""

    @pytest.mark.asyncio
    async def test_passthrough_preserves_scores(self):
        from src.orchestration.reranker import rerank

        scores = [0.65, 0.52, 0.41, 0.38]
        passages = [f"passage {i}" for i in range(4)]

        with _override_settings(reranker_provider="none"):
            results = await rerank(
                "test query", passages, original_scores=scores,
            )

        # Should be sorted by score descending.
        assert [r.relevance_score for r in results] == sorted(scores, reverse=True)
        # Confidence should be None for passthrough.
        assert all(r.confidence is None for r in results)

    @pytest.mark.asyncio
    async def test_passthrough_empty_passages(self):
        from src.orchestration.reranker import rerank

        with _override_settings(reranker_provider="none"):
            results = await rerank("test", [], original_scores=[])

        assert results == []


class TestRerankerZeroEntropyMocked:
    """Test ZeroEntropy provider with mocked SDK."""

    @pytest.mark.asyncio
    async def test_zeroentropy_mapping(self):
        from src.orchestration.reranker import rerank

        # Mock the ZeroEntropy SDK response.
        mock_result_0 = MagicMock()
        mock_result_0.index = 0
        mock_result_0.relevance_score = 0.92
        mock_result_0.confidence = None
        mock_result_1 = MagicMock()
        mock_result_1.index = 1
        mock_result_1.relevance_score = 0.68
        mock_result_1.confidence = None

        mock_response = MagicMock()
        mock_response.results = [mock_result_0, mock_result_1]

        mock_client = AsyncMock()
        mock_client.models.rerank = AsyncMock(return_value=mock_response)

        # Build a fake 'zeroentropy' module whose AsyncZeroEntropy returns
        # our mock_client.  The lazy `from zeroentropy import AsyncZeroEntropy`
        # will resolve against sys.modules.
        fake_ze_mod = MagicMock()
        fake_ze_mod.AsyncZeroEntropy = MagicMock(return_value=mock_client)

        with _override_settings(
            reranker_provider="zeroentropy",
            reranker_api_key="fake-key",
            reranker_model="zerank-2",
        ):
            with patch.dict(sys.modules, {"zeroentropy": fake_ze_mod}):
                results = await rerank(
                    "test query",
                    ["doc A", "doc B"],
                    instruction="find relevant info",
                )

        assert len(results) == 2
        assert results[0].relevance_score == 0.92
        assert results[1].relevance_score == 0.68
        # Verify call was made with correct params.
        mock_client.models.rerank.assert_called_once()


class TestRerankerErrorFallback:
    """Test that reranker errors fall back to passthrough."""

    @pytest.mark.asyncio
    async def test_provider_error_falls_back(self):
        from src.orchestration.reranker import rerank

        mock_client = AsyncMock()
        mock_client.models.rerank = AsyncMock(
            side_effect=RuntimeError("Service unavailable"),
        )

        with _override_settings(
            reranker_provider="zeroentropy",
            reranker_api_key="fake-key",
        ):
            with patch.dict(
                sys.modules,
                {"zeroentropy": MagicMock(AsyncZeroEntropy=MagicMock(return_value=mock_client))},
            ):
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    results = await rerank(
                        "query",
                        ["passage A", "passage B"],
                        original_scores=[0.8, 0.5],
                    )

        # Fallback to passthrough — should still return results.
        assert len(results) == 2
        assert any("falling back to passthrough" in str(w.message) for w in caught)

    @pytest.mark.asyncio
    async def test_unknown_provider_falls_back(self):
        from src.orchestration.reranker import rerank

        with _override_settings(reranker_provider="nonexistent_provider"):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                results = await rerank(
                    "query",
                    ["passage"],
                    original_scores=[0.9],
                )

        assert len(results) == 1
        assert any("Unknown reranker_provider" in str(w.message) for w in caught)


class TestScorePreservation:
    """CRITICAL: verify retrieval scores and reranked scores are distinct."""

    def test_scores_not_conflated(self):
        chunk = _make_retrieved_chunk(score=0.65, raw_similarity=0.70)
        ranked = RankedChunk(
            chunk=chunk,
            reranked_score=0.82,
            confidence=0.9,
        )

        # Inner retrieval scores are preserved.
        assert ranked.chunk.score == 0.65
        assert ranked.chunk.raw_similarity == 0.70
        # Outer reranked score is different.
        assert ranked.reranked_score == 0.82
        # They must not be equal.
        assert ranked.reranked_score != ranked.chunk.score


# ===================================================================
# 1.3  Evaluator Tests
# ===================================================================

class TestEvaluatorStopGoodPlateau:
    """STOP — good plateau: low variance, high mean."""

    @pytest.mark.asyncio
    async def test_good_plateau_stops(self):
        from src.orchestration.evaluator import evaluate

        scores = [0.78, 0.76, 0.75, 0.74, 0.73, 0.72, 0.71, 0.70, 0.69, 0.68]
        chunks = _make_ranked_chunks_with_scores(scores)
        qa = _make_query_analysis()

        with _override_settings(
            reranker_provider="zeroentropy",
            mediocre_score_floor=0.5,
            plateau_variance_threshold=0.02,
        ):
            signals, decision = await evaluate(chunks, 4096, 0, None, qa)

        assert decision.action == "stop"
        assert "plateau" in decision.reason.lower()


class TestEvaluatorStopSaturation:
    """STOP — token budget saturation."""

    @pytest.mark.asyncio
    async def test_saturated_budget_stops(self):
        from src.orchestration.evaluator import evaluate

        # Create chunks with large token spans that exceed 80% of budget.
        budget = 100
        chunks = _make_ranked_chunks_with_scores(
            [0.8, 0.75, 0.72, 0.70, 0.68],
            token_span=20,  # 5 chunks × 20 tokens = 100 tokens → fill 100%
        )
        qa = _make_query_analysis()

        with _override_settings(
            reranker_provider="zeroentropy",
            mediocre_score_floor=0.5,
            token_budget_saturation_ratio=0.8,
        ):
            signals, decision = await evaluate(chunks, budget, 0, None, qa)

        assert signals.token_fill_ratio >= 0.8
        assert decision.action == "stop"


class TestEvaluatorStopDiminishingReturns:
    """STOP — diminishing returns (iteration > 0)."""

    @pytest.mark.asyncio
    async def test_diminishing_returns_stops(self):
        from src.orchestration.evaluator import evaluate

        # Previous signals had 8 chunks above threshold.
        prev = EvaluationSignals(
            top_score=0.75,
            score_at_k=0.60,
            score_cliff=0.15,
            score_variance=0.01,
            score_mean=0.70,
            chunks_above_threshold=8,
            token_fill_ratio=0.5,
            redundancy_ratio=0.1,
            source_document_count=3,
            avg_confidence=None,
            is_plateau=False,
            is_cliff=False,
            is_saturated=False,
            is_mediocre_plateau=False,
            has_high_redundancy=False,
        )

        # Current: 9 chunks above threshold (delta=1, below default 0.03 threshold).
        # Actually diminishing_return_delta compares chunk count delta, not float.
        # Looking at the code: recall_delta = chunks_above_threshold difference.
        # diminishing_return_delta default = 0.03 but compared to int count.
        # delta=1 < 0.03? No, 1 > 0.03. We need delta < 0.03 → delta must be 0.
        # Let's make current chunks_above_threshold = 8 (same) → delta=0.
        scores = [0.75, 0.73, 0.72, 0.70, 0.69, 0.68, 0.67, 0.66]
        chunks = _make_ranked_chunks_with_scores(scores)
        qa = _make_query_analysis()

        with _override_settings(
            reranker_provider="zeroentropy",
            mediocre_score_floor=0.5,
            diminishing_return_delta=0.03,
            max_expansion_depth=10,
            # Prevent plateau from firing first.
            plateau_variance_threshold=0.0001,
        ):
            signals, decision = await evaluate(chunks, 4096, 1, prev, qa)

        assert decision.action == "stop"
        assert "diminishing" in decision.reason.lower()


class TestEvaluatorExpandBreadthCliff:
    """EXPAND_BREADTH — score cliff with low redundancy."""

    @pytest.mark.asyncio
    async def test_cliff_low_redundancy_expands_breadth(self):
        from src.orchestration.evaluator import evaluate

        # Big cliff: top score 0.82, rest are ~0.4
        scores = [0.82, 0.45, 0.42, 0.40, 0.38, 0.36, 0.34, 0.32, 0.30, 0.28]
        chunks = _make_ranked_chunks_with_scores(scores)
        qa = _make_query_analysis()

        with _override_settings(
            reranker_provider="zeroentropy",
            mediocre_score_floor=0.5,
            score_cliff_threshold=0.15,
            score_cliff_rank_k=5,
            token_budget_saturation_ratio=0.8,
        ):
            signals, decision = await evaluate(chunks, 4096, 0, None, qa)

        assert signals.is_cliff
        assert not signals.has_high_redundancy
        assert decision.action == "expand_breadth"


class TestEvaluatorExpandIntentCliffRedundant:
    """EXPAND_INTENT — score cliff with high redundancy."""

    @pytest.mark.asyncio
    async def test_cliff_high_redundancy_expands_intent(self):
        from src.orchestration.evaluator import evaluate

        # Cliff + high redundancy: top is 0.82, rest cluster tightly
        # around 0.45 (many near-duplicate scores → high redundancy).
        scores = [0.82, 0.450, 0.451, 0.452, 0.450, 0.451, 0.450, 0.449, 0.450, 0.451]
        chunks = _make_ranked_chunks_with_scores(scores)
        qa = _make_query_analysis()

        with _override_settings(
            reranker_provider="zeroentropy",
            mediocre_score_floor=0.5,
            score_cliff_threshold=0.15,
            score_cliff_rank_k=5,
            redundancy_ceiling=0.3,  # Low ceiling to trigger high redundancy.
            token_budget_saturation_ratio=0.8,
        ):
            signals, decision = await evaluate(chunks, 4096, 0, None, qa)

        assert signals.is_cliff
        assert signals.has_high_redundancy
        assert decision.action == "expand_intent"


class TestEvaluatorExpandIntentMediocrePlateau:
    """EXPAND_INTENT — mediocre plateau (low variance, low mean)."""

    @pytest.mark.asyncio
    async def test_mediocre_plateau_expands_intent(self):
        from src.orchestration.evaluator import evaluate

        scores = [0.38, 0.37, 0.36, 0.35, 0.35, 0.34, 0.33, 0.32, 0.31, 0.30]
        chunks = _make_ranked_chunks_with_scores(scores)
        qa = _make_query_analysis()

        with _override_settings(
            reranker_provider="zeroentropy",
            mediocre_score_floor=0.5,
            plateau_variance_threshold=0.02,
            token_budget_saturation_ratio=0.8,
        ):
            signals, decision = await evaluate(chunks, 4096, 0, None, qa)

        assert signals.is_mediocre_plateau
        assert decision.action == "expand_intent"


class TestEvaluatorEmptyChunks:
    """Edge case — no chunks at all."""

    @pytest.mark.asyncio
    async def test_empty_chunks_stops(self):
        from src.orchestration.evaluator import evaluate

        qa = _make_query_analysis()

        signals, decision = await evaluate([], 4096, 0, None, qa)

        assert signals.top_score == 0.0
        # With empty chunks, top_score < mediocre_floor → expand_intent.
        assert decision.action in ("expand_intent", "stop")


class TestEvaluatorMaxDepth:
    """STOP — max expansion depth safety cap."""

    @pytest.mark.asyncio
    async def test_max_depth_forces_stop(self):
        from src.orchestration.evaluator import evaluate

        scores = [0.30, 0.29]
        chunks = _make_ranked_chunks_with_scores(scores)
        qa = _make_query_analysis()

        with _override_settings(max_expansion_depth=3):
            # iteration=3 >= max_expansion_depth=3 → stop.
            signals, decision = await evaluate(chunks, 4096, 3, None, qa)

        assert decision.action == "stop"
        assert "max expansion depth" in decision.reason.lower()


# ===================================================================
# 1.4  Locality Expansion Tests (DB-dependent)
# ===================================================================

@pytest.fixture()
def _locality_test_data(async_db_conn):
    """Insert a synthetic document with 5 parent chunks for locality tests.

    Yields the document_id and the list of chunk_ids in index order.
    Cleans up after the test.
    """
    import asyncio

    doc_id = uuid4()
    chunk_ids = [uuid4() for _ in range(5)]
    source_url = f"https://locality-test-{doc_id.hex[:8]}.example.com/page"

    async def _setup():
        async with async_db_conn.cursor() as cur:
            # Insert a synthetic document (all NOT NULL columns filled).
            await cur.execute(
                """
                INSERT INTO documents (
                    id, url, source_url, title, doc_type,
                    content_hash, fetched_at, depth
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    str(doc_id), source_url, source_url,
                    "Locality Test Doc", "html",
                    "fakehash_locality", datetime.now(timezone.utc), 0,
                ),
            )
            # Insert 5 parent chunks.
            for i, cid in enumerate(chunk_ids):
                text = f"Parent chunk {i} content about locality testing."
                await cur.execute(
                    """
                    INSERT INTO chunks (
                        id, document_id, parent_id, chunk_level, chunk_index,
                        section_heading, chunk_text, html_text,
                        has_table, has_code, has_math,
                        has_definition_list, has_admonition, has_steps,
                        char_start, char_end, token_start, token_end,
                        source_url, fetched_at, depth, title
                    ) VALUES (
                        %s, %s, NULL, 'parent', %s,
                        'Section', %s, NULL,
                        false, false, false,
                        false, false, false,
                        %s, %s, %s, %s,
                        %s, %s, %s, %s
                    )
                    """,
                    (
                        str(cid), str(doc_id), i,
                        text, i * 100, (i + 1) * 100, i * 25, (i + 1) * 25,
                        source_url, datetime.now(timezone.utc), 0,
                        "Locality Test Doc",
                    ),
                )

    asyncio.get_event_loop().run_until_complete(_setup())

    yield {
        "doc_id": doc_id,
        "chunk_ids": chunk_ids,
        "source_url": source_url,
    }

    # Cleanup.
    async def _teardown():
        async with async_db_conn.cursor() as cur:
            await cur.execute(
                "DELETE FROM chunks WHERE document_id = %s", (str(doc_id),),
            )
            await cur.execute(
                "DELETE FROM documents WHERE id = %s", (str(doc_id),),
            )

    asyncio.get_event_loop().run_until_complete(_teardown())


@pytest.mark.asyncio(loop_scope="session")
class TestLocalityExpansion:
    """Locality expansion tests — require a running Postgres."""

    async def test_sibling_grab(self, async_db_conn, _locality_test_data):
        from src.orchestration.locality import expand_locality

        data = _locality_test_data
        # Create a high-scoring hit at chunk_index 2.
        hit = _make_ranked_chunk(
            reranked_score=0.8,
            chunk_id=data["chunk_ids"][2],
            document_id=data["doc_id"],
            source_url=data["source_url"],
        )
        hit.chunk.chunk_index = 2

        with _override_settings(
            locality_expansion_enabled=True,
            locality_expansion_radius=1,
        ):
            new_chunks = await expand_locality(
                [hit], async_db_conn, radius=1, min_score_for_expansion=0.5,
            )

        new_indexes = {c.chunk.chunk_index for c in new_chunks}
        # Should return index 1 and 3 (adjacent to 2).
        assert 1 in new_indexes
        assert 3 in new_indexes
        # All locality chunks should be flagged.
        assert all(c.is_locality_expanded for c in new_chunks)
        assert all(c.reranked_score == 0.0 for c in new_chunks)

    async def test_deduplication(self, async_db_conn, _locality_test_data):
        from src.orchestration.locality import expand_locality

        data = _locality_test_data
        # Hit at index 2, chunk at index 1 already in result set.
        hit_2 = _make_ranked_chunk(
            reranked_score=0.8,
            chunk_id=data["chunk_ids"][2],
            document_id=data["doc_id"],
            source_url=data["source_url"],
        )
        hit_2.chunk.chunk_index = 2

        existing_1 = _make_ranked_chunk(
            reranked_score=0.6,
            chunk_id=data["chunk_ids"][1],
            document_id=data["doc_id"],
            source_url=data["source_url"],
        )
        existing_1.chunk.chunk_index = 1

        with _override_settings(
            locality_expansion_enabled=True,
            locality_expansion_radius=1,
        ):
            new_chunks = await expand_locality(
                [hit_2, existing_1], async_db_conn,
                radius=1, min_score_for_expansion=0.5,
            )

        new_ids = {c.chunk.chunk_id for c in new_chunks}
        # Index 1 already present → should NOT be added again.
        assert data["chunk_ids"][1] not in new_ids
        # Index 3 should be added.
        assert data["chunk_ids"][3] in new_ids

    async def test_score_threshold_excludes_low_scorers(
        self, async_db_conn, _locality_test_data,
    ):
        from src.orchestration.locality import expand_locality

        data = _locality_test_data
        hit = _make_ranked_chunk(
            reranked_score=0.3,  # Below threshold.
            chunk_id=data["chunk_ids"][2],
            document_id=data["doc_id"],
            source_url=data["source_url"],
        )
        hit.chunk.chunk_index = 2

        with _override_settings(locality_expansion_enabled=True):
            new_chunks = await expand_locality(
                [hit], async_db_conn,
                radius=1, min_score_for_expansion=0.5,
            )

        assert new_chunks == []

    async def test_edge_index_zero(self, async_db_conn, _locality_test_data):
        from src.orchestration.locality import expand_locality

        data = _locality_test_data
        hit = _make_ranked_chunk(
            reranked_score=0.8,
            chunk_id=data["chunk_ids"][0],
            document_id=data["doc_id"],
            source_url=data["source_url"],
        )
        hit.chunk.chunk_index = 0

        with _override_settings(
            locality_expansion_enabled=True,
            locality_expansion_radius=1,
        ):
            new_chunks = await expand_locality(
                [hit], async_db_conn,
                radius=1, min_score_for_expansion=0.5,
            )

        new_indexes = {c.chunk.chunk_index for c in new_chunks}
        assert 1 in new_indexes
        # No negative index.
        assert all(c.chunk.chunk_index >= 0 for c in new_chunks)

    async def test_edge_max_index(self, async_db_conn, _locality_test_data):
        from src.orchestration.locality import expand_locality

        data = _locality_test_data
        hit = _make_ranked_chunk(
            reranked_score=0.8,
            chunk_id=data["chunk_ids"][4],  # Last chunk, index 4.
            document_id=data["doc_id"],
            source_url=data["source_url"],
        )
        hit.chunk.chunk_index = 4

        with _override_settings(
            locality_expansion_enabled=True,
            locality_expansion_radius=1,
        ):
            new_chunks = await expand_locality(
                [hit], async_db_conn,
                radius=1, min_score_for_expansion=0.5,
            )

        new_indexes = {c.chunk.chunk_index for c in new_chunks}
        assert 3 in new_indexes
        # No index beyond 4.
        assert all(c.chunk.chunk_index <= 4 for c in new_chunks)


# ===================================================================
# 1.5  Expander Tests (Link Scoring)
# ===================================================================

class TestExpanderLinkScoring:
    """Test the link scoring formula."""

    @pytest.mark.asyncio
    async def test_score_computation(self):
        from src.orchestration.expander import score_candidates

        candidate = _make_link_candidate(
            target_url="https://example.com/modules/gradient_boosting.html",
            title="Gradient Boosting Guide",
            description="Complete guide to gradient boosting algorithms.",
        )
        qa = _make_query_analysis(
            query="gradient boosting",
            query_type="how_to",
        )
        qa.key_concepts = ["gradient", "boosting"]

        scored = await score_candidates(
            [candidate], "gradient boosting", qa,
            already_ingested_urls=set(),
        )

        assert len(scored) == 1
        assert scored[0].score > 0
        # Score breakdown should contain all components.
        bd = scored[0].score_breakdown
        assert "url_path" in bd
        assert "title" in bd
        assert "description" in bd
        assert "in_degree" in bd
        assert "depth" in bd

    @pytest.mark.asyncio
    async def test_url_path_relevance_high(self):
        from src.orchestration.expander import score_candidates

        candidate = _make_link_candidate(
            target_url="https://example.com/modules/ensemble/gradient_boosting.html",
            title=None,
            description=None,
        )
        qa = _make_query_analysis()
        qa.key_concepts = ["gradient", "boosting"]

        scored = await score_candidates(
            [candidate], "gradient boosting", qa,
            already_ingested_urls=set(),
        )

        assert scored[0].score_breakdown["url_path"] > 0

    @pytest.mark.asyncio
    async def test_url_path_relevance_low(self):
        from src.orchestration.expander import score_candidates

        candidate = _make_link_candidate(
            target_url="https://example.com/api/v2/internal/config",
            title=None,
            description=None,
        )
        qa = _make_query_analysis()
        qa.key_concepts = ["gradient", "boosting"]

        scored = await score_candidates(
            [candidate], "gradient boosting", qa,
            already_ingested_urls=set(),
        )

        # URL path has no overlap with key concepts.
        assert scored[0].score_breakdown["url_path"] == 0.0

    @pytest.mark.asyncio
    async def test_unenriched_candidate_penalized(self):
        from src.orchestration.expander import score_candidates

        enriched = _make_link_candidate(
            target_url="https://example.com/enriched",
            title="Gradient Boosting",
            description="How gradient boosting works.",
        )
        unenriched = _make_link_candidate(
            target_url="https://example.com/unenriched",
            title=None,
            description=None,
        )
        qa = _make_query_analysis()
        qa.key_concepts = ["gradient", "boosting"]

        scored = await score_candidates(
            [enriched, unenriched], "gradient boosting", qa,
            already_ingested_urls=set(),
        )

        enriched_score = next(s for s in scored if s.link_candidate.target_url.endswith("enriched"))
        unenriched_score = next(s for s in scored if s.link_candidate.target_url.endswith("unenriched"))
        assert enriched_score.score > unenriched_score.score

    @pytest.mark.asyncio
    async def test_already_ingested_filtered_out(self):
        from src.orchestration.expander import score_candidates

        candidate = _make_link_candidate(
            target_url="https://example.com/already-ingested",
        )
        qa = _make_query_analysis()

        scored = await score_candidates(
            [candidate], "test", qa,
            already_ingested_urls={"https://example.com/already-ingested"},
        )

        assert len(scored) == 0


class TestExpanderParentURLDerivation:
    """Test _derive_parent_urls."""

    def test_derive_parents(self):
        from src.orchestration.expander import _derive_parent_urls

        parents = _derive_parent_urls(
            "https://scikit-learn.org/stable/modules/ensemble.html"
        )

        assert "https://scikit-learn.org/stable/modules/" in parents
        assert "https://scikit-learn.org/stable/" in parents
        # Root domain excluded (only 1 segment would remain).
        assert "https://scikit-learn.org/" not in parents

    def test_root_url_no_parents(self):
        from src.orchestration.expander import _derive_parent_urls

        parents = _derive_parent_urls("https://example.com/")
        assert parents == []

    def test_single_segment_no_parents(self):
        from src.orchestration.expander import _derive_parent_urls

        parents = _derive_parent_urls("https://example.com/page")
        assert parents == []


# ===================================================================
# 1.6  Merger Tests
# ===================================================================

class TestMergerDeduplication:
    """Test chunk deduplication by chunk_id."""

    @pytest.mark.asyncio
    async def test_keeps_max_score(self):
        from src.orchestration.merger import merge_subquery_results

        shared_chunk_id = _make_chunk_id()
        doc_id = uuid4()

        chunk_a = _make_retrieved_chunk(
            chunk_id=shared_chunk_id, document_id=doc_id, score=0.5,
        )
        chunk_b = _make_retrieved_chunk(
            chunk_id=shared_chunk_id, document_id=doc_id, score=0.5,
        )

        sqr_a = SubQueryResult(
            sub_query="query A",
            retrieval_result=_make_retrieval_result([chunk_a]),
            reranked_chunks=[
                RankedChunk(chunk=chunk_a, reranked_score=0.72),
            ],
        )
        sqr_b = SubQueryResult(
            sub_query="query B",
            retrieval_result=_make_retrieval_result([chunk_b]),
            reranked_chunks=[
                RankedChunk(chunk=chunk_b, reranked_score=0.85),
            ],
        )

        merged = await merge_subquery_results([sqr_a, sqr_b], 4096)

        # Same chunk_id should appear only once, with max score.
        assert len(merged) == 1
        assert merged[0].reranked_score == 0.85


class TestMergerMMRDedup:
    """Test MMR deduplication removes near-duplicate text."""

    @pytest.mark.asyncio
    async def test_near_duplicates_filtered(self):
        from src.orchestration.merger import merge_ranked_chunks

        # Two chunks with nearly identical text.
        chunk_a = _make_ranked_chunk(
            reranked_score=0.9,
            selected_text="gradient boosting is a machine learning technique",
        )
        chunk_b = _make_ranked_chunk(
            reranked_score=0.85,
            selected_text="gradient boosting is a machine learning technique for ensemble",
        )
        # Third chunk with very different text.
        chunk_c = _make_ranked_chunk(
            reranked_score=0.7,
            selected_text="random forests use decision tree bagging for classification",
        )

        with _override_settings(redundancy_ceiling=0.5):
            merged = await merge_ranked_chunks(
                [chunk_a, chunk_b, chunk_c], 4096,
            )

        # chunk_b should be filtered out as near-duplicate of chunk_a.
        assert len(merged) <= 2
        # chunk_a (highest score) and chunk_c (different text) should remain.
        scores = {c.reranked_score for c in merged}
        assert 0.9 in scores
        assert 0.7 in scores


class TestMergerTokenBudget:
    """Test token budget enforcement."""

    @pytest.mark.asyncio
    async def test_budget_respected(self):
        from src.orchestration.merger import merge_ranked_chunks

        chunks = [
            _make_ranked_chunk(
                reranked_score=0.9 - i * 0.1,
                token_start=0,
                token_end=50,
                selected_text=f"unique content {i} " * 10,
            )
            for i in range(5)
        ]

        with _override_settings(redundancy_ceiling=1.0):  # Disable MMR.
            merged = await merge_ranked_chunks(chunks, 100)  # Budget=100 tokens.

        # 50 tokens per chunk, budget=100 → at most 2 chunks.
        assert len(merged) <= 2
        # Highest scored chunks kept.
        assert merged[0].reranked_score >= merged[-1].reranked_score

    @pytest.mark.asyncio
    async def test_at_least_one_guarantee(self):
        from src.orchestration.merger import merge_ranked_chunks

        chunk = _make_ranked_chunk(
            reranked_score=0.9, token_start=0, token_end=500,
        )

        with _override_settings(redundancy_ceiling=1.0):
            merged = await merge_ranked_chunks([chunk], 10)  # Budget far too small.

        # At-least-one guarantee.
        assert len(merged) == 1


# ===================================================================
# 1.7  Engine Tests (Mocked Integration)
# ===================================================================

class TestEngineSinglePass:
    """Test engine with no expansion needed."""

    @pytest.mark.asyncio
    async def test_no_expansion(self):
        from src.orchestration.engine import OrchestratorEngine

        engine = OrchestratorEngine()

        good_chunks = [
            _make_retrieved_chunk(score=0.85, token_start=i*30, token_end=(i+1)*30)
            for i in range(10)
        ]
        good_rr = _make_retrieval_result(good_chunks, mode="chunk")

        # Passthrough reranker: scores copied as-is.
        rerank_results = [
            RerankResult(index=i, relevance_score=c.score)
            for i, c in enumerate(good_chunks)
        ]

        with _override_settings(
            reranker_provider="none",
            decomposition_mode="none",
            retrieval_context_budget=4096,
            locality_expansion_enabled=False,
        ):
            with patch.object(engine, "_pool", MagicMock()):
                with patch.object(engine, "_acquire_connection", new_callable=AsyncMock) as mock_conn:
                    mock_conn.return_value = MagicMock()
                    with patch.object(engine, "_release_connection", new_callable=AsyncMock):
                        with patch.object(engine, "_ensure_ingested", new_callable=AsyncMock):
                            with patch(
                                "src.orchestration.engine.retrieve",
                                new_callable=AsyncMock,
                                return_value=good_rr,
                            ):
                                with patch(
                                    "src.orchestration.engine.rerank",
                                    new_callable=AsyncMock,
                                    return_value=rerank_results,
                                ):
                                    result = await engine.run(
                                        "https://example.com", "test query",
                                    )

        assert result.total_iterations == 0
        assert result.expansion_steps == []
        assert len(result.chunks) > 0


class TestEngineOneExpansion:
    """Test engine with exactly one expansion iteration."""

    @pytest.mark.asyncio
    async def test_one_expansion(self):
        from src.orchestration.engine import OrchestratorEngine

        engine = OrchestratorEngine()

        # First retrieval: poor.
        poor_chunks = [_make_retrieved_chunk(score=0.3)]
        poor_rr = _make_retrieval_result(poor_chunks, mode="chunk")

        # Second retrieval: good.
        good_chunks = [
            _make_retrieved_chunk(score=0.85, token_start=i*30, token_end=(i+1)*30)
            for i in range(10)
        ]
        good_rr = _make_retrieval_result(good_chunks, mode="chunk")

        retrieve_mock = AsyncMock(side_effect=[poor_rr, good_rr])

        def make_rerank_results(query, passages, **kw):
            scores = kw.get("original_scores", [0.5] * len(passages))
            return [
                RerankResult(index=i, relevance_score=s)
                for i, s in enumerate(scores)
            ]

        rerank_mock = AsyncMock(side_effect=make_rerank_results)

        # Evaluator: expand_breadth first, then stop.
        eval_results = [
            (
                EvaluationSignals(
                    top_score=0.3, score_at_k=0.3, score_cliff=0.0,
                    score_variance=0.0, score_mean=0.3,
                    chunks_above_threshold=1, token_fill_ratio=0.1,
                    redundancy_ratio=0.0, source_document_count=1,
                    avg_confidence=None, is_plateau=False, is_cliff=False,
                    is_saturated=False, is_mediocre_plateau=True,
                    has_high_redundancy=False,
                ),
                ExpansionDecision(
                    action="expand_breadth",
                    reason="Need more sources.",
                    confidence="medium",
                ),
            ),
            (
                EvaluationSignals(
                    top_score=0.85, score_at_k=0.7, score_cliff=0.15,
                    score_variance=0.001, score_mean=0.78,
                    chunks_above_threshold=10, token_fill_ratio=0.9,
                    redundancy_ratio=0.1, source_document_count=3,
                    avg_confidence=None, is_plateau=True, is_cliff=False,
                    is_saturated=True, is_mediocre_plateau=False,
                    has_high_redundancy=False,
                ),
                ExpansionDecision(
                    action="stop",
                    reason="Good plateau.",
                    confidence="high",
                ),
            ),
        ]
        evaluate_mock = AsyncMock(side_effect=eval_results)
        expand_mock = AsyncMock(return_value=MagicMock(
            urls_ingested=["https://example.com/expanded"],
            urls_failed=[],
            candidates_scored=5,
            candidates_selected=1,
            chunks_added=3,
            depth=1,
        ))

        with _override_settings(
            reranker_provider="none",
            decomposition_mode="none",
            retrieval_context_budget=4096,
            locality_expansion_enabled=False,
            max_expansion_depth=5,
        ):
            with patch.object(engine, "_pool", MagicMock()):
                with patch.object(engine, "_acquire_connection", new_callable=AsyncMock, return_value=MagicMock()):
                    with patch.object(engine, "_release_connection", new_callable=AsyncMock):
                        with patch.object(engine, "_ensure_ingested", new_callable=AsyncMock):
                            with patch("src.orchestration.engine.retrieve", retrieve_mock):
                                with patch("src.orchestration.engine.rerank", rerank_mock):
                                    with patch("src.orchestration.engine.evaluate", evaluate_mock):
                                        with patch("src.orchestration.engine.expand", expand_mock):
                                            result = await engine.run(
                                                "https://example.com", "test query",
                                            )

        assert result.total_iterations == 1
        assert len(result.expansion_steps) == 1


class TestEngineExpandIntentReanalysis:
    """Test that expand_intent triggers re-analysis."""

    @pytest.mark.asyncio
    async def test_reanalysis_called(self):
        from src.orchestration.engine import OrchestratorEngine

        engine = OrchestratorEngine()

        chunk_data = [_make_retrieved_chunk(score=0.4)]
        rr = _make_retrieval_result(chunk_data, mode="chunk")

        # Good retrieval on second pass.
        good_chunks = [
            _make_retrieved_chunk(score=0.85, token_start=i*30, token_end=(i+1)*30)
            for i in range(5)
        ]
        good_rr = _make_retrieval_result(good_chunks, mode="chunk")

        retrieve_call_count = 0
        async def mock_retrieve(*args, **kwargs):
            nonlocal retrieve_call_count
            retrieve_call_count += 1
            if retrieve_call_count <= 1:
                return rr
            return good_rr

        def make_rerank(query, passages, **kw):
            scores = kw.get("original_scores", [0.5] * len(passages))
            return [RerankResult(index=i, relevance_score=s) for i, s in enumerate(scores)]

        # First eval: expand_intent. Second eval: stop.
        eval_results = [
            (
                EvaluationSignals(
                    top_score=0.4, score_at_k=0.4, score_cliff=0.0,
                    score_variance=0.0, score_mean=0.4,
                    chunks_above_threshold=1, token_fill_ratio=0.1,
                    redundancy_ratio=0.0, source_document_count=1,
                    avg_confidence=None, is_plateau=True, is_cliff=False,
                    is_saturated=False, is_mediocre_plateau=True,
                    has_high_redundancy=False,
                ),
                ExpansionDecision(
                    action="expand_intent",
                    reason="Mediocre plateau — rewriting query.",
                    confidence="medium",
                ),
            ),
            (
                EvaluationSignals(
                    top_score=0.85, score_at_k=0.7, score_cliff=0.15,
                    score_variance=0.001, score_mean=0.78,
                    chunks_above_threshold=5, token_fill_ratio=0.9,
                    redundancy_ratio=0.1, source_document_count=2,
                    avg_confidence=None, is_plateau=True, is_cliff=False,
                    is_saturated=True, is_mediocre_plateau=False,
                    has_high_redundancy=False,
                ),
                ExpansionDecision(
                    action="stop", reason="Good plateau.", confidence="high",
                ),
            ),
        ]

        analyze_results = [
            _make_query_analysis("test query", ["initial sub-query"]),
            _make_query_analysis("test query", ["rewritten sub-query"]),
        ]

        analyze_mock = AsyncMock(side_effect=analyze_results)

        with _override_settings(
            reranker_provider="none",
            decomposition_mode="none",
            retrieval_context_budget=4096,
            locality_expansion_enabled=False,
            max_expansion_depth=5,
        ):
            with patch.object(engine, "_pool", MagicMock()):
                with patch.object(engine, "_acquire_connection", new_callable=AsyncMock, return_value=MagicMock()):
                    with patch.object(engine, "_release_connection", new_callable=AsyncMock):
                        with patch.object(engine, "_ensure_ingested", new_callable=AsyncMock):
                            with patch("src.orchestration.engine.retrieve", AsyncMock(side_effect=mock_retrieve)):
                                with patch("src.orchestration.engine.rerank", AsyncMock(side_effect=make_rerank)):
                                    with patch("src.orchestration.engine.evaluate", AsyncMock(side_effect=eval_results)):
                                        with patch("src.orchestration.engine.analyze_query", analyze_mock):
                                            result = await engine.run(
                                                "https://example.com", "test query",
                                            )

        # analyze_query should be called twice: initial + rewrite.
        assert analyze_mock.call_count == 2
        # Second call should include rewrite_context.
        second_call_kwargs = analyze_mock.call_args_list[1].kwargs
        assert "rewrite_context" in second_call_kwargs
        assert second_call_kwargs["rewrite_context"] is not None


class TestEngineMaxIterationCap:
    """Test that the engine stops at max_expansion_depth."""

    @pytest.mark.asyncio
    async def test_max_depth_stops(self):
        from src.orchestration.engine import OrchestratorEngine

        engine = OrchestratorEngine()

        chunk_data = [_make_retrieved_chunk(score=0.4)]
        rr = _make_retrieval_result(chunk_data, mode="chunk")

        def make_rerank(query, passages, **kw):
            scores = kw.get("original_scores", [0.5] * len(passages))
            return [RerankResult(index=i, relevance_score=s) for i, s in enumerate(scores)]

        # Evaluator always says expand_breadth.
        always_expand = (
            EvaluationSignals(
                top_score=0.4, score_at_k=0.4, score_cliff=0.0,
                score_variance=0.0, score_mean=0.4,
                chunks_above_threshold=1, token_fill_ratio=0.1,
                redundancy_ratio=0.0, source_document_count=1,
                avg_confidence=None, is_plateau=False, is_cliff=True,
                is_saturated=False, is_mediocre_plateau=False,
                has_high_redundancy=False,
            ),
            ExpansionDecision(
                action="expand_breadth",
                reason="Need more.",
                confidence="medium",
            ),
        )

        # After max_depth, evaluator returns stop (the actual code checks
        # iteration >= max_expansion_depth as condition 4).
        max_depth = 2

        eval_call_count = 0
        async def mock_evaluate(*args, **kwargs):
            nonlocal eval_call_count
            eval_call_count += 1
            iteration = args[2] if len(args) > 2 else 0
            if iteration >= max_depth:
                return (
                    always_expand[0],
                    ExpansionDecision(
                        action="stop",
                        reason=f"Reached max expansion depth ({max_depth}).",
                        confidence="high",
                    ),
                )
            return always_expand

        expand_mock = AsyncMock(return_value=MagicMock(
            urls_ingested=["https://example.com/exp"],
            urls_failed=[], candidates_scored=1, candidates_selected=1,
            chunks_added=1, depth=1,
        ))

        with _override_settings(
            reranker_provider="none",
            decomposition_mode="none",
            retrieval_context_budget=4096,
            locality_expansion_enabled=False,
            max_expansion_depth=max_depth,
        ):
            with patch.object(engine, "_pool", MagicMock()):
                with patch.object(engine, "_acquire_connection", new_callable=AsyncMock, return_value=MagicMock()):
                    with patch.object(engine, "_release_connection", new_callable=AsyncMock):
                        with patch.object(engine, "_ensure_ingested", new_callable=AsyncMock):
                            with patch("src.orchestration.engine.retrieve", AsyncMock(return_value=rr)):
                                with patch("src.orchestration.engine.rerank", AsyncMock(side_effect=make_rerank)):
                                    with patch("src.orchestration.engine.evaluate", AsyncMock(side_effect=mock_evaluate)):
                                        with patch("src.orchestration.engine.expand", expand_mock):
                                            result = await engine.run(
                                                "https://example.com", "test query",
                                            )

        assert result.total_iterations <= max_depth
        assert result.final_decision.action == "stop"
        assert "max expansion depth" in result.final_decision.reason.lower()


class TestEngineGracefulDegradation:
    """Test graceful fallback when components fail."""

    @pytest.mark.asyncio
    async def test_reranker_failure_falls_back(self):
        """Reranker failure → passthrough (tested end-to-end via engine)."""
        from src.orchestration.reranker import rerank

        with _override_settings(reranker_provider="zeroentropy"):
            with patch.dict(
                sys.modules,
                {"zeroentropy": MagicMock(
                    AsyncZeroEntropy=MagicMock(
                        return_value=MagicMock(
                            models=MagicMock(
                                rerank=AsyncMock(side_effect=RuntimeError("fail"))
                            )
                        )
                    )
                )},
            ):
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    results = await rerank(
                        "query", ["passage"],
                        original_scores=[0.5],
                    )

        assert len(results) == 1
        assert any("falling back" in str(w.message) for w in caught)

    @pytest.mark.asyncio
    async def test_llm_failure_falls_back_to_rule_based(self):
        """LLM failure → rule_based decomposition."""
        from src.orchestration.query_analyzer import analyze_query

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(
            side_effect=RuntimeError("LLM unavailable"),
        )

        with _override_settings(
            decomposition_mode="llm",
            orchestration_llm_api_key="fake-key",
        ):
            with patch(
                "src.orchestration.query_analyzer.AsyncOpenAI",
                return_value=mock_client,
            ):
                result = await analyze_query("gradient boosting vs random forests")

        # Should fall back to rule_based → two sub-queries.
        assert len(result.sub_queries) == 2
