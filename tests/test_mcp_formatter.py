"""Unit tests for src.mcp_server.formatter."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import pytest

from src.mcp_server.formatter import (
    _build_citations,
    _build_expansion_trace,
    _build_sources,
    _build_stats,
    _common_url_prefix,
    format_result,
)
from src.orchestration.models import (
    ExpansionDecision,
    ExpansionStep,
    OrchestrationResult,
    OrchestrationTiming,
    QueryAnalysis,
    RankedChunk,
)
from src.retrieval.citations import CitationSpan
from src.retrieval.models import CorpusStats, RetrievedChunk


# ── Fixtures ──────────────────────────────────────────────────


def _make_chunk(
    *,
    source_url: str = "https://example.com/page",
    title: str = "Example Page",
    text: str = "Some content.",
    section: str | None = "Introduction",
    score: float = 0.8,
    reranked_score: float = 0.85,
    confidence: float | None = 0.9,
    surface: str = "markdown",
    is_locality_expanded: bool = False,
    source_sub_query: str | None = None,
    chunk_index: int = 0,
) -> RankedChunk:
    cid = uuid4()
    did = uuid4()
    return RankedChunk(
        chunk=RetrievedChunk(
            chunk_id=cid,
            document_id=did,
            parent_id=None,
            source_url=source_url,
            title=title,
            selected_text=text,
            surface=surface,
            section_heading=section,
            chunk_index=chunk_index,
            char_start=0,
            char_end=len(text),
            token_start=0,
            token_end=len(text) // 4,
            score=score,
            raw_similarity=score,
            depth=0,
            has_table=False,
            has_code=False,
            has_math=False,
            has_definition_list=False,
            has_admonition=False,
            has_steps=False,
            fetched_at=datetime.now(timezone.utc),
        ),
        reranked_score=reranked_score,
        confidence=confidence,
        is_locality_expanded=is_locality_expanded,
        source_sub_query=source_sub_query,
    )


def _make_result(
    chunks: list[RankedChunk] | None = None,
    citations: list[CitationSpan] | None = None,
    expansion_steps: list[ExpansionStep] | None = None,
    mode: str = "chunk",
    total_iterations: int = 0,
    total_urls_ingested: int = 1,
) -> OrchestrationResult:
    if chunks is None:
        chunks = [_make_chunk()]

    if citations is None:
        citations = []

    return OrchestrationResult(
        chunks=chunks,
        citations=citations or [],
        query_analysis=QueryAnalysis(
            original_query="test query",
            sub_queries=["test query"],
            query_type="factual",
            complexity="simple",
            key_concepts=["test"],
        ),
        expansion_steps=expansion_steps or [],
        corpus_stats=CorpusStats(
            total_documents=2,
            total_parent_chunks=20,
            total_tokens=5000,
            documents_matched=["https://example.com/page"],
        ),
        timing=OrchestrationTiming(
            query_analysis_ms=50.0,
            retrieval_ms=200.0,
            reranking_ms=100.0,
            expansion_ms=0.0,
            locality_ms=30.0,
            merge_ms=20.0,
            total_ms=400.0,
        ),
        mode=mode,
        final_decision=ExpansionDecision(
            action="stop",
            reason="score plateau",
            confidence="high",
        ),
        total_iterations=total_iterations,
        total_urls_ingested=total_urls_ingested,
    )


# ── Tests ─────────────────────────────────────────────────────


class TestBuildSources:
    def test_single_source(self):
        chunks = [_make_chunk()]
        source_map, section = _build_sources(chunks)
        assert source_map["https://example.com/page"] == 1
        assert "[1]" in section
        assert "Example Page" in section

    def test_multiple_sources(self):
        chunks = [
            _make_chunk(source_url="https://a.com", title="A"),
            _make_chunk(source_url="https://b.com", title="B"),
        ]
        source_map, section = _build_sources(chunks)
        assert len(source_map) == 2
        assert "[1]" in section
        assert "[2]" in section

    def test_section_headings_aggregated(self):
        chunks = [
            _make_chunk(section="Intro"),
            _make_chunk(section="Details"),
        ]
        _, section = _build_sources(chunks)
        assert "§ Intro" in section
        assert "§ Details" in section

    def test_empty_chunks(self):
        source_map, section = _build_sources([])
        assert len(source_map) == 0
        assert "(none)" in section


class TestFormatResult:
    def test_contains_all_sections(self):
        result = _make_result()
        text = format_result(result)
        assert "[SOURCES]" in text
        assert "[EVIDENCE]" in text
        assert "[STATS]" in text

    def test_single_chunk(self):
        result = _make_result(chunks=[_make_chunk()])
        text = format_result(result)
        assert "relevance: 0.85" in text
        assert "Some content." in text

    def test_zero_chunks(self):
        result = _make_result(chunks=[])
        text = format_result(result)
        assert "[SOURCES]" in text
        assert "(none)" in text or "(no evidence" in text

    def test_full_context_mode(self):
        result = _make_result(mode="full_context")
        text = format_result(result)
        assert "full_context" in text

    def test_locality_expanded_marker(self):
        chunk = _make_chunk(is_locality_expanded=True)
        result = _make_result(chunks=[chunk])
        text = format_result(result)
        assert "[adjacent context]" in text

    def test_sub_query_attribution(self):
        chunk = _make_chunk(source_sub_query="how does X work")
        result = _make_result(chunks=[chunk])
        text = format_result(result)
        assert "how does X work" in text

    def test_confidence_shown(self):
        chunk = _make_chunk(confidence=0.92)
        result = _make_result(chunks=[chunk])
        text = format_result(result)
        assert "confidence: 0.92" in text

    def test_no_confidence_when_none(self):
        chunk = _make_chunk(confidence=None)
        result = _make_result(chunks=[chunk])
        text = format_result(result)
        assert "confidence:" not in text

    def test_expansion_trace_absent_when_no_steps(self):
        result = _make_result(expansion_steps=[])
        text = format_result(result)
        assert "[EXPANSION TRACE]" not in text

    def test_citations_section(self):
        cit = CitationSpan(
            verbatim_text="A notable finding.",
            source_url="https://example.com/page",
            section_heading="Results",
            char_start=0,
            char_end=18,
            token_start=0,
            token_end=4,
            title="Example Page",
        )
        result = _make_result(citations=[cit])
        text = format_result(result)
        assert "[CITATIONS]" in text
        assert "A notable finding." in text

    def test_no_expansion_note(self):
        result = _make_result(total_iterations=0, total_urls_ingested=1)
        text = format_result(result)
        assert "No expansion performed" in text


class TestBuildStats:
    def test_stats_contain_timing(self):
        result = _make_result()
        stats = _build_stats(result)
        assert "400ms" in stats
        assert "analysis: 50ms" in stats
        assert "retrieval: 200ms" in stats

    def test_stats_contain_mode(self):
        result = _make_result(mode="chunk")
        stats = _build_stats(result)
        assert "Mode: chunk" in stats

    def test_full_context_annotation(self):
        result = _make_result(mode="full_context")
        stats = _build_stats(result)
        assert "entire corpus fits within token budget" in stats

    def test_stop_reason(self):
        result = _make_result()
        stats = _build_stats(result)
        assert "score plateau" in stats


class TestExpansionTrace:
    def test_single_step(self):
        step = ExpansionStep(
            iteration=1,
            depth=1,
            source_url="https://example.com/a",
            candidates_scored=5,
            candidates_expanded=["https://example.com/b"],
            candidates_failed=[],
            chunks_added=10,
            top_score_before=0.60,
            top_score_after=0.72,
            decision="stop",
            reason="score plateau",
            duration_ms=1200.0,
        )
        trace = _build_expansion_trace([step])
        assert "[EXPANSION TRACE]" in trace
        assert "+10 chunks" in trace
        assert "(seed)" in trace  # Root is labelled as seed.
        assert "depth 1" in trace

    def test_tree_structure_uses_box_drawing(self):
        step = ExpansionStep(
            iteration=1,
            depth=1,
            source_url="https://example.com/a",
            candidates_scored=3,
            candidates_expanded=[
                "https://example.com/b",
                "https://example.com/c",
            ],
            candidates_failed=[],
            chunks_added=5,
            top_score_before=0.5,
            top_score_after=0.7,
            decision="stop",
            reason="done",
            duration_ms=100.0,
        )
        trace = _build_expansion_trace([step])
        assert "├── " in trace or "└── " in trace

    def test_no_per_node_timing(self):
        """Timing and scores should NOT appear per-node (they belong in STATS)."""
        step = ExpansionStep(
            iteration=1,
            depth=1,
            source_url="https://example.com/a",
            candidates_scored=5,
            candidates_expanded=["https://example.com/b"],
            candidates_failed=[],
            chunks_added=10,
            top_score_before=0.60,
            top_score_after=0.72,
            decision="stop",
            reason="score plateau",
            duration_ms=1200.0,
        )
        trace = _build_expansion_trace([step])
        assert "1200ms" not in trace
        assert "score:" not in trace
        assert "0.60" not in trace

    def test_failed_url(self):
        step = ExpansionStep(
            iteration=1,
            depth=1,
            source_url="https://example.com/a",
            candidates_scored=3,
            candidates_expanded=[],
            candidates_failed=["https://example.com/fail"],
            chunks_added=0,
            top_score_before=0.5,
            top_score_after=0.5,
            decision="stop",
            reason="all candidates failed",
            duration_ms=500.0,
        )
        trace = _build_expansion_trace([step])
        assert "failed" in trace

    def test_stop_reason_annotated(self):
        step = ExpansionStep(
            iteration=1,
            depth=1,
            source_url="https://example.com/a",
            candidates_scored=3,
            candidates_expanded=["https://example.com/b"],
            candidates_failed=[],
            chunks_added=5,
            top_score_before=0.5,
            top_score_after=0.7,
            decision="stop",
            reason="score plateau",
            duration_ms=100.0,
        )
        trace = _build_expansion_trace([step])
        assert "stopped: score plateau" in trace

    def test_common_prefix_base_line(self):
        step = ExpansionStep(
            iteration=1,
            depth=1,
            source_url="https://docs.example.com/en/stable/intro",
            candidates_scored=3,
            candidates_expanded=["https://docs.example.com/en/stable/api"],
            candidates_failed=[],
            chunks_added=5,
            top_score_before=0.5,
            top_score_after=0.7,
            decision="stop",
            reason="done",
            duration_ms=100.0,
        )
        trace = _build_expansion_trace([step])
        assert "Base:" in trace

    def test_empty_steps(self):
        trace = _build_expansion_trace([])
        assert trace == ""


class TestCommonUrlPrefix:
    def test_shared_prefix(self):
        urls = [
            "https://example.com/docs/a.html",
            "https://example.com/docs/b.html",
        ]
        assert _common_url_prefix(urls) == "https://example.com/docs/"

    def test_no_common(self):
        urls = [
            "https://a.com/page",
            "https://b.com/page",
        ]
        prefix = _common_url_prefix(urls)
        assert prefix == "https://"

    def test_single_url(self):
        prefix = _common_url_prefix(["https://example.com/docs/page.html"])
        assert prefix == "https://example.com/docs/"

    def test_empty(self):
        assert _common_url_prefix([]) == ""


class TestBuildCitations:
    def test_citation_formatting(self):
        source_map = {"https://example.com/page": 1}
        cits = [
            CitationSpan(
                verbatim_text="Important quote.",
                source_url="https://example.com/page",
                section_heading="Intro",
                char_start=0,
                char_end=15,
                token_start=0,
                token_end=3,
                title="Example",
            )
        ]
        section = _build_citations(cits, source_map)
        assert "[1]" in section
        assert '"Important quote."' in section
        assert "§ Intro" in section

    def test_long_verbatim_truncated(self):
        source_map = {"https://example.com/page": 1}
        long_text = "x" * 500
        cits = [
            CitationSpan(
                verbatim_text=long_text,
                source_url="https://example.com/page",
                section_heading=None,
                char_start=0,
                char_end=500,
                token_start=0,
                token_end=125,
                title="Example",
            )
        ]
        section = _build_citations(cits, source_map)
        assert "..." in section
        # Should be truncated to ~300 chars.
        assert len(section) < 600
