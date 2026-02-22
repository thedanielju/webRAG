"""Orchestration-specific Pydantic models.

All data contracts shared across orchestration modules live here.
Models from other layers (RetrievedChunk, RetrievalResult, CorpusStats,
TimingInfo, CitationSpan, PersistedLinkCandidate) are imported — never
redefined.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import BaseModel

from src.retrieval.citations import CitationSpan
from src.retrieval.models import (
    CorpusStats,
    PersistedLinkCandidate,
    RetrievalResult,
    RetrievedChunk,
)


# ── Query Analysis ────────────────────────────────────────────────


class QueryAnalysis(BaseModel):
    """Output of query analysis."""

    original_query: str
    sub_queries: list[str]
    query_type: str  # "factual" | "comparison" | "how_to" | "exploratory"
    complexity: str  # "simple" | "moderate" | "complex"
    key_concepts: list[str]


# ── Reranking ─────────────────────────────────────────────────────


class RerankResult(BaseModel):
    """Single reranked result from any provider."""

    index: int  # Original position in the input list.
    relevance_score: float  # 0.0–1.0, calibrated for zerank-2.
    confidence: float | None = None  # ZeroEntropy-specific.


class RankedChunk(BaseModel):
    """A RetrievedChunk wrapped with reranker output.

    The inner ``chunk`` retains its original retrieval scores
    (``chunk.score`` and ``chunk.raw_similarity``).  The
    ``reranked_score`` is the cross-encoder relevance score produced
    by the reranker (or the retrieval score when provider="none").
    """

    chunk: RetrievedChunk
    reranked_score: float
    confidence: float | None = None
    is_locality_expanded: bool = False
    source_sub_query: str | None = None


# ── Evaluation ────────────────────────────────────────────────────


class EvaluationSignals(BaseModel):
    """Computed signals from a reranked retrieval result."""

    top_score: float
    score_at_k: float
    score_cliff: float
    score_variance: float
    score_mean: float
    chunks_above_threshold: int
    token_fill_ratio: float
    redundancy_ratio: float
    source_document_count: int
    avg_confidence: float | None = None
    is_plateau: bool
    is_cliff: bool
    is_saturated: bool
    is_mediocre_plateau: bool
    has_high_redundancy: bool


class ExpansionDecision(BaseModel):
    """The evaluator's recommendation for the next loop action."""

    action: str
    # "stop" | "expand_breadth" | "expand_recall" | "expand_intent"
    reason: str
    confidence: str  # "high" | "medium" | "low"


# ── Expansion ─────────────────────────────────────────────────────


class ScoredCandidate(BaseModel):
    """A link candidate with computed expansion-relevance score."""

    link_candidate: PersistedLinkCandidate
    score: float
    score_breakdown: dict[str, float]


class ExpansionOutcome(BaseModel):
    """Result of a single expansion iteration."""

    urls_attempted: list[str]
    urls_ingested: list[str]
    urls_failed: list[str]
    chunks_added: int
    candidates_scored: int
    candidates_selected: int
    depth: int


class ExpansionStep(BaseModel):
    """Record of a single expansion iteration for observability."""

    iteration: int
    depth: int
    source_url: str
    candidates_scored: int
    candidates_expanded: list[str]
    candidates_failed: list[str]
    chunks_added: int
    top_score_before: float
    top_score_after: float
    decision: str
    reason: str
    duration_ms: float = 0.0  # Wall-clock time for this iteration.


# ── Sub-query results ─────────────────────────────────────────────


class SubQueryResult(BaseModel):
    """Result from a single sub-query's retrieve → rerank cycle."""

    sub_query: str
    retrieval_result: RetrievalResult
    reranked_chunks: list[RankedChunk]


# ── Timing ────────────────────────────────────────────────────────


class OrchestrationTiming(BaseModel):
    """End-to-end timing breakdown."""

    query_analysis_ms: float = 0.0
    retrieval_ms: float = 0.0
    reranking_ms: float = 0.0
    expansion_ms: float = 0.0
    locality_ms: float = 0.0
    merge_ms: float = 0.0
    total_ms: float = 0.0


# ── Top-level output ──────────────────────────────────────────────


class OrchestrationResult(BaseModel):
    """Top-level output contract from orchestration to MCP layer."""

    chunks: list[RankedChunk]
    citations: list[CitationSpan]
    query_analysis: QueryAnalysis
    expansion_steps: list[ExpansionStep]
    corpus_stats: CorpusStats
    timing: OrchestrationTiming
    mode: str  # "full_context" | "chunk"
    final_decision: ExpansionDecision
    total_iterations: int
    total_urls_ingested: int


# ── Mutable orchestration state (dataclass, not Pydantic) ─────────


@dataclass
class OrchestrationState:
    """Mutable state tracked across the orchestration loop.

    This is a plain dataclass (not Pydantic) because it's mutated
    in-place across iterations — sets grow, lists append, signals
    get replaced.  Pydantic models are frozen by convention here.
    """

    # Query context (immutable after init).
    original_query: str
    seed_url: str
    intent: str | None = None
    known_context: str | None = None
    constraints: list[str] | None = None
    expansion_budget: int | None = None

    # Set after query analysis.
    query_analysis: QueryAnalysis | None = None

    # Iteration tracking.
    iteration: int = 0
    current_depth: int = 0

    # Corpus state (grows across iterations).
    ingested_urls: set[str] = field(default_factory=set)

    # Result state (updated each iteration).
    current_chunks: list[RankedChunk] = field(default_factory=list)
    current_signals: EvaluationSignals | None = None

    # History.
    expansion_steps: list[ExpansionStep] = field(default_factory=list)
    all_retrieval_results: list[RetrievalResult] = field(default_factory=list)
