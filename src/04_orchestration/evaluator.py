"""Retrieval quality evaluation and stop/expand decision engine.

After each reranking, ``evaluate()`` computes quality signals from the
score distribution and determines whether to stop or which expansion
strategy to pursue.

The decision matrix in ``_decide()`` is ordered by priority:
STOP conditions are checked first (cheap early exit), then EXPAND
conditions from most to least specific.  The numbered comments
(1–11) correspond to the conditions in the design doc §9.
"""

from __future__ import annotations

import statistics

from config import settings
from src.orchestration.models import (
    EvaluationSignals,
    ExpansionDecision,
    QueryAnalysis,
    RankedChunk,
)

# ── Public interface ──────────────────────────────────────────────


async def evaluate(
    reranked_chunks: list[RankedChunk],
    context_budget: int,
    iteration: int,
    previous_signals: EvaluationSignals | None,
    query_analysis: QueryAnalysis,
) -> tuple[EvaluationSignals, ExpansionDecision]:
    """Evaluate retrieval quality and decide next action.

    Returns
    -------
    (signals, decision)
        ``signals`` captures the current score distribution;
        ``decision`` recommends ``stop``, ``expand_breadth``,
        ``expand_recall``, or ``expand_intent``.
    """
    signals = _compute_signals(reranked_chunks, context_budget)
    decision = _decide(signals, iteration, previous_signals, query_analysis)
    return signals, decision


# ── Signal computation ────────────────────────────────────────────

def _scores(chunks: list[RankedChunk]) -> list[float]:
    """Extract the effective score list (reranked or raw retrieval)."""
    return [c.reranked_score for c in chunks]


def _compute_signals(
    chunks: list[RankedChunk],
    context_budget: int,
) -> EvaluationSignals:
    """Compute quality signals from reranked chunks."""
    if not chunks:
        return EvaluationSignals(
            top_score=0.0,
            score_at_k=0.0,
            score_cliff=0.0,
            score_variance=0.0,
            score_mean=0.0,
            chunks_above_threshold=0,
            token_fill_ratio=0.0,
            redundancy_ratio=0.0,
            source_document_count=0,
            avg_confidence=None,
            is_plateau=False,
            is_cliff=False,
            is_saturated=False,
            is_mediocre_plateau=False,
            has_high_redundancy=False,
        )

    scores = _scores(chunks)
    scores_sorted = sorted(scores, reverse=True)

    top_score = scores_sorted[0]
    k = min(settings.score_cliff_rank_k, len(scores_sorted)) - 1
    score_at_k = scores_sorted[k] if k >= 0 else 0.0
    score_cliff = top_score - score_at_k

    # Plateau metrics over top N.
    n = min(settings.plateau_top_n, len(scores_sorted))
    top_n_scores = scores_sorted[:n]
    score_variance = statistics.variance(top_n_scores) if n >= 2 else 0.0
    score_mean = statistics.mean(top_n_scores)

    # Chunks within (top_score - delta) band.
    delta = settings.diminishing_return_delta
    threshold = top_score - delta if top_score > delta else 0.0
    chunks_above_threshold = sum(1 for s in scores if s >= threshold)

    # Token fill ratio — count tokens in all chunks vs budget.
    total_tokens = sum(
        (c.chunk.token_end - c.chunk.token_start)
        if c.chunk.token_end is not None and c.chunk.token_start is not None
        else len((c.chunk.selected_text or "").split())
        for c in chunks
    )
    token_fill_ratio = total_tokens / max(context_budget, 1)

    # Lightweight redundancy proxy: score clustering suggests the
    # retriever is returning near-duplicate content.  Full text-based
    # dedup happens later in merger.py via MMR.
    redundancy_ratio = _estimate_redundancy_ratio(chunks)

    # Distinct source URLs.
    source_document_count = len({c.chunk.source_url for c in chunks})

    # Average confidence (ZeroEntropy-specific, may be None).
    confidences = [c.confidence for c in chunks if c.confidence is not None]
    avg_confidence = (
        statistics.mean(confidences) if confidences else None
    )

    # Derived booleans — these collapse multi-signal checks into
    # single flags that the decision matrix can branch on cleanly.
    mediocre_floor = _effective_mediocre_floor()

    is_plateau = score_variance < settings.plateau_variance_threshold
    is_cliff = score_cliff > settings.score_cliff_threshold
    is_saturated = token_fill_ratio > settings.token_budget_saturation_ratio
    is_mediocre_plateau = is_plateau and score_mean < mediocre_floor
    has_high_redundancy = redundancy_ratio > settings.redundancy_ceiling

    return EvaluationSignals(
        top_score=top_score,
        score_at_k=score_at_k,
        score_cliff=score_cliff,
        score_variance=score_variance,
        score_mean=score_mean,
        chunks_above_threshold=chunks_above_threshold,
        token_fill_ratio=token_fill_ratio,
        redundancy_ratio=redundancy_ratio,
        source_document_count=source_document_count,
        avg_confidence=avg_confidence,
        is_plateau=is_plateau,
        is_cliff=is_cliff,
        is_saturated=is_saturated,
        is_mediocre_plateau=is_mediocre_plateau,
        has_high_redundancy=has_high_redundancy,
    )


def _effective_mediocre_floor() -> float:
    """Return the mediocre score floor adjusted for calibration.

    Raw embedding similarity (reranker_provider='none') is not
    calibrated, so the floor is lowered from the configured value.
    """
    if settings.reranker_provider.lower() == "none":
        return min(settings.mediocre_score_floor, 0.35)
    return settings.mediocre_score_floor


def _estimate_redundancy_ratio(chunks: list[RankedChunk]) -> float:
    """Heuristic redundancy: fraction of chunks with near-duplicate scores.

    A proper MMR-based redundancy check uses embeddings, but this
    lightweight version detects score clusters as a proxy.  The full
    MMR dedup is in ``merger.py``.
    """
    if len(chunks) < 2:
        return 0.0

    scores = sorted([c.reranked_score for c in chunks], reverse=True)
    # Count consecutive pairs within a tiny epsilon.
    eps = 0.005
    duplicates = sum(
        1 for i in range(len(scores) - 1)
        if abs(scores[i] - scores[i + 1]) < eps
    )
    return duplicates / len(scores)


# ── Decision logic ────────────────────────────────────────────────


def _decide(
    signals: EvaluationSignals,
    iteration: int,
    previous_signals: EvaluationSignals | None,
    query_analysis: QueryAnalysis,  # noqa: ARG001  (reserved for future use)
) -> ExpansionDecision:
    """Apply the decision matrix (see design doc §9).

    The numbered conditions below map to the design-doc matrix.
    """

    # ── STOP CONDITIONS ───────────────────────────────────────

    # 4. Max depth reached (safety cap).
    if iteration >= settings.max_expansion_depth:
        return ExpansionDecision(
            action="stop",
            reason=f"Reached max expansion depth ({settings.max_expansion_depth}).",
            confidence="high",
        )

    # 1. Saturated + not mediocre + low redundancy.
    if (
        signals.is_saturated
        and not signals.is_mediocre_plateau
        and not signals.has_high_redundancy
    ):
        return ExpansionDecision(
            action="stop",
            reason=(
                f"Token budget sufficiently filled "
                f"(fill={signals.token_fill_ratio:.2f}), "
                f"redundancy={signals.redundancy_ratio:.2f}."
            ),
            confidence="high",
        )

    # 2. Diminishing returns (iteration > 0).
    if iteration > 0 and previous_signals is not None:
        recall_delta = (
            signals.chunks_above_threshold
            - previous_signals.chunks_above_threshold
        )
        if recall_delta < settings.diminishing_return_delta:
            return ExpansionDecision(
                action="stop",
                reason=(
                    f"Diminishing returns: recall proxy improved by only "
                    f"{recall_delta} (threshold={settings.diminishing_return_delta})."
                ),
                confidence="medium",
            )

    # 3. Good plateau.
    if signals.is_plateau and signals.score_mean >= _effective_mediocre_floor():
        return ExpansionDecision(
            action="stop",
            reason=(
                f"Good plateau: variance={signals.score_variance:.4f}, "
                f"mean={signals.score_mean:.3f}."
            ),
            confidence="high",
        )

    # ── EXPAND CONDITIONS ─────────────────────────────────────

    # 5. Cliff + high redundancy → EXPAND_INTENT.
    if signals.is_cliff and signals.has_high_redundancy:
        return ExpansionDecision(
            action="expand_intent",
            reason=(
                "Score cliff with high redundancy — "
                "stuck in one pocket, rewriting query."
            ),
            confidence="medium",
        )

    # 6. Cliff + low redundancy → EXPAND_BREADTH.
    if signals.is_cliff and not signals.has_high_redundancy:
        return ExpansionDecision(
            action="expand_breadth",
            reason=(
                "Score cliff with low redundancy — "
                "found one good source, need more sources."
            ),
            confidence="medium",
        )

    # 7. Mediocre plateau.
    #    With a tiny corpus (≤1 source doc), rephrasing against the
    #    same content won't help — grow the corpus first.
    if signals.is_mediocre_plateau:
        if signals.source_document_count <= 1:
            return ExpansionDecision(
                action="expand_breadth",
                reason=(
                    f"Mediocre plateau (mean={signals.score_mean:.3f} "
                    f"< floor={_effective_mediocre_floor():.2f}) with only "
                    f"{signals.source_document_count} source doc(s) — "
                    f"expanding corpus before rephrasing."
                ),
                confidence="medium",
            )
        return ExpansionDecision(
            action="expand_intent",
            reason=(
                f"Mediocre plateau: mean={signals.score_mean:.3f} "
                f"< floor={_effective_mediocre_floor():.2f}."
            ),
            confidence="medium",
        )

    # 8. Low scores everywhere.
    #    If the corpus is tiny (single source document), rephrasing won't
    #    help — expand_breadth to grow the corpus first.  Otherwise
    #    expand_intent to try a different query decomposition.
    if signals.top_score < _effective_mediocre_floor():
        if signals.source_document_count <= 1:
            return ExpansionDecision(
                action="expand_breadth",
                reason=(
                    f"Top score ({signals.top_score:.3f}) below mediocre floor "
                    f"({_effective_mediocre_floor():.2f}) with only "
                    f"{signals.source_document_count} source doc(s) — "
                    f"expanding corpus before rephrasing."
                ),
                confidence="low",
            )
        return ExpansionDecision(
            action="expand_intent",
            reason=(
                f"Top score ({signals.top_score:.3f}) below mediocre floor "
                f"({_effective_mediocre_floor():.2f}) — query may be misaligned."
            ),
            confidence="low",
        )

    # 9. Good scores + low token fill → EXPAND_BREADTH.
    if (
        signals.top_score >= _effective_mediocre_floor()
        and not signals.is_saturated
    ):
        return ExpansionDecision(
            action="expand_breadth",
            reason=(
                f"Quality hits exist (top={signals.top_score:.3f}) "
                f"but low fill ({signals.token_fill_ratio:.2f})."
            ),
            confidence="medium",
        )

    # 10. Low confidence (ZeroEntropy-specific) → EXPAND_BREADTH.
    if (
        signals.avg_confidence is not None
        and signals.avg_confidence < settings.confidence_floor
    ):
        return ExpansionDecision(
            action="expand_breadth",
            reason=(
                f"Low reranker confidence ({signals.avg_confidence:.3f} "
                f"< {settings.confidence_floor})."
            ),
            confidence="low",
        )

    # 11. Default fallback → STOP.
    return ExpansionDecision(
        action="stop",
        reason="No expansion condition triggered (default stop).",
        confidence="medium",
    )
