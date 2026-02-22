"""Result merging and MMR deduplication.

Merges results from multiple sub-query retrievals, deduplicates with
MMR (maximal marginal relevance), and applies a token budget.

MMR uses token-overlap Jaccard as a text-similarity proxy.  This is
cheaper than embedding cosine similarity and works well enough for
near-duplicate detection.  The redundancy_ceiling threshold controls
how similar two chunks can be before one is dropped.
"""

from __future__ import annotations

from config import settings
from src.orchestration.models import RankedChunk, SubQueryResult

# ── Public interface ──────────────────────────────────────────────


async def merge_subquery_results(
    subquery_results: list[SubQueryResult],
    context_budget: int,
) -> list[RankedChunk]:
    """Merge results from multiple sub-query retrievals.

    Steps:
      1. Union all chunks across sub-query results.
      2. For chunks appearing in multiple results, keep max reranked score.
      3. Apply MMR deduplication (``redundancy_ceiling`` from config).
      4. Sort by score descending.
      5. Apply token budget.
      6. Return merged, deduplicated, budget-fit result set.
    """
    if not subquery_results:
        return []

    # 1–2. Union by chunk_id, keeping the highest score.
    merged = _union_by_chunk_id(subquery_results)

    # 3. MMR deduplication.
    deduplicated = _mmr_dedup(merged)

    # 4. Sort by score descending.
    deduplicated.sort(key=lambda r: r.reranked_score, reverse=True)

    # 5. Apply token budget.
    return _apply_budget(deduplicated, context_budget)


async def merge_ranked_chunks(
    chunks: list[RankedChunk],
    context_budget: int,
) -> list[RankedChunk]:
    """Merge and deduplicate a flat list of RankedChunks.

    Convenience wrapper used when merging after locality expansion
    and in the final output assembly step.
    """
    if not chunks:
        return []

    # Deduplicate by chunk_id, keeping max score.
    by_id: dict[str, RankedChunk] = {}
    for c in chunks:
        key = str(c.chunk.chunk_id)
        existing = by_id.get(key)
        if existing is None or c.reranked_score > existing.reranked_score:
            by_id[key] = c

    merged = list(by_id.values())
    deduplicated = _mmr_dedup(merged)
    deduplicated.sort(key=lambda r: r.reranked_score, reverse=True)
    return _apply_budget(deduplicated, context_budget)


# ── Union ─────────────────────────────────────────────────────────


def _union_by_chunk_id(
    subquery_results: list[SubQueryResult],
) -> list[RankedChunk]:
    """Union chunks by ``chunk_id``, keeping the highest reranked score."""
    by_id: dict[str, RankedChunk] = {}

    for sqr in subquery_results:
        for rc in sqr.reranked_chunks:
            key = str(rc.chunk.chunk_id)
            existing = by_id.get(key)
            if existing is None or rc.reranked_score > existing.reranked_score:
                by_id[key] = rc

    return list(by_id.values())


# ── MMR deduplication ─────────────────────────────────────────────


def _mmr_dedup(chunks: list[RankedChunk]) -> list[RankedChunk]:
    """Greedy MMR-style deduplication based on text overlap.

    Uses token-overlap Jaccard similarity as a proxy for semantic
    similarity.  When full embeddings are available, cosine similarity
    would be preferable, but text overlap is functional and free.

    Chunks with ``similarity > redundancy_ceiling`` to an
    already-selected chunk are dropped.
    """
    ceiling = settings.redundancy_ceiling

    if not chunks or ceiling >= 1.0:
        return list(chunks)

    # Pre-tokenize.
    tokenized: list[set[str]] = [
        _tokenize(c.chunk.selected_text) for c in chunks
    ]

    selected: list[int] = []
    for i in range(len(chunks)):
        is_redundant = False
        for j in selected:
            sim = _jaccard(tokenized[i], tokenized[j])
            if sim > ceiling:
                is_redundant = True
                break
        if not is_redundant:
            selected.append(i)

    return [chunks[i] for i in selected]


def _tokenize(text: str) -> set[str]:
    """Lowercase word tokenization."""
    return set(text.lower().split())


def _jaccard(a: set[str], b: set[str]) -> float:
    """Jaccard similarity between two token sets."""
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# ── Budget enforcement ────────────────────────────────────────────


def _apply_budget(
    chunks: list[RankedChunk],
    context_budget: int,
) -> list[RankedChunk]:
    """Keep chunks until the token budget is exhausted.

    Assumes *chunks* are already sorted by descending score.
    """
    if context_budget <= 0:
        return list(chunks)

    result: list[RankedChunk] = []
    tokens_used = 0

    for c in chunks:
        chunk_tokens = _chunk_token_count(c)
        if tokens_used + chunk_tokens > context_budget and result:
            # Budget exceeded; stop adding.  The at-least-one guarantee
            # ensures the caller always gets something, even if a
            # single chunk exceeds the entire budget.
            break
        result.append(c)
        tokens_used += chunk_tokens

    return result


def _chunk_token_count(c: RankedChunk) -> int:
    """Estimate token count for a ranked chunk."""
    if (
        c.chunk.token_end is not None
        and c.chunk.token_start is not None
    ):
        return c.chunk.token_end - c.chunk.token_start
    # Fallback: word-count approximation.
    return len((c.chunk.selected_text or "").split())
