"""Adjacent chunk expansion (locality).

Fetches sibling parent chunks adjacent to high-scoring hits.
This is a cheap DB query — no embedding or API calls.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from psycopg import AsyncConnection

from config import settings
from src.orchestration.models import RankedChunk
from src.retrieval.models import RetrievedChunk

# ── Public interface ──────────────────────────────────────────────


async def expand_locality(
    high_score_chunks: list[RankedChunk],
    conn: AsyncConnection,
    *,
    radius: int | None = None,
    min_score_for_expansion: float | None = None,
) -> list[RankedChunk]:
    """Fetch sibling parent chunks adjacent to high-scoring hits.

    For each chunk with ``reranked_score >= min_score_for_expansion``:

    1. Query the ``chunks`` table for the same ``document_id``,
       ``chunk_level = 'parent'``, and ``chunk_index`` within
       ``± radius``.
    2. Exclude chunks already present in the result set.
    3. Return new chunks wrapped as ``RankedChunk`` with
       ``is_locality_expanded = True``.

    Parameters
    ----------
    high_score_chunks:
        The current set of reranked chunks.
    conn:
        An ``AsyncConnection`` with pgvector already registered.
    radius:
        Number of sibling chunks in each direction.
        Defaults to ``settings.locality_expansion_radius``.
    min_score_for_expansion:
        Only expand around chunks scoring at or above this.
    """
    if not settings.locality_expansion_enabled:
        return []

    effective_radius = radius if radius is not None else settings.locality_expansion_radius
    if effective_radius <= 0:
        return []

    effective_min_score = (
        min_score_for_expansion
        if min_score_for_expansion is not None
        else settings.min_score_for_expansion
    )

    # Identify expansion targets.
    existing_chunk_ids: set[UUID] = {
        c.chunk.chunk_id for c in high_score_chunks
    }
    targets = [
        c for c in high_score_chunks
        if c.reranked_score >= effective_min_score
    ]
    if not targets:
        return []

    # Build a list of (document_id, chunk_index_low, chunk_index_high)
    # ranges to query, then batch into a single SQL call.
    range_predicates: list[tuple[UUID, int, int]] = []
    for t in targets:
        low = max(0, t.chunk.chunk_index - effective_radius)
        high = t.chunk.chunk_index + effective_radius
        range_predicates.append((t.chunk.document_id, low, high))

    if not range_predicates:
        return []

    # Execute a single query covering all ranges via UNION-style OR.
    # One round-trip instead of N queries per target — important for
    # keeping locality expansion cheap.
    where_clauses: list[str] = []
    params: list[object] = []
    for doc_id, low, high in range_predicates:
        where_clauses.append(
            "(document_id = %s AND chunk_index BETWEEN %s AND %s)"
        )
        params.extend([str(doc_id), low, high])

    combined_where = " OR ".join(where_clauses)

    # Exclude already-present chunk IDs.
    exclude_placeholders = ", ".join(["%s"] * len(existing_chunk_ids))
    exclude_params = [str(cid) for cid in existing_chunk_ids]

    sql = f"""
        SELECT
            id, document_id, parent_id, chunk_level, chunk_index,
            section_heading, chunk_text, html_text,
            has_table, has_code, has_math,
            has_definition_list, has_admonition, has_steps,
            char_start, char_end, token_start, token_end,
            source_url, fetched_at, depth, title
        FROM chunks
        WHERE chunk_level = 'parent'
          AND ({combined_where})
          AND id NOT IN ({exclude_placeholders})
        ORDER BY document_id, chunk_index
    """  # noqa: S608

    all_params = params + exclude_params

    async with conn.cursor() as cur:
        await cur.execute(sql, all_params)
        rows = await cur.fetchall()

    # Map rows to RankedChunk with locality flag.
    new_chunks: list[RankedChunk] = []
    for row in rows:
        chunk_id = UUID(str(row[0])) if not isinstance(row[0], UUID) else row[0]
        if chunk_id in existing_chunk_ids:
            continue  # Safety check.

        doc_id = UUID(str(row[1])) if not isinstance(row[1], UUID) else row[1]
        parent_id_raw = row[2]
        parent_id = (
            UUID(str(parent_id_raw))
            if parent_id_raw is not None and not isinstance(parent_id_raw, UUID)
            else parent_id_raw
        )

        # Use html_text if available, else chunk_text.
        # html_text preserves structure (tables, code blocks) that
        # markdown flattening would lose.
        html_text = row[7]
        chunk_text = row[6]
        selected_text = html_text if html_text else chunk_text
        surface = "html" if html_text else "markdown"

        fetched_at = row[19]
        if isinstance(fetched_at, str):
            fetched_at = datetime.fromisoformat(fetched_at)

        retrieved = RetrievedChunk(
            chunk_id=chunk_id,
            document_id=doc_id,
            parent_id=parent_id,
            source_url=row[18],
            title=row[21],
            selected_text=selected_text,
            surface=surface,
            section_heading=row[5],
            chunk_index=row[4],
            char_start=row[14],
            char_end=row[15],
            token_start=row[16],
            token_end=row[17],
            score=0.0,  # No retrieval score for locality chunks.
            raw_similarity=None,
            depth=row[20],
            has_table=row[8],
            has_code=row[9],
            has_math=row[10],
            has_definition_list=row[11],
            has_admonition=row[12],
            has_steps=row[13],
            fetched_at=fetched_at,
        )

        new_chunks.append(
            RankedChunk(
                chunk=retrieved,
                reranked_score=0.0,  # Scored later during merge.
                is_locality_expanded=True,
            )
        )
        existing_chunk_ids.add(chunk_id)

    return new_chunks
