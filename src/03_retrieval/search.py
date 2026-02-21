from __future__ import annotations

"""Retrieval pipeline for WebRAG.

High-level flow:
1. Determine corpus mode (full_context vs chunk) from parent token span.
2. Return early without embedding when corpus is empty (cost guard).
3. Embed query once.
4. Execute either:
   - full_context parent load (ordered, budgeted), or
   - chunk-mode child ANN search + parent aggregation + budgeting.
5. Apply deterministic surface selection (html vs markdown) per returned parent.
6. Return a structured RetrievalResult for orchestration.
"""

from collections import defaultdict
from time import perf_counter
from typing import Any
import warnings

from psycopg import Connection
from psycopg.rows import dict_row

from config import settings
from src.indexing.embedder import embed_query
from src.retrieval.models import CorpusStats, RetrievedChunk, RetrievalResult, TimingInfo


def _normalize_source_urls(source_urls: list[str] | None) -> list[str] | None:
    if not source_urls:
        return None
    seen: set[str] = set()
    normalized: list[str] = []
    for source_url in source_urls:
        if source_url in seen:
            continue
        seen.add(source_url)
        normalized.append(source_url)
    return normalized or None


def _get_effective_context_budget(context_budget_override: int | None) -> int:
    if context_budget_override is None:
        return settings.retrieval_context_budget
    return max(1, int(context_budget_override))


def _get_effective_threshold(context_budget_override: int | None = None) -> int:
    """Clamp full-context threshold to budget if misconfigured."""
    budget = _get_effective_context_budget(context_budget_override)
    threshold = settings.retrieval_full_context_threshold
    if threshold > budget:
        warnings.warn(
            f"RETRIEVAL_FULL_CONTEXT_THRESHOLD ({threshold}) exceeds "
            f"RETRIEVAL_CONTEXT_BUDGET ({budget}). Clamping to budget.",
            stacklevel=2,
        )
        return budget
    return threshold


def _distance_threshold_from_similarity_floor() -> float:
    similarity_floor = max(0.0, min(1.0, settings.retrieval_similarity_floor))
    return 1.0 - similarity_floor


def _distance_to_similarity(distance: float) -> float:
    return max(0.0, min(1.0, 1.0 - distance))


def _depth_multiplier(depth: int) -> float:
    return max(
        1.0 - depth * settings.retrieval_depth_decay_rate,
        settings.retrieval_depth_floor,
    )


def _determine_mode(
    conn: Connection,
    source_urls: list[str] | None = None,
    context_budget_override: int | None = None,
) -> tuple[str, int]:
    """Determine retrieval mode from parent-token corpus size."""
    effective_threshold = _get_effective_threshold(context_budget_override)

    with conn.cursor() as cur:
        if source_urls:
            cur.execute(
                """
                SELECT COALESCE(SUM(token_end - token_start), 0) AS total_tokens
                FROM chunks
                WHERE chunk_level = 'parent'
                  AND source_url = ANY(%s)
                """,
                (source_urls,),
            )
        else:
            cur.execute(
                """
                SELECT COALESCE(SUM(token_end - token_start), 0) AS total_tokens
                FROM chunks
                WHERE chunk_level = 'parent'
                """
            )
        row = cur.fetchone()
    total_tokens = int(row[0] if row else 0)
    if total_tokens <= effective_threshold:
        return "full_context", total_tokens
    return "chunk", total_tokens


def _load_corpus_counts(conn: Connection, source_urls: list[str] | None) -> tuple[int, int]:
    with conn.cursor() as cur:
        if source_urls:
            cur.execute(
                """
                SELECT COUNT(*) AS total_documents
                FROM documents
                WHERE source_url = ANY(%s)
                """,
                (source_urls,),
            )
            total_documents = int(cur.fetchone()[0])
            cur.execute(
                """
                SELECT COUNT(*) AS total_parent_chunks
                FROM chunks
                WHERE chunk_level = 'parent'
                  AND source_url = ANY(%s)
                """,
                (source_urls,),
            )
        else:
            cur.execute("SELECT COUNT(*) AS total_documents FROM documents")
            total_documents = int(cur.fetchone()[0])
            cur.execute(
                """
                SELECT COUNT(*) AS total_parent_chunks
                FROM chunks
                WHERE chunk_level = 'parent'
                """
            )
        total_parent_chunks = int(cur.fetchone()[0])
    return total_documents, total_parent_chunks


def _choose_surface(row: dict[str, Any]) -> tuple[str, str]:
    """Choose the best render surface for a retrieved chunk row.

    HTML is preferred only when rich-content flags indicate it provides higher
    fidelity and html_text is available. Otherwise markdown is returned.
    """
    use_html = (
        bool(row["has_table"])
        or bool(row["has_code"])
        or bool(row["has_math"])
        or bool(row["has_definition_list"])
        or bool(row["has_admonition"])
    )
    if use_html and row["html_text"]:
        return str(row["html_text"]), "html"
    return str(row["chunk_text"]), "markdown"


def _row_to_retrieved_chunk(
    row: dict[str, Any],
    score: float,
    raw_similarity: float | None,
) -> RetrievedChunk:
    selected_text, surface = _choose_surface(row)
    return RetrievedChunk(
        chunk_id=row["id"],
        document_id=row["document_id"],
        parent_id=row["parent_id"],
        source_url=row["source_url"],
        title=row["title"],
        selected_text=selected_text,
        surface=surface,
        section_heading=row["section_heading"],
        chunk_index=int(row["chunk_index"]),
        char_start=int(row["char_start"]),
        char_end=int(row["char_end"]),
        token_start=int(row["token_start"]),
        token_end=int(row["token_end"]),
        score=float(score),
        raw_similarity=None if raw_similarity is None else float(raw_similarity),
        depth=int(row["depth"]),
        has_table=bool(row["has_table"]),
        has_code=bool(row["has_code"]),
        has_math=bool(row["has_math"]),
        has_definition_list=bool(row["has_definition_list"]),
        has_admonition=bool(row["has_admonition"]),
        has_steps=bool(row["has_steps"]),
        fetched_at=row["fetched_at"],
    )


def _group_chunks_by_source_ordered(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    grouped: dict[str, list[RetrievedChunk]] = defaultdict(list)
    source_order: list[str] = []
    seen: set[str] = set()
    for chunk in chunks:
        if chunk.source_url not in seen:
            seen.add(chunk.source_url)
            source_order.append(chunk.source_url)
        grouped[chunk.source_url].append(chunk)
    ordered: list[RetrievedChunk] = []
    for source_url in source_order:
        ordered.extend(sorted(grouped[source_url], key=lambda chunk: chunk.chunk_index))
    return ordered


def _query_full_context_parents(
    conn: Connection,
    source_urls: list[str] | None,
) -> list[dict[str, Any]]:
    with conn.cursor(row_factory=dict_row) as cur:
        if source_urls:
            cur.execute(
                """
                SELECT
                    id, document_id, parent_id, chunk_level,
                    chunk_index, section_heading,
                    chunk_text, html_text,
                    has_table, has_code, has_math,
                    has_definition_list, has_admonition, has_steps,
                    char_start, char_end, token_start, token_end,
                    source_url, fetched_at, depth, title
                FROM chunks
                WHERE chunk_level = 'parent'
                  AND source_url = ANY(%s)
                ORDER BY depth ASC, source_url, chunk_index ASC
                """,
                (source_urls,),
            )
        else:
            cur.execute(
                """
                SELECT
                    id, document_id, parent_id, chunk_level,
                    chunk_index, section_heading,
                    chunk_text, html_text,
                    has_table, has_code, has_math,
                    has_definition_list, has_admonition, has_steps,
                    char_start, char_end, token_start, token_end,
                    source_url, fetched_at, depth, title
                FROM chunks
                WHERE chunk_level = 'parent'
                ORDER BY depth ASC, source_url, chunk_index ASC
                """
            )
        return list(cur.fetchall())


def _run_hnsw_child_search(
    conn: Connection,
    query_embedding: list[float],
    source_urls: list[str] | None,
) -> list[dict[str, Any]]:
    """Run child ANN search on embedded child chunks.

    Notes:
    - Uses the partial HNSW index on child embeddings.
    - Applies similarity floor as a cosine distance threshold.
    - Sets transaction-local hnsw.ef_search via set_config().
    """
    distance_threshold = _distance_threshold_from_similarity_floor()
    with conn.cursor(row_factory=dict_row) as cur:
        # pgvector recall/speed control; set transaction-locally.
        # SET LOCAL doesn't accept bind placeholders in psycopg, so use
        # set_config(name, value, is_local=true) to stay parameterized.
        cur.execute(
            "SELECT set_config('hnsw.ef_search', %s, true)",
            (str(settings.retrieval_hnsw_ef_search),),
        )

        if source_urls:
            cur.execute(
                """
                SELECT
                    id, parent_id, document_id,
                    chunk_index, section_heading,
                    chunk_text, html_text,
                    has_table, has_code, has_math,
                    has_definition_list, has_admonition, has_steps,
                    char_start, char_end, token_start, token_end,
                    source_url, fetched_at, depth, title,
                    (embedding <=> %s::vector) AS distance
                FROM chunks
                WHERE chunk_level = 'child'
                  AND embedding IS NOT NULL
                  AND source_url = ANY(%s)
                  AND (embedding <=> %s::vector) < %s
                ORDER BY distance ASC
                LIMIT %s
                """,
                (
                    query_embedding,
                    source_urls,
                    query_embedding,
                    distance_threshold,
                    settings.retrieval_top_k_children_limit,
                ),
            )
        else:
            cur.execute(
                """
                SELECT
                    id, parent_id, document_id,
                    chunk_index, section_heading,
                    chunk_text, html_text,
                    has_table, has_code, has_math,
                    has_definition_list, has_admonition, has_steps,
                    char_start, char_end, token_start, token_end,
                    source_url, fetched_at, depth, title,
                    (embedding <=> %s::vector) AS distance
                FROM chunks
                WHERE chunk_level = 'child'
                  AND embedding IS NOT NULL
                  AND (embedding <=> %s::vector) < %s
                ORDER BY distance ASC
                LIMIT %s
                """,
                (
                    query_embedding,
                    query_embedding,
                    distance_threshold,
                    settings.retrieval_top_k_children_limit,
                ),
            )
        return list(cur.fetchall())


def _fetch_parents_by_ids(conn: Connection, parent_ids: list[Any]) -> dict[Any, dict[str, Any]]:
    if not parent_ids:
        return {}
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT
                id, document_id, parent_id, chunk_level,
                chunk_index, section_heading,
                chunk_text, html_text,
                has_table, has_code, has_math,
                has_definition_list, has_admonition, has_steps,
                char_start, char_end, token_start, token_end,
                source_url, fetched_at, depth, title
            FROM chunks
            WHERE id = ANY(%s)
            """,
            (parent_ids,),
        )
        rows = list(cur.fetchall())
    return {row["id"]: row for row in rows}


def _build_chunk_mode_results(
    conn: Connection,
    query_embedding: list[float],
    source_urls: list[str] | None,
    context_budget: int,
) -> list[RetrievedChunk]:
    """Build parent-level results from child ANN hits.

    Children are scored first, then aggregated to parents with max-score
    semantics. Parent results are budgeted by token span and always return at
    least one hit when ANN produced any candidate.
    """
    with conn.transaction():
        child_rows = _run_hnsw_child_search(conn, query_embedding, source_urls)

    if not child_rows:
        return []

    parent_best: dict[Any, tuple[float, float]] = {}
    for child_row in child_rows:
        parent_id = child_row["parent_id"]
        if parent_id is None:
            continue
        raw_similarity = _distance_to_similarity(float(child_row["distance"]))
        depth = int(child_row["depth"])
        adjusted_score = raw_similarity * _depth_multiplier(depth)
        previous = parent_best.get(parent_id)
        if previous is None or adjusted_score > previous[0]:
            parent_best[parent_id] = (adjusted_score, raw_similarity)

    ranked_parent_ids = sorted(
        parent_best.keys(),
        key=lambda parent_id: parent_best[parent_id][0],
        reverse=True,
    )
    parent_rows = _fetch_parents_by_ids(conn, ranked_parent_ids)

    # Context budget over parent token spans.
    selected: list[RetrievedChunk] = []
    used_tokens = 0
    for parent_id in ranked_parent_ids:
        parent_row = parent_rows.get(parent_id)
        if parent_row is None:
            continue
        token_span = max(0, int(parent_row["token_end"]) - int(parent_row["token_start"]))
        if selected and used_tokens + token_span > context_budget:
            continue
        score, raw_similarity = parent_best[parent_id]
        selected.append(_row_to_retrieved_chunk(parent_row, score, raw_similarity))
        used_tokens += token_span

    # Guarantee at least one result in chunk mode when there is any hit.
    if not selected and ranked_parent_ids:
        top_parent_id = ranked_parent_ids[0]
        top_row = parent_rows.get(top_parent_id)
        if top_row is not None:
            score, raw_similarity = parent_best[top_parent_id]
            selected.append(_row_to_retrieved_chunk(top_row, score, raw_similarity))

    return _group_chunks_by_source_ordered(selected)


def _build_full_context_results(
    conn: Connection,
    source_urls: list[str] | None,
    context_budget: int,
) -> list[RetrievedChunk]:
    """Load ordered parent chunks directly for small-corpus mode."""
    rows = _query_full_context_parents(conn, source_urls)
    if not rows:
        return []

    selected: list[RetrievedChunk] = []
    used_tokens = 0
    for row in rows:
        token_span = max(0, int(row["token_end"]) - int(row["token_start"]))
        if selected and used_tokens + token_span > context_budget:
            continue
        selected.append(_row_to_retrieved_chunk(row, score=1.0, raw_similarity=None))
        used_tokens += token_span

    return _group_chunks_by_source_ordered(selected)


def retrieve(
    conn: Connection,
    query: str,
    *,
    source_urls: list[str] | None = None,
    context_budget_override: int | None = None,
) -> RetrievalResult:
    """Retrieve supporting evidence chunks from the indexed corpus.

    Preconditions:
    - conn is an active psycopg3 connection to the WebRAG database
    - register_vector(conn) has already been called by the caller
    - schema has already been initialized by indexing

    Behavior:
    - Empty corpus short-circuits before embedding to avoid unnecessary cost.
    - Full-context mode returns ordered parent chunks with score=1.0.
    - Chunk mode performs child ANN search and returns ranked parent chunks.
    """
    started = perf_counter()
    normalized_source_urls = _normalize_source_urls(source_urls)
    context_budget = _get_effective_context_budget(context_budget_override)

    search_started = perf_counter()
    mode, total_tokens = _determine_mode(
        conn,
        source_urls=normalized_source_urls,
        context_budget_override=context_budget_override,
    )

    total_documents, total_parent_chunks = _load_corpus_counts(conn, normalized_source_urls)

    if total_tokens == 0:
        search_ms = (perf_counter() - search_started) * 1000.0
        total_ms = (perf_counter() - started) * 1000.0
        return RetrievalResult(
            mode=mode,
            chunks=[],
            query_embedding=[],
            corpus_stats=CorpusStats(
                total_documents=total_documents,
                total_parent_chunks=total_parent_chunks,
                total_tokens=total_tokens,
                documents_matched=[],
            ),
            timing=TimingInfo(
                embed_ms=0.0,
                search_ms=search_ms,
                total_ms=total_ms,
            ),
        )

    embed_started = perf_counter()
    query_vector = embed_query(query)
    embed_ms = (perf_counter() - embed_started) * 1000.0

    if mode == "full_context":
        chunks = _build_full_context_results(conn, normalized_source_urls, context_budget)
    else:
        chunks = _build_chunk_mode_results(
            conn,
            query_embedding=query_vector,
            source_urls=normalized_source_urls,
            context_budget=context_budget,
        )

    search_ms = (perf_counter() - search_started) * 1000.0
    total_ms = (perf_counter() - started) * 1000.0

    documents_matched: list[str] = []
    seen: set[str] = set()
    for chunk in chunks:
        if chunk.source_url in seen:
            continue
        seen.add(chunk.source_url)
        documents_matched.append(chunk.source_url)

    return RetrievalResult(
        mode=mode,
        chunks=chunks,
        query_embedding=query_vector,
        corpus_stats=CorpusStats(
            total_documents=total_documents,
            total_parent_chunks=total_parent_chunks,
            total_tokens=total_tokens,
            documents_matched=documents_matched,
        ),
        timing=TimingInfo(
            embed_ms=embed_ms,
            search_ms=search_ms,
            total_ms=total_ms,
        ),
    )
