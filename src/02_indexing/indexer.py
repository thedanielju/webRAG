from __future__ import annotations

# ────────────────────────────────────────────────────────────────
# indexer.py — Entry point for the indexing layer
#
# Public interface:
#   index_document(doc, depth) — index a single NormalizedDocument
#   index_batch(docs, depths)  — index multiple documents in one call
#
# Both are async coroutines.  Callers must await them.
#
# Pipeline phases (index_batch):
#   Phase 1 — Chunk preparation:
#       For each document, check the DB for an existing content_hash
#       (lightweight, no lock).  If the hash matches, skip the doc
#       (no-op dedup).  Otherwise, split into parent/child chunks,
#       detect rich-content flags, and compute token offsets.
#
#   Phase 2 — Embedding (concurrent):
#       Collect all child chunk texts across all staged documents and
#       pass them to embed_texts(), which dispatches concurrent async
#       HTTP batches to the embedding API via asyncio.gather().
#       See embedder.py header for concurrency and failure semantics.
#
#   Phase 3 — Transactional DB writes:
#       Inside a single Postgres transaction, recheck each document's
#       hash with FOR UPDATE (pessimistic lock), delete-and-reinsert
#       on mismatch, and bulk-insert all parent + child chunk rows
#       using executemany.  executemany sends all rows in a single
#       round-trip instead of one INSERT per row, reducing Phase 3
#       from ~3s to <0.5s for large batches.
#
# Async migration notes:
#   - DB: psycopg3 AsyncConnection replaces sync Connection.
#     register_vector_async replaces register_vector.
#   - Embedding: embed_texts() is now a native coroutine using
#     asyncio.gather() — the ThreadPoolExecutor overlap pattern
#     is removed since both embedding and DB I/O are async.
#   - Chunking remains synchronous (CPU-only, no I/O).
#
# Failure handling:
#   If any phase raises (embedding timeout, 429, DB constraint
#   violation, etc.), the exception propagates and the entire
#   index_batch call fails atomically — no partial writes reach
#   the database.  Phase 3's transaction ensures this for DB
#   writes; Phase 2's embed_texts() ensures it for embeddings.
# ────────────────────────────────────────────────────────────────

from datetime import datetime, timezone
import os
import time
from typing import TYPE_CHECKING
from uuid import uuid4

from pgvector.psycopg import register_vector_async
from psycopg import AsyncConnection

from config import settings
from src.indexing.chunker import build_chunks
from src.indexing.embedder import annotate_token_offsets, embed_texts
from src.indexing.models import Chunk
from src.indexing.schema import init_schema

if TYPE_CHECKING:
    from src.ingestion.service import NormalizedDocument


_SCHEMA_INITIALIZED = False
_INDEXING_DEBUG = os.getenv("INDEXING_DEBUG", "").strip() == "1"


async def _connect_db() -> AsyncConnection:
    if not settings.database_url:
        raise RuntimeError("DATABASE_URL is required for indexing.")
    # autocommit=True so that conn.transaction() manages its own
    # explicit BEGIN/COMMIT, rather than nesting inside an implicit
    # transaction that would never be committed.
    conn = await AsyncConnection.connect(settings.database_url, autocommit=True)
    return conn


async def _ensure_schema_initialized(conn: AsyncConnection) -> None:
    global _SCHEMA_INITIALIZED
    if _SCHEMA_INITIALIZED:
        return
    async with conn.cursor() as cur:
        await cur.execute(
            """
            SELECT
                to_regclass('public.documents') IS NOT NULL,
                to_regclass('public.chunks') IS NOT NULL,
                to_regclass('public.link_candidates') IS NOT NULL
            """
        )
        row = await cur.fetchone()
    schema_exists = bool(row and row[0] and row[1] and row[2])

    if schema_exists:
        _SCHEMA_INITIALIZED = True
        return

    # init_schema uses sync Connection — create a temporary one.
    # This only runs once on first use when the schema doesn't exist.
    from psycopg import connect as sync_connect
    with sync_connect(settings.database_url) as sync_conn:
        init_schema(sync_conn)
    _SCHEMA_INITIALIZED = True


def _canonical_source_url(doc: NormalizedDocument) -> str:
    return doc.source_url or doc.url


def _prepare_document_payload(doc: NormalizedDocument, depth: int) -> dict[str, object]:
    return {
        "id": uuid4(),
        "url": doc.url,
        "source_url": _canonical_source_url(doc),
        "title": doc.title,
        "description": doc.description,
        "language": doc.language,
        "status_code": doc.status_code,
        "published_time": doc.published_time,
        "modified_time": doc.modified_time,
        "doc_type": doc.doc_type,
        "content_hash": doc.content_hash,
        "fetched_at": doc.fetched_at or datetime.now(timezone.utc),
        "depth": depth,
    }


async def _fetch_existing_document(
    conn: AsyncConnection, source_url: str, for_update: bool = False
) -> tuple[str, str] | None:
    lock_clause = "FOR UPDATE" if for_update else ""
    async with conn.cursor() as cur:
        await cur.execute(
            f"""
            SELECT id::text, content_hash
            FROM documents
            WHERE source_url = %s
            {lock_clause}
            """,
            (source_url,),
        )
        row = await cur.fetchone()
    if row is None:
        return None
    return row[0], row[1]


async def _delete_document_cascade(conn: AsyncConnection, document_id: str) -> None:
    async with conn.cursor() as cur:
        await cur.execute("DELETE FROM documents WHERE id = %s", (document_id,))


async def _insert_document(conn: AsyncConnection, payload: dict[str, object]) -> None:
    async with conn.cursor() as cur:
        await cur.execute(
            """
            INSERT INTO documents (
                id, url, source_url, title, description, language, status_code,
                published_time, modified_time, doc_type, content_hash, fetched_at, depth
            )
            VALUES (
                %(id)s, %(url)s, %(source_url)s, %(title)s, %(description)s, %(language)s,
                %(status_code)s, %(published_time)s, %(modified_time)s, %(doc_type)s,
                %(content_hash)s, %(fetched_at)s, %(depth)s
            )
            """,
            payload,
        )


async def _insert_chunks(conn: AsyncConnection, chunks: list[Chunk]) -> None:
    """Bulk-insert chunk rows using executemany.

    executemany sends the parameterised INSERT once and streams all rows
    in a single client-server round-trip, compared to the previous
    row-by-row cur.execute() loop which issued N separate round-trips.
    For 1 000+ chunks this cuts Phase 3 write time from ~3s to < 0.5s.

    The semantic behaviour is identical: one INSERT per chunk, same
    column set, same parameter mapping.  The only difference is
    transport efficiency.
    """
    if not chunks:
        return
    for chunk in chunks:
        if chunk.document_id is None:
            raise ValueError("Chunk document_id must be set before insert.")

    params_list = [
        {
            "id": chunk.id,
            "document_id": chunk.document_id,
            "parent_id": chunk.parent_id,
            "chunk_level": chunk.chunk_level.value,
            "chunk_index": chunk.chunk_index,
            "section_heading": chunk.section_heading,
            "chunk_text": chunk.chunk_text,
            "html_text": chunk.html_text,
            "has_table": chunk.flags.has_table,
            "has_code": chunk.flags.has_code,
            "has_math": chunk.flags.has_math,
            "has_definition_list": chunk.flags.has_definition_list,
            "has_admonition": chunk.flags.has_admonition,
            "has_steps": chunk.flags.has_steps,
            "char_start": chunk.char_start,
            "char_end": chunk.char_end,
            "token_start": chunk.token_start,
            "token_end": chunk.token_end,
            "embedding": chunk.embedding,
            "source_url": chunk.source_url,
            "fetched_at": chunk.fetched_at,
            "depth": chunk.depth,
            "title": chunk.title,
        }
        for chunk in chunks
    ]

    async with conn.cursor() as cur:
        # executemany: single prepared statement, streamed rows — much
        # faster than N individual execute() calls for large chunk lists.
        await cur.executemany(
            """
            INSERT INTO chunks (
                id, document_id, parent_id, chunk_level, chunk_index, section_heading,
                chunk_text, html_text, has_table, has_code, has_math, has_definition_list,
                has_admonition, has_steps, char_start, char_end, token_start, token_end,
                embedding, source_url, fetched_at, depth, title
            )
            VALUES (
                %(id)s, %(document_id)s, %(parent_id)s, %(chunk_level)s, %(chunk_index)s,
                %(section_heading)s, %(chunk_text)s, %(html_text)s, %(has_table)s,
                %(has_code)s, %(has_math)s, %(has_definition_list)s, %(has_admonition)s,
                %(has_steps)s, %(char_start)s, %(char_end)s, %(token_start)s, %(token_end)s,
                %(embedding)s, %(source_url)s, %(fetched_at)s, %(depth)s, %(title)s
            )
            """,
            params_list,
        )


async def _insert_link_candidates(
    conn: AsyncConnection,
    document_id: object,
    source_url: str,
    links: list[str],
    depth: int,
) -> None:
    """Bulk-insert URL-only link candidate rows for a document.

    Uses ON CONFLICT DO NOTHING so re-indexing the same document
    (after content_hash change) gracefully skips duplicate links.
    Title and description start as NULL — they are populated later
    by enrich_link_candidates() when orchestration needs scoring
    metadata.
    """
    if not links:
        return

    params_list = [
        {
            "source_document_id": document_id,
            "source_url": source_url,
            "target_url": target_url,
            "depth": depth,
        }
        for target_url in links
    ]

    async with conn.cursor() as cur:
        await cur.executemany(
            """
            INSERT INTO link_candidates (
                source_document_id, source_url, target_url, depth
            )
            VALUES (
                %(source_document_id)s, %(source_url)s,
                %(target_url)s, %(depth)s
            )
            ON CONFLICT (source_document_id, target_url) DO NOTHING
            """,
            params_list,
        )


def _build_doc_chunks(
    doc: NormalizedDocument, depth: int, document_id: object
) -> tuple[list[Chunk], list[Chunk]]:
    markdown = doc.markdown or ""
    source_url = _canonical_source_url(doc)
    fetched_at = doc.fetched_at or datetime.now(timezone.utc)

    parent_chunks, child_chunks = build_chunks(
        markdown=markdown,
        html=doc.html,
        source_url=source_url,
        fetched_at=fetched_at,
        title=doc.title,
        depth=depth,
    )
    annotate_token_offsets(parent_chunks + child_chunks, markdown)

    # document_id is assigned once and propagated before DB insertion.
    for chunk in parent_chunks:
        chunk.document_id = document_id
    for chunk in child_chunks:
        chunk.document_id = document_id

    return parent_chunks, child_chunks


async def index_document(doc: NormalizedDocument, depth: int) -> None:
    """Index a single NormalizedDocument at the given crawl depth."""
    await index_batch([doc], [depth])


async def index_batch(docs: list[NormalizedDocument], depths: list[int]) -> None:
    """Chunk, embed, and store a batch of NormalizedDocuments atomically.

    Async coroutine — callers must await.  Embedding uses asyncio.gather()
    internally for concurrent API batches.  DB writes use AsyncConnection.
    """
    if len(docs) != len(depths):
        raise ValueError("docs and depths must have the same length.")
    if not docs:
        return

    conn = await _connect_db()
    try:
        total_started = time.perf_counter()
        await _ensure_schema_initialized(conn)
        await register_vector_async(conn)

        staged_items: list[dict[str, object]] = []
        phase1_started = time.perf_counter()
        if _INDEXING_DEBUG:
            print(f"index_batch debug: phase1 start docs={len(docs)}")

        # ── Phase 1: Chunk preparation ───────────────────────────
        # Chunking is CPU-only (no I/O) so it runs synchronously.
        # Dedup checks use async DB queries.
        for index, (doc, depth) in enumerate(zip(docs, depths), start=1):
            if _INDEXING_DEBUG:
                print(
                    "index_batch debug: phase1 doc-start",
                    f"{index}/{len(docs)}",
                    f"url={doc.source_url or doc.url}",
                )
            payload = _prepare_document_payload(doc, depth)
            source_url = str(payload["source_url"])
            existing = await _fetch_existing_document(conn, source_url, for_update=False)

            if existing is not None:
                _existing_id, existing_hash = existing
                if existing_hash == payload["content_hash"]:
                    if _INDEXING_DEBUG:
                        print(
                            "index_batch debug: phase1 doc-skip-noop",
                            f"{index}/{len(docs)}",
                            f"url={source_url}",
                        )
                    continue

            parent_chunks, child_chunks = _build_doc_chunks(
                doc=doc,
                depth=depth,
                document_id=payload["id"],
            )
            if _INDEXING_DEBUG:
                print(
                    "index_batch debug: phase1 doc-built",
                    f"{index}/{len(docs)}",
                    f"parents={len(parent_chunks)} children={len(child_chunks)}",
                )
            staged_items.append(
                {
                    "payload": payload,
                    "parent_chunks": parent_chunks,
                    "child_chunks": child_chunks,
                    "links": list(doc.links) if doc.links else [],
                }
            )

        if _INDEXING_DEBUG:
            print(
                "index_batch debug: phase1 done",
                f"staged_docs={len(staged_items)}",
                f"elapsed={time.perf_counter() - phase1_started:.2f}s",
            )

        if not staged_items:
            if _INDEXING_DEBUG:
                print("index_batch debug: nothing to index (all items were no-op).")
            return

        # ── Phase 2: Embedding (concurrent via asyncio.gather) ───
        # Collect all child texts, send to embed_texts() which fans
        # out batches concurrently via asyncio.gather internally.
        phase2_started = time.perf_counter()
        all_child_chunks: list[Chunk] = []
        all_child_texts: list[str] = []
        for item in staged_items:
            children = item["child_chunks"]
            all_child_chunks.extend(children)
            all_child_texts.extend(c.chunk_text for c in children)

        if _INDEXING_DEBUG:
            parent_count = sum(len(item["parent_chunks"]) for item in staged_items)
            print(
                "index_batch debug: staged",
                f"docs={len(staged_items)} parents={parent_count} children={len(all_child_chunks)}",
            )

        all_vectors = await embed_texts(all_child_texts)

        if len(all_vectors) != len(all_child_chunks):
            raise ValueError(
                "Embedding count mismatch after batch call: "
                f"expected {len(all_child_chunks)}, got {len(all_vectors)}."
            )
        for chunk, vector in zip(all_child_chunks, all_vectors):
            chunk.embedding = vector

        # ── Phase 3: Transactional DB writes (with recheck + lock) ─
        phase3_started = time.perf_counter()
        async with conn.transaction():
            for item in staged_items:
                payload = item["payload"]
                parent_chunks = item["parent_chunks"]
                child_chunks = item["child_chunks"]
                source_url = str(payload["source_url"])

                existing = await _fetch_existing_document(conn, source_url, for_update=True)
                if existing is not None:
                    existing_id, existing_hash = existing
                    if existing_hash == payload["content_hash"]:
                        continue
                    await _delete_document_cascade(conn, existing_id)

                await _insert_document(conn, payload)
                await _insert_chunks(conn, parent_chunks)
                await _insert_chunks(conn, child_chunks)

                # Persist outgoing links from this document for
                # orchestration link scoring.  URL-only rows;
                # enrichment happens on-demand via enrich_link_candidates().
                doc_links: list[str] = item.get("links", [])
                await _insert_link_candidates(
                    conn,
                    document_id=payload["id"],
                    source_url=str(payload["source_url"]),
                    links=doc_links,
                    depth=int(payload["depth"]),
                )

        if _INDEXING_DEBUG:
            phase1 = phase2_started - phase1_started
            phase2 = phase3_started - phase2_started
            phase3 = time.perf_counter() - phase3_started
            total = time.perf_counter() - total_started
            print(
                "index_batch debug timings:",
                f"phase1+2_chunk_embed={phase1:.2f}s phase2_assign={phase2:.2f}s phase3_db={phase3:.2f}s total={total:.2f}s",
            )
    finally:
        await conn.close()
