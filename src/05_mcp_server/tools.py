"""MCP tool definitions — ``answer``, ``search``, ``status``.

Three tools expose WebRAG's capabilities to reasoning models:

  ``answer``  — Full pipeline: ingest URL(s) → decompose query → retrieve →
                rerank → expand → format.  Slowest but most thorough.
  ``search``  — Fast path: query the existing corpus without scraping.
                Skips ingestion and expansion.  Useful for follow-up
                questions about already-indexed content.
  ``status``  — Introspection: report what's currently in the corpus
                (document count, chunk count, token totals, URLs).

Each tool function is registered on the ``FastMCP`` instance by
:mod:`server`.  Handlers receive a ``Context`` for progress
reporting and access the shared ``OrchestratorEngine`` via the
lifespan state dict.

Error handling:
  Every tool returns a string — never raises.  Exceptions are caught
  and formatted via ``errors.py`` so the reasoning model always gets
  structured feedback it can relay to the user.
"""

from __future__ import annotations

import asyncio
import logging
import sys

from mcp.server.fastmcp import Context
from pgvector.psycopg import register_vector_async
from psycopg import AsyncConnection

from config import settings
from src.mcp_server import errors
from src.mcp_server.formatter import format_result
from src.orchestration.engine import OrchestratorEngine
from src.orchestration.models import RankedChunk
from src.orchestration.reranker import rerank
from src.retrieval.citations import CitationSpan, extract_citation
from src.retrieval.models import CorpusStats, RetrievalResult
from src.retrieval.search import retrieve

logger = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────


def _get_engine(ctx: Context) -> OrchestratorEngine:
    """Retrieve the engine instance stored during lifespan startup."""
    # The lifespan yields a dict {"engine": engine} that FastMCP
    # stores on the context's lifespan_context attribute.
    return ctx.request_context.lifespan_context["engine"]


# ── answer ────────────────────────────────────────────────────


async def answer(
    url: str | list[str],
    query: str,
    ctx: Context,
    intent: str | None = None,
    known_context: str | None = None,
    constraints: list[str] | None = None,
    expansion_budget: int | None = None,
) -> str:
    """Query one or more web pages and their linked content for information.

    WebRAG will scrape each URL if not already indexed, decompose the
    query, retrieve and rerank relevant chunks, optionally expand to
    linked pages, and return cited evidence with source attribution.
    Use this when you need factual information grounded in specific
    web sources.
    """
    engine = _get_engine(ctx)

    # Normalise url to a list.
    urls: list[str] = [url] if isinstance(url, str) else list(url)
    if not urls:
        return errors.full_failure(ValueError("At least one URL is required."))

    primary_url = urls[0]

    try:
        # ── Multi-URL pre-ingestion ───────────────────────
        # When multiple URLs are provided, the first is the "primary"
        # (drives the orchestration loop) and the rest are pre-ingested
        # so their content is available during retrieval.  Progress
        # notifications keep the model informed of long ingestions.
        if len(urls) > 1:
            await ctx.report_progress(0, len(urls), "Ingesting additional URLs…")

            for i, extra_url in enumerate(urls[1:], 1):
                await ctx.info(f"Ingesting {extra_url}…")
                # Use the engine's internal _ensure_ingested via a
                # lightweight OrchestrationState.
                from src.orchestration.models import OrchestrationState

                state = OrchestrationState(
                    original_query=query,
                    seed_url=extra_url,
                )
                await engine._ensure_ingested(extra_url, state)
                await ctx.report_progress(i, len(urls), f"Ingested {extra_url}")

        # ── Main orchestration call ───────────────────────
        # Wrapped in asyncio.wait_for() to enforce the configurable
        # timeout.  Without this, a deeply-expanding orchestration
        # run on a large site could block the MCP connection
        # indefinitely.
        await ctx.info(f"Running orchestration on {primary_url}…")

        result = await asyncio.wait_for(
            engine.run(
                primary_url,
                query,
                intent=intent,
                known_context=known_context,
                constraints=constraints,
                expansion_budget=expansion_budget,
            ),
            timeout=settings.mcp_tool_timeout,
        )

    except asyncio.TimeoutError:
        logger.error("answer tool timed out after %ds", settings.mcp_tool_timeout)
        return errors.timeout(settings.mcp_tool_timeout)
    except Exception as exc:
        logger.error("answer tool failed: %r", exc, exc_info=True)
        return errors.full_failure(exc)

    # ── Empty results ─────────────────────────────────────
    if not result.chunks:
        return errors.empty_results(
            mode=result.mode,
            documents_searched=result.corpus_stats.total_documents,
            chunks_evaluated=result.corpus_stats.total_parent_chunks,
            total_ms=result.timing.total_ms,
        )

    # ── Format response ───────────────────────────────────
    await ctx.info(
        f"Formatting {len(result.chunks)} chunks "
        f"({result.timing.total_ms:.0f}ms total)…"
    )
    return format_result(result)


# ── search ────────────────────────────────────────────────────


async def search(
    query: str,
    ctx: Context,
    source_urls: list[str] | None = None,
    intent: str | None = None,
    top_k: int | None = None,
) -> str:
    """Search the existing WebRAG corpus without scraping new pages.

    Bypasses the full orchestration pipeline — goes directly to
    retrieve() + rerank() and formats the result.  This is much
    faster than ``answer`` because it skips:
      - URL ingestion / scraping
      - Query decomposition
      - Expansion loop (link scoring, crawling, re-indexing)

    Use this when content has already been indexed (e.g., from a
    previous ``answer`` call) and you want to ask a different question
    about the same material.
    """
    engine = _get_engine(ctx)
    effective_top_k = top_k or settings.reranker_top_n

    await ctx.info(f"Searching corpus… (top_k={effective_top_k})")

    try:
        conn = await engine._acquire_connection()
        try:
            # ── Retrieve ──────────────────────────────────
            rr: RetrievalResult = await retrieve(
                conn,
                query,
                source_urls=source_urls,
            )

            if rr.is_empty:
                return errors.empty_results(
                    mode=rr.mode,
                    documents_searched=rr.corpus_stats.total_documents,
                    chunks_evaluated=rr.corpus_stats.total_parent_chunks,
                    total_ms=rr.timing.total_ms,
                )

            # ── Rerank ────────────────────────────────────
            passages = [c.selected_text for c in rr.chunks[:effective_top_k]]
            original_scores = [c.score for c in rr.chunks[:effective_top_k]]

            rerank_results = await rerank(
                query,
                passages,
                instruction=intent,
                top_n=effective_top_k,
                original_scores=original_scores,
            )

            # Map back to RankedChunk.
            ranked: list[RankedChunk] = []
            for rr_item in rerank_results:
                if rr_item.index < len(rr.chunks):
                    chunk = rr.chunks[rr_item.index]
                    ranked.append(
                        RankedChunk(
                            chunk=chunk,
                            reranked_score=rr_item.relevance_score,
                            confidence=rr_item.confidence,
                        )
                    )

            # ── Citations ─────────────────────────────────
            citations: list[CitationSpan] = []
            for rc in ranked:
                c = rc.chunk
                cit = extract_citation(c, c.char_start, c.char_end)
                if cit is not None:
                    citations.append(cit)

        finally:
            await engine._release_connection(conn)

    except Exception as exc:
        logger.error("search tool failed: %r", exc, exc_info=True)
        return errors.full_failure(exc)

    # ── Build OrchestrationResult-like output ─────────────
    from src.orchestration.models import (
        ExpansionDecision,
        OrchestrationResult,
        OrchestrationTiming,
        QueryAnalysis,
    )

    result = OrchestrationResult(
        chunks=ranked,
        citations=citations,
        query_analysis=QueryAnalysis(
            original_query=query,
            sub_queries=[query],
            query_type=intent or "factual",
            complexity="simple",
            key_concepts=[],
        ),
        expansion_steps=[],
        corpus_stats=rr.corpus_stats,
        timing=OrchestrationTiming(
            retrieval_ms=rr.timing.search_ms,
            reranking_ms=0.0,
            total_ms=rr.timing.total_ms,
        ),
        mode=rr.mode,
        final_decision=ExpansionDecision(
            action="stop",
            reason="search tool — no expansion",
            confidence="high",
        ),
        total_iterations=0,
        total_urls_ingested=0,
    )

    return format_result(result)


# ── status ────────────────────────────────────────────────────


async def status(
    ctx: Context,
    source_url: str | None = None,
    include_urls: bool = True,
) -> str:
    """Check what content WebRAG currently has indexed.

    Returns document count, total tokens, indexed URLs with titles,
    and last-fetched timestamps.  Use this to understand what's
    available before deciding whether to call ``answer`` (which
    ingests new content) or ``search`` (which queries existing
    content).
    """
    engine = _get_engine(ctx)

    try:
        conn = await engine._acquire_connection()
        try:
            return await _query_corpus_status(conn, source_url, include_urls)
        finally:
            await engine._release_connection(conn)
    except Exception as exc:
        logger.error("status tool failed: %r", exc, exc_info=True)
        return errors.full_failure(exc)


async def _query_corpus_status(
    conn: AsyncConnection,
    source_url: str | None,
    include_urls: bool,
) -> str:
    """Execute status queries against the documents + chunks tables.

    Two modes:
      - Per-URL: returns title, chunk count, token count, and fetch time
        for a specific source URL.
      - Corpus-wide: returns aggregate counts plus an optional listing
        of all indexed URLs (sorted by most recently fetched first).
    """
    lines = ["[CORPUS STATUS]"]

    async with conn.cursor() as cur:
        if source_url:
            # Per-URL status.
            await cur.execute(
                """
                SELECT d.title,
                       COUNT(c.id)    AS chunk_count,
                       COALESCE(SUM(c.token_count), 0) AS total_tokens,
                       d.fetched_at
                FROM documents d
                LEFT JOIN chunks c ON d.id = c.document_id
                WHERE d.source_url = %s
                GROUP BY d.id, d.title, d.fetched_at
                """,
                (source_url,),
            )
            row = await cur.fetchone()
            if row is None:
                lines.append(f"URL not indexed: {source_url}")
                return "\n".join(lines)

            title, chunk_count, total_tokens, fetched_at = row
            fetched_str = fetched_at.strftime("%Y-%m-%d %H:%M UTC") if fetched_at else "unknown"
            lines.append(f"URL: {source_url}")
            lines.append(f"Title: {title or 'Untitled'}")
            lines.append(f"Chunks: {chunk_count}")
            lines.append(f"Tokens: {total_tokens:,}")
            lines.append(f"Fetched: {fetched_str}")
            return "\n".join(lines)

        # Corpus-wide status.
        await cur.execute(
            """
            SELECT COUNT(DISTINCT d.id),
                   COUNT(c.id),
                   COALESCE(SUM(c.token_count), 0)
            FROM documents d
            LEFT JOIN chunks c ON d.id = c.document_id
            """
        )
        row = await cur.fetchone()
        doc_count, chunk_count, total_tokens = row if row else (0, 0, 0)

        lines.append(f"Documents indexed: {doc_count}")
        lines.append(f"Total chunks: {chunk_count}")
        lines.append(f"Total tokens: {total_tokens:,}")

        if include_urls and doc_count > 0:
            lines.append("")
            lines.append("Indexed URLs:")
            await cur.execute(
                """
                SELECT d.source_url,
                       d.title,
                       COUNT(c.id)    AS chunk_count,
                       COALESCE(SUM(c.token_count), 0) AS total_tokens,
                       d.fetched_at
                FROM documents d
                LEFT JOIN chunks c ON d.id = c.document_id
                GROUP BY d.id, d.source_url, d.title, d.fetched_at
                ORDER BY d.fetched_at DESC NULLS LAST
                """
            )
            rows = await cur.fetchall()
            for url, title, chunks, tokens, fetched_at in rows:
                fetched_str = (
                    fetched_at.strftime("%Y-%m-%d %H:%M UTC")
                    if fetched_at
                    else "unknown"
                )
                lines.append(
                    f"- {title or 'Untitled'} — {url}\n"
                    f"  ({chunks} chunks, {tokens:,} tokens, fetched {fetched_str})"
                )

    return "\n".join(lines)
