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
import time
from typing import Any, Literal

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


def _normalize_research_mode(
    mode: str | None,
) -> Literal["fast", "auto", "deep"]:
    value = (mode or settings.mcp_default_research_mode or "fast").strip().lower()
    if value in {"fast", "auto", "deep"}:
        return value  # type: ignore[return-value]
    return "fast"


def _normalize_retrieval_mode(
    mode: str | None,
) -> Literal["chunk", "full_context", "auto"]:
    value = (mode or settings.mcp_default_retrieval_mode or "chunk").strip().lower()
    if value in {"chunk", "full_context", "auto"}:
        return value  # type: ignore[return-value]
    return "chunk"


def _resolve_answer_modes(
    *,
    research_mode: str | None,
    retrieval_mode: str | None,
    expansion_budget: int | None,
) -> tuple[Literal["fast", "auto", "deep"], Literal["chunk", "full_context", "auto"], int | None]:
    resolved_research = _normalize_research_mode(research_mode)
    resolved_retrieval = _normalize_retrieval_mode(retrieval_mode)

    effective_expansion_budget = expansion_budget
    if resolved_research == "fast" and expansion_budget is None:
        effective_expansion_budget = 0

    return resolved_research, resolved_retrieval, effective_expansion_budget


async def _mcp_progress_notifier(ctx: Context, phase: str, data: dict[str, Any]) -> None:
    """Translate engine phase callbacks into MCP progress/info notifications."""
    if phase == "corpus_prep_start":
        await ctx.info("Checking corpus / preparing seed URL…")
    elif phase == "ingestion_cache_hit":
        await ctx.info("Seed URL already indexed (cache hit).")
    elif phase == "ingestion_start":
        await ctx.info(f"Ingesting seed URL: {data.get('url', '')}")
    elif phase == "ingestion_done":
        await ctx.info("Seed ingestion complete.")
    elif phase == "query_analysis_start":
        await ctx.info("Analyzing query…")
    elif phase == "query_analysis_done":
        await ctx.info(
            f"Query analysis complete ({data.get('sub_queries', 1)} sub-query(s), "
            f"type={data.get('query_type', 'unknown')})."
        )
    elif phase == "initial_retrieval_start":
        await ctx.info(
            f"Running initial retrieval + reranking (mode={data.get('retrieval_mode', 'auto')})…"
        )
    elif phase == "initial_retrieval_done":
        await ctx.info(
            f"Initial retrieval complete ({data.get('chunks', 0)} chunks; "
            f"retrieval {float(data.get('retrieval_ms', 0.0)):.0f}ms, "
            f"rerank {float(data.get('reranking_ms', 0.0)):.0f}ms)."
        )
    elif phase == "evaluation_done":
        await ctx.info(
            f"Evaluator decision: {data.get('action', 'unknown')} "
            f"({data.get('reason', 'no reason')})"
        )
    elif phase == "expansion_skipped":
        await ctx.info(
            f"Expansion skipped (max_iterations={data.get('max_iterations', 0)}; "
            f"reason={data.get('reason', 'n/a')})."
        )
    elif phase == "expansion_iteration_start":
        await ctx.info(
            f"Expansion iteration {data.get('iteration', '?')} starting "
            f"({data.get('action', 'expand')})…"
        )
    elif phase == "expansion_iteration_done":
        await ctx.info(
            f"Expansion iteration {data.get('iteration', '?')} done: "
            f"selected={data.get('candidates_selected', 0)}, "
            f"chunks_added={data.get('chunks_added', 0)}, "
            f"score→{float(data.get('top_score_after', 0.0)):.2f}, "
            f"{float(data.get('duration_ms', 0.0)):.0f}ms."
        )
    elif phase == "locality_start":
        await ctx.info("Applying locality expansion…")
    elif phase == "merge_start":
        await ctx.info("Merging and deduplicating evidence…")
    elif phase == "citations_start":
        await ctx.info("Extracting citations…")
    elif phase == "run_done":
        await ctx.info(
            f"Orchestration complete ({float(data.get('total_ms', 0.0)):.0f}ms total)."
        )



# ── answer ────────────────────────────────────────────────────


async def answer(
    url: str | list[str],
    query: str,
    ctx: Context,
    intent: str | None = None,
    known_context: str | None = None,
    constraints: list[str] | None = None,
    expansion_budget: int | None = None,
    research_mode: Literal["fast", "auto", "deep"] | None = None,
    retrieval_mode: Literal["chunk", "full_context", "auto"] | None = None,
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
    resolved_research_mode, resolved_retrieval_mode, effective_expansion_budget = _resolve_answer_modes(
        research_mode=research_mode,
        retrieval_mode=retrieval_mode,
        expansion_budget=expansion_budget,
    )

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
        await ctx.info(
            f"Running orchestration on {primary_url} "
            f"(research_mode={resolved_research_mode}, retrieval_mode={resolved_retrieval_mode})…"
        )

        result = await asyncio.wait_for(
            engine.run(
                primary_url,
                query,
                intent=intent,
                known_context=known_context,
                constraints=constraints,
                expansion_budget=effective_expansion_budget,
                retrieval_mode=resolved_retrieval_mode,
                progress_callback=lambda phase, data: _mcp_progress_notifier(ctx, phase, data),
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
    await ctx.info("Building final MCP response (preserving citations/images)…")
    return format_result(
        result,
        research_mode=resolved_research_mode,
        retrieval_mode=resolved_retrieval_mode,
    )


# ── search ────────────────────────────────────────────────────


async def search(
    query: str,
    ctx: Context,
    source_urls: list[str] | None = None,
    intent: str | None = None,
    top_k: int | None = None,
    retrieval_mode: Literal["chunk", "full_context", "auto"] | None = None,
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
    resolved_retrieval_mode = _normalize_retrieval_mode(retrieval_mode)

    await ctx.info(
        f"Searching corpus… (top_k={effective_top_k}, retrieval_mode={resolved_retrieval_mode})"
    )

    try:
        conn = await engine._acquire_connection()
        try:
            # ── Retrieve ──────────────────────────────────
            rr: RetrievalResult = await retrieve(
                conn,
                query,
                source_urls=source_urls,
                mode_override=resolved_retrieval_mode,
            )

            if rr.is_empty:
                return errors.empty_results(
                    mode=rr.mode,
                    documents_searched=rr.corpus_stats.total_documents,
                    chunks_evaluated=rr.corpus_stats.total_parent_chunks,
                    total_ms=rr.timing.total_ms,
                )

            # ── Rerank ────────────────────────────────────
            passages: list[str] = []
            for c in rr.chunks[:effective_top_k]:
                if c.has_image and c.image_context_text:
                    passages.append(
                        f"{c.selected_text}\n\n[Image context]\n{c.image_context_text}"
                    )
                else:
                    passages.append(c.selected_text)
            original_scores = [c.score for c in rr.chunks[:effective_top_k]]

            rerank_t0 = time.perf_counter()
            rerank_results = await rerank(
                query,
                passages,
                instruction=intent,
                top_n=effective_top_k,
                original_scores=original_scores,
            )
            rerank_ms = (time.perf_counter() - rerank_t0) * 1000

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
            reranking_ms=rerank_ms,
            total_ms=rr.timing.total_ms + rerank_ms,
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

    return format_result(
        result,
        research_mode="fast",
        retrieval_mode=resolved_retrieval_mode,
    )


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
        errors.full_failure(exc)


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
                       COALESCE(SUM(c.token_end - c.token_start), 0) AS total_tokens,
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
                   COALESCE(SUM(c.token_end - c.token_start), 0)
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
                       COALESCE(SUM(c.token_end - c.token_start), 0) AS total_tokens,
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
