"""OrchestratorEngine — top-level orchestration entry point.

Coordinates the full retrieve-evaluate-expand loop:

  1. Corpus preparation (ensure URL is ingested).
  2. Query analysis / decomposition.
  3. Retrieve → rerank → evaluate loop.
  4. Expansion loop (breadth / recall / intent).
  5. Locality expansion.
  6. Final merge, dedup, citation extraction.
  7. Assemble and return ``OrchestrationResult``.
"""

from __future__ import annotations

import asyncio
import time
import warnings
from uuid import UUID

from pgvector.psycopg import register_vector_async
from psycopg import AsyncConnection
from psycopg_pool import AsyncConnectionPool

from config import settings
from src.indexing.indexer import index_batch
from src.ingestion.service import NormalizedDocument, ingest
from src.orchestration.evaluator import evaluate
from src.orchestration.expander import expand
from src.orchestration.locality import expand_locality
from src.orchestration.merger import merge_ranked_chunks, merge_subquery_results
from src.orchestration.models import (
    ExpansionDecision,
    ExpansionStep,
    OrchestrationResult,
    OrchestrationState,
    OrchestrationTiming,
    RankedChunk,
    SubQueryResult,
)
from src.orchestration.query_analyzer import analyze_query
from src.orchestration.reranker import rerank
from src.retrieval.citations import CitationSpan, extract_citation
from src.retrieval.models import CorpusStats, RetrievalResult
from src.retrieval.search import retrieve


class OrchestratorEngine:
    """Coordinates the full retrieve-evaluate-expand loop.

    Owns:
      - Connection pool management.
      - Query analysis dispatch.
      - The main iteration loop.
      - Output assembly.

    Does NOT own:
      - Individual evaluation logic (``evaluator.py``).
      - Reranking calls (``reranker.py``).
      - Expansion execution (``expander.py``).
    """

    def __init__(self) -> None:
        self._pool: AsyncConnectionPool | None = None

    # ── Pool lifecycle ────────────────────────────────────────

    async def start(self) -> None:
        """Open the connection pool.  Must be called before ``run()``."""
        if self._pool is not None:
            return
        self._pool = AsyncConnectionPool(
            conninfo=settings.database_url,
            min_size=2,
            max_size=10,
            open=False,
            kwargs={"autocommit": True},
        )
        await self._pool.open()

    async def stop(self) -> None:
        """Close the connection pool."""
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    async def _acquire_connection(self) -> AsyncConnection:
        """Get a connection from the pool, registering pgvector types.

        pgvector's register_vector_async must be called per-connection
        so psycopg knows how to encode/decode vector columns.
        """
        if self._pool is None:
            await self.start()
        assert self._pool is not None
        conn = await self._pool.getconn()
        await register_vector_async(conn)
        return conn

    async def _release_connection(self, conn: AsyncConnection) -> None:
        if self._pool is not None:
            await self._pool.putconn(conn)

    # ── Main entry point ──────────────────────────────────────

    async def run(
        self,
        url: str,
        query: str,
        *,
        intent: str | None = None,
        known_context: str | None = None,
        constraints: list[str] | None = None,
        expansion_budget: int | None = None,
    ) -> OrchestrationResult:
        """Execute the full orchestration pipeline for a single request."""
        total_start = time.perf_counter()
        timing = OrchestrationTiming()

        state = OrchestrationState(
            original_query=query,
            seed_url=url,
            intent=intent,
            known_context=known_context,
            constraints=constraints,
            expansion_budget=expansion_budget,
        )

        context_budget = settings.retrieval_context_budget

        conn = await self._acquire_connection()
        try:
            # ── Phase 1: Corpus Preparation ───────────────────
            await self._ensure_ingested(url, state)

            # ── Phase 2: Query Analysis ───────────────────────
            qa_start = time.perf_counter()
            state.query_analysis = await analyze_query(
                query,
                intent=intent,
                known_context=known_context,
                constraints=constraints,
            )
            timing.query_analysis_ms = (time.perf_counter() - qa_start) * 1000

            # ── Phase 3: Initial Retrieval + Reranking ────────
            ret_start = time.perf_counter()
            state.current_chunks = await self._retrieve_and_rerank(
                state.query_analysis.sub_queries,
                conn,
                state,
                intent=intent,
            )
            timing.retrieval_ms = (time.perf_counter() - ret_start) * 1000

            signals, decision = await evaluate(
                state.current_chunks,
                context_budget,
                0,
                None,
                state.query_analysis,
            )
            state.current_signals = signals

            # ── Phase 4: Expansion Loop ───────────────────────
            max_iterations = expansion_budget or settings.max_expansion_depth

            while (
                decision.action != "stop"
                and state.iteration < max_iterations
            ):
                state.iteration += 1
                top_score_before = signals.top_score
                outcome = None  # Track breadth expansion outcome.

                exp_start = time.perf_counter()

                if decision.action == "expand_breadth":
                    outcome = await expand(
                        state.seed_url,
                        query,
                        state.query_analysis,
                        conn,
                        already_ingested_urls=state.ingested_urls,
                        current_depth=state.current_depth,
                    )
                    state.ingested_urls.update(outcome.urls_ingested)
                    state.current_depth = outcome.depth

                    # Re-retrieve over expanded corpus.
                    state.current_chunks = await self._retrieve_and_rerank(
                        state.query_analysis.sub_queries,
                        conn,
                        state,
                        intent=intent,
                    )

                elif decision.action == "expand_recall":
                    # Re-retrieve with a doubled token budget to surface
                    # candidates that were just outside the original cutoff.
                    state.current_chunks = await self._retrieve_and_rerank(
                        state.query_analysis.sub_queries,
                        conn,
                        state,
                        intent=intent,
                        relaxed=True,
                    )

                elif decision.action == "expand_intent":
                    # Query decomposition didn't work well — re-analyze
                    # with feedback from the failed retrieval so the LLM
                    # (or rule engine) tries a different decomposition.
                    new_analysis = await analyze_query(
                        query,
                        intent=intent,
                        known_context=known_context,
                        constraints=constraints,
                        rewrite_context=(
                            f"Previous retrieval insufficient. "
                            f"Reason: {decision.reason}. "
                            f"Try alternative decomposition."
                        ),
                    )
                    state.query_analysis = new_analysis
                    state.current_chunks = await self._retrieve_and_rerank(
                        new_analysis.sub_queries,
                        conn,
                        state,
                        intent=intent,
                    )

                timing.expansion_ms += (time.perf_counter() - exp_start) * 1000

                # Re-evaluate.
                signals, decision = await evaluate(
                    state.current_chunks,
                    context_budget,
                    state.iteration,
                    state.current_signals,
                    state.query_analysis,
                )
                state.current_signals = signals

                # Log expansion step.
                state.expansion_steps.append(
                    ExpansionStep(
                        iteration=state.iteration,
                        depth=state.current_depth,
                        source_url=state.seed_url,
                        candidates_scored=(
                            outcome.candidates_scored if outcome else 0
                        ),
                        candidates_expanded=(
                            outcome.urls_ingested if outcome else []
                        ),
                        candidates_failed=(
                            outcome.urls_failed if outcome else []
                        ),
                        chunks_added=(
                            outcome.chunks_added if outcome else 0
                        ),
                        top_score_before=top_score_before,
                        top_score_after=signals.top_score,
                        decision=decision.action,
                        reason=decision.reason,
                    )
                )

            # ── Phase 5: Locality Expansion ───────────────────
            loc_start = time.perf_counter()
            if settings.locality_expansion_enabled:
                locality_chunks = await expand_locality(
                    state.current_chunks,
                    conn,
                    radius=settings.locality_expansion_radius,
                )
                if locality_chunks:
                    state.current_chunks.extend(locality_chunks)
            timing.locality_ms = (time.perf_counter() - loc_start) * 1000

            # ── Phase 6: Final Merge + Citation Assembly ──────
            merge_start = time.perf_counter()
            final_chunks = await merge_ranked_chunks(
                state.current_chunks, context_budget,
            )
            timing.merge_ms = (time.perf_counter() - merge_start) * 1000

            # Citations.
            citations = self._extract_citations(final_chunks)

            # Corpus stats from last retrieval.
            corpus_stats = self._aggregate_corpus_stats(
                state.all_retrieval_results,
            )

            # Determine mode.
            mode = "full_context"
            if state.all_retrieval_results:
                mode = state.all_retrieval_results[-1].mode

            timing.total_ms = (time.perf_counter() - total_start) * 1000

            return OrchestrationResult(
                chunks=final_chunks,
                citations=citations,
                query_analysis=state.query_analysis,
                expansion_steps=state.expansion_steps,
                corpus_stats=corpus_stats,
                timing=timing,
                mode=mode,
                final_decision=decision,
                total_iterations=state.iteration,
                total_urls_ingested=len(state.ingested_urls),
            )

        finally:
            await self._release_connection(conn)

    # ── Corpus preparation ────────────────────────────────────

    async def _ensure_ingested(
        self,
        url: str,
        state: OrchestrationState,
    ) -> None:
        """Ensure the seed URL is in the corpus (scrape + index if not)."""
        state.ingested_urls.add(url)

        # Check if documents table already has this URL.
        conn = await self._acquire_connection()
        try:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT EXISTS (SELECT 1 FROM documents WHERE source_url = %s)",
                    (url,),
                )
                row = await cur.fetchone()
                if row and row[0]:
                    return
        finally:
            await self._release_connection(conn)

        # Scrape and index.
        try:
            doc: NormalizedDocument = await ingest(url)
            await index_batch([doc], [0])  # Seed depth = 0.
        except Exception as exc:
            warnings.warn(
                f"Failed to ingest seed URL {url}: {exc!r}",
                stacklevel=2,
            )
            raise

    # ── Retrieve + rerank pipeline ────────────────────────────

    async def _retrieve_and_rerank(
        self,
        sub_queries: list[str],
        conn: AsyncConnection,
        state: OrchestrationState,
        *,
        intent: str | None = None,
        relaxed: bool = False,
    ) -> list[RankedChunk]:
        """Retrieve for each sub-query, rerank, merge.

        When *relaxed* is True the context budget is doubled to pull
        in more candidates before reranking.
        """
        budget_override = (
            settings.retrieval_context_budget * 2 if relaxed else None
        )

        # Source URLs: initially restrict to seed URL's document to avoid
        # noise from unrelated corpus content; after expansion starts,
        # open the search to all ingested documents.
        source_urls: list[str] | None = None
        if state.iteration == 0:
            source_urls = list(state.ingested_urls) if state.ingested_urls else None

        # Retrieve per sub-query (concurrently).
        async def _retrieve_one(sq: str) -> SubQueryResult:
            rr: RetrievalResult = await retrieve(
                conn,
                sq,
                source_urls=source_urls,
                context_budget_override=budget_override,
            )
            state.all_retrieval_results.append(rr)

            # Build passage texts for reranking.
            passages = [c.selected_text for c in rr.chunks]
            original_scores = [c.score for c in rr.chunks]

            # Build instruction for the reranker from intent.
            instruction = intent

            rerank_results = await rerank(
                sq,
                passages,
                instruction=instruction,
                original_scores=original_scores,
            )

            # Map rerank results back to RankedChunk.
            ranked: list[RankedChunk] = []
            for rr_item in rerank_results:
                if rr_item.index < len(rr.chunks):
                    chunk = rr.chunks[rr_item.index]
                    ranked.append(
                        RankedChunk(
                            chunk=chunk,
                            reranked_score=rr_item.relevance_score,
                            confidence=rr_item.confidence,
                            source_sub_query=sq,
                        )
                    )

            return SubQueryResult(
                sub_query=sq,
                retrieval_result=rr,
                reranked_chunks=ranked,
            )

        if len(sub_queries) == 1:
            results = [await _retrieve_one(sub_queries[0])]
        else:
            results = await asyncio.gather(
                *[_retrieve_one(sq) for sq in sub_queries]
            )

        # Merge sub-query results.
        context_budget = settings.retrieval_context_budget
        merged = await merge_subquery_results(list(results), context_budget)
        return merged

    # ── Citation extraction ───────────────────────────────────

    @staticmethod
    def _extract_citations(chunks: list[RankedChunk]) -> list[CitationSpan]:
        """Extract a citation from each chunk's full span."""
        citations: list[CitationSpan] = []
        for rc in chunks:
            c = rc.chunk
            citation = extract_citation(c, c.char_start, c.char_end)
            if citation is not None:
                citations.append(citation)
        return citations

    # ── Stats aggregation ─────────────────────────────────────

    @staticmethod
    def _aggregate_corpus_stats(
        retrieval_results: list[RetrievalResult],
    ) -> CorpusStats:
        """Aggregate corpus stats across all retrieval calls."""
        if not retrieval_results:
            return CorpusStats(
                total_documents=0,
                total_parent_chunks=0,
                total_tokens=0,
                documents_matched=[],
            )

        # Use the latest retrieval's stats as the most up-to-date.
        latest = retrieval_results[-1].corpus_stats

        # Aggregate documents_matched across all retrievals.
        all_docs: set[str] = set()
        for rr in retrieval_results:
            all_docs.update(rr.corpus_stats.documents_matched)

        return CorpusStats(
            total_documents=latest.total_documents,
            total_parent_chunks=latest.total_parent_chunks,
            total_tokens=latest.total_tokens,
            documents_matched=sorted(all_docs),
        )
