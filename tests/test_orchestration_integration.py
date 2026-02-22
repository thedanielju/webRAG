"""Integration tests for orchestration — gated behind RUN_SLOW_INTEGRATION.

These tests exercise the full OrchestratorEngine with real APIs
(reranker, LLM, Postgres, embedding service).  They are NOT intended
for CI — run manually with:

    RUN_SLOW_INTEGRATION=1 pytest tests/test_orchestration_integration.py -v -s
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

RUN_SLOW_INTEGRATION = os.getenv("RUN_SLOW_INTEGRATION", "").strip() == "1"

OUT_DIR = Path("agent_documentation")


def _save_result(result, *, filename_prefix: str) -> None:
    """Persist an OrchestrationResult to JSON + human-readable summary."""
    OUT_DIR.mkdir(exist_ok=True)
    top_score = result.chunks[0].reranked_score if result.chunks else 0.0

    output_json = {
        "query": result.query_analysis.original_query,
        "sub_queries": result.query_analysis.sub_queries,
        "query_type": result.query_analysis.query_type,
        "total_iterations": result.total_iterations,
        "total_urls_ingested": result.total_urls_ingested,
        "num_chunks": len(result.chunks),
        "top_score": top_score,
        "timing": result.timing.model_dump(),
        "expansion_steps": [
            step.model_dump() for step in result.expansion_steps
        ],
        "final_decision": result.final_decision.model_dump(),
        "chunk_previews": [
            {
                "source_url": c.chunk.source_url,
                "reranked_score": c.reranked_score,
                "confidence": c.confidence,
                "text_preview": c.chunk.selected_text[:200],
                "is_locality_expanded": c.is_locality_expanded,
                "source_sub_query": c.source_sub_query,
            }
            for c in result.chunks[:10]
        ],
    }

    json_path = OUT_DIR / f"{filename_prefix}.json"
    json_path.write_text(json.dumps(output_json, indent=2, default=str), encoding="utf-8")

    summary_lines = [
        f"Orchestration Integration Test — {filename_prefix}",
        "=" * 60,
        f"Query: {result.query_analysis.original_query}",
        f"Sub-queries: {result.query_analysis.sub_queries}",
        f"Query type: {result.query_analysis.query_type}",
        f"Total iterations: {result.total_iterations}",
        f"Total URLs ingested: {result.total_urls_ingested}",
        f"Chunks returned: {len(result.chunks)}",
        f"Top score: {top_score:.4f}",
        "",
        "Timing breakdown:",
        f"  query_analysis_ms: {result.timing.query_analysis_ms:.0f}",
        f"  retrieval_ms:      {result.timing.retrieval_ms:.0f}",
        f"  reranking_ms:      {result.timing.reranking_ms:.0f}",
        f"  expansion_ms:      {result.timing.expansion_ms:.0f}",
        f"  locality_ms:       {result.timing.locality_ms:.0f}",
        f"  merge_ms:          {result.timing.merge_ms:.0f}",
        f"  total_ms:          {result.timing.total_ms:.0f}",
    ]

    if result.expansion_steps:
        summary_lines.append("")
        summary_lines.append("Expansion steps:")
        for step in result.expansion_steps:
            summary_lines.append(
                f"  iter {step.iteration}: "
                f"score {step.top_score_before:.3f} → {step.top_score_after:.3f}  "
                f"| {step.decision} | {step.duration_ms:.0f} ms | "
                f"+{step.chunks_added} chunks | {step.reason[:60]}"
            )

    summary_lines.append("")
    summary_lines.append(f"Final decision: {result.final_decision.action} — {result.final_decision.reason}")
    summary_lines.append("")
    summary_lines.append("Top 5 chunks:")
    for i, c in enumerate(result.chunks[:5]):
        summary_lines.append(
            f"  {i+1}. [{c.reranked_score:.3f}] {c.chunk.source_url} "
            f"— {c.chunk.selected_text[:80]}..."
        )

    txt_path = OUT_DIR / f"{filename_prefix}.txt"
    txt_path.write_text("\n".join(summary_lines), encoding="utf-8")

    print(f"\n✅ JSON saved to {json_path}")
    print(f"✅ Summary saved to {txt_path}")


@pytest.mark.skipif(
    not RUN_SLOW_INTEGRATION,
    reason="Slow integration test — set RUN_SLOW_INTEGRATION=1 to run.",
)
@pytest.mark.asyncio(loop_scope="session")
class TestOrchestrationEndToEnd:
    """Full end-to-end orchestration through OrchestratorEngine.run()."""

    async def test_full_run(self, async_db_conn):
        from src.orchestration.engine import OrchestratorEngine

        engine = OrchestratorEngine()
        await engine.start()

        try:
            result = await engine.run(
                url="https://scikit-learn.org/stable/modules/ensemble.html",
                query="How does gradient boosting work?",
                intent="how_to",
            )
        finally:
            await engine.stop()

        # ---- Basic assertions ----
        assert result is not None
        assert len(result.chunks) > 0, "Should return at least one chunk."
        assert result.query_analysis is not None
        assert result.query_analysis.original_query == "How does gradient boosting work?"
        assert result.timing.total_ms > 0
        assert result.timing.retrieval_ms > 0
        assert result.timing.reranking_ms >= 0

        # ---- Chunk quality ----
        top_score = result.chunks[0].reranked_score
        assert top_score > 0, "Top chunk should have positive reranked_score."

        # ---- Sorted by score desc ----
        scores = [c.reranked_score for c in result.chunks]
        assert scores == sorted(scores, reverse=True), "Chunks must be sorted desc."

        _save_result(result, filename_prefix="orchestration_integration_output")

    async def test_multi_iteration_deep_expansion(self, async_db_conn):
        """Force 2-3 expansion rounds with a narrow query against a
        link-rich page.  Validates per-iteration timing and score
        progression observability.
        """
        from src.orchestration.engine import OrchestratorEngine

        engine = OrchestratorEngine()
        await engine.start()

        try:
            result = await engine.run(
                # A hub page with many outbound links.
                url="https://scikit-learn.org/stable/modules/ensemble.html",
                # Intentionally narrow query unlikely to be fully satisfied by
                # the seed page alone — should trigger expansion.
                query="What is the mathematical derivation of the shrinkage parameter in stochastic gradient boosting and how does it interact with subsampling ratios?",
                intent="explanation",
                expansion_budget=3,
            )
        finally:
            await engine.stop()

        # ---- Core assertions ----
        assert result is not None
        assert len(result.chunks) > 0
        assert result.timing.total_ms > 0

        # ---- Timing breakdown populated ----
        assert result.timing.retrieval_ms > 0
        # reranking_ms should be populated if a real reranker was used.
        assert result.timing.reranking_ms >= 0

        # ---- Expansion steps have duration_ms ----
        for step in result.expansion_steps:
            assert step.duration_ms > 0, (
                f"Iteration {step.iteration} should have positive duration_ms"
            )
            assert step.top_score_after >= 0

        # ---- Score progression logged ----
        if result.expansion_steps:
            print("\n📊 Score progression:")
            for step in result.expansion_steps:
                delta = step.top_score_after - step.top_score_before
                print(
                    f"  iter {step.iteration}: "
                    f"{step.top_score_before:.3f} → {step.top_score_after:.3f} "
                    f"(Δ{delta:+.3f}) | {step.duration_ms:.0f} ms | "
                    f"{step.decision}"
                )

        _save_result(result, filename_prefix="orchestration_deep_expansion_output")
