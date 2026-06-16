"""PAID, live, deep-mode recursion smoke for WebRAG.

A deep-mode sibling of ``scripts/smoke_free_path.py``.  Where the free smoke
proves the pipeline runs key-free on a single page, THIS smoke proves the
*deep* research mode genuinely RECURSES across multiple link levels and
reports reachability coverage — the definition-of-done gate for recursion.

It SPENDS REAL MONEY: Firecrawl scrape + /map credits and OpenAI embedding /
decomposition tokens.  It is deliberately self-capped to single-digit
Firecrawl credits via the env overrides applied at the top of ``main()``:

  - MAX_PAGES_PER_ANSWER = 6        → at most ~6 page scrapes (seed + expansion)
  - MAX_CANDIDATES_PER_ITERATION=1  → 1 scrape per expansion round, so depth
                                      accrues across rounds rather than width
  - FIRECRAWL_MAP_DEFAULT_LIMIT=30  → caps the single /map round-trip
  - EXPANSION_MAP_LIMIT = 30        → caps per-doc link enrichment
  - MAX_EXPANSION_DEPTH = 4         → lets the evaluator drive up to 4 rounds

Everything else (embeddings, reranker, decomposition, ingestion provider)
is left at the repo ``.env`` PAID defaults — OpenAI 1536-dim embeddings,
ZeroEntropy reranker, LLM decomposition, Firecrawl ingestion.

Acceptance (the gate this smoke verifies)
-----------------------------------------
PASS requires BOTH coverage machinery and recursion machinery to have run,
in either of two shapes:

  STRONG PASS — full multi-level descent:
    * formatted output has a "[SEARCH] depth N" line with N >= 2, AND
    * a "[COVERAGE] indexed N of ~M reachable pages" line, AND
    * at least one citation.

  ACCEPTABLE PASS — machinery proven, descent stopped early on quality:
    * the loop ran >= 1 expansion iteration drawing from the reachable
      frontier (so it CAN descend), AND
    * a "[COVERAGE]" line is present (reachability mapped the universe), AND
    * at least one citation,
    * with the actual depth reached + stop reason reported honestly.

A hard error or zero-expansion run is a FAIL.

Run (PowerShell), from the repo root::

    .venv\\Scripts\\python.exe scripts\\smoke_paid_deep.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

# ── Path + event-loop bootstrap (must precede project imports) ────────
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# The formatted answer contains Unicode (box-drawing chars in the expansion
# trace, "§" section markers).  The Windows console defaults to cp1252, which
# can't encode them and raises UnicodeEncodeError on print().  Force UTF-8 on
# stdout/stderr so the full answer prints verbatim.
for _stream in (sys.stdout, sys.stderr):
    _reconfigure = getattr(_stream, "reconfigure", None)
    if _reconfigure is not None:
        _reconfigure(encoding="utf-8", errors="replace")

# psycopg3 AsyncConnection needs the SelectorEventLoop on Windows (the
# default ProactorEventLoop is incompatible).  Mirror smoke_free_path.py.
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    # Tolerate stale socket fds during selector shutdown (WinError 10038),
    # the same harmless teardown race the test suite suppresses.
    import selectors as _selectors

    _original_select = _selectors.SelectSelector._select

    def _safe_select(self, r, w, _, timeout=None):  # type: ignore[override]
        try:
            return _original_select(self, r, w, _, timeout)
        except OSError:
            return [], [], []

    _selectors.SelectSelector._select = _safe_select  # type: ignore[assignment]


# ── Credit caps + DB target — set BEFORE importing config/settings ────
# pydantic-settings reads process env at import, but we also assign onto
# the live ``settings`` object after import as a belt-and-suspenders guard
# (the singleton may already be constructed by an earlier import).
_PAID_CAPS: dict[str, str] = {
    # Fresh, isolated DB so the 1536-dim OpenAI schema can't collide with
    # the 768-dim free smoke or the default 1536 ``webrag`` db.  127.0.0.1
    # (not 'localhost') dodges the Windows+py3.13 psycopg async-connect hang.
    "DATABASE_URL": "postgresql://webrag:webrag@127.0.0.1:5432/webrag_paid",
    # Hard page ceiling: seed + up to ~5 expansion scrapes.  The loop checks
    # this backstop at round entry, so it bounds total scrapes regardless of
    # how many rounds the evaluator wants.
    "MAX_PAGES_PER_ANSWER": "6",
    # One scrape per expansion round → depth accrues across rounds (genuine
    # N-level descent) instead of fanning out wide in a single round.
    "MAX_CANDIDATES_PER_ITERATION": "1",
    # Let the evaluator drive up to 4 expansion rounds before the depth cap.
    "MAX_EXPANSION_DEPTH": "4",
    # Cap the single reachability /map round-trip and per-doc link
    # enrichment so map credits stay tiny.
    "FIRECRAWL_MAP_DEFAULT_LIMIT": "30",
    "EXPANSION_MAP_LIMIT": "30",
}
for _k, _v in _PAID_CAPS.items():
    os.environ[_k] = _v

from config import settings  # noqa: E402

# Belt-and-suspenders: force the caps onto the live settings singleton in
# case it was constructed before the env was set above.
settings.database_url = _PAID_CAPS["DATABASE_URL"]
settings.max_pages_per_answer = int(_PAID_CAPS["MAX_PAGES_PER_ANSWER"])
settings.max_candidates_per_iteration = int(_PAID_CAPS["MAX_CANDIDATES_PER_ITERATION"])
settings.max_expansion_depth = int(_PAID_CAPS["MAX_EXPANSION_DEPTH"])
settings.firecrawl_map_default_limit = int(_PAID_CAPS["FIRECRAWL_MAP_DEFAULT_LIMIT"])
settings.expansion_map_limit = int(_PAID_CAPS["EXPANSION_MAP_LIMIT"])

from psycopg import connect as sync_connect  # noqa: E402

from src.indexing.schema import init_schema  # noqa: E402
from src.mcp_server.formatter import format_result  # noqa: E402
from src.orchestration.engine import OrchestratorEngine  # noqa: E402

# ── Target ────────────────────────────────────────────────────────────
# Mistune is a small, single-subdomain Python markdown-parser docs site.
# The landing page is a high-level overview; concrete API details (e.g. how
# custom plugins / directives are written, renderer internals) live on
# linked CHILD pages — so a specific implementation question is NOT fully
# answered by the seed and pushes the evaluator to expand into the reachable
# frontier.  Small origin → modest reachable_total → bounded /map cost.
SEED_URL = "https://mistune.lepture.com/"
QUERY = (
    "How do I write a custom directive plugin in mistune, and what methods "
    "must the directive class implement to parse and render it?"
)

RESEARCH_MODE = "deep"


def _section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def _preflight() -> None:
    """Refuse to run unless the PAID stack is actually configured.

    This is the inverse of the free smoke's preflight: we INSIST on the paid
    providers + 1536 dims + Firecrawl key, so a stray free override can't
    silently turn this into a no-cost (and non-representative) run.
    """
    problems: list[str] = []
    if settings.embedding_dimensions != 1536:
        problems.append(
            f"EMBEDDING_DIMENSIONS={settings.embedding_dimensions} (expected 1536 "
            "for the OpenAI paid embeddings)."
        )
    if "openai.com" not in settings.embedding_base_url:
        problems.append(
            f"EMBEDDING_BASE_URL={settings.embedding_base_url!r} is not the OpenAI "
            "endpoint — this smoke must exercise the paid embedding path."
        )
    if not settings.embedding_api_key:
        problems.append("EMBEDDING_API_KEY is not set (paid embeddings need it).")
    if not settings.firecrawl_api_key:
        problems.append(
            "FIRECRAWL_API_KEY is not set — deep-mode reachability needs Firecrawl."
        )
    if settings.ingestion_provider.lower() not in {"auto", "firecrawl"}:
        problems.append(
            f"INGESTION_PROVIDER={settings.ingestion_provider!r} (expected 'auto' or "
            "'firecrawl' so the Firecrawl scrape/map path runs)."
        )
    if not settings.reachability_enabled:
        problems.append(
            "REACHABILITY_ENABLED is false — deep-mode coverage mapping is disabled."
        )
    if "webrag_paid" not in settings.database_url:
        problems.append(
            f"DATABASE_URL={settings.database_url!r} is not the isolated webrag_paid db."
        )
    if problems:
        print("\n[SMOKE FAILED] paid-profile preconditions not met:", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        sys.exit(1)


def _init_schema() -> None:
    """Build the 1536-dim schema in webrag_paid if not already present."""
    _section("[1/4] Schema init")
    print(f"DB: {settings.database_url}")
    print(f"embedding_dimensions = {settings.embedding_dimensions}")
    with sync_connect(settings.database_url, autocommit=False) as conn:
        init_schema(conn)
    print("schema ready.")


async def _run() -> tuple[str, object]:
    _section("[2/4] Deep-mode ingest + recurse + answer (PAID)")
    print(f"seed   = {SEED_URL}")
    print(f"query  = {QUERY!r}")
    print(f"mode   = {RESEARCH_MODE}")
    print(
        f"caps   = max_pages={settings.max_pages_per_answer}, "
        f"max_candidates/iter={settings.max_candidates_per_iteration}, "
        f"max_depth={settings.max_expansion_depth}, "
        f"map_limit={settings.firecrawl_map_default_limit}"
    )

    engine = OrchestratorEngine()
    await engine.start()
    try:
        result = await engine.run(
            SEED_URL,
            QUERY,
            research_mode=RESEARCH_MODE,
            retrieval_mode="chunk",
            # expansion_budget=None → engine uses max_expansion_depth as the
            # iteration cap, letting the evaluator drive genuine multi-round
            # descent rather than forcing a fixed budget.
            expansion_budget=None,
        )
    finally:
        await engine.stop()

    _section("[3/4] Run summary (structured)")
    print(f"total_iterations   = {result.total_iterations}")
    print(f"max_depth_reached  = {result.max_depth_reached}")
    print(f"stop_reason        = {result.stop_reason}")
    print(f"total_urls_ingested= {result.total_urls_ingested}")
    print(f"reachable_total    = {result.reachable_total}")
    print(f"indexed_count      = {result.indexed_count}")
    print(f"coverage_ratio     = {result.coverage_ratio}")
    print(f"citations          = {len(result.citations)}")
    print(f"expansion_steps    = {len(result.expansion_steps)}")
    for i, step in enumerate(result.expansion_steps, 1):
        print(
            f"  step {i}: depth={step.depth} action={step.decision} "
            f"scored={step.candidates_scored} "
            f"ingested={step.candidates_expanded} "
            f"failed={step.candidates_failed} reason={step.reason!r}"
        )

    formatted = format_result(
        result, research_mode=RESEARCH_MODE, retrieval_mode="chunk"
    )
    return formatted, result


def _verdict(output: str, result: object) -> int:
    """Return process exit code: 0 = PASS, 1 = FAIL.

    Prints a clear PASS/FAIL line and the evidence behind it.
    """
    _section("[4/4] Verdict")

    depth = getattr(result, "max_depth_reached", 0)
    iterations = getattr(result, "total_iterations", 0)
    stop_reason = getattr(result, "stop_reason", "?")
    citations = len(getattr(result, "citations", []) or [])
    reachable_total = getattr(result, "reachable_total", None)

    has_search_line = "[SEARCH] depth" in output
    has_coverage_line = "[COVERAGE] indexed" in output
    has_citation = citations > 0 and "[CITATIONS]" in output

    print(f"depth reached      : {depth}")
    print(f"expansion rounds   : {iterations}")
    print(f"stop reason        : {stop_reason}")
    print(f"reachable_total    : {reachable_total}")
    print(f"citations present  : {has_citation} ({citations})")
    print(f"[SEARCH] line       : {has_search_line}")
    print(f"[COVERAGE] line     : {has_coverage_line}")

    # STRONG PASS: genuine multi-level descent with coverage + citation.
    strong = (
        has_search_line
        and depth >= 2
        and has_coverage_line
        and has_citation
    )
    # ACCEPTABLE PASS: machinery proven — >=1 expansion round drew from the
    # reachable frontier (coverage mapped), citation present — even if the
    # evaluator legitimately stopped before depth 2.
    acceptable = (
        iterations >= 1
        and has_coverage_line
        and reachable_total is not None
        and has_citation
    )

    if strong:
        print(
            f"\n[SMOKE PASSED — STRONG] deep mode recursed to depth {depth} "
            f"(>=2), reported coverage, and produced citations. "
            f"Stop reason: {stop_reason}."
        )
        return 0
    if acceptable:
        print(
            f"\n[SMOKE PASSED — ACCEPTABLE] recursion machinery proven: "
            f"{iterations} expansion round(s) drew from the reachable frontier "
            f"(reachable~{reachable_total}), coverage line present, citations "
            f"present. Descent reached depth {depth} before stopping "
            f"({stop_reason}); the loop CAN descend further on a target that "
            f"warrants it."
        )
        return 0

    # FAIL — say precisely what's missing.
    missing: list[str] = []
    if iterations < 1:
        missing.append("no expansion iteration ran (loop never descended)")
    if not has_coverage_line:
        missing.append("no [COVERAGE] line (reachability did not map / report)")
    if reachable_total is None:
        missing.append("reachable_total is None (deep map did not populate)")
    if not has_citation:
        missing.append("no citations (answer not grounded)")
    if not has_search_line:
        missing.append("no [SEARCH] depth line (no expansion steps recorded)")
    print(
        "\n[SMOKE FAILED] deep-mode recursion gate not met:\n  - "
        + "\n  - ".join(missing)
    )
    return 1


def main() -> None:
    _section("WebRAG PAID deep-mode recursion smoke")
    print("WARNING: this run spends real Firecrawl + OpenAI credits.")
    _preflight()
    _init_schema()
    formatted, result = asyncio.run(_run())

    _section("FULL FORMATTED ANSWER")
    print(formatted)

    code = _verdict(formatted, result)
    sys.exit(code)


if __name__ == "__main__":
    main()
