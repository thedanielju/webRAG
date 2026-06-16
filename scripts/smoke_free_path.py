"""Free / local-only end-to-end smoke test for WebRAG.

Proves the full ``answer`` pipeline runs with ZERO paid API keys, using only:
  - Ollama (``nomic-embed-text``, 768-dim) for embeddings,
  - the key-free raw-HTTP ingestion path,
  - rule-based query decomposition,
  - the no-op (passthrough) reranker.

It runs the SAME high-level flow the MCP ``answer`` tool uses
(``src/05_mcp_server/tools.py`` → ``OrchestratorEngine.run()`` →
``src/05_mcp_server/formatter.format_result``), then asserts the formatted
output carries at least one citation/evidence block sourced from the
ingested URL.  Exits non-zero if it does not.

Run it with the free profile env (see the "FREE / LOCAL-ONLY PROFILE" block
in ``blank.env``).  Example (PowerShell)::

    $env:DATABASE_URL="postgresql://webrag:webrag@localhost:5432/webrag_free"
    $env:EMBEDDING_BASE_URL="http://localhost:11434/v1"
    $env:EMBEDDING_API_KEY="ollama"
    $env:EMBEDDING_MODEL="nomic-embed-text"
    $env:EMBEDDING_DIMENSIONS="768"
    $env:RERANKER_PROVIDER="none"
    $env:DECOMPOSITION_MODE="rule_based"
    $env:INGESTION_PROVIDER="raw"
    .venv\\Scripts\\python.exe scripts\\smoke_free_path.py

IMPORTANT: the free path needs a FRESH database whose vector dimension
matches the model (768 for nomic-embed-text).  The default ``webrag``
database is built at 1536 and CANNOT hold 768-dim vectors; point
DATABASE_URL at a dedicated DB (e.g. ``webrag_free``).  init_schema() below
builds the vector column at ``EMBEDDING_DIMENSIONS`` on first run.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# ── Path + event-loop bootstrap (must precede project imports) ────────
# Make the repo root importable when run as ``python scripts/smoke_free_path.py``.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# psycopg3 AsyncConnection needs the SelectorEventLoop on Windows (the
# default ProactorEventLoop is incompatible).  The test suite sets this in
# conftest.py; a standalone script must do it itself.
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

from psycopg import connect as sync_connect

from config import settings
from src.indexing.schema import init_schema
from src.mcp_server.formatter import format_result
from src.orchestration.engine import OrchestratorEngine

# A small, deterministic, robots-friendly page: httpbin serves a fixed
# Moby-Dick excerpt at /html.  The raw fetcher extracts its <body> prose.
SEED_URL = "https://httpbin.org/html"
QUERY = "What is this text about?"


def _fail(message: str) -> None:
    print(f"\n[SMOKE FAILED] {message}", file=sys.stderr)
    sys.exit(1)


def _preflight_config() -> None:
    """Refuse to run if the env looks like the paid/default profile, so a
    stray key or the 1536 default DB can't silently corrupt the result."""
    problems: list[str] = []
    if not settings.database_url:
        problems.append("DATABASE_URL is not set.")
    if settings.embedding_dimensions != 768:
        problems.append(
            f"EMBEDDING_DIMENSIONS={settings.embedding_dimensions} (expected 768 "
            "for nomic-embed-text)."
        )
    if "11434" not in settings.embedding_base_url:
        problems.append(
            f"EMBEDDING_BASE_URL={settings.embedding_base_url!r} does not look like "
            "the local Ollama endpoint (http://localhost:11434/v1)."
        )
    if settings.ingestion_provider.lower() != "raw":
        problems.append(
            f"INGESTION_PROVIDER={settings.ingestion_provider!r} (expected 'raw' for "
            "the key-free fetch path)."
        )
    if settings.reranker_provider.lower() != "none":
        problems.append(
            f"RERANKER_PROVIDER={settings.reranker_provider!r} (expected 'none')."
        )
    if settings.decomposition_mode.lower() != "rule_based":
        problems.append(
            f"DECOMPOSITION_MODE={settings.decomposition_mode!r} (expected "
            "'rule_based')."
        )
    if problems:
        _fail(
            "free-profile preconditions not met — set the FREE / LOCAL-ONLY "
            "overrides via process env vars:\n  - " + "\n  - ".join(problems)
        )


def _normalize_localhost_dsn_for_windows() -> None:
    """Work around a Windows + Python 3.13 psycopg async-connect hang.

    On Windows the test suite/MCP server run on the SelectorEventLoop.  With
    Python 3.13, psycopg's *async* connection path stalls indefinitely while
    resolving the literal host ``localhost`` (which maps to both ``::1`` and
    ``127.0.0.1``); the synchronous path and the IP literal both connect
    instantly.  This is a host-environment quirk, NOT a WebRAG defect — the
    orchestration/ingestion code is untouched.

    To keep the documented invocation (DATABASE_URL=...@localhost:...) working
    out of the box, rewrite a bare ``localhost`` host to ``127.0.0.1`` for the
    duration of this run, and say so out loud so the substitution is visible.
    Non-Windows platforms and non-localhost hosts are left exactly as given.
    """
    if sys.platform != "win32":
        return
    from urllib.parse import urlsplit, urlunsplit

    parts = urlsplit(settings.database_url)
    if (parts.hostname or "").lower() != "localhost":
        return

    # Rebuild netloc with 127.0.0.1, preserving user:pass and :port.
    userinfo = ""
    if parts.username:
        userinfo = parts.username
        if parts.password:
            userinfo += f":{parts.password}"
        userinfo += "@"
    port = f":{parts.port}" if parts.port else ""
    new_netloc = f"{userinfo}127.0.0.1{port}"
    new_url = urlunsplit(
        (parts.scheme, new_netloc, parts.path, parts.query, parts.fragment)
    )
    print(
        "[note] Windows/py3.13 psycopg async-connect hangs on 'localhost'; "
        "using 127.0.0.1 for this run."
    )
    settings.database_url = new_url


def _init_schema() -> None:
    """Build the (768-dim) schema in the target DB if it isn't there yet.

    Uses a plain synchronous connection — init_schema() is sync DDL.  The
    vector column is created at settings.embedding_dimensions, so pointing
    DATABASE_URL at a fresh DB yields a 768-dim schema.
    """
    print(f"[1/4] Initialising schema in {settings.database_url}")
    print(f"      embedding_dimensions = {settings.embedding_dimensions}")
    with sync_connect(settings.database_url, autocommit=False) as conn:
        init_schema(conn)


async def _run_answer() -> str:
    print(f"[2/4] Ingesting + answering (raw fetch, local embeddings)")
    print(f"      seed = {SEED_URL}")
    print(f"      query = {QUERY!r}")
    engine = OrchestratorEngine()
    await engine.start()
    try:
        result = await engine.run(
            SEED_URL,
            QUERY,
            research_mode="fast",
            retrieval_mode="chunk",
            expansion_budget=0,
        )
    finally:
        await engine.stop()

    if not result.chunks:
        _fail(
            "orchestration returned no chunks — retrieval found nothing for the "
            "ingested page.  Check that the raw fetch produced content and that "
            "local embeddings succeeded."
        )

    print(f"[3/4] Formatting response ({len(result.chunks)} chunks)")
    return format_result(result, research_mode="fast", retrieval_mode="chunk")


def _verify_output(output: str) -> None:
    """Acceptance gate: the formatted answer must carry evidence/citations
    attributed to the ingested URL."""
    print("[4/4] Verifying citation / evidence presence")

    host = "httpbin.org"  # source_url may be the redirect-resolved final URL
    has_sources = "[SOURCES]" in output and host in output
    has_evidence = "[EVIDENCE]" in output and "(no evidence chunks)" not in output
    has_citations = "[CITATIONS]" in output

    if not has_sources:
        _fail(
            "no [SOURCES] entry attributed to the ingested host "
            f"({host}) found in the formatted output."
        )
    if not (has_evidence or has_citations):
        _fail(
            "formatted output has neither an [EVIDENCE] block nor a [CITATIONS] "
            "block — the answer is not grounded in the ingested page."
        )


def main() -> None:
    print("=" * 64)
    print("WebRAG free / local-only end-to-end smoke")
    print("=" * 64)
    _preflight_config()
    _normalize_localhost_dsn_for_windows()
    _init_schema()
    output = asyncio.run(_run_answer())

    print("\n" + "=" * 64)
    print("FORMATTED ANSWER")
    print("=" * 64)
    print(output)
    print("=" * 64 + "\n")

    _verify_output(output)
    print("[SMOKE PASSED] free/local-only answer returned grounded citations.")
    sys.exit(0)


if __name__ == "__main__":
    main()
