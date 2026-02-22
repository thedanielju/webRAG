"""WebRAG MCP server — entry point.

Creates the ``FastMCP`` instance, registers the lifespan context
manager (engine start/stop), registers tools, and runs the server
with the configured transport.

Usage::

    python -m src.mcp_server.server                   # stdio (default)
    python -m src.mcp_server.server --transport streamable-http
    MCP_TRANSPORT=streamable-http python -m src.mcp_server.server
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from mcp.server.fastmcp import FastMCP

from config import settings
from src.orchestration.engine import OrchestratorEngine
from src.mcp_server.tools import answer, search, status

logger = logging.getLogger(__name__)


# ── Lifespan ──────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastMCP) -> AsyncIterator[dict[str, Any]]:
    """Manage the ``OrchestratorEngine`` lifecycle.

    The engine is created, started, and stored in the lifespan
    context dict so that tool handlers can access it via
    ``ctx.request_context.lifespan_context["engine"]``.
    """
    engine = OrchestratorEngine()
    await engine.start()
    logger.info("OrchestratorEngine started")
    try:
        yield {"engine": engine}
    finally:
        await engine.stop()
        logger.info("OrchestratorEngine stopped")


# ── Server factory ────────────────────────────────────────────


def create_server() -> FastMCP:
    """Build and configure the ``FastMCP`` instance."""
    mcp = FastMCP(
        "WebRAG",
        instructions=(
            "WebRAG indexes web pages and retrieves cited evidence. "
            "Use the 'answer' tool to scrape + search a URL. "
            "Use 'search' to query already-indexed content. "
            "Use 'status' to check what's indexed."
        ),
        lifespan=lifespan,
        host=settings.mcp_host,
        port=settings.mcp_port,
        log_level=settings.mcp_log_level.upper(),
    )

    # Register tools.
    mcp.add_tool(
        answer,
        name="answer",
        description=(
            "Query one or more web pages and their linked content for "
            "information. WebRAG will scrape each URL if not already "
            "indexed, decompose the query, retrieve and rerank relevant "
            "chunks, optionally expand to linked pages, and return cited "
            "evidence with source attribution. Use this when you need "
            "factual information grounded in specific web sources."
        ),
    )
    mcp.add_tool(
        search,
        name="search",
        description=(
            "Search the existing WebRAG corpus for information without "
            "scraping new pages or expanding to linked content. Use this "
            "when you know the content has already been indexed (e.g., "
            "from a previous answer call) and want to ask a different "
            "question about the same material. Faster than answer because "
            "it skips ingestion and expansion."
        ),
    )
    mcp.add_tool(
        status,
        name="status",
        description=(
            "Check what content WebRAG currently has indexed. Returns "
            "document count, total tokens, indexed URLs with titles, and "
            "last-fetched timestamps. Use this to understand what's "
            "available before deciding whether to call answer (which "
            "ingests new content) or search (which queries existing "
            "content)."
        ),
    )

    return mcp


# ── Entry point ───────────────────────────────────────────────


def _configure_logging() -> None:
    """Route all logging to stderr (stdout is reserved for MCP protocol)."""
    root = logging.getLogger()
    root.setLevel(settings.mcp_log_level.upper())
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    root.addHandler(handler)


def main() -> None:
    parser = argparse.ArgumentParser(description="WebRAG MCP Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "streamable-http"],
        default=settings.mcp_transport,
        help="MCP transport (default: %(default)s)",
    )
    args = parser.parse_args()

    _configure_logging()

    transport: str = args.transport
    mcp = create_server()

    logger.info("Starting WebRAG MCP server (transport=%s)", transport)
    if transport == "streamable-http":
        logger.info("Listening on %s:%d", settings.mcp_host, settings.mcp_port)

    mcp.run(transport=transport)  # type: ignore[arg-type]


if __name__ == "__main__":
    main()
