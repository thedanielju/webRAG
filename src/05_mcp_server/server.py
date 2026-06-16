"""WebRAG MCP server — entry point.

Creates the ``FastMCP`` instance, registers the lifespan context
manager (engine start/stop), registers tools, and runs the server
with the configured transport.

Transport modes:
  - ``stdio`` (default): Communicates over stdin/stdout.  Used by
    desktop MCP clients like Claude Desktop and Cursor.  The client
    launches this process as a subprocess.
  - ``streamable-http``: HTTP-based transport for remote/hosted
    deployments.  Listens on ``MCP_HOST:MCP_PORT``.

Logging constraint:
  MCP's stdio transport uses stdout for protocol messages.  ALL
  application logging MUST go to stderr to avoid corrupting the
  protocol stream.  ``_configure_logging()`` enforces this.

Usage::

    python -m src.mcp_server.server                   # stdio (default)
    python -m src.mcp_server.server --transport streamable-http
    MCP_TRANSPORT=streamable-http python -m src.mcp_server.server
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import secrets
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

    FastMCP's lifespan pattern (borrowed from Starlette/FastAPI):
      - The context manager runs once when the server starts.
      - The yielded dict becomes ``ctx.request_context.lifespan_context``
        in every tool handler — this is how tools access the engine.
      - On server shutdown, the finally block stops the engine and
        closes the database connection pool cleanly.
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
    """Build and configure the ``FastMCP`` instance.

    The ``instructions`` string is sent to the model on connection
    and teaches it what tools are available and when to use each one.
    Tool descriptions are intentionally verbose — they serve as the
    model's only documentation for deciding which tool to call.
    """
    mcp = FastMCP(
        "WebRAG",
        instructions=(
            "WebRAG indexes web pages and retrieves cited evidence. "
            "Use the 'answer' tool to scrape + search a URL. "
            "Use 'search' to query already-indexed content. "
            "Use 'status' to check what's indexed. "
            "Prefer fast passes first (chunked retrieval, no expansion by default)."
            "\n\n"
            "RESPONSE FORMAT REQUIREMENTS — follow these for every WebRAG tool result:\n"
            "1. CITATIONS: Every factual claim in your response MUST include an "
            "inline citation referencing the source, e.g. [Source 1] or [1]. Use "
            "the [SOURCES] and [CITATIONS] sections to attribute claims. Never "
            "omit citations.\n"
            "2. EXPANSION TRACE: If the tool output contains an [EXPANSION TRACE] "
            "section, reproduce the ASCII tree diagram in your response so the "
            "user can see which linked pages were crawled and the traversal path.\n"
            "3. FOLLOW-UP OPTIONS: If the tool output contains a [FOLLOW-UP OPTIONS] "
            "section, present those options to the user at the end of your "
            "response. If a deep recursive subtree search is suggested, ASK the "
            "user before running it — do not auto-expand.\n"
            "4. IMAGES: If the tool output contains an [IMAGES] section, include "
            "relevant image references in your response.\n"
            "5. Always synthesize a natural-language answer first, then include "
            "citations, traces, and follow-up options. Do not dump raw tool output."
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
            "evidence with source attribution. Default behavior is a fast "
            "chunked pass without expansion; use research_mode='deep' and/or "
            "retrieval_mode='full_context' only when the user requests a "
            "slower, more exhaustive search. Use this when you need "
            "factual information grounded in specific web sources.\n\n"
            "OUTPUT FORMAT: The response contains these sections:\n"
            "- [PRESENTATION GUIDE]: Instructions for formatting your response\n"
            "- [SOURCES]: Numbered source list — reference these as [Source N]\n"
            "- [EVIDENCE]: Ranked evidence chunks with relevance scores\n"
            "- [IMAGES]: Image links with alt text (when images are found)\n"
            "- [EXPANSION TRACE]: ASCII tree of crawled pages (when expansion occurred)\n"
            "- [CITATIONS]: Exact quotes for attribution — always include these\n"
            "- [STATS]: Performance and retrieval metadata\n"
            "- [FOLLOW-UP OPTIONS]: Suggested next steps to offer the user"
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
            "it skips ingestion and expansion.\n\n"
            "OUTPUT FORMAT: Same section structure as the answer tool "
            "([SOURCES], [EVIDENCE], [CITATIONS], [STATS], [FOLLOW-UP OPTIONS]). "
            "Always include citations and follow-up options in your response."
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
    """Route all logging to stderr.

    CRITICAL: stdout is reserved for the MCP JSON-RPC protocol stream.
    Any stray print() or stdout log would corrupt the protocol and
    crash the client connection.  This function ensures every logger
    in the process writes to stderr instead.

    The guard prevents duplicate handlers when ``main()`` is called
    more than once (e.g. in tests or interactive sessions).
    """
    root = logging.getLogger()
    root.setLevel(settings.mcp_log_level.upper())
    if root.handlers:
        return  # Already configured — avoid duplicate output.
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    root.addHandler(handler)


def _make_auth_middleware(token: str):
    """Build Starlette ASGI middleware that enforces Bearer token auth.

    Returns a middleware class (not an instance) suitable for passing
    to ``Starlette(middleware=...)``.  Requests without a valid
    ``Authorization: Bearer <token>`` header receive a 401 response.

    The comparison uses ``secrets.compare_digest`` to prevent
    timing-based side-channel attacks on the token value.
    """
    from starlette.middleware import Middleware
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.requests import Request
    from starlette.responses import Response

    class BearerAuthMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request: Request, call_next):
            auth_header = request.headers.get("authorization", "")
            if not auth_header.startswith("Bearer "):
                return Response("Unauthorized", status_code=401)
            provided = auth_header[7:]  # Strip "Bearer " prefix.
            if not secrets.compare_digest(provided, token):
                return Response("Unauthorized", status_code=401)
            return await call_next(request)

    return Middleware(BearerAuthMiddleware)


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

    if transport == "streamable-http" and settings.mcp_auth_token:
        # Wrap the Starlette app with bearer-token auth middleware,
        # then run with uvicorn directly instead of mcp.run().
        import uvicorn
        from starlette.applications import Starlette
        from starlette.routing import Mount

        inner_app = mcp.streamable_http_app()
        app = Starlette(
            routes=[Mount("/", app=inner_app)],
            middleware=[_make_auth_middleware(settings.mcp_auth_token)],
        )

        logger.info(
            "Auth enabled — listening on %s:%d",
            settings.mcp_host, settings.mcp_port,
        )
        uvicorn.run(
            app,
            host=settings.mcp_host,
            port=settings.mcp_port,
            log_level=settings.mcp_log_level.lower(),
        )
    else:
        if transport == "streamable-http":
            logger.info(
                "Listening on %s:%d (no auth — MCP_AUTH_TOKEN not set)",
                settings.mcp_host, settings.mcp_port,
            )
        mcp.run(transport=transport)  # type: ignore[arg-type]


if __name__ == "__main__":
    main()
