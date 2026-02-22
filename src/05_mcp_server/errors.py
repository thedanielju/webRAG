"""Error response templates for MCP tool calls.

Each function returns a pre-formatted error string that the reasoning
model (Claude, GPT, etc.) can read.  The text explicitly instructs
the model NOT to hallucinate an answer — it should report the error
to the user instead.

Error responses are raised as ``ToolError`` (from FastMCP) so the
MCP protocol marks them with ``is_error=True``.  This tells the
reasoning model the tool call failed, preventing it from treating
the error text as valid evidence.

Design rationale:
  MCP tools return plain text, not exceptions.  The reasoning model
  treats the tool output as context for its response.  If we returned
  a terse "500 Internal Server Error", the model might ignore it and
  fabricate an answer from training data.  By returning a structured
  [ERROR] block with diagnostic detail and a clear "do not answer
  from memory" instruction, we keep the model honest and give the
  user actionable feedback.
"""

from __future__ import annotations

from typing import NoReturn

from mcp.server.fastmcp.exceptions import ToolError


def full_failure(exc: BaseException) -> NoReturn:
    """Format a complete orchestration failure.

    Includes the exception class and message so the model can relay
    specifics to the user (e.g. "ConnectionRefusedError: ...").
    Raises ``ToolError`` so the MCP response has ``is_error=True``.
    """
    raise ToolError(
        "[ERROR]\n"
        "WebRAG encountered an error during retrieval.\n"
        "\n"
        f"Error type: {type(exc).__name__}\n"
        f"Details: {exc}\n"
        "\n"
        "The knowledge retrieval service was unable to complete this query. "
        "Possible causes:\n"
        "- The URL may be unreachable or return an error.\n"
        "- An external API (embedding, reranker) may be temporarily unavailable.\n"
        "- The database connection may have failed.\n"
        "\n"
        "Please inform the user of this error. "
        "Do not attempt to answer from memory — "
        "the tool was called because specific source material was needed."
    )


def timeout(seconds: int | float) -> NoReturn:
    """Format a timeout error.

    Suggests concrete remedies (skip expansion, use a narrower URL)
    so the model can offer the user a path forward.
    Raises ``ToolError`` so the MCP response has ``is_error=True``.
    """
    raise ToolError(
        "[ERROR]\n"
        f"WebRAG timed out after {int(seconds)}s.\n"
        "\n"
        "The orchestration pipeline did not complete within the allowed time. "
        "This may indicate:\n"
        "- A very large page requiring extensive indexing.\n"
        "- Many expansion iterations with slow external APIs.\n"
        "- Network latency to external services.\n"
        "\n"
        "Suggestion: Retry with expansion_budget=0 to skip expansion, "
        "or try a more specific URL."
    )


def empty_results(
    *,
    mode: str,
    documents_searched: int,
    chunks_evaluated: int,
    total_ms: float,
) -> str:
    """Format a response when orchestration returns zero chunks.

    Uses the same [SOURCES]/[EVIDENCE]/[STATS] section structure as
    successful responses so the model's parsing logic doesn't need
    a separate code path for "empty but not broken" results.
    """
    return (
        "[SOURCES]\n"
        "(none)\n"
        "\n"
        "[EVIDENCE]\n"
        "No relevant content was found for this query in the specified URL.\n"
        "\n"
        "[STATS]\n"
        f"Mode: {mode}\n"
        f"Documents searched: {documents_searched}\n"
        f"Chunks evaluated: {chunks_evaluated}\n"
        "Expansion iterations: 0\n"
        "Stop reason: No chunks above relevance threshold.\n"
        f"Total time: {total_ms:.0f}ms"
    )
