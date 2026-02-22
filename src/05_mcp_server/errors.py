"""Error response templates for MCP tool calls.

Each function returns a pre-formatted error string that the reasoning
model can read.  The text instructs the model NOT to hallucinate —
it should report the error to the user instead.
"""

from __future__ import annotations


def full_failure(exc: BaseException) -> str:
    """Format a complete orchestration failure."""
    return (
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


def timeout(seconds: int | float) -> str:
    """Format a timeout error."""
    return (
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
    """Format a response when orchestration returns zero chunks."""
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
