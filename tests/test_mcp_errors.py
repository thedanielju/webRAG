"""Unit tests for src.mcp_server.errors."""

from __future__ import annotations

import pytest

from mcp.server.fastmcp.exceptions import ToolError
from src.mcp_server.errors import empty_results, full_failure, timeout


class TestFullFailure:
    def test_raises_tool_error(self):
        with pytest.raises(ToolError):
            full_failure(RuntimeError("DB connection refused"))

    def test_contains_error_header(self):
        with pytest.raises(ToolError, match=r"\[ERROR\]"):
            full_failure(RuntimeError("DB connection refused"))

    def test_contains_exception_type(self):
        with pytest.raises(ToolError, match="ValueError"):
            full_failure(ValueError("bad url"))

    def test_contains_exception_message(self):
        with pytest.raises(ToolError, match="host unreachable"):
            full_failure(ConnectionError("host unreachable"))

    def test_instructs_model_not_to_hallucinate(self):
        with pytest.raises(ToolError, match="Do not attempt to answer from memory"):
            full_failure(RuntimeError("oops"))


class TestTimeout:
    def test_raises_tool_error(self):
        with pytest.raises(ToolError):
            timeout(120)

    def test_contains_seconds(self):
        with pytest.raises(ToolError, match="120s"):
            timeout(120)

    def test_contains_error_header(self):
        with pytest.raises(ToolError, match=r"\[ERROR\]"):
            timeout(60)

    def test_suggestion_present(self):
        with pytest.raises(ToolError, match="expansion_budget=0"):
            timeout(30)

    def test_float_truncated(self):
        with pytest.raises(ToolError, match="120s"):
            timeout(120.7)


class TestEmptyResults:
    def test_contains_sources_none(self):
        msg = empty_results(
            mode="chunk",
            documents_searched=3,
            chunks_evaluated=42,
            total_ms=450.0,
        )
        assert "[SOURCES]\n(none)" in msg

    def test_contains_stats(self):
        msg = empty_results(
            mode="chunk",
            documents_searched=3,
            chunks_evaluated=42,
            total_ms=450.0,
        )
        assert "Documents searched: 3" in msg
        assert "Chunks evaluated: 42" in msg

    def test_no_relevant_content_note(self):
        msg = empty_results(
            mode="full_context",
            documents_searched=1,
            chunks_evaluated=0,
            total_ms=100.0,
        )
        assert "No relevant content" in msg
