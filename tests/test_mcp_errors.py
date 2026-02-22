"""Unit tests for src.mcp_server.errors."""

from __future__ import annotations

from src.mcp_server.errors import empty_results, full_failure, timeout


class TestFullFailure:
    def test_contains_error_header(self):
        msg = full_failure(RuntimeError("DB connection refused"))
        assert msg.startswith("[ERROR]")

    def test_contains_exception_type(self):
        msg = full_failure(ValueError("bad url"))
        assert "ValueError" in msg

    def test_contains_exception_message(self):
        msg = full_failure(ConnectionError("host unreachable"))
        assert "host unreachable" in msg

    def test_instructs_model_not_to_hallucinate(self):
        msg = full_failure(RuntimeError("oops"))
        assert "Do not attempt to answer from memory" in msg


class TestTimeout:
    def test_contains_seconds(self):
        msg = timeout(120)
        assert "120s" in msg

    def test_contains_error_header(self):
        msg = timeout(60)
        assert "[ERROR]" in msg

    def test_suggestion_present(self):
        msg = timeout(30)
        assert "expansion_budget=0" in msg

    def test_float_truncated(self):
        msg = timeout(120.7)
        assert "120s" in msg


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
