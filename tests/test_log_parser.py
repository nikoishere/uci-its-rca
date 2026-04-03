"""Tests for rca.log_parser — reverse-chunk log reader."""

from __future__ import annotations

from pathlib import Path

import pytest

from rca.log_parser import LogParser


class TestLogParserWithSampleLogs:
    """Integration-style tests using the sample log files."""

    def test_keyerror_detected(self, keyerror_log: Path) -> None:
        result = LogParser().parse(keyerror_log)
        assert result.found is True
        # Parser reads backwards; hits "KeyError:" before "Traceback" header
        assert "Error" in result.failure_type
        assert "trip_mode" in result.failure_line

    def test_oom_detected(self, oom_log: Path) -> None:
        result = LogParser().parse(oom_log)
        assert result.found is True
        assert result.failure_type == "MemoryError"
        assert "MemoryError" in result.stack_trace

    def test_missing_file_detected(self, missing_file_log: Path) -> None:
        result = LogParser().parse(missing_file_log)
        assert result.found is True
        assert (
            "FileNotFoundError" in result.failure_line
            or "Traceback" in result.failure_line
        )

    def test_clean_log_no_failure(self, clean_log: Path) -> None:
        result = LogParser().parse(clean_log)
        assert result.found is False
        assert result.failure_type == ""
        assert result.stack_trace == ""

    def test_log_path_stored(self, keyerror_log: Path) -> None:
        result = LogParser().parse(keyerror_log)
        assert str(keyerror_log) in result.log_path


class TestLogParserChunkBoundary:
    """Verify the carry-buffer logic works across chunk boundaries."""

    def test_tiny_chunk_size(self, keyerror_log: Path) -> None:
        """Use a very small chunk size to force many boundary crossings."""
        parser = LogParser(chunk_size=64, context_lines=10)
        result = parser.parse(keyerror_log)
        assert result.found is True
        assert "Error" in result.failure_type

    def test_context_lines_limit(self, keyerror_log: Path) -> None:
        parser = LogParser(context_lines=5)
        result = parser.parse(keyerror_log)
        assert result.found is True
        # context_window should have at most ~5 lines before failure + lines after
        assert len(result.context_window) <= 20


class TestLogParserEdgeCases:
    def test_nonexistent_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            LogParser().parse(tmp_path / "does_not_exist.log")

    def test_empty_file(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty.log"
        empty.write_text("")
        result = LogParser().parse(empty)
        assert result.found is False

    def test_single_error_line(self, tmp_path: Path) -> None:
        log = tmp_path / "one_line.log"
        log.write_text("ERROR: something went wrong\n")
        result = LogParser().parse(log)
        assert result.found is True
        assert result.failure_type == "Error"

    def test_noise_lines_are_skipped(self, tmp_path: Path) -> None:
        """Lines that are logger setup (e.g. logging.ERROR) should not trigger."""
        log = tmp_path / "noise.log"
        log.write_text(
            "logging.ERROR = 40\nlogger.error('test setup complete')\nINFO: all good\n"
        )
        result = LogParser().parse(log)
        assert result.found is False

    def test_stack_trace_extraction(self, keyerror_log: Path) -> None:
        result = LogParser().parse(keyerror_log)
        assert "Traceback (most recent call last)" in result.stack_trace
        assert "KeyError" in result.stack_trace
