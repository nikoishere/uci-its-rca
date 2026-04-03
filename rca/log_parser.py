"""
Reverse-chunk log parser for large ActivitySim simulation logs.

Design:
  - Opens the file in binary mode and reads it backwards in fixed-size chunks
    so memory usage is O(chunk_size + context_window), not O(file_size).
  - A carry buffer merges partial lines that span chunk boundaries.
  - Stops scanning as soon as the failure region plus enough context is found,
    so we never touch the irrelevant early parts of a multi-hundred-MB log.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator

from config import settings
from rca.models import LogParseResult

# Ordered by specificity; first match wins.
FAILURE_MARKERS: list[tuple[str, str]] = [
    ("Traceback (most recent call last)", "Python Traceback"),
    ("MemoryError", "MemoryError"),
    ("KeyboardInterrupt", "KeyboardInterrupt"),
    ("Segmentation fault", "Segmentation Fault"),
    ("Killed", "OOM Kill"),
    ("FATAL", "Fatal"),
    ("Exception:", "Exception"),
    ("Error:", "Error"),
    ("ERROR", "Error"),
]

# Suppress false positives: lines that mention error keywords in comments or
# logger setup code rather than actual failures.
_NOISE_RE = re.compile(
    r"^\s*(#|logging\.(ERROR|WARNING)|logger\.(error|warning)|log\.(error|warning))",
    re.IGNORECASE,
)


class LogParser:
    def __init__(
        self,
        chunk_size: int = settings.CHUNK_SIZE,
        context_lines: int = settings.CONTEXT_LINES,
    ) -> None:
        self.chunk_size = chunk_size
        self.context_lines = context_lines

    # ── Public ────────────────────────────────────────────────────────────────

    def parse(self, log_path: Path) -> LogParseResult:
        """
        Scan log_path backwards and return context around the last failure.

        The returned context_window is in chronological order (oldest first)
        and contains up to context_lines lines before the failure plus all
        lines after it that were read during the backwards scan.
        """
        log_path = Path(log_path)
        if not log_path.exists():
            raise FileNotFoundError(f"Log not found: {log_path}")

        # collected[0] = last line of file, collected[-1] = earliest line read
        collected: list[str] = []
        failure_idx_in_collected: int | None = None
        failure_type = ""
        failure_line = ""

        for line in self._iter_lines_reverse(log_path):
            collected.append(line)

            if failure_idx_in_collected is None and not _NOISE_RE.search(line):
                for marker, label in FAILURE_MARKERS:
                    if marker in line:
                        failure_idx_in_collected = len(collected) - 1
                        failure_type = label
                        failure_line = line.strip()
                        break
            elif failure_idx_in_collected is not None:
                # lines_before counts how many chronologically-earlier lines
                # we have collected after finding the failure marker.
                lines_before = len(collected) - 1 - failure_idx_in_collected
                if lines_before >= self.context_lines:
                    break

        if failure_idx_in_collected is None:
            return LogParseResult(found=False, log_path=str(log_path))

        # Flip to chronological order.
        collected.reverse()
        # After reversal, the failure line is at this index.
        failure_idx = len(collected) - 1 - failure_idx_in_collected

        # Window: everything from (failure - context_lines) to end of collected.
        start = max(0, failure_idx - self.context_lines)
        context_window = collected[start:]

        stack_trace = self._extract_stack_trace(context_window, failure_idx - start)

        return LogParseResult(
            found=True,
            failure_type=failure_type,
            failure_line=failure_line,
            context_window=context_window,
            stack_trace=stack_trace,
            log_path=str(log_path),
        )

    # ── Private ───────────────────────────────────────────────────────────────

    def _iter_lines_reverse(self, path: Path) -> Iterator[str]:
        """
        Yield decoded lines from path in reverse order without loading
        the whole file into memory.

        Chunk boundary handling: the carry buffer holds the incomplete
        beginning of the current chunk and is prepended to the next
        (earlier) chunk so no line is silently truncated.
        """
        with open(path, "rb") as f:
            f.seek(0, 2)
            remaining = f.tell()
            carry = b""

            while remaining > 0:
                read_size = min(self.chunk_size, remaining)
                remaining -= read_size
                f.seek(remaining)
                # Prepend carry so the split line from the previous chunk
                # is completed before we split on newlines.
                chunk = f.read(read_size) + carry
                lines = chunk.split(b"\n")
                # lines[0] may be an incomplete line at the chunk boundary;
                # save it and continue in the next iteration.
                carry = lines[0]
                for line in reversed(lines[1:]):
                    yield line.decode("utf-8", errors="replace").rstrip("\r")

            if carry:
                yield carry.decode("utf-8", errors="replace").rstrip("\r")

    def _extract_stack_trace(self, lines: list[str], failure_idx: int) -> str:
        """
        Walk backwards from failure_idx looking for a Python traceback header.
        If found, collect from the header through the exception message line.
        Falls back to a tight window around failure_idx if no header is found.
        """
        traceback_start: int | None = None
        for i in range(failure_idx, -1, -1):
            if "Traceback (most recent call last)" in lines[i]:
                traceback_start = i
                break

        if traceback_start is None:
            start = max(0, failure_idx - 10)
            end = min(len(lines), failure_idx + 3)
            return "\n".join(lines[start:end])

        trace: list[str] = []
        i = traceback_start
        # Collect up to 80 lines; stop after the exception message line
        # (first non-indented, non-empty line that follows the header).
        while i < len(lines) and i < traceback_start + 80:
            trace.append(lines[i])
            if i > traceback_start and lines[i] and not lines[i][0].isspace():
                break
            i += 1

        return "\n".join(trace)
