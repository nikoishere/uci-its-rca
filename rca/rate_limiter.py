"""
Thread-safe token-bucket rate limiter.

Used to cap outgoing OpenAI API calls so the agent plays well on
shared lab machines that may have organisation-level RPM quotas.
"""

from __future__ import annotations

import threading
import time


class TokenBucket:
    """
    Classic token-bucket algorithm.

    Tokens refill at `rate_per_minute / 60` per second up to `burst`
    (defaults to rate_per_minute, i.e. a one-minute burst window).
    """

    def __init__(
        self,
        rate_per_minute: float,
        burst: float | None = None,
    ) -> None:
        self._rate = rate_per_minute / 60.0
        self._capacity = burst if burst is not None else rate_per_minute
        self._tokens = self._capacity
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()

    # ── Public ────────────────────────────────────────────────────────────────

    def consume(self, tokens: float = 1.0) -> bool:
        """Try to consume tokens. Returns True on success, False if empty."""
        with self._lock:
            self._refill()
            if self._tokens >= tokens:
                self._tokens -= tokens
                return True
            return False

    def wait_and_consume(
        self, tokens: float = 1.0, poll_interval: float = 0.25
    ) -> None:
        """Block until tokens are available, then consume them."""
        while not self.consume(tokens):
            time.sleep(poll_interval)

    # ── Private ───────────────────────────────────────────────────────────────

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self._capacity, self._tokens + elapsed * self._rate)
        self._last_refill = now
