"""Tests for rca.rate_limiter — token bucket."""
from __future__ import annotations

import time

from rca.rate_limiter import TokenBucket


class TestTokenBucket:
    def test_initial_tokens_available(self) -> None:
        bucket = TokenBucket(rate_per_minute=60)
        assert bucket.consume() is True

    def test_burst_capacity(self) -> None:
        bucket = TokenBucket(rate_per_minute=60, burst=3)
        assert bucket.consume() is True
        assert bucket.consume() is True
        assert bucket.consume() is True
        assert bucket.consume() is False

    def test_refill(self) -> None:
        bucket = TokenBucket(rate_per_minute=6000, burst=1)
        assert bucket.consume() is True
        assert bucket.consume() is False
        # At 6000/min = 100/sec, sleeping 0.02s should refill ~2 tokens
        time.sleep(0.02)
        assert bucket.consume() is True

    def test_wait_and_consume(self) -> None:
        bucket = TokenBucket(rate_per_minute=6000, burst=1)
        bucket.consume()  # drain
        t0 = time.monotonic()
        bucket.wait_and_consume(poll_interval=0.01)
        elapsed = time.monotonic() - t0
        # Should have waited a tiny bit for refill
        assert elapsed < 1.0

    def test_consume_multiple_tokens(self) -> None:
        bucket = TokenBucket(rate_per_minute=60, burst=5)
        assert bucket.consume(3) is True
        assert bucket.consume(3) is False
        assert bucket.consume(2) is True
