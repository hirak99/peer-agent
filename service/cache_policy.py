"""Recency gate for prompt-cache breakpoints (provider-agnostic).

Cache breakpoints both read the cache and (re)write it at a surcharge; a
write nobody re-reads before the TTL is pure cost. Breakeven needs only a
low hit probability (~0.2 for Anthropic's 1.25x writes / 0.05x reads), so
the gate attaches by default and suspends only on sustained idleness:
consecutive over-window gaps showing nobody is reading.

Thresholds are deliberately required, not defaulted: the freshness window
tracks the provider's cache TTL and the idle count tunes so that after N
consecutive misses P(next request hits) < breakeven (0.2083 on Anthropic).
Each backend declares its own pair.

Any backend with similar cache pricing can share this: construct a
CacheGate, call should_cache() once per request, and attach its own
breakpoint shapes when it returns True.
"""

import time


class CacheGate:
    """Per-backend request pacer deciding if cache breakpoints pay off."""

    def __init__(
        self,
        freshness_window_s: float,
        idle_disable_after: int,
    ) -> None:
        self._freshness_window_s = freshness_window_s
        self._idle_disable_after = idle_disable_after
        self._last_request_time: float | None = None
        self._idle_streak = 0

    def should_cache(self, now: float | None = None) -> bool:
        """Reports whether this request should carry cache breakpoints.

        True unless consecutive over-window gaps show sustained idleness;
        a lone slow gap still attaches, since the next call usually reads.
        Records every request so slow cadences naturally suspend caching.
        """
        current = time.monotonic() if now is None else now
        if (
            self._last_request_time is None
            or current - self._last_request_time <= self._freshness_window_s
        ):
            self._idle_streak = 0
        else:
            self._idle_streak += 1
        self._last_request_time = current
        return self._idle_streak < self._idle_disable_after
