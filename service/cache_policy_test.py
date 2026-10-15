"""Tests for the provider-agnostic prompt-cache recency gate."""

from service import cache_policy


def _gate() -> cache_policy.CacheGate:
    """Builds a gate with Anthropic-like thresholds (explicit, no defaults)."""
    return cache_policy.CacheGate(freshness_window_s=300.0, idle_disable_after=3)


def test_gate_attaches_by_default() -> None:
    """Cold gates and brisk cadences always carry breakpoints."""
    gate = _gate()
    assert gate.should_cache(now=0.0) is True
    assert gate.should_cache(now=10.0) is True


def test_gate_survives_a_lone_idle_gap() -> None:
    """One slow gap still attaches; the next call usually reads the write."""
    gate = _gate()
    assert gate.should_cache(now=0.0) is True
    assert gate.should_cache(now=400.0) is True


def test_gate_suspends_on_sustained_idleness() -> None:
    """Three consecutive over-window gaps suspend; brisk cadence re-arms."""
    gate = _gate()
    assert gate.should_cache(now=0.0) is True
    assert gate.should_cache(now=400.0) is True
    assert gate.should_cache(now=800.0) is True
    assert gate.should_cache(now=1200.0) is False
    assert gate.should_cache(now=1210.0) is True


def test_gate_honors_custom_thresholds() -> None:
    """Providers with different TTL economics tune the counts."""
    gate = cache_policy.CacheGate(freshness_window_s=60.0, idle_disable_after=1)
    assert gate.should_cache(now=0.0) is True
    assert gate.should_cache(now=120.0) is False
    assert gate.should_cache(now=130.0) is True
