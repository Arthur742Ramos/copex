"""Tests for circuit_breaker module."""

from __future__ import annotations

import time

import pytest

from copex.circuit_breaker import (
    DEFAULT_FALLBACK_CHAINS,
    ModelAwareBreaker,
    SlidingWindowBreaker,
)
from copex.exceptions import CircuitBreakerOpen

# ---------------------------------------------------------------------------
# SlidingWindowBreaker
# ---------------------------------------------------------------------------

class TestSlidingWindowBreaker:
    """Tests for SlidingWindowBreaker."""

    def test_init_defaults(self) -> None:
        b = SlidingWindowBreaker()
        assert b.window_size == 10
        assert b.threshold == 0.5
        assert b.cooldown_seconds == 60.0
        assert b.failure_rate == 0.0
        assert not b.is_open
        assert not b.is_half_open

    def test_init_custom(self) -> None:
        b = SlidingWindowBreaker(window_size=5, threshold=0.8, cooldown_seconds=30)
        assert b.window_size == 5
        assert b.threshold == 0.8
        assert b.cooldown_seconds == 30.0

    def test_invalid_threshold(self) -> None:
        with pytest.raises(ValueError, match="threshold"):
            SlidingWindowBreaker(threshold=0)
        with pytest.raises(ValueError, match="threshold"):
            SlidingWindowBreaker(threshold=1.5)
        with pytest.raises(ValueError, match="threshold"):
            SlidingWindowBreaker(threshold=-0.1)

    def test_invalid_window_size(self) -> None:
        with pytest.raises(ValueError, match="window_size"):
            SlidingWindowBreaker(window_size=0)
        with pytest.raises(ValueError, match="window_size"):
            SlidingWindowBreaker(window_size=-1)

    def test_invalid_cooldown_seconds(self) -> None:
        with pytest.raises(ValueError, match="cooldown_seconds"):
            SlidingWindowBreaker(cooldown_seconds=0)
        with pytest.raises(ValueError, match="cooldown_seconds"):
            SlidingWindowBreaker(cooldown_seconds=-5)

    def test_record_success(self) -> None:
        b = SlidingWindowBreaker(window_size=5)
        b.record_success()
        assert b.failure_rate == 0.0

    def test_record_failure(self) -> None:
        b = SlidingWindowBreaker(window_size=5)
        b.record_failure()
        assert b.failure_rate == 1.0

    def test_failure_rate_mixed(self) -> None:
        b = SlidingWindowBreaker(window_size=4)
        b.record_success()
        b.record_failure()
        b.record_success()
        b.record_failure()
        assert b.failure_rate == 0.5

    def test_opens_on_threshold(self) -> None:
        b = SlidingWindowBreaker(window_size=4, threshold=0.5, cooldown_seconds=10)
        b.record_failure()
        b.record_failure()
        # window_size // 2 == 2, so after 2 samples with 100% failure rate > 0.5
        assert b.is_open

    def test_does_not_open_below_min_samples(self) -> None:
        b = SlidingWindowBreaker(window_size=10, threshold=0.5, cooldown_seconds=10)
        b.record_failure()
        # Only 1 sample, need window_size // 2 = 5
        assert not b.is_open

    def test_check_raises_when_open(self) -> None:
        b = SlidingWindowBreaker(window_size=4, threshold=0.5, cooldown_seconds=100)
        b.record_failure()
        b.record_failure()
        b.record_failure()
        with pytest.raises(CircuitBreakerOpen, match="Circuit breaker open"):
            b.check()

    def test_check_passes_when_closed(self) -> None:
        b = SlidingWindowBreaker()
        b.check()  # Should not raise

    def test_half_open_after_cooldown(self) -> None:
        b = SlidingWindowBreaker(window_size=2, threshold=0.5, cooldown_seconds=0.01)
        b.record_failure()
        b.record_failure()
        assert b.is_open
        time.sleep(0.02)
        assert b.is_half_open
        assert not b.is_open

    def test_check_resets_after_cooldown(self) -> None:
        b = SlidingWindowBreaker(window_size=2, threshold=0.5, cooldown_seconds=0.01)
        b.record_failure()
        b.record_failure()
        time.sleep(0.02)
        b.check()  # Should not raise, resets state

    def test_success_in_half_open_closes_circuit(self) -> None:
        b = SlidingWindowBreaker(window_size=2, threshold=0.5, cooldown_seconds=0.01)
        b.record_failure()
        b.record_failure()
        time.sleep(0.02)
        b.record_success()
        assert not b.is_open
        assert not b.is_half_open

    def test_reset(self) -> None:
        b = SlidingWindowBreaker(window_size=2, threshold=0.5, cooldown_seconds=100)
        b.record_failure()
        b.record_failure()
        assert b.is_open
        b.reset()
        assert not b.is_open
        assert b.failure_rate == 0.0

    def test_sliding_window_evicts_old_entries(self) -> None:
        b = SlidingWindowBreaker(window_size=3, threshold=0.5)
        b.record_failure()
        b.record_failure()
        b.record_failure()
        assert b.failure_rate == 1.0
        # Push successes to evict failures
        b.record_success()
        b.record_success()
        b.record_success()
        assert b.failure_rate == 0.0


# ---------------------------------------------------------------------------
# ModelAwareBreaker
# ---------------------------------------------------------------------------

class TestModelAwareBreaker:
    """Tests for ModelAwareBreaker."""

    def test_per_model_isolation(self) -> None:
        mab = ModelAwareBreaker(window_size=2, threshold=0.5, cooldown_seconds=100)
        mab.record_failure("model-a")
        mab.record_failure("model-a")
        assert mab.is_open("model-a")
        assert not mab.is_open("model-b")

    def test_check_raises_for_open_model(self) -> None:
        mab = ModelAwareBreaker(window_size=2, threshold=0.5, cooldown_seconds=100)
        mab.record_failure("model-a")
        mab.record_failure("model-a")
        with pytest.raises(CircuitBreakerOpen):
            mab.check("model-a")
        mab.check("model-b")  # Should not raise

    def test_record_success(self) -> None:
        mab = ModelAwareBreaker()
        mab.record_success("model-a")
        assert not mab.is_open("model-a")

    def test_reset_single_model(self) -> None:
        mab = ModelAwareBreaker(window_size=2, threshold=0.5, cooldown_seconds=100)
        mab.record_failure("model-a")
        mab.record_failure("model-a")
        mab.record_failure("model-b")
        mab.record_failure("model-b")
        mab.reset("model-a")
        assert not mab.is_open("model-a")
        assert mab.is_open("model-b")

    def test_reset_all(self) -> None:
        mab = ModelAwareBreaker(window_size=2, threshold=0.5, cooldown_seconds=100)
        mab.record_failure("model-a")
        mab.record_failure("model-a")
        mab.reset()
        status = mab.get_status()
        assert len(status) == 0

    def test_get_status(self) -> None:
        mab = ModelAwareBreaker()
        mab.record_success("model-a")
        mab.record_failure("model-b")
        status = mab.get_status()
        assert "model-a" in status
        assert "model-b" in status
        assert status["model-a"]["failure_rate"] == 0.0
        assert status["model-b"]["failure_rate"] == 1.0

    def test_is_open_unknown_model(self) -> None:
        mab = ModelAwareBreaker()
        assert not mab.is_open("nonexistent")

    def test_get_available_model_preferred(self) -> None:
        mab = ModelAwareBreaker()
        result = mab.get_available_model("model-a", ["model-b"])
        assert result == "model-a"

    def test_get_available_model_fallback(self) -> None:
        mab = ModelAwareBreaker(window_size=2, threshold=0.5, cooldown_seconds=100)
        mab.record_failure("model-a")
        mab.record_failure("model-a")
        result = mab.get_available_model("model-a", ["model-b", "model-c"])
        assert result == "model-b"

    def test_get_available_model_all_open(self) -> None:
        mab = ModelAwareBreaker(window_size=2, threshold=0.5, cooldown_seconds=100)
        for m in ["model-a", "model-b"]:
            mab.record_failure(m)
            mab.record_failure(m)
        result = mab.get_available_model("model-a", ["model-b"])
        assert result is None

    def test_get_available_model_no_fallback(self) -> None:
        mab = ModelAwareBreaker(window_size=2, threshold=0.5, cooldown_seconds=100)
        mab.record_failure("model-a")
        mab.record_failure("model-a")
        result = mab.get_available_model("model-a")
        assert result is None


# ---------------------------------------------------------------------------
# DEFAULT_FALLBACK_CHAINS
# ---------------------------------------------------------------------------

def test_default_fallback_chains_exist() -> None:
    assert isinstance(DEFAULT_FALLBACK_CHAINS, dict)
    assert len(DEFAULT_FALLBACK_CHAINS) > 0
    # All values should be lists of strings
    for key, chain in DEFAULT_FALLBACK_CHAINS.items():
        assert isinstance(key, str)
        assert isinstance(chain, list)
        assert all(isinstance(m, str) for m in chain)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestSlidingWindowBreakerEdgeCases:
    """Edge case tests for SlidingWindowBreaker."""

    def test_check_resets_window_on_half_open(self) -> None:
        """After cooldown, check() should clear the window for fresh tracking."""
        b = SlidingWindowBreaker(window_size=4, threshold=0.5, cooldown_seconds=0.01)
        b.record_failure()
        b.record_failure()
        assert b.is_open
        time.sleep(0.02)
        b.check()  # Transition to half-open → closed
        assert len(b._window) == 0
        assert b.failure_rate == 0.0

    def test_threshold_exactly_one(self) -> None:
        """Threshold of 1.0 means only opens at 100% failure rate."""
        b = SlidingWindowBreaker(window_size=4, threshold=1.0, cooldown_seconds=100)
        b.record_failure()
        b.record_failure()
        b.record_success()  # Not 100%
        assert not b.is_open
        b.record_failure()
        # Still not 100%: 3/4 = 75%
        assert not b.is_open

    def test_window_size_one(self) -> None:
        """Minimum window size of 1 should work correctly."""
        b = SlidingWindowBreaker(window_size=1, threshold=1.0, cooldown_seconds=100)
        b.record_failure()
        assert b.is_open

    def test_rapid_success_failure_alternation(self) -> None:
        """Alternating success/failure should maintain correct rate."""
        b = SlidingWindowBreaker(window_size=4, threshold=0.5, cooldown_seconds=100)
        for _ in range(10):
            b.record_success()
            b.record_failure()
        assert b.failure_rate == 0.5

    def test_multiple_resets(self) -> None:
        """Multiple resets should be safe."""
        b = SlidingWindowBreaker(window_size=2, threshold=0.5, cooldown_seconds=100)
        b.reset()
        b.reset()
        assert not b.is_open
        assert b.failure_rate == 0.0


class TestModelAwareBreakerEdgeCases:
    """Edge case tests for ModelAwareBreaker."""

    def test_reset_nonexistent_model(self) -> None:
        """Resetting a model that was never used should be safe."""
        mab = ModelAwareBreaker()
        mab.reset("nonexistent")  # Should not raise

    def test_get_status_empty(self) -> None:
        """Status of empty breaker should be empty dict."""
        mab = ModelAwareBreaker()
        assert mab.get_status() == {}

    def test_fallback_skips_open_models(self) -> None:
        """Fallback should skip multiple open models."""
        mab = ModelAwareBreaker(window_size=2, threshold=0.5, cooldown_seconds=100)
        mab.record_failure("m-a")
        mab.record_failure("m-a")
        mab.record_failure("m-b")
        mab.record_failure("m-b")
        result = mab.get_available_model("m-a", ["m-b", "m-c"])
        assert result == "m-c"

    def test_check_creates_breaker_lazily(self) -> None:
        """Checking a new model should create its breaker."""
        mab = ModelAwareBreaker()
        mab.check("new-model")  # Should not raise
        assert "new-model" in mab._breakers
