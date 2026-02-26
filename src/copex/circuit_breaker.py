"""Circuit breaker implementations for resilient request handling."""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from typing import Any

logger = logging.getLogger(__name__)

# Circuit breaker defaults
CB_FAILURE_THRESHOLD = 5
CB_COOLDOWN_SECONDS = 60.0


class SlidingWindowBreaker:
    """Circuit breaker using a sliding window of recent request outcomes.

    Unlike simple consecutive failure counting, this tracks a window of
    the last N requests and opens the circuit when the failure rate
    exceeds a threshold.

    Args:
        window_size: Number of recent requests to track (default: 10)
        threshold: Failure rate threshold to open circuit (default: 0.5 = 50%)
        cooldown_seconds: Seconds to wait before half-open state (default: 60)

    Example:
        breaker = SlidingWindowBreaker(window_size=10, threshold=0.5)
        breaker.check()  # Raises if circuit is open
        breaker.record_success()
        breaker.record_failure()
    """

    def __init__(
        self,
        window_size: int = 10,
        threshold: float = 0.5,
        cooldown_seconds: float = 60.0,
    ) -> None:
        if not 0 < threshold <= 1:
            raise ValueError("threshold must be between 0 and 1")
        if window_size < 1:
            raise ValueError("window_size must be >= 1")

        self.window_size = window_size
        self.threshold = threshold
        self.cooldown_seconds = cooldown_seconds

        # Sliding window: True = success, False = failure
        self._window: deque[bool] = deque(maxlen=window_size)
        self._opened_at: float | None = None
        self._lock = asyncio.Lock()

    @property
    def failure_rate(self) -> float:
        """Current failure rate in the sliding window."""
        if not self._window:
            return 0.0
        failures = sum(1 for success in self._window if not success)
        return failures / len(self._window)

    @property
    def is_open(self) -> bool:
        """Check if circuit is currently open."""
        if self._opened_at is None:
            return False
        elapsed = time.monotonic() - self._opened_at
        return elapsed < self.cooldown_seconds

    @property
    def is_half_open(self) -> bool:
        """Check if circuit is in half-open state (cooldown elapsed)."""
        if self._opened_at is None:
            return False
        elapsed = time.monotonic() - self._opened_at
        return elapsed >= self.cooldown_seconds

    def check(self) -> None:
        """Check circuit state; raise RuntimeError if open."""
        if self._opened_at is not None:
            elapsed = time.monotonic() - self._opened_at
            if elapsed < self.cooldown_seconds:
                remaining = self.cooldown_seconds - elapsed
                raise RuntimeError(
                    f"Circuit breaker open (failure rate {self.failure_rate:.0%}). "
                    f"Retry in {remaining:.0f}s."
                )
            # Cooldown elapsed - half-open: reset for retry
            self._opened_at = None
            self._window.clear()

    def record_success(self) -> None:
        """Record a successful request."""
        self._window.append(True)
        if self._opened_at is not None:
            # Successful request in half-open state closes the circuit
            self._opened_at = None

    def record_failure(self) -> None:
        """Record a failed request, potentially opening the circuit."""
        self._window.append(False)
        # Only evaluate after window has enough samples
        if len(self._window) >= self.window_size // 2:
            if self.failure_rate >= self.threshold:
                if self._opened_at is None:
                    self._opened_at = time.monotonic()
                    logger.warning(
                        "Circuit breaker opened: failure rate %.0f%% "
                        "exceeds threshold %.0f%%",
                        self.failure_rate * 100,
                        self.threshold * 100,
                    )

    def reset(self) -> None:
        """Reset the circuit breaker to closed state."""
        self._window.clear()
        self._opened_at = None


class ModelAwareBreaker:
    """Per-model circuit breakers for granular failure isolation.

    Different models may have different reliability characteristics.
    This class maintains separate circuit breakers per model.

    Args:
        window_size: Window size for each breaker
        threshold: Failure threshold for each breaker
        cooldown_seconds: Cooldown for each breaker

    Example:
        breakers = ModelAwareBreaker()
        breakers.check("gpt-5.2-codex")
        breakers.record_success("gpt-5.2-codex")
        breakers.record_failure("claude-opus-4.5")
    """

    def __init__(
        self,
        window_size: int = 10,
        threshold: float = 0.5,
        cooldown_seconds: float = 60.0,
    ) -> None:
        self.window_size = window_size
        self.threshold = threshold
        self.cooldown_seconds = cooldown_seconds
        self._breakers: dict[str, SlidingWindowBreaker] = {}

    def _get_breaker(self, model: str) -> SlidingWindowBreaker:
        """Get or create a circuit breaker for a model."""
        if model not in self._breakers:
            self._breakers[model] = SlidingWindowBreaker(
                window_size=self.window_size,
                threshold=self.threshold,
                cooldown_seconds=self.cooldown_seconds,
            )
        return self._breakers[model]

    def check(self, model: str) -> None:
        """Check if circuit is open for a model."""
        self._get_breaker(model).check()

    def record_success(self, model: str) -> None:
        """Record a successful request for a model."""
        self._get_breaker(model).record_success()

    def record_failure(self, model: str) -> None:
        """Record a failed request for a model."""
        self._get_breaker(model).record_failure()

    def reset(self, model: str | None = None) -> None:
        """Reset breaker(s). If model is None, reset all."""
        if model is None:
            self._breakers.clear()
        elif model in self._breakers:
            self._breakers[model].reset()

    def get_status(self) -> dict[str, dict[str, Any]]:
        """Get status of all breakers."""
        return {
            model: {
                "failure_rate": breaker.failure_rate,
                "is_open": breaker.is_open,
                "is_half_open": breaker.is_half_open,
                "window_size": len(breaker._window),
            }
            for model, breaker in self._breakers.items()
        }

    def is_open(self, model: str) -> bool:
        """Check if circuit is open for a model (without raising)."""
        breaker = self._breakers.get(model)
        return breaker.is_open if breaker else False

    def get_available_model(
        self,
        preferred: str,
        fallback_chain: list[str] | None = None,
    ) -> str | None:
        """Get the first available model from the fallback chain.

        If the preferred model's circuit is open, returns the first
        model in the fallback chain whose circuit is closed or
        half-open. Returns None if all circuits are open.

        Args:
            preferred: The preferred model to use
            fallback_chain: Optional list of fallback models in order

        Returns:
            The model to use, or None if all are unavailable

        Example:
            model = breaker.get_available_model(
                "claude-opus-4.5",
                fallback_chain=["claude-sonnet-4.5", "claude-haiku-4.5"]
            )
        """
        # Check preferred model first
        if not self.is_open(preferred):
            return preferred

        # Try fallback chain
        if fallback_chain:
            for fallback in fallback_chain:
                if not self.is_open(fallback):
                    logger.info(
                        "Circuit open for %s, falling back to %s",
                        preferred,
                        fallback,
                    )
                    return fallback

        return None


# Default fallback chains for common model families
DEFAULT_FALLBACK_CHAINS: dict[str, list[str]] = {
    # Claude family: opus -> sonnet -> haiku
    "claude-opus-4.6": ["claude-opus-4.6-fast", "claude-opus-4.6-1m", "claude-opus-4.5", "claude-sonnet-4.5", "claude-haiku-4.5"],
    "claude-opus-4.6-fast": ["claude-opus-4.6", "claude-opus-4.6-1m", "claude-opus-4.5", "claude-sonnet-4.5", "claude-haiku-4.5"],
    "claude-opus-4.6-1m": ["claude-opus-4.6", "claude-opus-4.6-fast", "claude-opus-4.5", "claude-sonnet-4.5", "claude-haiku-4.5"],
    "claude-opus-4.5": ["claude-sonnet-4.5", "claude-haiku-4.5"],
    "claude-sonnet-4.5": ["claude-sonnet-4", "claude-haiku-4.5"],
    "claude-sonnet-4": ["claude-haiku-4.5"],
    # GPT family: codex -> codex-max -> regular
    "gpt-5.2-codex": ["gpt-5.1-codex", "gpt-5.2", "gpt-5.1"],
    "gpt-5.1-codex": ["gpt-5.1-codex-max", "gpt-5.1", "gpt-5"],
    "gpt-5.2": ["gpt-5.1", "gpt-5"],
    "gpt-5.1": ["gpt-5", "gpt-5-mini"],
    "gpt-5": ["gpt-5-mini", "gpt-4.1"],
}
