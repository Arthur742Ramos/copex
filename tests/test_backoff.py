"""Tests for backoff module."""

from __future__ import annotations

import asyncio

import pytest

from copex.backoff import (
    AdaptiveRetry,
    BackoffStrategy,
    ErrorCategory,
    RetryState,
    categorize_error,
    with_retry,
)
from copex.exceptions import RateLimitError, RetryError


# ---------------------------------------------------------------------------
# BackoffStrategy
# ---------------------------------------------------------------------------

class TestBackoffStrategy:
    """Tests for BackoffStrategy dataclass."""

    def test_defaults(self) -> None:
        s = BackoffStrategy()
        assert s.base_delay == 1.0
        assert s.max_delay == 60.0
        assert s.multiplier == 2.0
        assert s.jitter == 0.1
        assert s.max_retries == 5
        assert s.retryable is True

    def test_invalid_base_delay(self) -> None:
        with pytest.raises(ValueError, match="base_delay"):
            BackoffStrategy(base_delay=-1)

    def test_invalid_max_delay(self) -> None:
        with pytest.raises(ValueError, match="max_delay"):
            BackoffStrategy(max_delay=-1)

    def test_invalid_multiplier(self) -> None:
        with pytest.raises(ValueError, match="multiplier"):
            BackoffStrategy(multiplier=-1)

    def test_invalid_jitter_low(self) -> None:
        with pytest.raises(ValueError, match="jitter"):
            BackoffStrategy(jitter=-0.1)

    def test_invalid_jitter_high(self) -> None:
        with pytest.raises(ValueError, match="jitter"):
            BackoffStrategy(jitter=1.5)

    def test_invalid_max_retries(self) -> None:
        with pytest.raises(ValueError, match="max_retries"):
            BackoffStrategy(max_retries=-1)

    def test_compute_delay_first_attempt(self) -> None:
        s = BackoffStrategy(base_delay=1.0, multiplier=2.0, jitter=0)
        assert s.compute_delay(1) == 1.0

    def test_compute_delay_exponential(self) -> None:
        s = BackoffStrategy(base_delay=1.0, multiplier=2.0, jitter=0)
        assert s.compute_delay(1) == 1.0
        assert s.compute_delay(2) == 2.0
        assert s.compute_delay(3) == 4.0
        assert s.compute_delay(4) == 8.0

    def test_compute_delay_capped(self) -> None:
        s = BackoffStrategy(base_delay=1.0, multiplier=2.0, max_delay=5.0, jitter=0)
        assert s.compute_delay(10) == 5.0

    def test_compute_delay_with_jitter(self) -> None:
        s = BackoffStrategy(base_delay=10.0, multiplier=1.0, jitter=0.5)
        delays = [s.compute_delay(1) for _ in range(50)]
        # All delays should be within [5, 15]
        assert all(5.0 <= d <= 15.0 for d in delays)
        # With 50 samples, we should see some variation
        assert min(delays) < max(delays)

    def test_compute_delay_never_negative(self) -> None:
        s = BackoffStrategy(base_delay=0.01, multiplier=1.0, jitter=1.0)
        delays = [s.compute_delay(1) for _ in range(100)]
        assert all(d >= 0 for d in delays)

    def test_zero_jitter(self) -> None:
        s = BackoffStrategy(base_delay=1.0, jitter=0)
        assert s.compute_delay(1) == 1.0

    def test_zero_base_delay(self) -> None:
        s = BackoffStrategy(base_delay=0, jitter=0)
        assert s.compute_delay(1) == 0.0

    def test_large_attempt_capped(self) -> None:
        s = BackoffStrategy(base_delay=1.0, multiplier=2.0, max_delay=60.0, jitter=0)
        # Even with attempt=100, should be capped at max_delay
        assert s.compute_delay(100) == 60.0


# ---------------------------------------------------------------------------
# categorize_error
# ---------------------------------------------------------------------------

class TestCategorizeError:
    """Tests for error categorization."""

    def test_rate_limit_error_type(self) -> None:
        assert categorize_error(RateLimitError("too many")) == ErrorCategory.RATE_LIMIT

    def test_rate_limit_message(self) -> None:
        assert categorize_error(Exception("rate limit exceeded")) == ErrorCategory.RATE_LIMIT
        assert categorize_error(Exception("HTTP 429 too many requests")) == ErrorCategory.RATE_LIMIT

    def test_network_error_type(self) -> None:
        assert categorize_error(ConnectionError("refused")) == ErrorCategory.NETWORK
        assert categorize_error(TimeoutError("timed out")) == ErrorCategory.NETWORK

    def test_network_error_message(self) -> None:
        assert categorize_error(Exception("connection refused")) == ErrorCategory.NETWORK
        assert categorize_error(Exception("socket timeout")) == ErrorCategory.NETWORK

    def test_auth_error_message(self) -> None:
        assert categorize_error(Exception("HTTP 401 unauthorized")) == ErrorCategory.AUTH
        assert categorize_error(Exception("HTTP error 403 forbidden")) == ErrorCategory.AUTH

    def test_server_error_message(self) -> None:
        assert categorize_error(Exception("HTTP status 500")) == ErrorCategory.SERVER
        assert categorize_error(Exception("internal server error")) == ErrorCategory.SERVER
        assert categorize_error(Exception("service unavailable")) == ErrorCategory.SERVER

    def test_client_error_message(self) -> None:
        assert categorize_error(Exception("HTTP error 400")) == ErrorCategory.CLIENT
        assert categorize_error(Exception("HTTP status 404")) == ErrorCategory.CLIENT

    def test_unknown_error(self) -> None:
        assert categorize_error(Exception("something weird")) == ErrorCategory.UNKNOWN

    def test_no_false_positive_line_500(self) -> None:
        # "line 500" should not match as a server error
        result = categorize_error(Exception("error at line 500"))
        assert result != ErrorCategory.SERVER

    def test_status_code_429(self) -> None:
        exc = RuntimeError("error")
        exc.status_code = 429  # type: ignore[attr-defined]
        assert categorize_error(exc) == ErrorCategory.RATE_LIMIT

    def test_status_code_401(self) -> None:
        exc = RuntimeError("error")
        exc.status_code = 401  # type: ignore[attr-defined]
        assert categorize_error(exc) == ErrorCategory.AUTH

    def test_status_code_500(self) -> None:
        exc = RuntimeError("error")
        exc.status_code = 500  # type: ignore[attr-defined]
        assert categorize_error(exc) == ErrorCategory.SERVER

    def test_status_code_404(self) -> None:
        exc = RuntimeError("error")
        exc.status_code = 404  # type: ignore[attr-defined]
        assert categorize_error(exc) == ErrorCategory.CLIENT

    def test_status_code_non_int_ignored(self) -> None:
        exc = RuntimeError("error")
        exc.status_code = "429"  # type: ignore[attr-defined]
        assert categorize_error(exc) == ErrorCategory.UNKNOWN


# ---------------------------------------------------------------------------
# RetryState
# ---------------------------------------------------------------------------

class TestRetryState:
    """Tests for RetryState tracking."""

    def test_initial_state(self) -> None:
        s = RetryState()
        assert s.attempt == 0
        assert s.total_delay == 0.0
        assert len(s.errors) == 0

    def test_record_error(self) -> None:
        s = RetryState()
        err = Exception("test")
        s.record_error(err, ErrorCategory.NETWORK)
        assert s.attempt == 1
        assert len(s.errors) == 1
        assert s.category_counts[ErrorCategory.NETWORK] == 1

    def test_record_delay(self) -> None:
        s = RetryState()
        s.record_delay(1.5)
        s.record_delay(2.5)
        assert s.total_delay == 4.0

    def test_elapsed(self) -> None:
        s = RetryState()
        assert s.elapsed >= 0


# ---------------------------------------------------------------------------
# AdaptiveRetry
# ---------------------------------------------------------------------------

class TestAdaptiveRetry:
    """Tests for AdaptiveRetry handler."""

    @pytest.mark.asyncio
    async def test_success_no_retry(self) -> None:
        retry = AdaptiveRetry()
        async def _ok() -> str:
            return "ok"

        result = await retry.execute(_ok)
        assert result == "ok"

    @pytest.mark.asyncio
    async def test_success_on_first_try(self) -> None:
        async def operation() -> str:
            return "hello"

        retry = AdaptiveRetry()
        result = await retry.execute(operation)
        assert result == "hello"

    @pytest.mark.asyncio
    async def test_retry_then_success(self) -> None:
        call_count = 0

        async def operation() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("connection timeout")
            return "ok"

        retry = AdaptiveRetry(
            strategies={
                cat: BackoffStrategy(base_delay=0.01, max_delay=0.02, max_retries=5, jitter=0)
                for cat in ErrorCategory
            }
        )
        result = await retry.execute(operation)
        assert result == "ok"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_max_retries_exceeded(self) -> None:
        async def operation() -> str:
            raise Exception("connection timeout")

        retry = AdaptiveRetry(
            strategies={
                cat: BackoffStrategy(base_delay=0.01, max_delay=0.02, max_retries=2, jitter=0)
                for cat in ErrorCategory
            }
        )
        with pytest.raises(RetryError, match="Max retries"):
            await retry.execute(operation)

    @pytest.mark.asyncio
    async def test_non_retryable_error(self) -> None:
        async def operation() -> str:
            raise Exception("HTTP error 401 unauthorized")

        retry = AdaptiveRetry()
        with pytest.raises(RetryError, match="Non-retryable"):
            await retry.execute(operation)

    @pytest.mark.asyncio
    async def test_max_total_time(self) -> None:
        async def operation() -> str:
            raise Exception("connection timeout")

        retry = AdaptiveRetry(
            max_total_time=0.05,
            strategies={
                cat: BackoffStrategy(base_delay=0.02, max_delay=0.05, max_retries=100, jitter=0)
                for cat in ErrorCategory
            },
        )
        with pytest.raises(RetryError, match="Max total time"):
            await retry.execute(operation)

    @pytest.mark.asyncio
    async def test_on_retry_callback(self) -> None:
        callbacks: list[tuple[int, Exception, float]] = []
        call_count = 0

        async def operation() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise Exception("connection timeout")
            return "ok"

        retry = AdaptiveRetry(
            on_retry=lambda a, e, d: callbacks.append((a, e, d)),
            strategies={
                cat: BackoffStrategy(base_delay=0.01, max_delay=0.02, max_retries=5, jitter=0)
                for cat in ErrorCategory
            },
        )
        await retry.execute(operation)
        assert len(callbacks) == 1
        assert callbacks[0][0] == 1  # attempt

    @pytest.mark.asyncio
    async def test_rate_limit_retry_after_capped(self) -> None:
        call_count = 0

        async def operation() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise RateLimitError("rate limited", retry_after=9999)
            return "ok"

        retry = AdaptiveRetry(
            strategies={
                ErrorCategory.RATE_LIMIT: BackoffStrategy(
                    base_delay=0.01, max_delay=0.05, max_retries=5, jitter=0
                ),
                **{
                    cat: BackoffStrategy(base_delay=0.01, max_delay=0.05, max_retries=5, jitter=0)
                    for cat in ErrorCategory
                    if cat != ErrorCategory.RATE_LIMIT
                },
            }
        )
        # retry_after=9999 should be capped to max_delay=0.05
        result = await retry.execute(operation)
        assert result == "ok"

    @pytest.mark.asyncio
    async def test_retryable_exceptions_override(self) -> None:
        call_count = 0

        class CustomError(Exception):
            pass

        async def operation() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise CustomError("custom")
            return "ok"

        retry = AdaptiveRetry(
            strategies={
                cat: BackoffStrategy(base_delay=0.01, max_delay=0.02, max_retries=5, jitter=0)
                for cat in ErrorCategory
            }
        )
        result = await retry.execute(operation, retryable_exceptions=[CustomError])
        assert result == "ok"


# ---------------------------------------------------------------------------
# with_retry convenience function
# ---------------------------------------------------------------------------

class TestWithRetry:
    """Tests for the with_retry convenience function."""

    @pytest.mark.asyncio
    async def test_success(self) -> None:
        async def op() -> int:
            return 42

        result = await with_retry(op)
        assert result == 42

    @pytest.mark.asyncio
    async def test_retries_then_succeeds(self) -> None:
        call_count = 0

        async def op() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise Exception("connection error")
            return "done"

        result = await with_retry(op, base_delay=0.01, max_delay=0.02)
        assert result == "done"
