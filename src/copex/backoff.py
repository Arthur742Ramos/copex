"""
Adaptive Retry Backoff - Intelligent retry with per-error-type backoff strategies.

Provides exponential backoff, jitter, and special handling for rate limits.
"""

from __future__ import annotations

import asyncio
import logging
import random
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from copex.exceptions import RateLimitError, RetryError

logger = logging.getLogger(__name__)


class ErrorCategory(str, Enum):
    """Categories of errors for backoff strategy selection."""

    RATE_LIMIT = "rate_limit"  # 429 / rate limit errors
    NETWORK = "network"  # Connection errors, timeouts
    SERVER = "server"  # 5xx server errors
    AUTH = "auth"  # Authentication errors (usually non-retryable)
    CLIENT = "client"  # 4xx client errors (usually non-retryable)
    TRANSIENT = "transient"  # Other transient errors
    UNKNOWN = "unknown"  # Unclassified errors


@dataclass
class BackoffStrategy:
    """Configuration for a backoff strategy."""

    base_delay: float = 1.0  # Base delay in seconds
    max_delay: float = 60.0  # Maximum delay cap
    multiplier: float = 2.0  # Exponential multiplier
    jitter: float = 0.1  # Random jitter factor (0-1)
    max_retries: int = 5  # Maximum retry attempts
    retryable: bool = True  # Whether errors of this type are retryable

    def compute_delay(self, attempt: int) -> float:
        """Compute delay for a given attempt number (1-indexed).

        Args:
            attempt: The attempt number (1 = first retry)

        Returns:
            Delay in seconds before next retry
        """
        # Exponential backoff: base * multiplier^(attempt-1)
        delay = self.base_delay * (self.multiplier ** (attempt - 1))

        # Apply cap
        delay = min(delay, self.max_delay)

        # Apply jitter
        if self.jitter > 0:
            jitter_range = delay * self.jitter
            delay += random.uniform(-jitter_range, jitter_range)

        return max(0, delay)


# Default strategies per error category
DEFAULT_STRATEGIES: dict[ErrorCategory, BackoffStrategy] = {
    ErrorCategory.RATE_LIMIT: BackoffStrategy(
        base_delay=5.0,  # Start with 5s for rate limits
        max_delay=120.0,  # Cap at 2 minutes
        multiplier=2.0,
        jitter=0.2,
        max_retries=10,  # More retries for rate limits
    ),
    ErrorCategory.NETWORK: BackoffStrategy(
        base_delay=1.0,
        max_delay=30.0,
        multiplier=2.0,
        jitter=0.3,
        max_retries=5,
    ),
    ErrorCategory.SERVER: BackoffStrategy(
        base_delay=2.0,
        max_delay=60.0,
        multiplier=2.0,
        jitter=0.2,
        max_retries=5,
    ),
    ErrorCategory.AUTH: BackoffStrategy(
        base_delay=1.0,
        max_delay=5.0,
        multiplier=1.0,
        max_retries=1,  # Auth errors usually shouldn't be retried
        retryable=False,
    ),
    ErrorCategory.CLIENT: BackoffStrategy(
        base_delay=1.0,
        max_delay=5.0,
        multiplier=1.0,
        max_retries=1,
        retryable=False,
    ),
    ErrorCategory.TRANSIENT: BackoffStrategy(
        base_delay=1.0,
        max_delay=30.0,
        multiplier=2.0,
        jitter=0.2,
        max_retries=5,
    ),
    ErrorCategory.UNKNOWN: BackoffStrategy(
        base_delay=1.0,
        max_delay=30.0,
        multiplier=2.0,
        jitter=0.3,
        max_retries=3,
    ),
}


# Compiled regex patterns for efficient error categorization (v1.9.0)
# Using context-aware patterns to avoid false positives (e.g., "line 500" won't match)
_RE_RATE_LIMIT = re.compile(
    r"\brate\s*limit|(?:^|http\s*|status\s*|code\s*|error\s*)429\b|too many requests", re.IGNORECASE
)
_RE_NETWORK = re.compile(r"\b(connection|timeout|network|refused|socket)\b", re.IGNORECASE)
_RE_AUTH = re.compile(
    r"(?:^|http\s*|status\s*|code\s*|error\s*)(401|403)\b|\bunauthorized\b|\bforbidden\b",
    re.IGNORECASE,
)
# Match HTTP 5xx status codes in HTTP context (not just any "500")
_RE_SERVER = re.compile(
    r"(?:^|http\s*|status\s*|code\s*|error\s*)(5\d{2})\b|internal server|service unavailable|bad gateway",
    re.IGNORECASE,
)
# Match HTTP 4xx status codes (excluding 401, 403, 429) in HTTP context
# Note: "not found" and "bad request" only match if they appear as HTTP error descriptions
_RE_CLIENT = re.compile(
    r"(?:^|http\s*|status\s*|code\s*|error\s*)(4(?:0[02-8]|1[0-79]|[2-9]\d))\b|"
    r"(?:^|http\s*|error\s*)(?:bad request|not found)\b",
    re.IGNORECASE,
)

# Sets for O(1) type name lookups
_NETWORK_TYPES = frozenset(["connection", "timeout", "network", "socket"])
_AUTH_TYPES = frozenset(["auth", "permission", "forbidden"])


def categorize_error(error: Exception) -> ErrorCategory:
    """Categorize an exception for backoff strategy selection.

    Uses compiled regex patterns for efficient matching and avoids
    false positives (e.g., "error at line 500" won't match SERVER).
    Optimized from O(100) iteration to O(1) regex matching (v1.9.0).

    Args:
        error: The exception to categorize

    Returns:
        The error category
    """
    # Fast path: check exception type first
    if isinstance(error, RateLimitError):
        return ErrorCategory.RATE_LIMIT

    error_type = type(error).__name__.lower()
    error_msg = str(error).lower()

    # Check type name for network/auth errors (O(1) set lookup)
    if any(x in error_type for x in _NETWORK_TYPES):
        return ErrorCategory.NETWORK
    if any(x in error_type for x in _AUTH_TYPES):
        return ErrorCategory.AUTH

    # Check for rate limit patterns
    if _RE_RATE_LIMIT.search(error_msg):
        return ErrorCategory.RATE_LIMIT

    # Check for network errors in message
    if _RE_NETWORK.search(error_msg):
        return ErrorCategory.NETWORK

    # Check for auth errors in message
    if _RE_AUTH.search(error_msg):
        return ErrorCategory.AUTH

    # Check for server errors (5xx)
    if _RE_SERVER.search(error_msg):
        return ErrorCategory.SERVER

    # Check for client errors (4xx excluding 401, 403, 429)
    if _RE_CLIENT.search(error_msg):
        return ErrorCategory.CLIENT

    # Default to unknown
    return ErrorCategory.UNKNOWN


@dataclass
class RetryState:
    """State tracking for retry attempts."""

    attempt: int = 0
    total_delay: float = 0.0
    errors: list[Exception] = field(default_factory=list)
    category_counts: dict[ErrorCategory, int] = field(default_factory=dict)
    started_at: float = field(default_factory=time.time)

    def record_error(self, error: Exception, category: ErrorCategory) -> None:
        """Record an error occurrence."""
        self.attempt += 1
        self.errors.append(error)
        self.category_counts[category] = self.category_counts.get(category, 0) + 1

    def record_delay(self, delay: float) -> None:
        """Record a delay that was applied."""
        self.total_delay += delay

    @property
    def elapsed(self) -> float:
        """Total elapsed time including delays."""
        return time.time() - self.started_at


class AdaptiveRetry:
    """Adaptive retry handler with per-error-type backoff.

    Usage:
        retry = AdaptiveRetry()

        @retry.wrap
        async def my_operation():
            # ... operation that might fail

        # Or manually:
        async with retry.context() as ctx:
            while True:
                try:
                    result = await operation()
                    break
                except Exception as e:
                    await ctx.handle_error(e)
    """

    def __init__(
        self,
        strategies: dict[ErrorCategory, BackoffStrategy] | None = None,
        *,
        max_total_time: float | None = None,
        on_retry: Callable[[int, Exception, float], None] | None = None,
    ) -> None:
        """Initialize adaptive retry handler.

        Args:
            strategies: Custom strategies per error category
            max_total_time: Maximum total time for all retries (None = unlimited)
            on_retry: Callback called before each retry (attempt, error, delay)
        """
        self.strategies = {**DEFAULT_STRATEGIES, **(strategies or {})}
        self.max_total_time = max_total_time
        self.on_retry = on_retry

    def get_strategy(self, category: ErrorCategory) -> BackoffStrategy:
        """Get the backoff strategy for an error category."""
        return self.strategies.get(category, self.strategies[ErrorCategory.UNKNOWN])

    async def execute(
        self,
        operation: Callable[[], Any],
        *,
        retryable_exceptions: Sequence[type[Exception]] | None = None,
    ) -> Any:
        """Execute an operation with adaptive retry.

        Args:
            operation: Async callable to execute
            retryable_exceptions: Additional exception types to retry

        Returns:
            The operation result

        Raises:
            RetryError: If all retries are exhausted
        """
        state = RetryState()
        retryable = set(retryable_exceptions or [])

        while True:
            try:
                return await operation()
            except Exception as e:  # Catch-all: retry handler must intercept any error
                category = categorize_error(e)
                strategy = self.get_strategy(category)
                state.record_error(e, category)

                # Check if retryable
                should_retry = strategy.retryable or type(e) in retryable

                if not should_retry:
                    logger.debug(f"Non-retryable error category: {category}")
                    raise RetryError(
                        f"Non-retryable error: {e}",
                        attempts=state.attempt,
                        last_error=e,
                    ) from e

                # Check max retries for this category
                category_attempts = state.category_counts[category]
                if category_attempts > strategy.max_retries:
                    logger.debug(f"Max retries exceeded for {category}")
                    raise RetryError(
                        f"Max retries ({strategy.max_retries}) exceeded for {category}",
                        attempts=state.attempt,
                        last_error=e,
                    ) from e

                # Check total time
                if self.max_total_time and state.elapsed >= self.max_total_time:
                    raise RetryError(
                        f"Max total time ({self.max_total_time}s) exceeded",
                        attempts=state.attempt,
                        last_error=e,
                    ) from e

                # Compute and apply delay
                delay = strategy.compute_delay(category_attempts)

                # Special handling for rate limits with retry-after header
                if isinstance(e, RateLimitError) and e.retry_after:
                    delay = max(delay, e.retry_after)
                    logger.info(f"Rate limit: waiting {e.retry_after}s (retry-after header)")

                logger.info(
                    f"Retry {state.attempt}: {category.value} error, waiting {delay:.1f}s ({e})"
                )

                if self.on_retry:
                    self.on_retry(state.attempt, e, delay)

                state.record_delay(delay)
                await asyncio.sleep(delay)

    def wrap(
        self,
        retryable_exceptions: Sequence[type[Exception]] | None = None,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Decorator to wrap a function with adaptive retry.

        Args:
            retryable_exceptions: Additional exception types to retry

        Returns:
            Decorator function
        """

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            async def wrapper(*args: Any, **kwargs: Any) -> Any:
                return await self.execute(
                    lambda: func(*args, **kwargs),
                    retryable_exceptions=retryable_exceptions,
                )

            return wrapper

        return decorator


# Convenience function
async def with_retry(
    operation: Callable[[], Any],
    *,
    max_retries: int = 5,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
) -> Any:
    """Execute an operation with simple retry logic.

    Args:
        operation: Async callable to execute
        max_retries: Maximum number of retries
        base_delay: Base delay in seconds
        max_delay: Maximum delay cap

    Returns:
        The operation result
    """
    retry = AdaptiveRetry(
        strategies={
            cat: BackoffStrategy(
                base_delay=base_delay,
                max_delay=max_delay,
                max_retries=max_retries,
            )
            for cat in ErrorCategory
        }
    )
    return await retry.execute(operation)
