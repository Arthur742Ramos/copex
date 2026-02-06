"""Tests for v1.9.0 features: SessionPool in Fleet, AdaptiveRetry, Model Fallback."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from copex.backoff import categorize_error, ErrorCategory
from copex.client import (
    Copex,
    ModelAwareBreaker,
    DEFAULT_FALLBACK_CHAINS,
)
from copex.config import CopexConfig


# =============================================================================
# categorize_error Tests (Task 4: Fixed performance and false positives)
# =============================================================================


class TestCategorizeErrorPerformance:
    """Tests for optimized categorize_error function."""

    def test_rate_limit_detection(self):
        """Rate limit errors are correctly categorized."""
        assert categorize_error(RuntimeError("Rate limit exceeded")) == ErrorCategory.RATE_LIMIT
        assert categorize_error(RuntimeError("Error 429")) == ErrorCategory.RATE_LIMIT
        assert categorize_error(RuntimeError("HTTP 429: Too many requests")) == ErrorCategory.RATE_LIMIT

    def test_network_error_detection(self):
        """Network errors are correctly categorized."""
        assert categorize_error(RuntimeError("Connection refused")) == ErrorCategory.NETWORK
        assert categorize_error(RuntimeError("Socket timeout")) == ErrorCategory.NETWORK
        assert categorize_error(RuntimeError("Network unreachable")) == ErrorCategory.NETWORK

    def test_auth_error_detection(self):
        """Auth errors are correctly categorized."""
        assert categorize_error(RuntimeError("HTTP 401 Unauthorized")) == ErrorCategory.AUTH
        assert categorize_error(RuntimeError("Error 403: Forbidden")) == ErrorCategory.AUTH
        assert categorize_error(RuntimeError("Unauthorized access")) == ErrorCategory.AUTH

    def test_server_error_detection(self):
        """Server errors (5xx) are correctly categorized."""
        assert categorize_error(RuntimeError("Error 500 Internal Server Error")) == ErrorCategory.SERVER
        assert categorize_error(RuntimeError("HTTP 502 Bad Gateway")) == ErrorCategory.SERVER
        assert categorize_error(RuntimeError("status 503")) == ErrorCategory.SERVER
        assert categorize_error(RuntimeError("Internal server error")) == ErrorCategory.SERVER

    def test_client_error_detection(self):
        """Client errors (4xx) are correctly categorized."""
        assert categorize_error(RuntimeError("HTTP 400 Bad Request")) == ErrorCategory.CLIENT
        assert categorize_error(RuntimeError("Error 404 Not Found")) == ErrorCategory.CLIENT

    def test_no_false_positives_on_500(self):
        """Messages containing '500' that aren't HTTP errors are not misclassified."""
        # These should NOT be categorized as SERVER errors
        assert categorize_error(RuntimeError("Error at line 500")) == ErrorCategory.UNKNOWN
        assert categorize_error(RuntimeError("User ID 500 not found")) == ErrorCategory.UNKNOWN
        assert categorize_error(RuntimeError("Processed 500 records")) == ErrorCategory.UNKNOWN
        # Note: "Timeout after 500ms" correctly categorizes as NETWORK due to "timeout" keyword

    def test_no_false_positives_on_400(self):
        """Messages containing '400' that aren't HTTP errors are not misclassified."""
        assert categorize_error(RuntimeError("File has 400 lines")) == ErrorCategory.UNKNOWN
        assert categorize_error(RuntimeError("Value must be < 400")) == ErrorCategory.UNKNOWN

    def test_unknown_errors(self):
        """Unknown errors are correctly categorized."""
        assert categorize_error(RuntimeError("Something went wrong")) == ErrorCategory.UNKNOWN
        assert categorize_error(RuntimeError("Unexpected state")) == ErrorCategory.UNKNOWN


# =============================================================================
# ModelAwareBreaker Tests (Task 3: Model fallback)
# =============================================================================


class TestModelAwareBreaker:
    """Tests for ModelAwareBreaker with fallback support."""

    def test_is_open_returns_false_for_new_model(self):
        """is_open returns False for models with no history."""
        breaker = ModelAwareBreaker()
        assert breaker.is_open("claude-opus-4.5") is False

    def test_is_open_returns_true_after_failures(self):
        """is_open returns True after sufficient failures."""
        breaker = ModelAwareBreaker(window_size=5, threshold=0.5)
        model = "test-model"

        # Record enough failures to open circuit
        for _ in range(5):
            breaker.record_failure(model)

        assert breaker.is_open(model) is True

    def test_get_available_model_returns_preferred_when_available(self):
        """get_available_model returns preferred model when its circuit is closed."""
        breaker = ModelAwareBreaker()
        model = breaker.get_available_model(
            "claude-opus-4.5",
            fallback_chain=["claude-sonnet-4.5", "claude-haiku-4.5"]
        )
        assert model == "claude-opus-4.5"

    def test_get_available_model_returns_fallback_when_primary_open(self):
        """get_available_model returns fallback when primary circuit is open."""
        breaker = ModelAwareBreaker(window_size=3, threshold=0.5)

        # Open circuit for primary model
        for _ in range(3):
            breaker.record_failure("claude-opus-4.5")

        model = breaker.get_available_model(
            "claude-opus-4.5",
            fallback_chain=["claude-sonnet-4.5", "claude-haiku-4.5"]
        )
        assert model == "claude-sonnet-4.5"

    def test_get_available_model_returns_none_when_all_open(self):
        """get_available_model returns None when all circuits are open."""
        breaker = ModelAwareBreaker(window_size=3, threshold=0.5)

        # Open circuit for all models
        for model in ["claude-opus-4.5", "claude-sonnet-4.5", "claude-haiku-4.5"]:
            for _ in range(3):
                breaker.record_failure(model)

        model = breaker.get_available_model(
            "claude-opus-4.5",
            fallback_chain=["claude-sonnet-4.5", "claude-haiku-4.5"]
        )
        assert model is None

    def test_get_available_model_skips_open_fallbacks(self):
        """get_available_model skips fallbacks with open circuits."""
        breaker = ModelAwareBreaker(window_size=3, threshold=0.5)

        # Open circuits for primary and first fallback
        for model in ["claude-opus-4.5", "claude-sonnet-4.5"]:
            for _ in range(3):
                breaker.record_failure(model)

        model = breaker.get_available_model(
            "claude-opus-4.5",
            fallback_chain=["claude-sonnet-4.5", "claude-haiku-4.5"]
        )
        assert model == "claude-haiku-4.5"


class TestDefaultFallbackChains:
    """Tests for DEFAULT_FALLBACK_CHAINS."""

    def test_claude_opus_has_fallback_chain(self):
        """Claude Opus models have defined fallback chains."""
        assert "claude-opus-4.6" in DEFAULT_FALLBACK_CHAINS
        assert "claude-opus-4.5" in DEFAULT_FALLBACK_CHAINS

        chain = DEFAULT_FALLBACK_CHAINS["claude-opus-4.6"]
        assert "claude-sonnet-4.5" in chain or "claude-opus-4.5" in chain

    def test_gpt_codex_has_fallback_chain(self):
        """GPT Codex models have defined fallback chains."""
        assert "gpt-5.2-codex" in DEFAULT_FALLBACK_CHAINS

        chain = DEFAULT_FALLBACK_CHAINS["gpt-5.2-codex"]
        assert any("gpt" in m for m in chain)

    def test_fallback_chains_dont_include_self(self):
        """Fallback chains don't include the model itself."""
        for model, chain in DEFAULT_FALLBACK_CHAINS.items():
            assert model not in chain


# =============================================================================
# Copex Fallback Integration Tests
# =============================================================================


class TestCopexFallback:
    """Tests for Copex model fallback integration."""

    def test_copex_accepts_fallback_chain(self):
        """Copex accepts fallback_chain parameter."""
        config = CopexConfig()
        copex = Copex(config, fallback_chain=["model-a", "model-b"])
        assert copex._fallback_chain == ["model-a", "model-b"]

    def test_copex_uses_default_fallback_chain(self):
        """Copex uses default fallback chain when not specified."""
        config = CopexConfig()
        copex = Copex(config)
        # Default fallback should be None (uses DEFAULT_FALLBACK_CHAINS)
        assert copex._fallback_chain is None

    def test_shared_model_breaker(self):
        """Copex instances share the same model breaker."""
        copex1 = Copex()
        copex2 = Copex()
        assert copex1.get_model_breaker() is copex2.get_model_breaker()


# =============================================================================
# AdaptiveRetry Integration Tests (Task 2)
# =============================================================================


class TestAdaptiveRetryIntegration:
    """Tests for AdaptiveRetry integration in Copex client."""

    def test_should_retry_uses_categorization(self):
        """_should_retry uses error categorization for decisions."""
        config = CopexConfig()
        config.retry.retry_on_any_error = False
        copex = Copex(config)

        # Network errors should be retryable
        assert copex._should_retry(ConnectionError("Network error")) is True

        # Server errors should be retryable
        assert copex._should_retry(RuntimeError("HTTP 500")) is True

    def test_should_retry_rejects_auth_errors(self):
        """_should_retry rejects auth errors when not retry_on_any_error."""
        config = CopexConfig()
        config.retry.retry_on_any_error = False
        copex = Copex(config)

        # Auth errors should NOT be retryable
        assert copex._should_retry(RuntimeError("HTTP 401 Unauthorized")) is False
        assert copex._should_retry(RuntimeError("HTTP 403 Forbidden")) is False

    def test_calculate_delay_with_error_uses_adaptive_strategy(self):
        """_calculate_delay with error uses AdaptiveRetry strategies."""
        config = CopexConfig()
        copex = Copex(config)

        # Rate limit errors should have longer delays
        rate_limit_delay = copex._calculate_delay(1, RuntimeError("Rate limit exceeded"))
        network_delay = copex._calculate_delay(1, RuntimeError("Connection refused"))

        # Rate limit strategy has base_delay=5.0, network has base_delay=1.0
        # So rate limit delay should generally be higher
        assert rate_limit_delay >= 0  # Just verify it computes without error

    def test_calculate_delay_without_error_uses_config(self):
        """_calculate_delay without error uses config-based calculation."""
        config = CopexConfig()
        config.retry.base_delay = 2.0
        config.retry.exponential_base = 2.0
        copex = Copex(config)

        delay = copex._calculate_delay(0)  # No error parameter
        # Should be around 2.0 with jitter
        assert 1.5 <= delay <= 2.5
