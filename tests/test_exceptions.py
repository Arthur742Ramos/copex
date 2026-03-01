"""Tests for exceptions module."""

from __future__ import annotations

import pytest

from copex.exceptions import (
    AllModelsUnavailable,
    AuthenticationError,
    CircuitBreakerOpen,
    ConfigError,
    ConfigurationError,
    ConnectionError,
    CopexError,
    MCPError,
    PlanExecutionError,
    RateLimitError,
    RetryError,
    SecurityError,
    SessionError,
    SessionRecoveryFailed,
    StreamingError,
    TimeoutError,
    ToolExecutionError,
    ValidationError,
)


class TestCopexError:
    """Tests for the base CopexError."""

    def test_basic_message(self) -> None:
        e = CopexError("something failed")
        assert str(e) == "something failed"
        assert e.message == "something failed"

    def test_with_context(self) -> None:
        e = CopexError("oops", context={"model": "gpt-5"})
        assert "model='gpt-5'" in str(e)
        assert e.context == {"model": "gpt-5"}

    def test_empty_context(self) -> None:
        e = CopexError("oops")
        assert e.context == {}

    def test_is_exception(self) -> None:
        assert issubclass(CopexError, Exception)


class TestConfigError:
    def test_inherits_copex_error(self) -> None:
        e = ConfigError("bad config")
        assert isinstance(e, CopexError)


class TestMCPError:
    def test_with_tool_and_server(self) -> None:
        e = MCPError("fail", tool_name="read_file", server_name="fs")
        assert e.tool_name == "read_file"
        assert e.server_name == "fs"
        assert "tool_name='read_file'" in str(e)

    def test_without_extras(self) -> None:
        e = MCPError("fail")
        assert e.tool_name is None
        assert e.server_name is None


class TestRetryError:
    def test_with_last_error(self) -> None:
        cause = ValueError("bad value")
        e = RetryError("retries exhausted", attempts=3, last_error=cause)
        assert e.attempts == 3
        assert e.last_error is cause
        assert "attempts=3" in str(e)

    def test_without_last_error(self) -> None:
        e = RetryError("done", attempts=1)
        assert e.last_error is None


class TestPlanExecutionError:
    def test_with_step_info(self) -> None:
        e = PlanExecutionError(
            "step failed",
            step_index=2,
            step_name="build",
            original_error=RuntimeError("build error"),
        )
        assert e.step_index == 2
        assert e.step_name == "build"
        assert e.original_error is not None


class TestValidationError:
    def test_with_field(self) -> None:
        e = ValidationError("invalid", field_name="model", value="bad")
        assert e.field_name == "model"
        assert e.value == "bad"
        assert "field='model'" in str(e)


class TestSecurityError:
    def test_with_violation_type(self) -> None:
        e = SecurityError("path traversal", violation_type="path_traversal")
        assert e.violation_type == "path_traversal"


class TestTimeoutError:
    def test_with_details(self) -> None:
        e = TimeoutError("timed out", timeout_seconds=30, operation="send")
        assert e.timeout_seconds == 30
        assert e.operation == "send"


class TestRateLimitError:
    def test_with_retry_after(self) -> None:
        e = RateLimitError("too many", retry_after=60.0)
        assert e.retry_after == 60.0
        assert "retry_after=60.0" in str(e)

    def test_without_retry_after(self) -> None:
        e = RateLimitError("too many")
        assert e.retry_after is None


class TestConnectionError:
    def test_with_host_port(self) -> None:
        e = ConnectionError("refused", host="localhost", port=8080)
        assert e.host == "localhost"
        assert e.port == 8080


class TestInheritanceHierarchy:
    """Verify exception hierarchy for catch-all patterns."""

    def test_all_inherit_from_copex_error(self) -> None:
        subclasses = [
            ConfigError,
            MCPError,
            RetryError,
            PlanExecutionError,
            ValidationError,
            SecurityError,
            TimeoutError,
            AuthenticationError,
            RateLimitError,
            ConnectionError,
            CircuitBreakerOpen,
            SessionError,
            SessionRecoveryFailed,
            ToolExecutionError,
            StreamingError,
            ConfigurationError,
            AllModelsUnavailable,
        ]
        for cls in subclasses:
            assert issubclass(cls, CopexError), f"{cls.__name__} should inherit CopexError"

    def test_session_recovery_inherits_session_error(self) -> None:
        assert issubclass(SessionRecoveryFailed, SessionError)

    def test_catch_all_works(self) -> None:
        """Verify except CopexError catches all custom exceptions."""
        with pytest.raises(CopexError):
            raise CircuitBreakerOpen("open")
        with pytest.raises(CopexError):
            raise RateLimitError("limited")
        with pytest.raises(CopexError):
            raise SessionRecoveryFailed("failed")
