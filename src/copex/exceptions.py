"""
Copex Exception Hierarchy.

All custom exceptions inherit from CopexError for unified error handling.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class CopexError(Exception):
    """Base exception for Copex errors.

    All Copex exceptions inherit from this class to allow catching
    any Copex-related error with a single except clause.

    Attributes:
        message: Human-readable error description
        context: Additional context for debugging
    """

    def __init__(self, message: str, context: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.context = context or {}
        self._log_error()

    def _log_error(self) -> None:
        """Log the error creation at debug level.

        Errors are logged at debug to avoid noise from expected/retried errors.
        Callers should log at appropriate level when handling the exception.
        """
        logger.debug(
            f"{self.__class__.__name__}: {self.message}",
            extra={"error_context": self.context},
        )

    def __str__(self) -> str:
        if self.context:
            ctx = ", ".join(f"{k}={v!r}" for k, v in self.context.items())
            return f"{self.message} [{ctx}]"
        return self.message


class ConfigError(CopexError):
    """Raised for configuration errors.

    Examples:
        - Invalid model name
        - Invalid reasoning level
        - Malformed config file
        - Missing required configuration
    """


class MCPError(CopexError):
    """Raised for MCP (Model Context Protocol) related errors.

    Examples:
        - MCP server connection failure
        - Tool execution failure
        - Invalid MCP response
        - MCP timeout
    """

    def __init__(
        self,
        message: str,
        tool_name: str | None = None,
        server_name: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        ctx = context or {}
        if tool_name:
            ctx["tool_name"] = tool_name
        if server_name:
            ctx["server_name"] = server_name
        super().__init__(message, ctx)
        self.tool_name = tool_name
        self.server_name = server_name


class RetryError(CopexError):
    """Raised when all retry attempts have been exhausted.

    Attributes:
        attempts: Number of attempts made
        last_error: The last error that occurred
    """

    def __init__(
        self,
        message: str,
        attempts: int,
        last_error: Exception | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        ctx = context or {}
        ctx["attempts"] = attempts
        if last_error:
            ctx["last_error"] = str(last_error)
        super().__init__(message, ctx)
        self.attempts = attempts
        self.last_error = last_error


class PlanExecutionError(CopexError):
    """Raised when plan execution fails.

    Attributes:
        step_index: The step that failed (0-indexed)
        step_name: Name of the failed step
        original_error: The underlying error
    """

    def __init__(
        self,
        message: str,
        step_index: int | None = None,
        step_name: str | None = None,
        original_error: Exception | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        ctx = context or {}
        if step_index is not None:
            ctx["step_index"] = step_index
        if step_name:
            ctx["step_name"] = step_name
        if original_error:
            ctx["original_error"] = str(original_error)
        super().__init__(message, ctx)
        self.step_index = step_index
        self.step_name = step_name
        self.original_error = original_error


class ValidationError(CopexError):
    """Raised for input validation errors.

    Examples:
        - Invalid file path
        - Invalid model name
        - Invalid parameter value
    """

    def __init__(
        self,
        message: str,
        field_name: str | None = None,
        value: Any = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        ctx = context or {}
        if field_name:
            ctx["field"] = field_name
        if value is not None:
            ctx["value"] = repr(value)
        super().__init__(message, ctx)
        self.field_name = field_name
        self.value = value


class SecurityError(CopexError):
    """Raised for security-related violations.

    Examples:
        - Path traversal attempt
        - Disallowed environment variable
        - Subprocess command injection attempt
    """

    def __init__(
        self,
        message: str,
        violation_type: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        ctx = context or {}
        if violation_type:
            ctx["violation_type"] = violation_type
        super().__init__(message, ctx)
        self.violation_type = violation_type


class TimeoutError(CopexError):
    """Raised when an operation times out.

    Attributes:
        timeout_seconds: The timeout value that was exceeded
        operation: Description of the operation that timed out
    """

    def __init__(
        self,
        message: str,
        timeout_seconds: float | None = None,
        operation: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        ctx = context or {}
        if timeout_seconds is not None:
            ctx["timeout_seconds"] = timeout_seconds
        if operation:
            ctx["operation"] = operation
        super().__init__(message, ctx)
        self.timeout_seconds = timeout_seconds
        self.operation = operation


class AuthenticationError(CopexError):
    """Raised for authentication/authorization failures.

    Examples:
        - GitHub token expired
        - Invalid credentials
        - Insufficient permissions
    """


class RateLimitError(CopexError):
    """Raised when rate limits are exceeded.

    Attributes:
        retry_after: Seconds to wait before retrying (if known)
    """

    def __init__(
        self,
        message: str,
        retry_after: float | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        ctx = context or {}
        if retry_after is not None:
            ctx["retry_after"] = retry_after
        super().__init__(message, ctx)
        self.retry_after = retry_after


class ConnectionError(CopexError):
    """Raised for network connection errors.

    Attributes:
        host: The host that couldn't be reached
        port: The port number
    """

    def __init__(
        self,
        message: str,
        host: str | None = None,
        port: int | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        ctx = context or {}
        if host:
            ctx["host"] = host
        if port:
            ctx["port"] = port
        super().__init__(message, ctx)
        self.host = host
        self.port = port


class CircuitBreakerOpen(CopexError):
    """Raised when the circuit breaker is open and requests are rejected."""


class SessionError(CopexError):
    """Raised when session creation or management fails."""


class SessionRecoveryFailed(SessionError):
    """Raised when session recovery after an error fails."""


class AllModelsUnavailable(CopexError):
    """Raised when all models (primary + fallbacks) are unavailable."""


class ToolExecutionError(CopexError):
    """Raised when a tool execution fails."""


class StreamingError(CopexError):
    """Raised when streaming encounters an unrecoverable error."""


class ConfigurationError(CopexError):
    """Raised when configuration is invalid."""
