"""
Security utilities for Copex.

Provides input sanitization, environment variable filtering, and path validation
to prevent security vulnerabilities in subprocess execution and file operations.
"""

from __future__ import annotations

import os
import re
import shlex
from pathlib import Path
from typing import Any, Mapping, Sequence

from copex.exceptions import SecurityError

# Environment variables that are safe to pass to subprocesses
ENV_ALLOWLIST: frozenset[str] = frozenset(
    {
        # System essentials
        "PATH",
        "HOME",
        "USER",
        "SHELL",
        "TERM",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "TZ",
        "TMPDIR",
        "TEMP",
        "TMP",
        # Python
        "PYTHONPATH",
        "PYTHONHOME",
        "VIRTUAL_ENV",
        # Development tools
        "EDITOR",
        "VISUAL",
        "GIT_AUTHOR_NAME",
        "GIT_AUTHOR_EMAIL",
        "GIT_COMMITTER_NAME",
        "GIT_COMMITTER_EMAIL",
        # Node.js
        "NODE_ENV",
        "NODE_PATH",
        "NPM_CONFIG_PREFIX",
        # MCP-specific
        "MCP_SERVER_NAME",
        # XDG
        "XDG_CONFIG_HOME",
        "XDG_DATA_HOME",
        "XDG_CACHE_HOME",
        "XDG_RUNTIME_DIR",
    }
)

# Patterns that indicate potentially dangerous input
DANGEROUS_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\$\{[^}]+\}"),  # ${VAR} expansion
    re.compile(r"\$\([^)]+\)"),  # $(command) substitution
    re.compile(r"`[^`]+`"),  # `command` substitution
    re.compile(r";\s*"),  # Command chaining with ;
    re.compile(r"\|\s*"),  # Pipe chaining
    re.compile(r"&&\s*"),  # Logical AND chaining
    re.compile(r"\|\|\s*"),  # Logical OR chaining
    re.compile(r">\s*"),  # Output redirection
    re.compile(r"<\s*"),  # Input redirection
    re.compile(r"\n"),  # Newline injection
)

# Characters that should be escaped in shell arguments
SHELL_SPECIAL_CHARS: frozenset[str] = frozenset(
    {
        " ",
        "\t",
        "\n",
        "$",
        "`",
        "\\",
        '"',
        "'",
        "|",
        "&",
        ";",
        "(",
        ")",
        "<",
        ">",
        "!",
        "?",
        "*",
        "[",
        "]",
        "#",
        "~",
        "=",
        "%",
        "{",
        "}",
        "^",
    }
)


def filter_env_vars(
    env: Mapping[str, str] | None = None,
    *,
    allowlist: frozenset[str] | None = None,
    include_prefixes: Sequence[str] | None = None,
) -> dict[str, str]:
    """Filter environment variables to only include safe ones.

    Args:
        env: Environment dict to filter (defaults to os.environ)
        allowlist: Set of allowed variable names (defaults to ENV_ALLOWLIST)
        include_prefixes: Additional prefixes to allow (e.g., ["COPEX_", "MY_"])

    Returns:
        Filtered environment dictionary

    Example:
        >>> safe_env = filter_env_vars(os.environ, include_prefixes=["COPEX_"])
    """
    if env is None:
        env = os.environ
    if allowlist is None:
        allowlist = ENV_ALLOWLIST

    prefixes = tuple(include_prefixes) if include_prefixes else ()

    result: dict[str, str] = {}
    for key, value in env.items():
        if key in allowlist:
            result[key] = value
        elif prefixes and key.startswith(prefixes):
            result[key] = value

    return result


def sanitize_command_arg(arg: str, *, allow_paths: bool = True) -> str:
    """Sanitize a single command argument.

    Checks for dangerous patterns that could lead to command injection.

    Args:
        arg: The argument to sanitize
        allow_paths: Whether to allow path-like arguments

    Returns:
        The sanitized argument (unchanged if safe)

    Raises:
        SecurityError: If the argument contains dangerous patterns
    """
    for pattern in DANGEROUS_PATTERNS:
        if pattern.search(arg):
            raise SecurityError(
                f"Dangerous pattern detected in command argument: {pattern.pattern}",
                violation_type="command_injection",
                context={"argument": arg[:100]},  # Truncate for logging
            )

    return arg


def sanitize_command(
    command: Sequence[str],
    *,
    allow_paths: bool = True,
) -> list[str]:
    """Sanitize a command and its arguments.

    Args:
        command: Command as a list of strings
        allow_paths: Whether to allow path-like arguments

    Returns:
        Sanitized command list

    Raises:
        SecurityError: If any argument contains dangerous patterns
    """
    if not command:
        raise SecurityError(
            "Empty command not allowed",
            violation_type="empty_command",
        )

    return [sanitize_command_arg(arg, allow_paths=allow_paths) for arg in command]


def validate_path(
    path: str | Path,
    *,
    base_dir: str | Path | None = None,
    must_exist: bool = False,
    allow_absolute: bool = True,
) -> Path:
    """Validate a file path for security.

    Prevents path traversal attacks and ensures paths are within allowed directories.

    Args:
        path: Path to validate
        base_dir: If provided, path must be under this directory
        must_exist: Whether the path must exist
        allow_absolute: Whether to allow absolute paths

    Returns:
        Validated and resolved Path object

    Raises:
        SecurityError: If path validation fails
    """
    path = Path(path)

    # Check for null bytes (path traversal via null byte injection)
    if "\x00" in str(path):
        raise SecurityError(
            "Null byte in path not allowed",
            violation_type="path_injection",
            context={"path": str(path)[:100]},
        )

    # Check for absolute paths if not allowed
    if not allow_absolute and path.is_absolute():
        raise SecurityError(
            "Absolute paths not allowed",
            violation_type="absolute_path",
            context={"path": str(path)},
        )

    # Resolve to absolute path for traversal check
    try:
        resolved = path.resolve()
    except (OSError, ValueError) as e:
        raise SecurityError(
            f"Invalid path: {e}",
            violation_type="invalid_path",
            context={"path": str(path)},
        ) from e

    # Check for path traversal
    if base_dir is not None:
        base_resolved = Path(base_dir).resolve()
        try:
            resolved.relative_to(base_resolved)
        except ValueError:
            raise SecurityError(
                f"Path traversal detected: {path} is not under {base_dir}",
                violation_type="path_traversal",
                context={"path": str(path), "base_dir": str(base_dir)},
            )

    # Check existence
    if must_exist and not resolved.exists():
        raise SecurityError(
            f"Path does not exist: {path}",
            violation_type="path_not_found",
            context={"path": str(path)},
        )

    return resolved


def safe_shell_quote(arg: str) -> str:
    """Safely quote a string for shell use.

    Uses shlex.quote for proper escaping.

    Args:
        arg: The argument to quote

    Returns:
        Properly quoted string safe for shell use
    """
    return shlex.quote(arg)


def validate_mcp_tool_name(name: str) -> str:
    """Validate an MCP tool name.

    Tool names should be alphanumeric with underscores/hyphens only.

    Args:
        name: The tool name to validate

    Returns:
        The validated tool name

    Raises:
        SecurityError: If the tool name is invalid
    """
    if not name:
        raise SecurityError(
            "Empty tool name not allowed",
            violation_type="invalid_tool_name",
        )

    if not re.match(r"^[a-zA-Z][a-zA-Z0-9_-]*$", name):
        raise SecurityError(
            f"Invalid tool name format: {name}",
            violation_type="invalid_tool_name",
            context={"tool_name": name},
        )

    return name


def validate_json_value(value: Any, max_depth: int = 10) -> Any:
    """Validate a value for safe JSON serialization.

    Prevents circular references and excessively deep nesting.

    Args:
        value: Value to validate
        max_depth: Maximum nesting depth allowed

    Returns:
        The validated value

    Raises:
        SecurityError: If the value is unsafe for serialization
    """

    def _validate(v: Any, depth: int, seen: set[int]) -> None:
        if depth > max_depth:
            raise SecurityError(
                f"Value too deeply nested (max {max_depth})",
                violation_type="deep_nesting",
            )

        obj_id = id(v)
        if obj_id in seen:
            raise SecurityError(
                "Circular reference detected",
                violation_type="circular_reference",
            )

        if isinstance(v, dict):
            seen.add(obj_id)
            for key, val in v.items():
                if not isinstance(key, str):
                    raise SecurityError(
                        "Dict keys must be strings",
                        violation_type="invalid_key_type",
                    )
                _validate(val, depth + 1, seen)
            seen.discard(obj_id)
        elif isinstance(v, (list, tuple)):
            seen.add(obj_id)
            for item in v:
                _validate(item, depth + 1, seen)
            seen.discard(obj_id)
        elif not isinstance(v, (str, int, float, bool, type(None))):
            raise SecurityError(
                f"Unsupported type for JSON: {type(v).__name__}",
                violation_type="invalid_json_type",
            )

    _validate(value, 0, set())
    return value
