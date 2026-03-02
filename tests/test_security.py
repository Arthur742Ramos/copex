"""Tests for copex.security — input sanitization, path validation, env filtering."""

from __future__ import annotations

from pathlib import Path

import pytest

from copex.exceptions import SecurityError
from copex.security import (
    ENV_ALLOWLIST,
    filter_env_vars,
    safe_shell_quote,
    sanitize_command,
    sanitize_command_arg,
    validate_json_value,
    validate_mcp_tool_name,
    validate_path,
)

# ── filter_env_vars ──────────────────────────────────────────────────

class TestFilterEnvVars:
    def test_filters_to_allowlist(self):
        env = {"PATH": "/usr/bin", "SECRET_KEY": "abc", "HOME": "/home/user"}
        result = filter_env_vars(env)
        assert "PATH" in result
        assert "HOME" in result
        assert "SECRET_KEY" not in result

    def test_include_prefixes(self):
        env = {"COPEX_DEBUG": "1", "MY_VAR": "2", "SECRET": "3"}
        result = filter_env_vars(env, include_prefixes=["COPEX_", "MY_"])
        assert "COPEX_DEBUG" in result
        assert "MY_VAR" in result
        assert "SECRET" not in result

    def test_custom_allowlist(self):
        env = {"CUSTOM": "val", "PATH": "/usr/bin"}
        result = filter_env_vars(env, allowlist=frozenset({"CUSTOM"}))
        assert "CUSTOM" in result
        assert "PATH" not in result

    def test_defaults_to_os_environ(self):
        result = filter_env_vars()
        # Should return a dict with only allowlisted vars from os.environ
        assert isinstance(result, dict)
        for key in result:
            assert key in ENV_ALLOWLIST

    def test_empty_env(self):
        result = filter_env_vars({})
        assert result == {}


# ── sanitize_command_arg ─────────────────────────────────────────────

class TestSanitizeCommandArg:
    def test_safe_arg_passes(self):
        assert sanitize_command_arg("hello") == "hello"

    def test_safe_path_passes(self):
        assert sanitize_command_arg("/usr/bin/python3") == "/usr/bin/python3"

    @pytest.mark.parametrize(
        "dangerous",
        [
            "${HOME}",
            "$(whoami)",
            "`id`",
            "foo; rm -rf /",
            "foo | cat",
            "foo && evil",
            "foo || evil",
            "foo > /etc/passwd",
            "foo < /dev/null",
            "foo\nbar",
        ],
    )
    def test_dangerous_patterns_rejected(self, dangerous):
        with pytest.raises(SecurityError) as exc_info:
            sanitize_command_arg(dangerous)
        assert exc_info.value.violation_type == "command_injection"


# ── sanitize_command ─────────────────────────────────────────────────

class TestSanitizeCommand:
    def test_safe_command(self):
        result = sanitize_command(["python3", "-c", "print('hi')"])
        assert result == ["python3", "-c", "print('hi')"]

    def test_empty_command_rejected(self):
        with pytest.raises(SecurityError) as exc_info:
            sanitize_command([])
        assert exc_info.value.violation_type == "empty_command"

    def test_dangerous_arg_in_command(self):
        with pytest.raises(SecurityError):
            sanitize_command(["echo", "$(whoami)"])


# ── validate_path ────────────────────────────────────────────────────

class TestValidatePath:
    def test_valid_path(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hi")
        result = validate_path(str(f), must_exist=True)
        assert result == f.resolve()

    def test_nonexistent_must_exist(self, tmp_path):
        with pytest.raises(SecurityError) as exc_info:
            validate_path(tmp_path / "nope.txt", must_exist=True)
        assert exc_info.value.violation_type == "path_not_found"

    def test_null_byte_rejected(self):
        with pytest.raises(SecurityError) as exc_info:
            validate_path("/tmp/test\x00.txt")
        assert exc_info.value.violation_type == "path_injection"

    def test_absolute_path_rejected(self):
        with pytest.raises(SecurityError) as exc_info:
            validate_path("/etc/passwd", allow_absolute=False)
        assert exc_info.value.violation_type == "absolute_path"

    def test_path_traversal_rejected(self, tmp_path):
        with pytest.raises(SecurityError) as exc_info:
            validate_path("../../etc/passwd", base_dir=tmp_path)
        assert exc_info.value.violation_type == "path_traversal"

    def test_path_within_base_dir(self, tmp_path):
        child = tmp_path / "sub" / "file.txt"
        child.parent.mkdir(parents=True, exist_ok=True)
        child.touch()
        result = validate_path(child, base_dir=tmp_path, must_exist=True)
        assert result == child.resolve()

    def test_relative_path_allowed(self):
        result = validate_path("relative/path.txt")
        assert isinstance(result, Path)


# ── validate_mcp_tool_name ───────────────────────────────────────────

class TestValidateMcpToolName:
    def test_valid_names(self):
        assert validate_mcp_tool_name("my_tool") == "my_tool"
        assert validate_mcp_tool_name("run-test") == "run-test"
        assert validate_mcp_tool_name("Tool123") == "Tool123"

    def test_empty_name(self):
        with pytest.raises(SecurityError) as exc_info:
            validate_mcp_tool_name("")
        assert exc_info.value.violation_type == "invalid_tool_name"

    def test_invalid_format(self):
        with pytest.raises(SecurityError):
            validate_mcp_tool_name("123invalid")
        with pytest.raises(SecurityError):
            validate_mcp_tool_name("has spaces")
        with pytest.raises(SecurityError):
            validate_mcp_tool_name("has.dots")


# ── validate_json_value ──────────────────────────────────────────────

class TestValidateJsonValue:
    def test_primitives(self):
        assert validate_json_value("hello") == "hello"
        assert validate_json_value(42) == 42
        assert validate_json_value(3.14) == 3.14
        assert validate_json_value(True) is True
        assert validate_json_value(None) is None

    def test_nested_dict(self):
        data = {"a": {"b": [1, 2, 3]}}
        assert validate_json_value(data) == data

    def test_deep_nesting_rejected(self):
        # Build deeply nested dict
        d: dict = {}
        current = d
        for i in range(15):
            current["next"] = {}
            current = current["next"]
        with pytest.raises(SecurityError) as exc_info:
            validate_json_value(d, max_depth=10)
        assert exc_info.value.violation_type == "deep_nesting"

    def test_circular_reference_rejected(self):
        d: dict = {"a": 1}
        d["self"] = d  # type: ignore[assignment]
        with pytest.raises(SecurityError) as exc_info:
            validate_json_value(d)
        assert exc_info.value.violation_type == "circular_reference"

    def test_non_string_key_rejected(self):
        with pytest.raises(SecurityError) as exc_info:
            validate_json_value({1: "value"})
        assert exc_info.value.violation_type == "invalid_key_type"

    def test_unsupported_type_rejected(self):
        with pytest.raises(SecurityError) as exc_info:
            validate_json_value({"key": object()})
        assert exc_info.value.violation_type == "invalid_json_type"


# ── safe_shell_quote ─────────────────────────────────────────────────

class TestSafeShellQuote:
    def test_simple_string(self):
        assert safe_shell_quote("hello") == "hello"

    def test_special_chars_quoted(self):
        result = safe_shell_quote("hello world")
        assert " " not in result or result.startswith("'")

    def test_empty_string(self):
        result = safe_shell_quote("")
        assert result == "''"
