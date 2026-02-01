"""Configuration management for Copex."""

import json
import logging
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from pydantic import BaseModel, Field, field_validator

from copex.models import Model, ReasoningEffort

_UNSET = object()
_COPILOT_CLI_CACHE: str | None | object = _UNSET
_LOG_LEVELS = {
    "critical": logging.CRITICAL,
    "error": logging.ERROR,
    "warning": logging.WARNING,
    "warn": logging.WARNING,
    "info": logging.INFO,
    "debug": logging.DEBUG,
}
_LOG_RECORD_KEYS = {
    "name",
    "msg",
    "args",
    "levelname",
    "levelno",
    "pathname",
    "filename",
    "module",
    "exc_info",
    "exc_text",
    "stack_info",
    "lineno",
    "funcName",
    "created",
    "msecs",
    "relativeCreated",
    "thread",
    "threadName",
    "processName",
    "process",
}


def find_copilot_cli() -> str | None:
    """Auto-detect the Copilot CLI path across platforms.

    Searches in order:
    1. shutil.which('copilot') - system PATH
    2. Common npm global locations
    3. Common installation paths

    Returns the path if found, None otherwise.
    """
    env_override = os.environ.get("COPEX_COPILOT_CLI")
    if env_override:
        return env_override

    global _COPILOT_CLI_CACHE
    if _COPILOT_CLI_CACHE is not _UNSET:
        return _COPILOT_CLI_CACHE  # type: ignore[return-value]

    # First try PATH (works on all platforms)
    cli_path = shutil.which("copilot")
    if cli_path:
        # On Windows, prefer .cmd over .ps1 for subprocess compatibility
        if sys.platform == "win32" and cli_path.endswith(".ps1"):
            cmd_path = cli_path.replace(".ps1", ".cmd")
            if os.path.exists(cmd_path):
                _COPILOT_CLI_CACHE = cmd_path
                return cmd_path
        _COPILOT_CLI_CACHE = cli_path
        return cli_path

    # Platform-specific common locations
    if sys.platform == "win32":
        # Windows locations
        candidates = [
            Path(os.environ.get("APPDATA", "")) / "npm" / "copilot.cmd",
            Path(os.environ.get("LOCALAPPDATA", "")) / "npm" / "copilot.cmd",
            Path.home() / "AppData" / "Roaming" / "npm" / "copilot.cmd",
            Path.home() / ".npm-global" / "copilot.cmd",
        ]
        # Also check USERPROFILE
        if "USERPROFILE" in os.environ:
            candidates.append(Path(os.environ["USERPROFILE"]) / "AppData" / "Roaming" / "npm" / "copilot.cmd")
    elif sys.platform == "darwin":
        # macOS locations
        candidates = [
            Path.home() / ".npm-global" / "bin" / "copilot",
            Path("/usr/local/bin/copilot"),
            Path("/opt/homebrew/bin/copilot"),
            Path.home() / ".nvm" / "versions" / "node",  # Check NVM later
        ]
    else:
        # Linux locations
        candidates = [
            Path.home() / ".npm-global" / "bin" / "copilot",
            Path("/usr/local/bin/copilot"),
            Path("/usr/bin/copilot"),
            Path.home() / ".local" / "bin" / "copilot",
        ]

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            _COPILOT_CLI_CACHE = str(candidate)
            return str(candidate)

    # Check for NVM installations (macOS/Linux)
    if sys.platform != "win32":
        nvm_dir = Path.home() / ".nvm" / "versions" / "node"
        if nvm_dir.exists():
            # Find latest node version
            versions = sorted(nvm_dir.iterdir(), reverse=True)
            for version in versions:
                copilot_path = version / "bin" / "copilot"
                if copilot_path.exists():
                    _COPILOT_CLI_CACHE = str(copilot_path)
                    return str(copilot_path)

    _COPILOT_CLI_CACHE = None
    return None


def _parse_log_level(level: str) -> int:
    normalized = level.strip().lower()
    if normalized in _LOG_LEVELS:
        return _LOG_LEVELS[normalized]
    valid = ", ".join(sorted({k for k in _LOG_LEVELS if k != "warn"}))
    raise ValueError(f"Invalid log level: {level}. Valid: {valid}")


def _coerce_log_levels(levels: dict[str, str]) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for name, level in levels.items():
        _parse_log_level(level)
        normalized[name] = level.strip().lower()
    return normalized


def _iter_log_levels(
    default_level: str,
    per_component: dict[str, str],
) -> Iterable[int]:
    yield _parse_log_level(default_level)
    for level in per_component.values():
        yield _parse_log_level(level)


class _StructuredFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname.lower(),
            "logger": record.name,
            "message": record.getMessage(),
        }
        extras = {
            key: value
            for key, value in record.__dict__.items()
            if key not in _LOG_RECORD_KEYS
        }
        for key, value in extras.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                payload[key] = value
            else:
                payload[key] = str(value)
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack_info"] = record.stack_info
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(config: "CopexConfig") -> None:
    """Configure structured logging for CLI usage."""
    root_logger = logging.getLogger()
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    handler: logging.Handler
    if config.log_file:
        handler = logging.FileHandler(config.log_file)
    else:
        handler = logging.StreamHandler()
    handler.setLevel(logging.NOTSET)
    handler.setFormatter(_StructuredFormatter())
    root_logger.addHandler(handler)

    min_level = min(_iter_log_levels(config.log_level, config.log_levels))
    root_logger.setLevel(min_level)

    for name, level in config.log_levels.items():
        logging.getLogger(name).setLevel(_parse_log_level(level))


UI_THEMES = {"default", "midnight", "mono", "sunset"}
UI_DENSITIES = {"compact", "extended"}


class RetryConfig(BaseModel):
    """Retry configuration."""

    max_retries: int = Field(default=5, ge=1, le=20, description="Maximum retry attempts")
    max_auto_continues: int = Field(default=3, ge=0, le=10, description="Maximum auto-continue cycles after exhausting retries")
    base_delay: float = Field(default=1.0, ge=0.1, description="Base delay between retries (seconds)")
    max_delay: float = Field(default=30.0, ge=1.0, description="Maximum delay between retries (seconds)")
    exponential_base: float = Field(default=2.0, ge=1.5, description="Exponential backoff multiplier")
    retry_on_any_error: bool = Field(
        default=True, description="Retry and auto-continue on any error"
    )
    retry_on_errors: list[str] = Field(
        default=["500", "502", "503", "504", "Internal Server Error", "rate limit"],
        description="Error patterns to retry on (only used if retry_on_any_error=False)",
    )


def get_user_state_path() -> Path:
    """Get path to user state file."""
    return Path.home() / ".copex" / "state.json"


def load_last_model() -> Model | None:
    """Load the last used model from user state."""
    state_path = get_user_state_path()
    if not state_path.exists():
        return None
    try:
        import json
        with open(state_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        model_value = data.get("last_model")
        if model_value:
            return Model(model_value)
    except (ValueError, OSError, json.JSONDecodeError):
        pass
    return None


def load_last_reasoning_effort() -> ReasoningEffort | None:
    """Load the last used reasoning effort from user state."""
    state_path = get_user_state_path()
    if not state_path.exists():
        return None
    try:
        import json
        with open(state_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        value = data.get("last_reasoning_effort")
        if value:
            return ReasoningEffort(value)
    except (ValueError, OSError, json.JSONDecodeError):
        pass
    return None


def save_last_model(model: Model) -> None:
    """Save the last used model to user state."""
    import json
    state_path = get_user_state_path()
    state_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load existing state
    data: dict[str, Any] = {}
    if state_path.exists():
        try:
            with open(state_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            pass
    
    # Update and save
    data["last_model"] = model.value
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def save_last_reasoning_effort(reasoning_effort: ReasoningEffort) -> None:
    """Save the last used reasoning effort to user state."""
    import json
    state_path = get_user_state_path()
    state_path.parent.mkdir(parents=True, exist_ok=True)

    data: dict[str, Any] = {}
    if state_path.exists():
        try:
            with open(state_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            pass

    data["last_reasoning_effort"] = reasoning_effort.value
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


class CopexConfig(BaseModel):
    """Main configuration for Copex client."""

    model: Model = Field(default=Model.CLAUDE_OPUS_4_5, description="Model to use")
    reasoning_effort: ReasoningEffort = Field(
        default=ReasoningEffort.XHIGH, description="Reasoning effort level"
    )
    streaming: bool = Field(default=True, description="Enable streaming responses")
    retry: RetryConfig = Field(default_factory=RetryConfig, description="Retry configuration")

    # Client options
    cli_path: str | None = Field(default=None, description="Path to Copilot CLI executable")
    cli_url: str | None = Field(default=None, description="URL of existing CLI server")
    cwd: str | None = Field(default=None, description="Working directory for CLI process")
    auto_start: bool = Field(default=True, description="Auto-start CLI server")
    auto_restart: bool = Field(default=True, description="Auto-restart on crash")
    log_level: str = Field(default="warning", description="Log level")
    log_levels: dict[str, str] = Field(
        default_factory=dict,
        description="Per-component log levels (e.g., {'copex.client': 'debug'})",
    )
    log_file: str | None = Field(
        default=None,
        description="Optional log file path (structured JSON)",
    )

    # Session options
    timeout: float = Field(default=300.0, ge=10.0, description="Inactivity timeout (seconds) - resets on each event")
    auto_continue: bool = Field(
        default=True, description="Auto-send 'Keep going' on interruption/error"
    )
    continue_prompt: str = Field(
        default="Keep going", description="Prompt to send on auto-continue"
    )
    recovery_prompt_max_chars: int = Field(
        default=8000,
        ge=100,
        description="Max chars for recovery prompt context",
    )
    auth_refresh_interval: float = Field(
        default=3300.0,
        ge=0.0,
        description="Seconds between auth refreshes (0 disables)",
    )
    auth_refresh_buffer: float = Field(
        default=300.0,
        ge=0.0,
        description="Seconds before interval to refresh auth",
    )
    auth_refresh_on_error: bool = Field(
        default=True,
        description="Refresh auth on unauthorized/expired errors",
    )

    # Skills and capabilities
    skills: list[str] = Field(
        default_factory=list,
        description="Skills to enable (e.g., ['code-review', 'azure-openai'])"
    )
    instructions: str | None = Field(
        default=None,
        description="Custom instructions for the session"
    )
    instructions_file: str | None = Field(
        default=None,
        description="Path to instructions file (.md)"
    )

    # MCP configuration
    mcp_servers: list[dict[str, Any]] = Field(
        default_factory=list,
        description="MCP server configurations"
    )
    mcp_config_file: str | None = Field(
        default=None,
        description="Path to MCP config JSON file"
    )
    mcp_http_timeout: float = Field(
        default=10.0,
        ge=1.0,
        description="HTTP MCP timeout in seconds",
    )
    mcp_http_max_retries: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Max HTTP MCP retries",
    )
    mcp_http_retry_base_delay: float = Field(
        default=0.5,
        ge=0.1,
        description="Base delay for HTTP MCP retries",
    )
    mcp_http_retry_max_delay: float = Field(
        default=5.0,
        ge=0.1,
        description="Max delay for HTTP MCP retries",
    )
    mcp_health_check_path: str | None = Field(
        default=None,
        description="Optional HTTP MCP health check path",
    )

    # Tool filtering
    available_tools: list[str] | None = Field(
        default=None,
        description="Whitelist of tools to enable (None = all)"
    )
    excluded_tools: list[str] = Field(
        default_factory=list,
        description="Blacklist of tools to disable"
    )

    # UI options
    ui_theme: str = Field(
        default="default",
        description="UI theme (default, midnight, mono, sunset)"
    )
    ui_density: str = Field(
        default="extended",
        description="UI density (compact or extended)"
    )
    ui_ascii_icons: bool = Field(
        default=False,
        description="Use ASCII icons instead of Unicode"
    )

    # Streaming/backpressure
    stream_queue_max_size: int = Field(
        default=1000,
        ge=10,
        description="Max items in streaming queue before dropping",
    )
    stream_drop_mode: str = Field(
        default="drop_oldest",
        description="Streaming queue overflow strategy (drop_oldest, drop_newest)",
    )

    @field_validator("ui_theme")
    @classmethod
    def _validate_ui_theme(cls, value: str) -> str:
        if value not in UI_THEMES:
            valid = ", ".join(sorted(UI_THEMES))
            raise ValueError(f"Invalid ui_theme. Valid: {valid}")
        return value

    @field_validator("ui_density")
    @classmethod
    def _validate_ui_density(cls, value: str) -> str:
        if value not in UI_DENSITIES:
            valid = ", ".join(sorted(UI_DENSITIES))
            raise ValueError(f"Invalid ui_density. Valid: {valid}")
        return value

    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, value: str) -> str:
        _parse_log_level(value)
        return value.strip().lower()

    @field_validator("log_levels")
    @classmethod
    def _validate_log_levels(cls, value: dict[str, str]) -> dict[str, str]:
        return _coerce_log_levels(value)

    @field_validator("stream_drop_mode")
    @classmethod
    def _validate_stream_drop_mode(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in {"drop_oldest", "drop_newest"}:
            raise ValueError("stream_drop_mode must be drop_oldest or drop_newest")
        return normalized

    @classmethod
    def from_file(cls, path: str | Path) -> "CopexConfig":
        """Load configuration from TOML file."""
        import tomllib

        path = Path(path)
        if not path.exists():
            return cls()

        with open(path, "rb") as f:
            data = tomllib.load(f)

        return cls(**data)

    @classmethod
    def default_path(cls) -> Path:
        """Get default config file path."""
        return Path.home() / ".config" / "copex" / "config.toml"

    def to_client_options(self) -> dict[str, Any]:
        """Convert to CopilotClient options."""
        opts: dict[str, Any] = {
            "auto_start": self.auto_start,
            "auto_restart": self.auto_restart,
            "log_level": self.log_level,
        }

        # Use provided cli_path or auto-detect
        cli_path = self.cli_path or find_copilot_cli()
        if cli_path:
            opts["cli_path"] = cli_path

        if self.cli_url:
            opts["cli_url"] = self.cli_url
        if self.cwd:
            opts["cwd"] = self.cwd
        return opts

    def to_session_options(self) -> dict[str, Any]:
        """Convert to create_session options."""
        opts: dict[str, Any] = {
            "model": self.model.value,
            "model_reasoning_effort": self.reasoning_effort.value,
            "streaming": self.streaming,
        }

        # Skills
        if self.skills:
            opts["skills"] = self.skills

        # Instructions
        if self.instructions:
            opts["instructions"] = self.instructions
        elif self.instructions_file:
            instructions_path = Path(self.instructions_file)
            if not instructions_path.exists():
                raise FileNotFoundError(f"Instructions file not found: {instructions_path}")
            with open(instructions_path, "r", encoding="utf-8") as f:
                opts["instructions"] = f.read()

        # MCP servers
        if self.mcp_servers:
            opts["mcp_servers"] = self._apply_mcp_defaults(self.mcp_servers)
        elif self.mcp_config_file:
            config_path = Path(self.mcp_config_file)
            if not config_path.exists():
                raise FileNotFoundError(f"MCP config file not found: {config_path}")
            import json
            with open(config_path, "r", encoding="utf-8") as f:
                mcp_data = json.load(f)
                if "servers" in mcp_data:
                    opts["mcp_servers"] = self._apply_mcp_defaults(
                        list(mcp_data["servers"].values())
                    )

        # Tool filtering
        if self.available_tools is not None:
            opts["available_tools"] = self.available_tools
        if self.excluded_tools:
            opts["excluded_tools"] = self.excluded_tools

        return opts

    def _apply_mcp_defaults(self, servers: list[dict[str, Any]]) -> list[dict[str, Any]]:
        enriched: list[dict[str, Any]] = []
        for server in servers:
            payload = dict(server)
            payload.setdefault("http_timeout", self.mcp_http_timeout)
            payload.setdefault("http_max_retries", self.mcp_http_max_retries)
            payload.setdefault("http_retry_base_delay", self.mcp_http_retry_base_delay)
            payload.setdefault("http_retry_max_delay", self.mcp_http_retry_max_delay)
            if self.mcp_health_check_path and "health_check_path" not in payload:
                payload["health_check_path"] = self.mcp_health_check_path
            enriched.append(payload)
        return enriched
