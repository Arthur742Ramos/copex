"""Configuration management for Copex."""

import os
import shutil
import sys
import warnings
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from copex.models import Model, ReasoningEffort, normalize_reasoning_effort, parse_reasoning_effort


def _parse_node_version(path: Path) -> tuple[int, ...]:
    """Parse a Node version directory name (e.g. 'v18.17.0') into a sortable tuple."""
    name = path.name.lstrip("v")
    try:
        return tuple(int(x) for x in name.split("."))
    except (ValueError, AttributeError):
        return (0,)


def find_copilot_cli() -> str | None:
    """Auto-detect the Copilot CLI path across platforms.

    Searches in order:
    1. shutil.which('copilot') - system PATH
    2. Common npm global locations
    3. Common installation paths

    Returns the path if found, None otherwise.
    """
    # First try PATH (works on all platforms)
    cli_path = shutil.which("copilot")
    if cli_path:
        # On Windows, prefer .cmd over .ps1 for subprocess compatibility
        if sys.platform == "win32" and cli_path.endswith(".ps1"):
            cmd_path = cli_path.replace(".ps1", ".cmd")
            if os.path.exists(cmd_path):
                return cmd_path
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
            candidates.append(
                Path(os.environ["USERPROFILE"]) / "AppData" / "Roaming" / "npm" / "copilot.cmd"
            )
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
            return str(candidate)

    # Check for NVM installations (macOS/Linux)
    if sys.platform != "win32":
        nvm_dir = Path.home() / ".nvm" / "versions" / "node"
        if nvm_dir.exists():
            # Find latest node version
            versions = sorted(nvm_dir.iterdir(), key=_parse_node_version, reverse=True)
            for version in versions:
                copilot_path = version / "bin" / "copilot"
                if copilot_path.exists():
                    return str(copilot_path)

    return None


UI_THEMES = {"default", "midnight", "mono", "sunset"}
UI_DENSITIES = {"compact", "extended"}


class RetryConfig(BaseModel):
    """Retry configuration."""

    max_retries: int = Field(default=5, ge=1, le=20, description="Maximum retry attempts")
    max_auto_continues: int = Field(
        default=3, ge=0, le=10, description="Maximum auto-continue cycles after exhausting retries"
    )
    base_delay: float = Field(
        default=1.0, ge=0.1, description="Base delay between retries (seconds)"
    )
    max_delay: float = Field(
        default=30.0, ge=1.0, description="Maximum delay between retries (seconds)"
    )
    exponential_base: float = Field(
        default=2.0, ge=1.5, description="Exponential backoff multiplier"
    )
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


class CopexConfig(BaseModel):
    """Main configuration for Copex client."""

    model: Model = Field(default=Model.CLAUDE_OPUS_4_6, description="Model to use")
    reasoning_effort: ReasoningEffort = Field(
        default=ReasoningEffort.XHIGH, description="Reasoning effort level"
    )
    streaming: bool = Field(default=True, description="Enable streaming responses")
    retry: RetryConfig = Field(default_factory=RetryConfig, description="Retry configuration")

    # Client options
    use_cli: bool = Field(default=False, description="Use CLI subprocess instead of SDK (supports all models)")
    cli_path: str | None = Field(default=None, description="Path to Copilot CLI executable")
    cli_url: str | None = Field(default=None, description="URL of existing CLI server")
    cwd: str | None = Field(default=None, description="Working directory for CLI process")
    auto_start: bool = Field(default=True, description="Auto-start CLI server")
    auto_restart: bool = Field(default=True, description="Auto-restart on crash")
    log_level: str = Field(default="warning", description="Log level")

    # Session options
    timeout: float = Field(
        default=300.0, ge=10.0, description="Inactivity timeout (seconds) - resets on each event"
    )
    auto_continue: bool = Field(
        default=True, description="Auto-send 'Keep going' on interruption/error"
    )
    continue_prompt: str = Field(
        default="Keep going", description="Prompt to send on auto-continue"
    )

    # Skills and capabilities
    skills: list[str] = Field(
        default_factory=list, description="Skills to enable (e.g., ['code-review', 'azure-openai'])"
    )
    skill_directories: list[str] = Field(
        default_factory=list, description="Explicit skill directories to load"
    )
    disabled_skills: list[str] = Field(
        default_factory=list, description="Skills to disable"
    )
    auto_discover_skills: bool = Field(
        default=True, description="Auto-discover skills from repo and user dirs"
    )
    working_directory: str | None = Field(
        default=None, description="Working directory for SDK (defaults to cwd)"
    )
    instructions: str | None = Field(
        default=None, description="Custom instructions for the session"
    )
    instructions_file: str | None = Field(
        default=None, description="Path to instructions file (.md)"
    )

    # MCP configuration
    mcp_servers: list[dict[str, Any]] = Field(
        default_factory=list, description="MCP server configurations"
    )
    mcp_config_file: str | None = Field(default=None, description="Path to MCP config JSON file")

    # Tool filtering
    available_tools: list[str] | None = Field(
        default=None, description="Whitelist of tools to enable (None = all)"
    )
    excluded_tools: list[str] = Field(
        default_factory=list, description="Blacklist of tools to disable"
    )

    # UI options
    ui_theme: str = Field(
        default="default", description="UI theme (default, midnight, mono, sunset)"
    )
    ui_density: str = Field(default="extended", description="UI density (compact or extended)")

    @field_validator("reasoning_effort", mode="before")
    @classmethod
    def _parse_reasoning_effort_aliases(cls, value: Any) -> Any:
        # Accept short aliases like "xh".
        try:
            parsed = parse_reasoning_effort(value)
        except Exception:
            return value
        return parsed if parsed is not None else value

    @model_validator(mode="after")
    def _cap_reasoning_by_model(self) -> "CopexConfig":
        normalized, warning = normalize_reasoning_effort(self.model, self.reasoning_effort)
        if warning and normalized != self.reasoning_effort:
            # Only warn when the user explicitly set reasoning_effort.
            # (Avoid noisy warnings when defaults get normalized.)
            fields_set = getattr(self, "__pydantic_fields_set__", set())
            if "reasoning_effort" in fields_set:
                warnings.warn(warning, UserWarning)
            self.reasoning_effort = normalized
        return self

    @field_validator("skill_directories", mode="after")
    @classmethod
    def _warn_nonexistent_skill_dirs(cls, value: list[str]) -> list[str]:
        for d in value:
            p = Path(d)
            if not p.exists():
                warnings.warn(f"Skill directory does not exist: {d}", UserWarning)
        return value

    @model_validator(mode="after")
    def _apply_env_overrides(self) -> "CopexConfig":
        env_model = os.environ.get("COPEX_MODEL")
        if env_model:
            try:
                self.model = Model(env_model)
            except ValueError:
                valid = ", ".join(m.value for m in Model)
                warnings.warn(
                    f"COPEX_MODEL='{env_model}' is not valid. Valid: {valid}",
                    UserWarning,
                )

        env_reasoning = os.environ.get("COPEX_REASONING")
        if env_reasoning:
            try:
                parsed = parse_reasoning_effort(env_reasoning)
                if parsed is not None:
                    self.reasoning_effort = parsed
            except ValueError:
                valid = ", ".join(r.value for r in ReasoningEffort)
                warnings.warn(
                    f"COPEX_REASONING='{env_reasoning}' is not valid. Valid: {valid}",
                    UserWarning,
                )
        return self

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

    @classmethod
    def from_file(cls, path: str | Path) -> "CopexConfig":
        """Load configuration from TOML file."""
        try:
            import tomllib  # Python 3.11+
        except Exception:
            import tomli as tomllib  # type: ignore

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
        from copex.models import _NO_REASONING_MODELS
        from copex.skills import SkillDiscovery

        model_id = self.model.value
        opts: dict[str, Any] = {
            "model": model_id,
            "streaming": self.streaming,
        }
        # Only include reasoning_effort for models that support it
        if model_id not in _NO_REASONING_MODELS:
            opts["reasoning_effort"] = self.reasoning_effort.value

        # Skills (named skills like 'code-review')
        if self.skills:
            opts["skills"] = self.skills

        # Skill directories - auto-discover + explicit
        discovery = SkillDiscovery(
            explicit_dirs=[Path(d) for d in self.skill_directories],
            disabled_skills=set(self.disabled_skills),
            auto_discover=self.auto_discover_skills,
        )
        skill_dirs = discovery.get_skill_directories_for_sdk()
        if skill_dirs:
            opts["skill_directories"] = skill_dirs

        # Disabled skills
        if self.disabled_skills:
            opts["disabled_skills"] = self.disabled_skills

        # Working directory (defaults to cwd)
        opts["working_directory"] = self.working_directory or str(Path.cwd())

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
            opts["mcp_servers"] = self.mcp_servers
        elif self.mcp_config_file:
            config_path = Path(self.mcp_config_file)
            if not config_path.exists():
                raise FileNotFoundError(f"MCP config file not found: {config_path}")
            import json

            with open(config_path, "r", encoding="utf-8") as f:
                mcp_data = json.load(f)
                if "servers" in mcp_data:
                    opts["mcp_servers"] = list(mcp_data["servers"].values())

        # Tool filtering
        if self.available_tools is not None:
            opts["available_tools"] = self.available_tools
        if self.excluded_tools:
            opts["excluded_tools"] = self.excluded_tools

        return opts


def make_client(config: CopexConfig) -> Any:
    """Create the appropriate client based on configuration.

    When ``use_cli`` is True, returns a CopilotCLI instance that wraps
    the Copilot CLI subprocess directly (supports all CLI models).
    Otherwise returns a Copex instance using the SDK.
    """
    if config.use_cli:
        from copex.cli_client import CopilotCLI
        return CopilotCLI(config)
    from copex.client import Copex
    return Copex(config)
