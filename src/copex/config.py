"""Configuration management for Copex."""

import os
import shutil
import sys
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from copex.models import Model, ReasoningEffort


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
                    return str(copilot_path)
    
    return None


class RetryConfig(BaseModel):
    """Retry configuration."""

    max_retries: int = Field(default=5, ge=1, le=20, description="Maximum retry attempts")
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


class CopexConfig(BaseModel):
    """Main configuration for Copex client."""

    model: Model = Field(default=Model.GPT_5_2_CODEX, description="Model to use")
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

    # Session options
    timeout: float = Field(default=300.0, ge=10.0, description="Response timeout (seconds)")
    auto_continue: bool = Field(
        default=True, description="Auto-send 'Keep going' on interruption/error"
    )
    continue_prompt: str = Field(
        default="Keep going", description="Prompt to send on auto-continue"
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

    # Tool filtering
    available_tools: list[str] | None = Field(
        default=None,
        description="Whitelist of tools to enable (None = all)"
    )
    excluded_tools: list[str] = Field(
        default_factory=list,
        description="Blacklist of tools to disable"
    )

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
            try:
                with open(self.instructions_file, "r", encoding="utf-8") as f:
                    opts["instructions"] = f.read()
            except Exception:
                pass

        # MCP servers
        if self.mcp_servers:
            opts["mcp_servers"] = self.mcp_servers
        elif self.mcp_config_file:
            try:
                import json
                with open(self.mcp_config_file, "r", encoding="utf-8") as f:
                    mcp_data = json.load(f)
                    if "servers" in mcp_data:
                        opts["mcp_servers"] = list(mcp_data["servers"].values())
            except Exception:
                pass

        # Tool filtering
        if self.available_tools is not None:
            opts["available_tools"] = self.available_tools
        if self.excluded_tools:
            opts["excluded_tools"] = self.excluded_tools

        return opts
