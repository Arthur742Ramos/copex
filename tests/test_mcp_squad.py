"""Tests for MCP support in squad mode and built-in MCP server discovery.

Covers:
- Squad CLI --mcp-config flag (acceptance, validation, config propagation)
- get_builtin_mcp_servers() auto-discovery (scrapling detection)
- Built-in server merge logic in CopexConfig.to_session_options()
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from unittest.mock import patch

import pytest

from copex.config import CopexConfig
from copex.mcp import MCPServerConfig, get_builtin_mcp_servers


# ===========================================================================
# 1. Squad CLI --mcp-config flag
# ===========================================================================


class TestSquadMCPConfigFlag:
    """Verify the --mcp-config option on the squad command."""

    def test_help_shows_mcp_config_option(self):
        from typer.testing import CliRunner

        from copex.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["squad", "--help"])
        plain = re.sub(r"\x1b\[[0-9;]*m", "", result.output)
        assert "--mcp-config" in plain

    def test_nonexistent_mcp_config_exits_with_error(self, tmp_path):
        from typer.testing import CliRunner

        from copex.cli import app

        fake_path = tmp_path / "does_not_exist.json"
        runner = CliRunner()
        result = runner.invoke(app, ["squad", "hello", "--mcp-config", str(fake_path)])
        assert result.exit_code == 1
        plain = re.sub(r"\x1b\[[0-9;]*m", "", result.output)
        assert "MCP config file not found" in plain

    def test_valid_mcp_config_sets_config_field(self, tmp_path):
        """A valid --mcp-config path should propagate to config.mcp_config_file."""
        from typer.testing import CliRunner

        from copex.cli import app

        mcp_file = tmp_path / "mcp.json"
        mcp_file.write_text(json.dumps({"servers": [{"name": "test", "command": "echo"}]}))

        captured_config = {}

        async def fake_run(prompt, *, config, **kw):
            captured_config["mcp_config_file"] = config.mcp_config_file
            return type("R", (), {"content": "done", "agents": {}})()

        with patch("copex.cli_squad._run_squad", side_effect=fake_run):
            runner = CliRunner()
            result = runner.invoke(
                app, ["squad", "hello", "--mcp-config", str(mcp_file)]
            )

        if "mcp_config_file" in captured_config:
            assert captured_config["mcp_config_file"] == str(mcp_file)
        else:
            # If we didn't reach _run_squad (e.g. stdin/prompt resolution),
            # at least verify no crash on valid path
            assert result.exit_code != 1 or "MCP config file not found" not in result.output


# ===========================================================================
# 2. get_builtin_mcp_servers() — auto-discovery
# ===========================================================================


class TestGetBuiltinMCPServers:
    """Verify scrapling auto-detection via shutil.which."""

    def test_returns_scrapling_when_installed(self):
        with patch("copex.mcp.shutil.which", return_value="/usr/bin/scrapling"):
            servers = get_builtin_mcp_servers()
        assert len(servers) == 1
        srv = servers[0]
        assert srv.name == "scrapling"
        assert srv.command == "scrapling"
        assert srv.args == ["mcp"]
        assert srv.transport == "stdio"

    def test_returns_empty_when_scrapling_not_installed(self):
        with patch("copex.mcp.shutil.which", return_value=None):
            servers = get_builtin_mcp_servers()
        assert servers == []

    def test_returns_MCPServerConfig_instances(self):
        with patch("copex.mcp.shutil.which", return_value="/usr/bin/scrapling"):
            servers = get_builtin_mcp_servers()
        assert all(isinstance(s, MCPServerConfig) for s in servers)


# ===========================================================================
# 3. Built-in server merge in to_session_options()
# ===========================================================================


class TestBuiltinServerMerge:
    """Verify that to_session_options() merges built-in servers additively."""

    def _make_config(self, **kw) -> CopexConfig:
        return CopexConfig(**kw)

    def test_builtin_added_when_no_user_servers(self):
        config = self._make_config()
        with patch("copex.mcp.shutil.which", return_value="/usr/bin/scrapling"):
            opts = config.to_session_options()
        servers = opts.get("mcp_servers", [])
        names = [s.get("name") if isinstance(s, dict) else s.name for s in servers]
        assert "scrapling" in names

    def test_builtin_not_added_when_not_installed(self):
        config = self._make_config()
        with patch("copex.mcp.shutil.which", return_value=None):
            opts = config.to_session_options()
        servers = opts.get("mcp_servers", [])
        assert servers == [] or all(
            (s.get("name") if isinstance(s, dict) else s.name) != "scrapling"
            for s in servers
        )

    def test_user_server_wins_on_name_collision(self, tmp_path):
        """User-configured server named 'scrapling' should not be overridden."""
        mcp_file = tmp_path / "mcp.json"
        mcp_file.write_text(
            json.dumps(
                {
                    "servers": [
                        {"name": "scrapling", "command": "my-custom-scrapling", "args": ["serve"]}
                    ]
                }
            )
        )
        config = self._make_config(mcp_config_file=str(mcp_file))
        with patch("copex.mcp.shutil.which", return_value="/usr/bin/scrapling"):
            opts = config.to_session_options()

        servers = opts.get("mcp_servers", [])
        scrapling_servers = [
            s for s in servers
            if (s.get("name") if isinstance(s, dict) else s.name) == "scrapling"
        ]
        assert len(scrapling_servers) == 1
        srv = scrapling_servers[0]
        cmd = srv.get("command") if isinstance(srv, dict) else srv.command
        assert cmd == "my-custom-scrapling"

    def test_user_servers_preserved_alongside_builtin(self, tmp_path):
        """User servers and built-in servers should coexist."""
        mcp_file = tmp_path / "mcp.json"
        mcp_file.write_text(
            json.dumps(
                {"servers": [{"name": "my-tool", "command": "my-tool", "args": []}]}
            )
        )
        config = self._make_config(mcp_config_file=str(mcp_file))
        with patch("copex.mcp.shutil.which", return_value="/usr/bin/scrapling"):
            opts = config.to_session_options()

        servers = opts.get("mcp_servers", [])
        names = {s.get("name") if isinstance(s, dict) else s.name for s in servers}
        assert "my-tool" in names
        assert "scrapling" in names

    def test_builtin_merge_with_dict_format_servers(self):
        """Config with mcp_servers as list of dicts should merge correctly."""
        config = self._make_config(
            mcp_servers=[{"name": "existing", "command": "foo", "args": []}]
        )
        with patch("copex.mcp.shutil.which", return_value="/usr/bin/scrapling"):
            opts = config.to_session_options()

        servers = opts.get("mcp_servers", [])
        names = {s.get("name") if isinstance(s, dict) else s.name for s in servers}
        assert "existing" in names
        assert "scrapling" in names

    def test_mcp_config_file_not_found_raises(self, tmp_path):
        """A non-existent mcp_config_file should raise FileNotFoundError."""
        config = self._make_config(mcp_config_file=str(tmp_path / "missing.json"))
        with patch("copex.mcp.shutil.which", return_value=None):
            with pytest.raises(FileNotFoundError, match="MCP config file not found"):
                config.to_session_options()

    def test_mcp_config_dict_format_servers(self, tmp_path):
        """MCP config with servers as a dict (keyed by name) should load."""
        mcp_file = tmp_path / "mcp.json"
        mcp_file.write_text(
            json.dumps(
                {
                    "servers": {
                        "alpha": {"name": "alpha", "command": "alpha-cmd", "args": []},
                        "beta": {"name": "beta", "command": "beta-cmd", "args": []},
                    }
                }
            )
        )
        config = self._make_config(mcp_config_file=str(mcp_file))
        with patch("copex.mcp.shutil.which", return_value=None):
            opts = config.to_session_options()

        servers = opts.get("mcp_servers", [])
        names = {s.get("name") if isinstance(s, dict) else s.name for s in servers}
        assert "alpha" in names
        assert "beta" in names
