"""Monkey-patch CopilotClient to remove --no-auto-update from CLI startup args.

The Copilot SDK passes ``--no-auto-update`` when spawning the CLI server in
headless mode.  This prevents the binary from fetching the latest model list
from the Copilot backend, causing newer models (e.g. ``claude-opus-4.6``,
``claude-opus-4.6-fast``) to silently fall back to ``claude-sonnet-4.5``.

We monkey-patch ``CopilotClient._start_cli_server`` to strip that flag so
the CLI always operates with the up-to-date model catalogue.
"""

from __future__ import annotations

import asyncio
import logging

from copilot import CopilotClient

logger = logging.getLogger(__name__)

_original_start_cli_server = CopilotClient._start_cli_server  # type: ignore[attr-defined]


async def _patched_start_cli_server(self: CopilotClient) -> None:
    """Wrapper that removes ``--no-auto-update`` before starting the CLI."""
    import os
    import re
    import subprocess

    cli_path = self.options["cli_path"]

    if not os.path.exists(cli_path):
        raise RuntimeError(f"Copilot CLI not found at {cli_path}")

    # Build args WITHOUT --no-auto-update
    args = ["--headless", "--log-level", self.options["log_level"]]

    if self.options.get("github_token"):
        args.extend(["--auth-token-env", "COPILOT_SDK_AUTH_TOKEN"])
    if not self.options.get("use_logged_in_user", True):
        args.append("--no-auto-login")

    if cli_path.endswith(".js"):
        args = ["node", cli_path] + args
    else:
        args = [cli_path] + args

    env = self.options.get("env")
    if env is None:
        env = dict(os.environ)
    else:
        env = dict(env)

    if self.options.get("github_token"):
        env["COPILOT_SDK_AUTH_TOKEN"] = self.options["github_token"]

    if self.options["use_stdio"]:
        args.append("--stdio")
        self._process = subprocess.Popen(
            args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
            cwd=self.options["cwd"],
            env=env,
        )
    else:
        if self.options["port"] > 0:
            args.extend(["--port", str(self.options["port"])])
        self._process = subprocess.Popen(
            args,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=self.options["cwd"],
            env=env,
        )

    if self.options["use_stdio"]:
        return

    loop = asyncio.get_event_loop()
    process = self._process

    async def read_port() -> None:
        if not process or not process.stdout:
            raise RuntimeError("Process not started or stdout not available")
        while True:
            line = await loop.run_in_executor(None, process.stdout.readline)
            if not line:
                raise RuntimeError("CLI process exited before announcing port")
            line_str = line.decode() if isinstance(line, bytes) else line
            match = re.search(r"listening on port (\d+)", line_str, re.IGNORECASE)
            if match:
                self._actual_port = int(match.group(1))
                return

    try:
        await asyncio.wait_for(read_port(), timeout=10.0)
    except asyncio.TimeoutError:
        raise RuntimeError("Timeout waiting for CLI server to start") from None


def patch_copilot_client() -> None:
    """Apply the monkey-patch to CopilotClient."""
    CopilotClient._start_cli_server = _patched_start_cli_server  # type: ignore[attr-defined]
    logger.debug("Patched CopilotClient._start_cli_server to remove --no-auto-update")


# Auto-apply on import
patch_copilot_client()
