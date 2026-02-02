"""
Monkey-patch for the Copilot SDK to support model_reasoning_effort.

The official SDK doesn't pass through model_reasoning_effort to the CLI,
so we patch create_session to inject it into the payload.
"""

from __future__ import annotations

import functools
from typing import Any, Optional

_patched = False


def patch_sdk() -> None:
    """Apply the monkey-patch to support model_reasoning_effort."""
    global _patched
    if _patched:
        return

    from copilot import client as copilot_client

    _original_create_session = copilot_client.CopilotClient.create_session

    @functools.wraps(_original_create_session)
    async def _patched_create_session(
        self: Any, config: Optional[dict[str, Any]] = None
    ) -> Any:
        """Patched create_session that passes model_reasoning_effort."""
        if config and "model_reasoning_effort" in config:
            # The SDK builds payload manually and doesn't include this field.
            # We need to intercept the _client.request call.
            # Store it on the client instance temporarily.
            self._copex_reasoning_effort = config["model_reasoning_effort"]
        else:
            self._copex_reasoning_effort = None

        return await _original_create_session(self, config)

    # Also patch the underlying request to inject reasoning_effort
    _original_request = copilot_client.JsonRpcClient.request if hasattr(copilot_client, 'JsonRpcClient') else None
    
    # Alternative: patch at the jsonrpc level
    from copilot import jsonrpc as copilot_jsonrpc
    
    _original_jsonrpc_request = copilot_jsonrpc.JsonRpcClient.request

    @functools.wraps(_original_jsonrpc_request)
    async def _patched_jsonrpc_request(self: Any, method: str, params: Any) -> Any:
        """Patched request that injects model_reasoning_effort for session.create."""
        if method == "session.create" and isinstance(params, dict):
            # Check if the parent client has reasoning effort set
            # We need to find a way to get this... let's use a module-level var
            if _pending_reasoning_effort:
                params["model_reasoning_effort"] = _pending_reasoning_effort
        return await _original_jsonrpc_request(self, method, params)

    copilot_jsonrpc.JsonRpcClient.request = _patched_jsonrpc_request
    copilot_client.CopilotClient.create_session = _patched_create_session
    _patched = True


# Module-level storage for pending reasoning effort
_pending_reasoning_effort: str | None = None


def set_reasoning_effort(effort: str | None) -> None:
    """Set the reasoning effort to be injected into the next session.create call."""
    global _pending_reasoning_effort
    _pending_reasoning_effort = effort


def clear_reasoning_effort() -> None:
    """Clear the pending reasoning effort."""
    global _pending_reasoning_effort
    _pending_reasoning_effort = None
