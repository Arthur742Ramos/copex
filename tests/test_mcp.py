from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from copex.mcp import MCPClient, MCPManager, MCPServerConfig, load_mcp_config


class FakeTransport:
    def __init__(self) -> None:
        self.sent: list[dict] = []

    async def connect(self) -> None:
        return None

    async def disconnect(self) -> None:
        return None

    async def send(self, message: dict) -> dict:
        self.sent.append(message)
        method = message.get("method")
        if method == "tools/list":
            return {"tools": [{"name": "search", "description": "Search", "inputSchema": {"type": "object"}}]}
        if method == "resources/list":
            return {"resources": [{"uri": "file://doc", "name": "doc", "mimeType": "text/plain"}]}
        if method == "tools/call":
            return {"content": [{"text": "ok"}]}
        if method == "resources/read":
            return {"contents": [{"text": "data"}]}
        return {}


@pytest.mark.asyncio
async def test_mcp_client_list_and_call(monkeypatch) -> None:
    config = MCPServerConfig(name="demo", command="echo")
    client = MCPClient(config)

    async def connect_stub(self):
        self._transport = FakeTransport()
        await self._refresh_capabilities()

    monkeypatch.setattr(MCPClient, "connect", connect_stub, raising=False)

    await client.connect()
    tools = await client.list_tools()
    resources = await client.list_resources()
    result = await client.call_tool("search", {"q": "hi"})
    content = await client.read_resource("file://doc")

    assert tools[0].name == "search"
    assert resources[0].uri == "file://doc"
    assert result == "ok"
    assert content == "data"


@pytest.mark.asyncio
async def test_mcp_manager_tools_and_call(monkeypatch) -> None:
    manager = MCPManager()
    manager.add_server(MCPServerConfig(name="demo", command="echo"))

    async def connect_stub(self):
        self._transport = FakeTransport()
        await self._refresh_capabilities()

    monkeypatch.setattr(MCPClient, "connect", connect_stub, raising=False)

    await manager.connect_all()
    tools = manager.get_all_tools()
    assert tools[0]["name"] == "demo:search"
    result = await manager.call_tool("demo:search", {"q": "hi"})
    assert result == "ok"

    await manager.disconnect_all()


def test_load_mcp_config(tmp_path: Path) -> None:
    config_path = tmp_path / "mcp.json"
    config_path.write_text(json.dumps({
        "servers": {
            "demo": {"command": "echo", "args": ["hello"]},
        }
    }))

    configs = load_mcp_config(config_path)
    assert len(configs) == 1
    assert configs[0].name == "demo"
