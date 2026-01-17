"""
MCP (Model Context Protocol) Integration for Copex.

Enables:
- Connecting to external MCP servers
- Exposing tools from MCP servers to Copex
- Running a Copex MCP server for other clients

Based on the MCP specification for tool/resource sharing.
"""

from __future__ import annotations

import asyncio
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable
from pathlib import Path


@dataclass
class MCPTool:
    """Definition of an MCP tool."""

    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[..., Awaitable[Any]] | None = None


@dataclass
class MCPResource:
    """Definition of an MCP resource."""

    uri: str
    name: str
    description: str = ""
    mime_type: str = "text/plain"


@dataclass
class MCPServerConfig:
    """Configuration for an MCP server connection."""

    name: str
    command: str | list[str]  # Command to launch server
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    cwd: str | None = None
    
    # Connection settings
    transport: str = "stdio"  # "stdio" or "http"
    url: str | None = None  # For HTTP transport
    
    # Behavior
    auto_start: bool = True
    restart_on_crash: bool = True


class MCPTransport(ABC):
    """Abstract base for MCP transport."""

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection."""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection."""
        pass

    @abstractmethod
    async def send(self, message: dict[str, Any]) -> dict[str, Any]:
        """Send a JSON-RPC message and get response."""
        pass

    @abstractmethod
    async def receive(self) -> dict[str, Any]:
        """Receive a message."""
        pass


class StdioTransport(MCPTransport):
    """MCP transport over stdio (subprocess)."""

    def __init__(self, config: MCPServerConfig):
        self.config = config
        self._process: asyncio.subprocess.Process | None = None
        self._request_id = 0
        self._pending: dict[int, asyncio.Future] = {}
        self._reader_task: asyncio.Task | None = None

    async def connect(self) -> None:
        """Start the MCP server process."""
        cmd = self.config.command
        if isinstance(cmd, str):
            cmd = [cmd] + self.config.args
        else:
            cmd = list(cmd) + self.config.args

        self._process = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=self.config.cwd,
            env={**dict(__import__("os").environ), **self.config.env} if self.config.env else None,
        )

        # Start reader task
        self._reader_task = asyncio.create_task(self._reader_loop())

        # Initialize connection
        await self._initialize()

    async def _initialize(self) -> None:
        """Send MCP initialization."""
        response = await self.send({
            "jsonrpc": "2.0",
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {},
                    "resources": {},
                },
                "clientInfo": {
                    "name": "copex",
                    "version": "0.1.0",
                },
            },
        })
        
        # Send initialized notification
        await self._write({
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
        })

    async def _reader_loop(self) -> None:
        """Read responses from the server."""
        if not self._process or not self._process.stdout:
            return

        buffer = ""
        while True:
            try:
                line = await self._process.stdout.readline()
                if not line:
                    break

                text = line.decode("utf-8").strip()
                if not text:
                    continue

                try:
                    message = json.loads(text)
                except json.JSONDecodeError:
                    continue

                # Handle response
                if "id" in message:
                    request_id = message["id"]
                    if request_id in self._pending:
                        future = self._pending.pop(request_id)
                        if "error" in message:
                            future.set_exception(
                                RuntimeError(message["error"].get("message", "Unknown error"))
                            )
                        else:
                            future.set_result(message.get("result"))

            except asyncio.CancelledError:
                break
            except Exception:
                continue

    async def _write(self, message: dict[str, Any]) -> None:
        """Write a message to the server."""
        if not self._process or not self._process.stdin:
            raise RuntimeError("Not connected")

        data = json.dumps(message) + "\n"
        self._process.stdin.write(data.encode("utf-8"))
        await self._process.stdin.drain()

    async def send(self, message: dict[str, Any]) -> dict[str, Any]:
        """Send a request and wait for response."""
        self._request_id += 1
        request_id = self._request_id

        message = {**message, "id": request_id}

        future: asyncio.Future = asyncio.get_event_loop().create_future()
        self._pending[request_id] = future

        await self._write(message)

        try:
            result = await asyncio.wait_for(future, timeout=30.0)
            return result
        except asyncio.TimeoutError:
            self._pending.pop(request_id, None)
            raise

    async def receive(self) -> dict[str, Any]:
        """Receive is handled by reader loop."""
        raise NotImplementedError("Use send() for request/response")

    async def disconnect(self) -> None:
        """Stop the server process."""
        if self._reader_task:
            self._reader_task.cancel()
            try:
                await self._reader_task
            except asyncio.CancelledError:
                pass

        if self._process:
            self._process.terminate()
            try:
                await asyncio.wait_for(self._process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                self._process.kill()
            self._process = None


class MCPClient:
    """
    Client for connecting to MCP servers.

    Usage:
        config = MCPServerConfig(
            name="my-server",
            command="npx",
            args=["-y", "@my/mcp-server"],
        )

        client = MCPClient(config)
        await client.connect()

        # List available tools
        tools = await client.list_tools()

        # Call a tool
        result = await client.call_tool("search", {"query": "hello"})

        await client.disconnect()
    """

    def __init__(self, config: MCPServerConfig):
        self.config = config
        self._transport: MCPTransport | None = None
        self._tools: list[MCPTool] = []
        self._resources: list[MCPResource] = []

    @property
    def connected(self) -> bool:
        """Check if connected."""
        return self._transport is not None

    async def connect(self) -> None:
        """Connect to the MCP server."""
        if self.config.transport == "stdio":
            self._transport = StdioTransport(self.config)
        else:
            raise ValueError(f"Unsupported transport: {self.config.transport}")

        await self._transport.connect()

        # Fetch available tools and resources
        await self._refresh_capabilities()

    async def _refresh_capabilities(self) -> None:
        """Refresh list of tools and resources."""
        if not self._transport:
            return

        # Get tools
        try:
            result = await self._transport.send({
                "jsonrpc": "2.0",
                "method": "tools/list",
                "params": {},
            })
            self._tools = [
                MCPTool(
                    name=t["name"],
                    description=t.get("description", ""),
                    input_schema=t.get("inputSchema", {}),
                )
                for t in result.get("tools", [])
            ]
        except Exception:
            self._tools = []

        # Get resources
        try:
            result = await self._transport.send({
                "jsonrpc": "2.0",
                "method": "resources/list",
                "params": {},
            })
            self._resources = [
                MCPResource(
                    uri=r["uri"],
                    name=r["name"],
                    description=r.get("description", ""),
                    mime_type=r.get("mimeType", "text/plain"),
                )
                for r in result.get("resources", [])
            ]
        except Exception:
            self._resources = []

    async def disconnect(self) -> None:
        """Disconnect from the MCP server."""
        if self._transport:
            await self._transport.disconnect()
            self._transport = None

    async def list_tools(self) -> list[MCPTool]:
        """Get list of available tools."""
        return self._tools.copy()

    async def list_resources(self) -> list[MCPResource]:
        """Get list of available resources."""
        return self._resources.copy()

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        """
        Call a tool on the MCP server.

        Args:
            name: Tool name
            arguments: Tool arguments

        Returns:
            Tool result
        """
        if not self._transport:
            raise RuntimeError("Not connected")

        result = await self._transport.send({
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": name,
                "arguments": arguments,
            },
        })

        # Extract content from result
        content = result.get("content", [])
        if content and len(content) == 1:
            return content[0].get("text", content[0])
        return content

    async def read_resource(self, uri: str) -> str:
        """
        Read a resource from the MCP server.

        Args:
            uri: Resource URI

        Returns:
            Resource content
        """
        if not self._transport:
            raise RuntimeError("Not connected")

        result = await self._transport.send({
            "jsonrpc": "2.0",
            "method": "resources/read",
            "params": {"uri": uri},
        })

        contents = result.get("contents", [])
        if contents:
            return contents[0].get("text", "")
        return ""

    def get_copex_tools(self) -> list[dict[str, Any]]:
        """
        Get tools formatted for Copex session.

        Returns:
            List of tool definitions for create_session
        """
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.input_schema,
            }
            for tool in self._tools
        ]


class MCPManager:
    """
    Manages multiple MCP server connections.

    Usage:
        manager = MCPManager()

        # Add servers
        manager.add_server(MCPServerConfig(
            name="github",
            command="npx",
            args=["-y", "@github/mcp-server"],
        ))

        # Connect all
        await manager.connect_all()

        # Get all tools across servers
        all_tools = manager.get_all_tools()

        # Call a tool (auto-routes to correct server)
        result = await manager.call_tool("github:search_repos", {...})

        await manager.disconnect_all()
    """

    def __init__(self):
        self._servers: dict[str, MCPServerConfig] = {}
        self._clients: dict[str, MCPClient] = {}

    def add_server(self, config: MCPServerConfig) -> None:
        """Add an MCP server configuration."""
        self._servers[config.name] = config

    def remove_server(self, name: str) -> None:
        """Remove an MCP server."""
        self._servers.pop(name, None)

    async def connect(self, name: str) -> MCPClient:
        """Connect to a specific server."""
        config = self._servers.get(name)
        if not config:
            raise ValueError(f"Unknown server: {name}")

        client = MCPClient(config)
        await client.connect()
        self._clients[name] = client
        return client

    async def connect_all(self) -> None:
        """Connect to all configured servers."""
        for name in self._servers:
            if name not in self._clients:
                await self.connect(name)

    async def disconnect(self, name: str) -> None:
        """Disconnect from a specific server."""
        client = self._clients.pop(name, None)
        if client:
            await client.disconnect()

    async def disconnect_all(self) -> None:
        """Disconnect from all servers."""
        for name in list(self._clients.keys()):
            await self.disconnect(name)

    def get_client(self, name: str) -> MCPClient | None:
        """Get a connected client by name."""
        return self._clients.get(name)

    def get_all_tools(self) -> list[dict[str, Any]]:
        """Get all tools across all connected servers."""
        tools = []
        for server_name, client in self._clients.items():
            for tool in client._tools:
                tools.append({
                    "name": f"{server_name}:{tool.name}",
                    "description": f"[{server_name}] {tool.description}",
                    "parameters": tool.input_schema,
                    "_server": server_name,
                    "_tool": tool.name,
                })
        return tools

    async def call_tool(self, qualified_name: str, arguments: dict[str, Any]) -> Any:
        """
        Call a tool by qualified name (server:tool).

        Args:
            qualified_name: "server_name:tool_name"
            arguments: Tool arguments

        Returns:
            Tool result
        """
        if ":" not in qualified_name:
            raise ValueError(f"Expected qualified name 'server:tool', got: {qualified_name}")

        server_name, tool_name = qualified_name.split(":", 1)
        client = self._clients.get(server_name)

        if not client:
            raise ValueError(f"Server not connected: {server_name}")

        return await client.call_tool(tool_name, arguments)


def load_mcp_config(path: Path | str | None = None) -> list[MCPServerConfig]:
    """
    Load MCP server configurations from file.

    Default locations:
    - .copex/mcp.json (project)
    - ~/.copex/mcp.json (global)

    Config format:
    {
        "servers": {
            "github": {
                "command": "npx",
                "args": ["-y", "@github/mcp-server"],
                "env": {"GITHUB_TOKEN": "..."}
            }
        }
    }
    """
    if path:
        config_path = Path(path)
    else:
        # Try project, then global
        project_path = Path(".copex/mcp.json")
        global_path = Path.home() / ".copex" / "mcp.json"

        if project_path.exists():
            config_path = project_path
        elif global_path.exists():
            config_path = global_path
        else:
            return []

    if not config_path.exists():
        return []

    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    servers = []
    for name, config in data.get("servers", {}).items():
        servers.append(MCPServerConfig(
            name=name,
            command=config["command"],
            args=config.get("args", []),
            env=config.get("env", {}),
            cwd=config.get("cwd"),
            transport=config.get("transport", "stdio"),
            url=config.get("url"),
            auto_start=config.get("auto_start", True),
            restart_on_crash=config.get("restart_on_crash", True),
        ))

    return servers
