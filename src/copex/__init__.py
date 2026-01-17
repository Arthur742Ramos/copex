"""Copex - Copilot Extended: A resilient wrapper for GitHub Copilot SDK."""

# Checkpointing
from copex.checkpoint import Checkpoint, CheckpointedRalph, CheckpointStore
from copex.client import Copex
from copex.config import CopexConfig, find_copilot_cli

# MCP integration
from copex.mcp import MCPClient, MCPManager, MCPServerConfig, MCPTool, load_mcp_config

# Metrics
from copex.metrics import MetricsCollector, RequestMetrics, SessionMetrics, get_collector
from copex.models import Model, ReasoningEffort

# Persistence
from copex.persistence import Message, PersistentSession, SessionData, SessionStore

# Ralph Wiggum loops
from copex.ralph import RalphConfig, RalphState, RalphWiggum, ralph_loop

# Parallel tools
from copex.tools import ParallelToolExecutor, ToolRegistry, ToolResult

__all__ = [
    # Core
    "Copex",
    "CopexConfig",
    "Model",
    "ReasoningEffort",
    "find_copilot_cli",
    # Ralph
    "RalphWiggum",
    "RalphConfig",
    "RalphState",
    "ralph_loop",
    # Persistence
    "SessionStore",
    "PersistentSession",
    "Message",
    "SessionData",
    # Checkpointing
    "CheckpointStore",
    "Checkpoint",
    "CheckpointedRalph",
    # Metrics
    "MetricsCollector",
    "RequestMetrics",
    "SessionMetrics",
    "get_collector",
    # Tools
    "ToolRegistry",
    "ParallelToolExecutor",
    "ToolResult",
    # MCP
    "MCPClient",
    "MCPManager",
    "MCPServerConfig",
    "MCPTool",
    "load_mcp_config",
]
__version__ = "0.3.0"
