"""Copex - Copilot Extended: A resilient wrapper for GitHub Copilot SDK."""

from copex.client import Copex
from copex.config import CopexConfig, find_copilot_cli
from copex.models import Model, ReasoningEffort

# Ralph Wiggum loops
from copex.ralph import RalphWiggum, RalphConfig, RalphState, ralph_loop

# Persistence
from copex.persistence import SessionStore, PersistentSession, Message, SessionData

# Checkpointing
from copex.checkpoint import CheckpointStore, Checkpoint, CheckpointedRalph

# Metrics
from copex.metrics import MetricsCollector, RequestMetrics, SessionMetrics, get_collector

# Parallel tools
from copex.tools import ToolRegistry, ParallelToolExecutor, ToolResult

# MCP integration
from copex.mcp import MCPClient, MCPManager, MCPServerConfig, MCPTool, load_mcp_config

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
__version__ = "0.1.0"
