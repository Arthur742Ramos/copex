"""Copex - Copilot Extended: A resilient wrapper for GitHub Copilot SDK."""

# Checkpointing
from copex.checkpoint import Checkpoint, CheckpointedRalph, CheckpointStore
from copex.client import Copex
from copex.config import CopexConfig, configure_logging, find_copilot_cli

# MCP integration
from copex.mcp import MCPClient, MCPManager, MCPServerConfig, MCPTool, load_mcp_config

# Metrics
from copex.metrics import MetricsCollector, RequestMetrics, SessionMetrics, get_collector
from copex.models import Model, ReasoningEffort, TokenUsage

# Persistence
from copex.persistence import Message, PersistentSession, SessionData, SessionStore

# Plan mode
from copex.plan import Plan, PlanExecutor, PlanState, PlanStep, StepStatus

# Progress reporting
from copex.progress import (
    PlanProgressReporter,
    ProgressItem,
    ProgressReporter,
    ProgressState,
    ProgressStatus,
)

# Ralph Wiggum loops
from copex.ralph import RalphConfig, RalphState, RalphWiggum, ralph_loop

# Parallel tools
from copex.tools import ParallelToolExecutor, ToolRegistry, ToolResult

# UI Components
from copex.ui_components import (
    CodeBlock,
    CollapsibleGroup,
    CollapsibleSection,
    DiffDisplay,
    DiffLine,
    DiffLineType,
    PlainTextRenderer,
    RichProgressReporter,
    StepInfo,
    StepProgress,
    StepStatus,
    TokenUsageDisplay,
    ToolCallGroup,
    ToolCallPanel,
    ToolStatus,
    extract_code_blocks,
    get_plain_console,
    is_terminal,
    render_markdown_with_syntax,
)

# UI / Theme
from copex.ui import (
    CopexUI,
    Icons,
    StreamingContext,
    TerminalInfo,
    Theme,
    THEME_PRESETS,
    apply_theme,
    auto_apply_theme,
    get_recommended_theme,
    get_theme_for_terminal,
    get_theme_preview,
    is_light_theme,
    list_themes,
    supports_256_color,
    supports_true_color,
)

__all__ = [
    # Core
    "Copex",
    "CopexConfig",
    "configure_logging",
    "Model",
    "ReasoningEffort",
    "TokenUsage",
    "find_copilot_cli",
    # Ralph
    "RalphWiggum",
    "RalphConfig",
    "RalphState",
    "ralph_loop",
    # Plan
    "Plan",
    "PlanStep",
    "PlanExecutor",
    "PlanState",
    "StepStatus",
    # Progress
    "ProgressReporter",
    "ProgressState",
    "ProgressItem",
    "ProgressStatus",
    "PlanProgressReporter",
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
    # UI Components
    "CodeBlock",
    "CollapsibleGroup",
    "CollapsibleSection",
    "DiffDisplay",
    "DiffLine",
    "DiffLineType",
    "PlainTextRenderer",
    "RichProgressReporter",
    "StepInfo",
    "StepProgress",
    "StepStatus",
    "TokenUsageDisplay",
    "ToolCallGroup",
    "ToolCallPanel",
    "ToolStatus",
    "extract_code_blocks",
    "get_plain_console",
    "is_terminal",
    "render_markdown_with_syntax",
    # UI / Theme
    "CopexUI",
    "Icons",
    "StreamingContext",
    "TerminalInfo",
    "Theme",
    "THEME_PRESETS",
    "apply_theme",
    "auto_apply_theme",
    "get_recommended_theme",
    "get_theme_for_terminal",
    "get_theme_preview",
    "is_light_theme",
    "list_themes",
    "supports_256_color",
    "supports_true_color",
]
__version__ = "0.9.0"
