"""Copex - Copilot Extended: A resilient wrapper for GitHub Copilot SDK."""

from __future__ import annotations

# ruff: noqa: F401
from typing import TYPE_CHECKING, Any


__version__ = "2.23.0"

# Re-export core components for convenience
from .agent import AgentResult, AgentSession, AgentTurn
from .approval import (
    ApprovalAction,
    ApprovalGate,
    ApprovalMode,
    ApprovalWorkflow,
    AuditEntry,
    AuditLogger,
    ChangePreview,
    ChangeStats,
    RiskAssessor,
)
from .backoff import AdaptiveRetry, with_retry
from .cache import StepCache, clear_global_cache, get_cache
from .checkpoint import CheckpointedRalph, CheckpointStore
from .cli_client import CopilotCLI
from .client import Copex
from .conditions import Condition, ConditionContext, all_of, any_of, none_of, when
from .config import CopexConfig, find_copilot_cli, make_client
from .edits import (
    EditBatchResult,
    EditFormat,
    EditOperation,
    UndoResult,
    VerificationReport,
    apply_edit_operations,
    apply_edit_text,
    list_undo_history,
    parse_structured_edits,
    run_verification,
    undo_last_edit_batch,
)

# Export new modules
from .exceptions import (
    AllModelsUnavailable,
    AuthenticationError,
    CircuitBreakerOpen,
    ConfigError,
    ConfigurationError,
    CopexConnectionError,
    CopexError,
    CopexTimeoutError,
    MCPError,
    PlanExecutionError,
    RateLimitError,
    RetryError,
    SecurityError,
    SessionError,
    SessionRecoveryFailed,
    StreamingError,
    ToolExecutionError,
    ValidationError,
)
from .fleet import (
    AdaptiveConcurrency,
    Fleet,
    FleetConfig,
    FleetCoordinator,
    FleetMailbox,
    FleetResult,
    FleetSummary,
    FleetTask,
    summarize_fleet_results,
)
from .fleet_store import FleetStore, RunRecord, TaskRecord
from .metrics import MetricsCollector, RequestMetrics, SessionMetrics
from .models import (
    Model,
    ReasoningEffort,
    discover_models,
    get_available_models,
    model_supports_reasoning,
    no_reasoning_models,
    refresh_model_capabilities,
    resolve_model,
)
from .persistence import PersistentSession, SessionStore
from .skills import SkillDiscovery, SkillInfo, get_skill_content, list_skills
from .squad import (
    SquadAgent,
    SquadAgentResult,
    SquadAggregationStrategy,
    SquadCoordinator,
    SquadResult,
    SquadRole,
    SquadTeam,
)
from .stats import RunStats, StatsTracker, load_start_commit, load_state, save_start_commit
from .templates import (
    StepInstance,
    StepTemplate,
    TemplateRegistry,
    build_workflow,
    create_step,
    deploy_workflow,
    get_registry,
    test_workflow,
)
from .visualization import render_ascii, render_mermaid, visualize_plan

if TYPE_CHECKING:
    from .repo_map import RelevantFile, RepoMap, RepoMapFile

__all__ = [
    # Core API
    "Copex",
    "CopexConfig",
    "CopilotCLI",
    "make_client",
    "find_copilot_cli",
    "Model",
    "ReasoningEffort",
    "resolve_model",
    "discover_models",
    "get_available_models",
    "model_supports_reasoning",
    "refresh_model_capabilities",
    "no_reasoning_models",
    # Main orchestration
    "AgentSession",
    "AgentTurn",
    "AgentResult",
    "Fleet",
    "FleetTask",
    "FleetConfig",
    "FleetCoordinator",
    "FleetMailbox",
    "FleetResult",
    "FleetSummary",
    "summarize_fleet_results",
    "SquadCoordinator",
    "SquadTeam",
    "SquadAgent",
    "SquadRole",
    "SquadResult",
    "SquadAgentResult",
    "SquadAggregationStrategy",
    # Runtime support
    "SessionStore",
    "PersistentSession",
    "CheckpointStore",
    "CheckpointedRalph",
    "MetricsCollector",
    "RequestMetrics",
    "SessionMetrics",
    "list_skills",
    "get_skill_content",
    "SkillDiscovery",
    "SkillInfo",
    # Exception API
    "CopexError",
    "ConfigError",
    "ConfigurationError",
    "ValidationError",
    "SecurityError",
    "AuthenticationError",
    "RateLimitError",
    "RetryError",
    "MCPError",
    "PlanExecutionError",
    "CircuitBreakerOpen",
    "SessionError",
    "SessionRecoveryFailed",
    "AllModelsUnavailable",
    "ToolExecutionError",
    "StreamingError",
    "CopexTimeoutError",
    "CopexConnectionError",
    # NOTE: TimeoutError and ConnectionError are deprecated aliases kept for backward
    # compatibility but intentionally excluded from __all__ to avoid shadowing builtins.
    # Users should use CopexTimeoutError and CopexConnectionError instead.
    # Approval
    "ApprovalAction",
    "ApprovalGate",
    "ApprovalMode",
    "ApprovalWorkflow",
    "AuditEntry",
    "AuditLogger",
    "ChangePreview",
    "ChangeStats",
    "RiskAssessor",
    # Backoff / retry
    "AdaptiveRetry",
    "with_retry",
    # Cache
    "StepCache",
    "clear_global_cache",
    "get_cache",
    # Conditions
    "Condition",
    "ConditionContext",
    "all_of",
    "any_of",
    "none_of",
    "when",
    # Edits
    "EditBatchResult",
    "EditFormat",
    "EditOperation",
    "UndoResult",
    "VerificationReport",
    "apply_edit_operations",
    "apply_edit_text",
    "list_undo_history",
    "parse_structured_edits",
    "run_verification",
    "undo_last_edit_batch",
    # Fleet extras
    "AdaptiveConcurrency",
    "FleetStore",
    "RunRecord",
    "TaskRecord",
    # Stats
    "RunStats",
    "StatsTracker",
    "load_start_commit",
    "load_state",
    "save_start_commit",
    # Templates
    "StepInstance",
    "StepTemplate",
    "TemplateRegistry",
    "build_workflow",
    "create_step",
    "deploy_workflow",
    "get_registry",
    "test_workflow",
    # Visualization
    "render_ascii",
    "render_mermaid",
    "visualize_plan",
    # Repo map (lazy)
    "RelevantFile",
    "RepoMap",
    "RepoMapFile",
]


def __getattr__(name: str) -> Any:
    if name in {"RepoMap", "RepoMapFile", "RelevantFile"}:
        from .repo_map import RelevantFile, RepoMap, RepoMapFile

        return {
            "RepoMap": RepoMap,
            "RepoMapFile": RepoMapFile,
            "RelevantFile": RelevantFile,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
