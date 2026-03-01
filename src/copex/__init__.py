"""Copex - Copilot Extended: A resilient wrapper for GitHub Copilot SDK."""

__version__ = "2.10.0"

# Re-export core components for convenience
from .agent import AgentResult, AgentSession, AgentTurn
from .backoff import AdaptiveRetry, BackoffStrategy, ErrorCategory, with_retry
from .cache import StepCache, clear_global_cache, get_cache
from .checkpoint import CheckpointedRalph, CheckpointStore
from .cli_client import CopilotCLI
from .client import Copex
from .conditions import Condition, ConditionContext, all_of, any_of, when
from .config import CopexConfig, find_copilot_cli, make_client

# Export new modules
from .exceptions import (
    AllModelsUnavailable,
    AuthenticationError,
    CircuitBreakerOpen,
    ConfigError,
    ConfigurationError,
    ConnectionError,
    CopexError,
    MCPError,
    PlanExecutionError,
    RateLimitError,
    RetryError,
    SecurityError,
    SessionError,
    SessionRecoveryFailed,
    StreamingError,
    TimeoutError,
    ToolExecutionError,
    ValidationError,
)
from .fleet import (
    AdaptiveConcurrency,
    Fleet,
    FleetConfig,
    FleetCoordinator,
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
from .squad import SquadAgent, SquadAgentResult, SquadCoordinator, SquadResult, SquadRole, SquadTeam
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

__all__ = [
    # Agent
    "AgentSession",
    "AgentTurn",
    "AgentResult",
    # Core
    "Copex",
    "CopilotCLI",
    "CopexConfig",
    "make_client",
    "Model",
    "ReasoningEffort",
    "discover_models",
    "get_available_models",
    "model_supports_reasoning",
    "no_reasoning_models",
    "refresh_model_capabilities",
    "resolve_model",
    # Skills
    "SkillDiscovery",
    "SkillInfo",
    "find_copilot_cli",
    "get_skill_content",
    "list_skills",
    # Exceptions
    "AllModelsUnavailable",
    "AuthenticationError",
    "CircuitBreakerOpen",
    "ConfigError",
    "ConfigurationError",
    "ConnectionError",
    "CopexError",
    "MCPError",
    "PlanExecutionError",
    "RateLimitError",
    "RetryError",
    "SecurityError",
    "SessionError",
    "SessionRecoveryFailed",
    "StreamingError",
    "TimeoutError",
    "ToolExecutionError",
    "ValidationError",
    # Retry/Backoff
    "AdaptiveRetry",
    "with_retry",
    "BackoffStrategy",
    "ErrorCategory",
    # Caching
    "StepCache",
    "get_cache",
    "clear_global_cache",
    # Conditions
    "Condition",
    "ConditionContext",
    "when",
    "all_of",
    "any_of",
    # Templates
    "StepTemplate",
    "StepInstance",
    "TemplateRegistry",
    "get_registry",
    "create_step",
    "test_workflow",
    "build_workflow",
    "deploy_workflow",
    # Visualization
    "visualize_plan",
    "render_ascii",
    "render_mermaid",
    # Fleet
    "AdaptiveConcurrency",
    "Fleet",
    "FleetConfig",
    "FleetCoordinator",
    "FleetResult",
    "FleetSummary",
    "FleetTask",
    "FleetStore",
    "RunRecord",
    "TaskRecord",
    "summarize_fleet_results",
    # Persistence
    "SessionStore",
    "PersistentSession",
    # Checkpointing
    "CheckpointStore",
    "CheckpointedRalph",
    # Stats
    "RunStats",
    "StatsTracker",
    "save_start_commit",
    "load_start_commit",
    "load_state",
    # Squad
    "SquadAgent",
    "SquadAgentResult",
    "SquadCoordinator",
    "SquadResult",
    "SquadRole",
    "SquadTeam",
    # Metrics
    "MetricsCollector",
    "RequestMetrics",
    "SessionMetrics",
]
