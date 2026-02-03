"""Copex - Copilot Extended: A resilient wrapper for GitHub Copilot SDK."""

__version__ = "1.1.0"

# Re-export core components for convenience
from .client import Copex
from .config import CopexConfig, find_copilot_cli
from .models import Model, ReasoningEffort
from .skills import SkillDiscovery, SkillInfo, get_skill_content, list_skills

# Export new modules
from .exceptions import (
    CopexError,
    ConfigError,
    MCPError,
    RetryError,
    PlanExecutionError,
    ValidationError,
    SecurityError,
    TimeoutError,
    AuthenticationError,
    RateLimitError,
    ConnectionError,
)
from .backoff import AdaptiveRetry, with_retry, BackoffStrategy, ErrorCategory
from .cache import StepCache, get_cache, clear_global_cache
from .conditions import Condition, ConditionContext, when, all_of, any_of
from .templates import (
    StepTemplate,
    StepInstance,
    TemplateRegistry,
    get_registry,
    create_step,
    test_workflow,
    build_workflow,
    deploy_workflow,
)
from .visualization import visualize_plan, render_ascii, render_mermaid

__all__ = [
    # Core
    "Copex",
    "CopexConfig",
    "Model",
    "ReasoningEffort",
    # Skills
    "SkillDiscovery",
    "SkillInfo",
    "find_copilot_cli",
    "get_skill_content",
    "list_skills",
    # Exceptions
    "CopexError",
    "ConfigError",
    "MCPError",
    "RetryError",
    "PlanExecutionError",
    "ValidationError",
    "SecurityError",
    "TimeoutError",
    "AuthenticationError",
    "RateLimitError",
    "ConnectionError",
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
]
