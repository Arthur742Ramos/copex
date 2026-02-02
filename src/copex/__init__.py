"""Copex - Copilot Extended: A resilient wrapper for GitHub Copilot SDK."""

__version__ = "1.0.2"

# Re-export core components for convenience
from .client import Copex
from .config import CopexConfig, find_copilot_cli
from .models import Model, ReasoningEffort

__all__ = [
    "Copex",
    "CopexConfig",
    "Model",
    "ReasoningEffort",
    "find_copilot_cli",
]
