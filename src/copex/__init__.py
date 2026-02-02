"""Copex - Copilot Extended: A resilient wrapper for GitHub Copilot SDK."""

__version__ = "0.12.1"

# Re-export core components for convenience
from .client import Copex  # noqa: F401
from .config import CopexConfig  # noqa: F401

__all__ = ["Copex", "CopexConfig"]
