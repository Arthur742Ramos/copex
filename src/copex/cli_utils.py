"""Shared CLI utilities."""

from __future__ import annotations

import typer

from copex.approval import normalize_approval_mode
from copex.config import CopexConfig


def apply_approval_flags(
    config: CopexConfig,
    *,
    auto_approve: bool = False,
    approve: bool = False,
    dry_run: bool = False,
    audit: bool = False,
    default_auto: bool = False,
) -> None:
    selected = int(auto_approve) + int(approve) + int(dry_run)
    if selected > 1:
        raise typer.BadParameter("Use only one of --auto-approve, --approve, or --dry-run")

    if dry_run:
        config.approval_mode = "dry-run"
    elif approve:
        config.approval_mode = "approve"
    elif auto_approve or default_auto:
        config.approval_mode = "auto-approve"
    else:
        config.approval_mode = normalize_approval_mode(config.approval_mode).value

    if audit:
        config.audit = True
