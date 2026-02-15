"""
Multi-task fleet mode — run tasks from a JSONL file in parallel.

Each line in the JSONL file is a JSON object with:
  - task (required): The task prompt string
  - model (optional): Model override for this task
  - reasoning (optional): Reasoning effort override (low/medium/high/xhigh)

Usage:
    copex fleet -f tasks.jsonl --parallel 3
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from copex.fleet import FleetTask
from copex.models import Model, ReasoningEffort, parse_reasoning_effort

logger = logging.getLogger(__name__)


@dataclass
class MultiFleetSpec:
    """Parsed multi-task JSONL specification."""

    tasks: list[FleetTask]
    source_path: Path


def load_jsonl_tasks(path: Path) -> list[FleetTask]:
    """Load fleet tasks from a JSONL file.

    Each line must be a JSON object with at least a 'task' field.
    Optional fields: 'id', 'model', 'reasoning', 'depends_on', 'cwd',
    'timeout', 'skills', 'exclude_tools'.

    Returns:
        List of FleetTask objects ready for fleet execution.

    Raises:
        ValueError: If the file is malformed or a line is missing 'task'.
    """
    if not path.exists():
        raise ValueError(f"JSONL file not found: {path}")

    tasks: list[FleetTask] = []
    seen_ids: set[str] = set()

    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("//"):
                continue  # skip blank lines and comments

            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_no} of {path}: {exc.msg}"
                ) from exc

            if not isinstance(obj, dict):
                raise ValueError(
                    f"Line {line_no} of {path}: expected a JSON object, got {type(obj).__name__}"
                )

            task_prompt = obj.get("task")
            if not task_prompt or not isinstance(task_prompt, str):
                raise ValueError(
                    f"Line {line_no} of {path}: missing or invalid 'task' field"
                )

            # Generate or use explicit task id
            task_id = obj.get("id", f"jsonl-{line_no}")
            if task_id in seen_ids:
                raise ValueError(
                    f"Line {line_no} of {path}: duplicate task id '{task_id}'"
                )
            seen_ids.add(task_id)

            # Parse optional model
            model: Model | None = None
            if "model" in obj:
                try:
                    model = Model(obj["model"])
                except ValueError:
                    logger.warning(
                        "Line %d: unknown model '%s', using default", line_no, obj["model"]
                    )

            # Parse optional reasoning effort
            reasoning: ReasoningEffort | None = None
            if "reasoning" in obj:
                try:
                    reasoning = parse_reasoning_effort(obj["reasoning"])
                except (ValueError, AttributeError):
                    logger.warning(
                        "Line %d: invalid reasoning '%s', using default",
                        line_no,
                        obj["reasoning"],
                    )

            # Parse optional fields
            depends_on = obj.get("depends_on", [])
            if isinstance(depends_on, str):
                depends_on = [depends_on]

            cwd = obj.get("cwd")
            timeout_sec = obj.get("timeout")
            if timeout_sec is not None:
                timeout_sec = float(timeout_sec)

            skills_dirs = obj.get("skills", [])
            if isinstance(skills_dirs, str):
                skills_dirs = [skills_dirs]

            exclude_tools = obj.get("exclude_tools", [])
            if isinstance(exclude_tools, str):
                exclude_tools = [exclude_tools]

            tasks.append(
                FleetTask(
                    id=task_id,
                    prompt=task_prompt,
                    model=model,
                    reasoning_effort=reasoning,
                    depends_on=depends_on,
                    cwd=cwd,
                    timeout_sec=timeout_sec,
                    skills_dirs=skills_dirs,
                    exclude_tools=exclude_tools,
                )
            )

    if not tasks:
        raise ValueError(f"No tasks found in {path}")

    logger.info("Loaded %d tasks from %s", len(tasks), path)
    return tasks
