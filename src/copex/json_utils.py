"""Utilities for robustly extracting JSON from LLM responses."""

from __future__ import annotations

import json
import re


def extract_json_array(text: str) -> list:
    """Robustly extract a JSON array from LLM output.

    Handles: markdown code fences, mixed text around JSON,
    truncated strings, and other common LLM output issues.
    """
    text = text.strip()

    # Strategy 1: Direct parse
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass

    # Strategy 2: Extract from markdown code fences
    for pattern in [r'```json\s*\n?(.*?)```', r'```\s*\n?(.*?)```']:
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            try:
                result = json.loads(match.strip())
                if isinstance(result, list):
                    return result
            except json.JSONDecodeError:
                continue

    # Strategy 3: Find [ ... ] bracket pair (complete or truncated)
    first_bracket = text.find("[")
    if first_bracket == -1:
        raise ValueError(
            f"Could not extract JSON array from LLM response (length={len(text)})"
        )

    last_bracket = text.rfind("]")
    if last_bracket > first_bracket:
        candidate = text[first_bracket:last_bracket + 1]
        try:
            result = json.loads(candidate)
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass
    else:
        # No closing bracket - definitely truncated
        candidate = text[first_bracket:]

    # Strategy 4: Repair truncated JSON - find last complete top-level object
    depth_curly = 0
    in_str = False
    escape_next = False
    last_complete_obj_end = -1
    for i, ch in enumerate(candidate):
        if escape_next:
            escape_next = False
            continue
        if in_str:
            if ch == "\\":
                escape_next = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
            continue
        if ch == "{":
            depth_curly += 1
        elif ch == "}":
            depth_curly -= 1
            if depth_curly == 0:
                last_complete_obj_end = i

    if last_complete_obj_end > 0:
        truncated = (
            candidate[:last_complete_obj_end + 1].rstrip().rstrip(",") + "]"
        )
        try:
            result = json.loads(truncated)
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

    raise ValueError(
        f"Could not extract JSON array from LLM response (length={len(text)})"
    )
