from __future__ import annotations

import warnings

import pytest

from copex.config import CopexConfig
from copex.models import Model, ReasoningEffort, normalize_reasoning_effort


def test_normalize_reasoning_effort_opus_xhigh_downgrades_to_high() -> None:
    effort, warning_msg = normalize_reasoning_effort("opus", "xhigh")
    assert effort == ReasoningEffort.HIGH
    assert warning_msg is not None


def test_config_claude_opus_4_5_xhigh_forced_to_none() -> None:
    with pytest.warns(UserWarning):
        cfg = CopexConfig(model=Model.CLAUDE_OPUS_4_5, reasoning_effort="xhigh")
    assert cfg.reasoning_effort == ReasoningEffort.NONE


def test_config_claude_opus_4_5_xh_alias_forced_to_none() -> None:
    with pytest.warns(UserWarning):
        cfg = CopexConfig(model=Model.CLAUDE_OPUS_4_5, reasoning_effort="xh")
    assert cfg.reasoning_effort == ReasoningEffort.NONE


def test_config_gpt_5_2_xhigh_stays_xhigh() -> None:
    with warnings.catch_warnings(record=True) as rec:
        cfg = CopexConfig(model=Model.GPT_5_2, reasoning_effort="xhigh")
    assert cfg.reasoning_effort == ReasoningEffort.XHIGH
    assert rec == []


def test_config_gpt_5_2_codex_xhigh_stays_xhigh() -> None:
    with warnings.catch_warnings(record=True) as rec:
        cfg = CopexConfig(model=Model.GPT_5_2_CODEX, reasoning_effort="xhigh")
    assert cfg.reasoning_effort == ReasoningEffort.XHIGH
    assert rec == []


def test_normalize_reasoning_effort_gpt_5_3_xhigh_stays_xhigh() -> None:
    effort, warning_msg = normalize_reasoning_effort("gpt-5.3", "xhigh")
    assert effort == ReasoningEffort.XHIGH
    assert warning_msg is None


def test_normalize_reasoning_effort_gpt_6_xhigh_stays_xhigh() -> None:
    effort, warning_msg = normalize_reasoning_effort("gpt-6", "xhigh")
    assert effort == ReasoningEffort.XHIGH
    assert warning_msg is None
