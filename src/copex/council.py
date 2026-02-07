"""Council workflow: Multi-model deliberation with chair arbitration.

This module provides:
- Custom model selection for investigators and chair
- Debate rounds where investigators can revise opinions
- Specialist presets for focused analysis
- Tie-breaker escalation for uncertain decisions
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from copex.models import Model, ReasoningEffort


class CouncilPreset(str, Enum):
    """Specialist presets for council analysis focus."""
    
    SECURITY = "security"
    ARCHITECTURE = "architecture"
    REFACTOR = "refactor"
    REVIEW = "review"


# Preset-specific system prompt additions
PRESET_PROMPTS: dict[CouncilPreset, str] = {
    CouncilPreset.SECURITY: (
        "Focus your analysis on SECURITY aspects:\n"
        "- Authentication and authorization vulnerabilities\n"
        "- Input validation and injection attacks (SQL, XSS, command injection)\n"
        "- Data exposure and privacy concerns\n"
        "- Cryptographic weaknesses\n"
        "- Secure defaults and principle of least privilege\n"
        "- Rate limiting and DoS protection\n"
        "- Dependency vulnerabilities\n"
    ),
    CouncilPreset.ARCHITECTURE: (
        "Focus your analysis on ARCHITECTURE aspects:\n"
        "- Design patterns and anti-patterns\n"
        "- Scalability and performance implications\n"
        "- Modularity and separation of concerns\n"
        "- API design and contracts\n"
        "- Database schema and data modeling\n"
        "- Service boundaries and coupling\n"
        "- Future extensibility\n"
    ),
    CouncilPreset.REFACTOR: (
        "Focus your analysis on CODE QUALITY aspects:\n"
        "- DRY (Don't Repeat Yourself) violations\n"
        "- Naming conventions and clarity\n"
        "- Function/method length and complexity\n"
        "- Code organization and structure\n"
        "- Error handling patterns\n"
        "- Test coverage and testability\n"
        "- Technical debt identification\n"
    ),
    CouncilPreset.REVIEW: (
        "Provide a BALANCED CODE REVIEW covering:\n"
        "- Correctness and logic errors\n"
        "- Performance considerations\n"
        "- Security basics\n"
        "- Code style and readability\n"
        "- Edge cases and error handling\n"
        "- Documentation quality\n"
    ),
}


@dataclass
class CouncilConfig:
    """Configuration for council workflow."""
    
    # Model overrides
    investigator_model: Model | None = None  # Applies to all 3 investigators
    codex_model: Model | None = None         # Override for Codex only
    gemini_model: Model | None = None        # Override for Gemini only
    opus_model: Model | None = None          # Override for Opus investigator only
    chair_model: Model = Model.CLAUDE_OPUS_4_6
    
    # Reasoning effort
    reasoning_effort: ReasoningEffort = ReasoningEffort.HIGH
    
    # Features
    debate: bool = False                     # Enable debate rounds
    preset: CouncilPreset | None = None      # Specialist preset
    escalate: bool = False                   # Enable tie-breaker escalation
    
    def get_codex_model(self) -> Model:
        """Get effective model for Codex investigator."""
        if self.codex_model:
            return self.codex_model
        if self.investigator_model:
            return self.investigator_model
        return Model.GPT_5_2_CODEX
    
    def get_gemini_model(self) -> Model:
        """Get effective model for Gemini investigator."""
        if self.gemini_model:
            return self.gemini_model
        if self.investigator_model:
            return self.investigator_model
        return Model.GEMINI_3_PRO
    
    def get_opus_model(self) -> Model:
        """Get effective model for Opus investigator."""
        if self.opus_model:
            return self.opus_model
        if self.investigator_model:
            return self.investigator_model
        return Model.CLAUDE_OPUS_4_6


def _get_preset_prefix(preset: CouncilPreset | None) -> str:
    """Get the preset-specific prompt prefix."""
    if preset is None:
        return ""
    return PRESET_PROMPTS.get(preset, "") + "\n"


def _build_investigator_prompt(
    role: str,
    task: str,
    preset: CouncilPreset | None = None,
    round_num: int = 1,
    prior_opinions: dict[str, str] | None = None,
) -> str:
    """Build prompt for an investigator."""
    preset_prefix = _get_preset_prefix(preset)
    
    if round_num == 1:
        # First round: independent analysis
        return (
            f"You are council member {role}. Analyze the task independently.\n\n"
            f"{preset_prefix}"
            f"Task:\n{task}\n\n"
            "Output format:\n"
            "1. Recommendation\n"
            "2. Key reasoning\n"
            "3. Risks and unknowns\n"
            "4. Confidence (0-1)"
        )
    else:
        # Debate round: see others' opinions and revise
        others_text = ""
        if prior_opinions:
            for name, opinion_ref in prior_opinions.items():
                if name.lower() != role.lower():
                    others_text += f"\n{name}'s opinion:\n{opinion_ref}\n"
        
        return (
            f"You are council member {role}. This is ROUND 2 (debate round).\n\n"
            f"{preset_prefix}"
            f"Task:\n{task}\n\n"
            "You have seen the other investigators' initial opinions:\n"
            f"{others_text}\n"
            "After considering their perspectives, revise or strengthen your position.\n\n"
            "Output format:\n"
            "1. Revised recommendation (or confirmation of original)\n"
            "2. How others' perspectives influenced your thinking\n"
            "3. Key points of agreement/disagreement\n"
            "4. Updated confidence (0-1)"
        )


def _build_chair_prompt(
    task: str,
    include_debate: bool = False,
    escalate: bool = False,
) -> str:
    """Build prompt for the chair."""
    base = (
        "You are Opus acting as chair. Produce the final decision for the council.\n\n"
        f"Task:\n{task}\n\n"
    )
    
    if include_debate:
        submissions = (
            "Initial opinions:\n"
            "Codex status: {{task:codex-opinion.success}}\n"
            "Codex error: {{task:codex-opinion.error}}\n"
            "{{task:codex-opinion.content}}\n\n"
            "Gemini status: {{task:gemini-opinion.success}}\n"
            "Gemini error: {{task:gemini-opinion.error}}\n"
            "{{task:gemini-opinion.content}}\n\n"
            "Opus independent status: {{task:opus-opinion.success}}\n"
            "Opus independent error: {{task:opus-opinion.error}}\n"
            "{{task:opus-opinion.content}}\n\n"
            "Revised opinions after debate:\n"
            "Codex revised: {{task:codex-opinion-r2.success}}\n"
            "{{task:codex-opinion-r2.content}}\n\n"
            "Gemini revised: {{task:gemini-opinion-r2.success}}\n"
            "{{task:gemini-opinion-r2.content}}\n\n"
            "Opus revised: {{task:opus-opinion-r2.success}}\n"
            "{{task:opus-opinion-r2.content}}\n\n"
        )
    else:
        submissions = (
            "Submissions:\n"
            "Codex status: {{task:codex-opinion.success}}\n"
            "Codex error: {{task:codex-opinion.error}}\n"
            "{{task:codex-opinion.content}}\n\n"
            "Gemini status: {{task:gemini-opinion.success}}\n"
            "Gemini error: {{task:gemini-opinion.error}}\n"
            "{{task:gemini-opinion.content}}\n\n"
            "Opus independent status: {{task:opus-opinion.success}}\n"
            "Opus independent error: {{task:opus-opinion.error}}\n"
            "{{task:opus-opinion.content}}\n\n"
        )
    
    confidence_note = ""
    if escalate:
        confidence_note = (
            "\nIMPORTANT: If you are uncertain about the decision or confidence is below 0.7, "
            "explicitly state 'UNCERTAIN' at the start of your response. "
            "This will trigger a re-evaluation with higher reasoning effort.\n\n"
        )
    
    output_format = (
        "Output format:\n"
        "1. Final decision\n"
        "2. Why this decision wins over alternatives\n"
        "3. Concrete next steps\n"
        "4. Confidence (0-1)"
    )
    
    return base + submissions + confidence_note + output_format


def _build_escalation_prompt(
    task: str,
    chair_response: str,
) -> str:
    """Build prompt for escalated tie-breaker."""
    return (
        "You are Opus acting as chair with MAXIMUM reasoning power.\n\n"
        "The initial chair decision was uncertain. Re-analyze with deeper reasoning.\n\n"
        f"Task:\n{task}\n\n"
        f"Initial chair response:\n{chair_response}\n\n"
        "Provide a definitive decision with thorough justification.\n\n"
        "Output format:\n"
        "1. Final definitive decision\n"
        "2. Comprehensive reasoning\n"
        "3. Concrete next steps\n"
        "4. Confidence (should be high after deeper analysis)"
    )


def build_council_tasks(
    task: str,
    config: CouncilConfig | None = None,
) -> list[Any]:
    """Create council workflow task graph with enhancements.
    
    Args:
        task: The task/problem for the council to analyze
        config: Council configuration with model overrides and features
        
    Returns:
        List of FleetTask objects for the council workflow
    """
    from copex.fleet import DependencyFailurePolicy, FleetTask
    
    if config is None:
        config = CouncilConfig()
    
    tasks: list[FleetTask] = []
    high = config.reasoning_effort
    
    # Round 1: Independent opinions
    codex_prompt = _build_investigator_prompt(
        "Codex", task, config.preset, round_num=1
    )
    gemini_prompt = _build_investigator_prompt(
        "Gemini", task, config.preset, round_num=1
    )
    opus_opinion_prompt = _build_investigator_prompt(
        "Opus", task, config.preset, round_num=1
    )
    
    tasks.extend([
        FleetTask(
            id="codex-opinion",
            prompt=codex_prompt,
            model=config.get_codex_model(),
            reasoning_effort=high,
        ),
        FleetTask(
            id="gemini-opinion",
            prompt=gemini_prompt,
            model=config.get_gemini_model(),
            reasoning_effort=high,
        ),
        FleetTask(
            id="opus-opinion",
            prompt=opus_opinion_prompt,
            model=config.get_opus_model(),
            reasoning_effort=high,
        ),
    ])
    
    chair_depends_on = ["codex-opinion", "gemini-opinion", "opus-opinion"]
    
    # Round 2: Debate (optional)
    if config.debate:
        # Each investigator sees others' opinions and can revise
        prior_opinions = {
            "Codex": "{{task:codex-opinion.content}}",
            "Gemini": "{{task:gemini-opinion.content}}",
            "Opus": "{{task:opus-opinion.content}}",
        }
        
        codex_r2_prompt = _build_investigator_prompt(
            "Codex", task, config.preset, round_num=2, prior_opinions=prior_opinions
        )
        gemini_r2_prompt = _build_investigator_prompt(
            "Gemini", task, config.preset, round_num=2, prior_opinions=prior_opinions
        )
        opus_r2_prompt = _build_investigator_prompt(
            "Opus", task, config.preset, round_num=2, prior_opinions=prior_opinions
        )
        
        tasks.extend([
            FleetTask(
                id="codex-opinion-r2",
                prompt=codex_r2_prompt,
                depends_on=["codex-opinion", "gemini-opinion", "opus-opinion"],
                model=config.get_codex_model(),
                reasoning_effort=high,
                on_dependency_failure=DependencyFailurePolicy.CONTINUE,
            ),
            FleetTask(
                id="gemini-opinion-r2",
                prompt=gemini_r2_prompt,
                depends_on=["codex-opinion", "gemini-opinion", "opus-opinion"],
                model=config.get_gemini_model(),
                reasoning_effort=high,
                on_dependency_failure=DependencyFailurePolicy.CONTINUE,
            ),
            FleetTask(
                id="opus-opinion-r2",
                prompt=opus_r2_prompt,
                depends_on=["codex-opinion", "gemini-opinion", "opus-opinion"],
                model=config.get_opus_model(),
                reasoning_effort=high,
                on_dependency_failure=DependencyFailurePolicy.CONTINUE,
            ),
        ])
        
        chair_depends_on.extend(["codex-opinion-r2", "gemini-opinion-r2", "opus-opinion-r2"])
    
    # Chair final decision
    chair_prompt = _build_chair_prompt(
        task,
        include_debate=config.debate,
        escalate=config.escalate,
    )
    
    tasks.append(
        FleetTask(
            id="opus-chair-final",
            prompt=chair_prompt,
            depends_on=chair_depends_on,
            model=config.chair_model,
            reasoning_effort=high,
            on_dependency_failure=DependencyFailurePolicy.CONTINUE,
        )
    )
    
    # Escalation task (only included if escalate is enabled)
    # This task will be conditionally run based on chair response
    if config.escalate:
        escalation_prompt = (
            "You are Opus acting as chair with MAXIMUM reasoning power.\n\n"
            "The initial chair decision was uncertain. Re-analyze with deeper reasoning.\n\n"
            f"Task:\n{task}\n\n"
            "Initial chair response:\n{{task:opus-chair-final.content}}\n\n"
            "Only produce output if the initial response contains 'UNCERTAIN' or "
            "if the stated confidence is below 0.7.\n\n"
            "If the initial decision was confident (no 'UNCERTAIN', confidence >= 0.7), "
            "simply respond: 'ESCALATION_NOT_NEEDED'\n\n"
            "Otherwise, provide a definitive decision with thorough justification:\n"
            "1. Final definitive decision\n"
            "2. Comprehensive reasoning\n"
            "3. Concrete next steps\n"
            "4. Confidence (should be high after deeper analysis)"
        )
        
        tasks.append(
            FleetTask(
                id="opus-chair-escalated",
                prompt=escalation_prompt,
                depends_on=["opus-chair-final"],
                model=config.chair_model,
                reasoning_effort=ReasoningEffort.XHIGH,
                on_dependency_failure=DependencyFailurePolicy.CONTINUE,
            )
        )
    
    return tasks
