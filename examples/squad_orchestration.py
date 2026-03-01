"""Squad orchestration example — multi-agent team coordination.

Demonstrates:
- SquadCoordinator for team orchestration
- Lead architect analysis, Developer implementation, Tester verification
- Auto-discovery of project context
- JSON output for machine consumption
"""

import asyncio
from copex import SquadCoordinator, CopexConfig, Model


async def squad_simple():
    """Run a squad on a simple task."""
    config = CopexConfig(model=Model.GPT_5_2_CODEX)

    async with SquadCoordinator(config) as squad:
        # Squad auto-creates: Lead, Developer, Tester, Docs
        result = await squad.run("Build a simple TODO list CLI app")

        print("=== Squad Results ===")
        print(f"Final output:\n{result.final_content}\n")
        print(f"Total duration: {result.total_duration_ms}ms")
        print(f"Cost estimate: ${result.estimated_cost:.4f}" if result.estimated_cost else "")


async def squad_with_context():
    """Run squad with auto-discovered project context."""
    config = CopexConfig(model=Model.CLAUDE_OPUS_4_6)

    async with SquadCoordinator(config) as squad:
        # Squad auto-discovers: README.md, pyproject.toml, src structure, conventions
        result = await squad.run(
            "Add a new authentication module following project patterns. "
            "Write tests and update documentation."
        )

        print("Squad task breakdown and execution:")
        print(result.final_content)


async def squad_json_output():
    """Get squad results in JSON format for machine consumption."""
    config = CopexConfig(model=Model.GPT_5_2_CODEX)

    async with SquadCoordinator(config) as squad:
        result = await squad.run("Build a REST API with CRUD operations")

        # Output JSON for downstream processing
        print(result.to_json(indent=2))


async def squad_custom_team():
    """Create a custom squad team with specific agents."""
    from copex import SquadTeam, SquadAgent, SquadRole

    # Use default team (Lead, Developer, Tester, Docs)
    team = SquadTeam.default()

    config = CopexConfig(model=Model.GPT_5_2_CODEX)

    async with SquadCoordinator(config, team=team) as squad:
        result = await squad.run("Refactor the logging module")

        for agent in team.agents:
            print(f"{agent.emoji} {agent.name} ({agent.role.value})")
        print(f"\nResult: {result.final_content[:200]}...")


if __name__ == "__main__":
    asyncio.run(squad_simple())
    # Uncomment to try other examples:
    # asyncio.run(squad_with_context())
    # asyncio.run(squad_json_output())
    # asyncio.run(squad_custom_team())
