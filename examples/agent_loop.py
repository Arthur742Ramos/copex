"""Agent loop example — iterative tool-based reasoning.

Demonstrates:
- AgentSession for turn-based interaction
- Tool detection and JSON Lines output
- Streaming vs batched results
"""

import asyncio
from copex import AgentSession, Copex, CopexConfig, Model


async def agent_loop_example():
    """Run an agent loop with tool calls."""
    config = CopexConfig(model=Model.GPT_5_2_CODEX)

    async with Copex(config) as client:
        # Create agent session with max 10 turns
        agent = AgentSession(client, max_turns=10)

        # Run agent on a prompt
        prompt = "Write a Python function to compute Fibonacci numbers. Test it with a few examples."
        result = await agent.run(prompt)

        print(f"Agent completed in {result.total_turns} turns")
        print(f"Final content:\n{result.final_content}\n")

        # Print each turn
        for turn in result.turns:
            print(f"Turn {turn.turn}:")
            print(f"  Content: {turn.content[:100]}...")
            if turn.tool_calls:
                print(f"  Tool calls: {len(turn.tool_calls)}")
            print()


async def agent_json_output():
    """Output JSON Lines for machine consumption (one JSON per turn)."""
    config = CopexConfig(model=Model.GPT_5_2_CODEX)

    async with Copex(config) as client:
        agent = AgentSession(client, max_turns=5)
        result = await agent.run("Implement a simple HTTP server")

        # JSON Lines output (one JSON object per line)
        for turn in result.turns:
            print(turn.to_json())


async def agent_streaming():
    """Stream agent turns as they complete."""
    config = CopexConfig(model=Model.CLAUDE_OPUS_4_6)

    async with Copex(config) as client:
        agent = AgentSession(client, max_turns=3)

        async for turn in agent.run_streaming("Build a calculator CLI"):
            print(f"Turn {turn.turn}: {turn.stop_reason}")
            if turn.tool_calls:
                print(f"  → {len(turn.tool_calls)} tool calls")


if __name__ == "__main__":
    asyncio.run(agent_loop_example())
    # Uncomment to try other examples:
    # asyncio.run(agent_json_output())
    # asyncio.run(agent_streaming())
