"""Streaming example: Get real-time chunks as they arrive."""

import asyncio

from copex import Copex, CopexConfig, Model, ReasoningEffort


async def main():
    config = CopexConfig(
        model=Model.GPT_5_2_CODEX,
        reasoning_effort=ReasoningEffort.XHIGH,
        streaming=True,
    )

    async with Copex(config) as copilot:
        print("Streaming response:\n")

        async for chunk in copilot.stream("Write a Python function to find prime numbers"):
            if chunk.type == "reasoning":
                # Show reasoning in dim text
                print(f"\033[2m{chunk.delta}\033[0m", end="", flush=True)
            elif chunk.type == "message":
                if chunk.is_final:
                    print("\n\n--- Complete ---")
                else:
                    print(chunk.delta, end="", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
