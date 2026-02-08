"""Retry demo: Simulates handling of transient errors."""

import asyncio

from copex import Copex, CopexConfig
from copex.config import RetryConfig


async def main():
    # Configure aggressive retry for unreliable connections
    config = CopexConfig(
        retry=RetryConfig(
            max_retries=10,
            base_delay=0.5,
            max_delay=30.0,
            exponential_base=2.0,
            retry_on_errors=[
                "500",
                "502",
                "503",
                "504",
                "Internal Server Error",
                "rate limit",
                "timeout",
                "connection reset",
            ],
        ),
        auto_continue=True,
        continue_prompt="Keep going",
        timeout=120.0,
    )

    async with Copex(config) as copilot:

        def on_chunk(chunk):
            if chunk.type == "system":
                print(f"[SYSTEM] {chunk.delta}")
            elif chunk.type == "message" and not chunk.is_final:
                print(chunk.delta, end="", flush=True)

        response = await copilot.send(
            "Write a comprehensive guide to Python async programming",
            on_chunk=on_chunk,
        )

        print(f"\n\n--- Stats ---")
        print(f"Retries: {response.retries}")
        print(f"Auto-continues: {response.auto_continues}")
        print(f"Response length: {len(response.content)} chars")


if __name__ == "__main__":
    asyncio.run(main())
