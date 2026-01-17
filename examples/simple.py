"""Simple example: Send a prompt and get a response."""

import asyncio

from robust_copilot import RobustCopilot


async def main():
    # Default: gpt-5.2-codex with xhigh reasoning, auto-retry enabled
    async with RobustCopilot() as copilot:
        response = await copilot.chat("What is the time complexity of quicksort?")
        print(response)


if __name__ == "__main__":
    asyncio.run(main())
