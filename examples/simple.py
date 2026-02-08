"""Simple example: Send a prompt and get a response."""

import asyncio

from copex import Copex


async def main():
    # Default: claude-opus-4.6 with xhigh reasoning, auto-retry enabled
    async with Copex() as copilot:
        response = await copilot.chat("What is the time complexity of quicksort?")
        print(response)


if __name__ == "__main__":
    asyncio.run(main())
