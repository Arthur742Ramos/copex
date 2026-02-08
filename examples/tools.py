"""Custom tools example: Define tools that Copilot can invoke."""

import asyncio
from datetime import datetime

from pydantic import BaseModel, Field
from copilot import define_tool

from copex import Copex


# Define tool parameters using Pydantic
class GetTimeParams(BaseModel):
    timezone: str = Field(default="UTC", description="Timezone name (e.g., 'UTC', 'US/Eastern')")


class CalculateParams(BaseModel):
    expression: str = Field(description="Mathematical expression to evaluate")


# Define tools using the @define_tool decorator
@define_tool(description="Get the current date and time")
async def get_current_time(params: GetTimeParams) -> str:
    # In production, you'd use pytz or similar for proper timezone handling
    now = datetime.now()
    return f"Current time ({params.timezone}): {now.strftime('%Y-%m-%d %H:%M:%S')}"


@define_tool(description="Safely evaluate a mathematical expression")
async def calculate(params: CalculateParams) -> str:
    try:
        # Safe eval for simple math expressions
        allowed = set("0123456789+-*/.() ")
        if not all(c in allowed for c in params.expression):
            return f"Error: Invalid characters in expression"
        result = eval(params.expression)  # Only safe because we validated
        return f"{params.expression} = {result}"
    except Exception as e:
        return f"Error calculating: {e}"


async def main():
    async with Copex() as copilot:
        # Provide tools when sending the prompt
        response = await copilot.send(
            "What time is it? Also, what's 2^10 + 15*3?",
            tools=[get_current_time, calculate],
        )
        print(response.content)


if __name__ == "__main__":
    asyncio.run(main())
