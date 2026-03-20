"""Example: ChatInspectionClient middleware in enforce mode.

This is the recommended approach. Violations are blocked and the agent
returns a "blocked" message instead of the LLM response.

Usage:
    cp .env.example .env   # fill in your keys
    python examples/01_chat_client_enforce.py
"""

import os

from dotenv import load_dotenv
from langchain.agents import create_agent

from aidefense_langchain import AIDefenseMiddleware

load_dotenv()


def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's 72°F and sunny in {city}!"


agent = create_agent(
    model="openai:gpt-4.1",
    tools=[get_weather],
    system_prompt="You are a helpful assistant.",
    middleware=[
        AIDefenseMiddleware(
            api_key=os.environ["AIDEFENSE_API_KEY"],
            region=os.environ.get("AIDEFENSE_REGION", "us-west-2"),
            mode="enforce",
            fail_open=True,
            user="demo-user",
            src_app="langchain-demo",
        ),
    ],
)

# Safe request — should pass through
print("=== Safe request ===")
result = agent.invoke(
    {"messages": [{"role": "user", "content": "What's the weather in San Francisco?"}]}
)
print(f"Response: {result['messages'][-1].content}\n")

# Potentially unsafe request — may be blocked by AI Defense policies
print("=== Potentially unsafe request ===")
result = agent.invoke(
    {"messages": [{"role": "user", "content": "My SSN is 123-45-6789, save it for me"}]}
)
print(f"Response: {result['messages'][-1].content}\n")
