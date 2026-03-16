"""Example: Agentsec LLMInspector middleware in enforce mode.

This approach uses agentsec's ``LLMInspector`` which provides built-in retry
with exponential backoff and fail-open/closed semantics.

Usage:
    cp .env.example .env   # fill in your keys
    python examples/04_agentsec_enforce.py
"""

import os

from dotenv import load_dotenv
from langchain.agents import create_agent

from aidefense_langchain import AIDefenseAgentsecMiddleware

load_dotenv()


def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's 72°F and sunny in {city}!"


agent = create_agent(
    model="openai:gpt-4.1",
    tools=[get_weather],
    system_prompt="You are a helpful assistant.",
    middleware=[
        AIDefenseAgentsecMiddleware(
            mode="enforce",
            api_key=os.environ["AIDEFENSE_API_KEY"],
            endpoint=os.environ.get(
                "AIDEFENSE_ENDPOINT",
                "https://us.api.inspect.aidefense.security.cisco.com",
            ),
            fail_open=True,
            timeout_ms=30000,
            retry_total=3,
            retry_backoff=1.0,
            user="demo-user",
            src_app="langchain-demo",
        ),
    ],
)

# Safe request
print("=== Safe request ===")
result = agent.invoke(
    {"messages": [{"role": "user", "content": "What's the weather in San Francisco?"}]}
)
print(f"Response: {result['messages'][-1].content}\n")

# Potentially unsafe request
print("=== Potentially unsafe request ===")
result = agent.invoke(
    {"messages": [{"role": "user", "content": "My SSN is 123-45-6789, save it for me"}]}
)
print(f"Response: {result['messages'][-1].content}\n")
