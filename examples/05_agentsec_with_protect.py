"""Example: Agentsec middleware using protect() for configuration.

When your application already uses ``agentsec.protect()`` with a YAML
config, the agentsec middleware can pick up configuration automatically.

Usage:
    cp .env.example .env   # fill in your keys
    python examples/05_agentsec_with_protect.py
"""

import os

from dotenv import load_dotenv
from langchain.agents import create_agent

from aidefense.runtime import agentsec
from aidefense_langchain import AIDefenseAgentsecMiddleware

load_dotenv()


def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's 72°F and sunny in {city}!"


# Configure agentsec globally — the middleware will inherit settings
agentsec.protect(
    api_mode={
        "llm": {
            "mode": "enforce",
            "endpoint": os.environ.get(
                "AIDEFENSE_ENDPOINT",
                "https://us.api.inspect.aidefense.security.cisco.com",
            ),
            "api_key": os.environ["AIDEFENSE_API_KEY"],
        }
    },
    patch_clients=False,  # no monkey-patching needed — using middleware instead
)

agent = create_agent(
    model="openai:gpt-4.1",
    tools=[get_weather],
    system_prompt="You are a helpful assistant.",
    middleware=[
        # No api_key/endpoint needed — inherited from protect()
        AIDefenseAgentsecMiddleware(mode="enforce"),
    ],
)

print("=== Using protect() configuration ===\n")
result = agent.invoke(
    {"messages": [{"role": "user", "content": "What's the weather in San Francisco?"}]}
)
print(f"Response: {result['messages'][-1].content}\n")
