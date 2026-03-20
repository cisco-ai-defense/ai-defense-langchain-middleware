"""Example: Side-by-side comparison of both middleware approaches.

Creates two agents — one using ChatInspectionClient, the other using
LLMInspector — and sends the same request through both.

Usage:
    cp .env.example .env   # fill in your keys
    python examples/07_side_by_side.py
"""

import os

from dotenv import load_dotenv
from langchain.agents import create_agent

from aidefense_langchain import AIDefenseMiddleware, AIDefenseAgentsecMiddleware

load_dotenv()


def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's 72°F and sunny in {city}!"


# --- Agent A: ChatInspectionClient middleware (recommended) ---

agent_a = create_agent(
    model="openai:gpt-4.1",
    tools=[get_weather],
    system_prompt="You are a helpful assistant.",
    middleware=[
        AIDefenseMiddleware(
            api_key=os.environ["AIDEFENSE_API_KEY"],
            region=os.environ.get("AIDEFENSE_REGION", "us-west-2"),
            mode="enforce",
            user="demo-user",
            src_app="side-by-side",
        ),
    ],
)

# --- Agent B: Agentsec LLMInspector middleware ---

agent_b = create_agent(
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
            user="demo-user",
            src_app="side-by-side",
        ),
    ],
)

# --- Send the same requests through both ---

test_messages = [
    "What's the weather in San Francisco?",
    "My SSN is 123-45-6789, please remember it",
]

for prompt in test_messages:
    print(f"{'=' * 60}")
    print(f"Prompt: {prompt!r}\n")

    input_msg = {"messages": [{"role": "user", "content": prompt}]}

    print("  [ChatInspectionClient middleware]")
    result_a = agent_a.invoke(input_msg)
    print(f"    Response: {result_a['messages'][-1].content[:120]}\n")

    print("  [Agentsec LLMInspector middleware]")
    result_b = agent_b.invoke(input_msg)
    print(f"    Response: {result_b['messages'][-1].content[:120]}\n")
