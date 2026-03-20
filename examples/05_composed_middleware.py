"""Example: Composing AI Defense with other LangChain middleware.

Shows how AIDefenseMiddleware fits into a middleware stack alongside other
built-in LangChain middleware (summarization, tool monitoring, etc.).

Usage:
    cp .env.example .env   # fill in your keys
    python examples/06_composed_middleware.py
"""

import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import (
    before_model,
    after_model,
    AgentState,
)
from langgraph.runtime import Runtime
from typing import Any

from aidefense_langchain import AIDefenseMiddleware

load_dotenv()


# --- Custom logging middleware (decorator style) ---

@before_model
def log_before(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    msg_count = len(state["messages"])
    last = state["messages"][-1].content[:80] if state["messages"] else "<empty>"
    print(f"  [LOG] before_model: {msg_count} messages, last: {last!r}")
    return None


@after_model
def log_after(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    last = state["messages"][-1].content[:80] if state["messages"] else "<empty>"
    print(f"  [LOG] after_model: response: {last!r}")
    return None


# --- Tools ---

def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's 72°F and sunny in {city}!"


def send_email(to: str, body: str) -> str:
    """Send an email."""
    return f"Email sent to {to}"


# --- Agent with composed middleware ---

agent = create_agent(
    model="openai:gpt-4.1",
    tools=[get_weather, send_email],
    system_prompt="You are a helpful assistant.",
    middleware=[
        # Security runs first (outermost)
        AIDefenseMiddleware(
            api_key=os.environ["AIDEFENSE_API_KEY"],
            region=os.environ.get("AIDEFENSE_REGION", "us-west-2"),
            mode="enforce",
            user="demo-user",
            src_app="composed-demo",
        ),
        # Logging
        log_before,
        log_after,
    ],
)

print("=== Composed middleware: AI Defense + custom logging ===\n")

result = agent.invoke(
    {"messages": [{"role": "user", "content": "What's the weather in NYC?"}]}
)
print(f"\nFinal response: {result['messages'][-1].content}")
