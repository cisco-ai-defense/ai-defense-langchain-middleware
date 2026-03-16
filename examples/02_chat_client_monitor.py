"""Example: ChatInspectionClient middleware in monitor mode.

Violations are logged and the optional ``on_violation`` callback is invoked,
but the request is never blocked.

Usage:
    cp .env.example .env   # fill in your keys
    python examples/02_chat_client_monitor.py
"""

import os

from dotenv import load_dotenv
from langchain.agents import create_agent

from aidefense_langchain import AIDefenseMiddleware

load_dotenv()

violations: list = []


def record_violation(result, direction: str) -> None:
    """Callback that records violations for later analysis."""
    classifications = [c.value for c in result.classifications] if result.classifications else []
    entry = {
        "direction": direction,
        "is_safe": result.is_safe,
        "action": result.action.value,
        "classifications": classifications,
        "severity": result.severity.value if result.severity else None,
        "event_id": result.event_id,
    }
    violations.append(entry)
    print(f"  [VIOLATION CALLBACK] {entry}")


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
            region=os.environ.get("AIDEFENSE_REGION", "us"),
            mode="monitor",
            on_violation=record_violation,
            user="demo-user",
            src_app="langchain-demo",
        ),
    ],
)

print("=== Monitor mode: violations are logged but not blocked ===\n")

result = agent.invoke(
    {"messages": [{"role": "user", "content": "My SSN is 123-45-6789, save it for me"}]}
)
print(f"\nResponse: {result['messages'][-1].content}")
print(f"\nTotal violations recorded: {len(violations)}")
for v in violations:
    print(f"  - {v}")
