"""Example: ChatInspectionClient middleware with specific rules enabled.

Demonstrates enabling only specific AI Defense rules (e.g. PII, Prompt
Injection) rather than using the default policy.

Usage:
    cp .env.example .env   # fill in your keys
    python examples/03_chat_client_with_rules.py
"""

import os

from dotenv import load_dotenv
from langchain.agents import create_agent

from aidefense_langchain import AIDefenseMiddleware

load_dotenv()


def lookup_account(name: str) -> str:
    """Look up an account by name."""
    return f"Account for {name}: balance $1,234.56"


agent = create_agent(
    model="openai:gpt-4.1",
    tools=[lookup_account],
    system_prompt="You are a helpful financial assistant.",
    middleware=[
        AIDefenseMiddleware(
            api_key=os.environ["AIDEFENSE_API_KEY"],
            region=os.environ.get("AIDEFENSE_REGION", "us-west-2"),
            mode="enforce",
            rules=[
                "PII",
                "Prompt Injection",
                {"rule_name": "PCI", "entity_types": ["Credit Card Number"]},
            ],
            user="demo-user",
            src_app="finance-app",
        ),
    ],
)

print("=== Request with PII + Prompt Injection rules ===\n")

result = agent.invoke(
    {"messages": [{"role": "user", "content": "Look up the account for Alice"}]}
)
print(f"Response: {result['messages'][-1].content}\n")
