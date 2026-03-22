# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Example: Inspecting tool / MCP calls with AIDefenseToolMiddleware.

This example shows how to use the tool inspection middleware to scan
tool call requests and responses for policy violations.

The middleware uses ``MCPInspectionClient`` under the hood and integrates
with LangChain's ``wrap_tool_call`` hook — it wraps every tool call with
pre-call (request) and post-call (response) inspection.

Usage:
    export AI_DEFENSE_API_KEY="your-api-key"
    python examples/08_tool_inspection_enforce.py
"""

import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI

from aidefense_langchain import AIDefenseMiddleware, AIDefenseToolMiddleware

load_dotenv()


@tool
def search_database(query: str) -> str:
    """Search a database for information."""
    return f"Results for: {query} — Found 3 matching records."


@tool
def send_email(recipient: str, subject: str, body: str) -> str:
    """Send an email to a recipient."""
    return f"Email sent to {recipient} with subject '{subject}'."


def on_tool_violation(result, tool_name, direction):
    """Custom callback for tool policy violations."""
    print(f"\n[TOOL VIOLATION] tool={tool_name} direction={direction}")
    if result.result:
        print(f"  classifications: {result.result.classifications}")
        print(f"  is_safe: {result.result.is_safe}")


def main():
    api_key = os.environ["AI_DEFENSE_API_KEY"]

    llm_middleware = AIDefenseMiddleware(
        api_key=api_key,
        region="us-west-2",
        mode="enforce",
    )

    tool_middleware = AIDefenseToolMiddleware(
        api_key=api_key,
        region="us-west-2",
        mode="enforce",
        inspect_requests=True,
        inspect_responses=True,
        on_violation=on_tool_violation,
    )

    model = ChatOpenAI(model="gpt-4.1")
    agent = create_agent(
        model=model,
        tools=[search_database, send_email],
        middleware=[llm_middleware, tool_middleware],
    )

    print("=== Full LLM + Tool Inspection (Enforce) ===\n")

    result = agent.invoke(
        {"messages": [{"role": "user", "content": "Search for customer records"}]},
    )

    for msg in result["messages"]:
        print(f"[{msg.type}] {msg.content[:200]}")


if __name__ == "__main__":
    main()
