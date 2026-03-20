# Copyright 2026 Cisco Systems, Inc. and its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""Example: Tool inspection using agentsec's MCPInspector.

Uses ``AIDefenseAgentsecToolMiddleware`` which delegates to ``MCPInspector``
for retry logic, exponential backoff, and fail-open/closed semantics.

This variant can also inherit configuration from ``agentsec.protect()``
if the environment is already initialized with agentsec.

Usage:
    export AI_DEFENSE_API_KEY="your-api-key"
    python examples/09_tool_inspection_agentsec.py
"""

import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI

from aidefense_langchain import (
    AIDefenseAgentsecMiddleware,
    AIDefenseAgentsecToolMiddleware,
)

load_dotenv()


@tool
def read_file(path: str) -> str:
    """Read a file from the filesystem."""
    return f"Contents of {path}: [simulated file content]"


@tool
def execute_query(sql: str) -> str:
    """Execute a SQL query against the database."""
    return f"Query result for: {sql} — 5 rows returned."


def on_tool_violation(decision, tool_name, direction):
    """Custom callback for tool policy violations."""
    print(f"\n[TOOL VIOLATION] tool={tool_name} direction={direction}")
    print(f"  action: {decision.action}")


def main():
    api_key = os.environ.get("AI_DEFENSE_API_KEY")

    llm_middleware = AIDefenseAgentsecMiddleware(
        mode="enforce",
        api_key=api_key,
    )

    tool_middleware = AIDefenseAgentsecToolMiddleware(
        mode="enforce",
        api_key=api_key,
        fail_open=True,
        inspect_requests=True,
        inspect_responses=True,
        on_violation=on_tool_violation,
    )

    model = ChatOpenAI(model="gpt-4.1")
    agent = create_agent(
        model=model,
        tools=[read_file, execute_query],
        middleware=[llm_middleware, tool_middleware],
    )

    print("=== Agentsec Tool Inspection (Enforce) ===\n")

    result = agent.invoke(
        {"messages": [{"role": "user", "content": "Read the config file and run a query"}]},
    )

    for msg in result["messages"]:
        print(f"[{msg.type}] {msg.content[:200]}")


if __name__ == "__main__":
    main()
