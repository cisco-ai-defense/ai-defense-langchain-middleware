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
