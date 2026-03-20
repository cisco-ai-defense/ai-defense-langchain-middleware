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
