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
            region=os.environ.get("AIDEFENSE_REGION", "us-west-2"),
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
