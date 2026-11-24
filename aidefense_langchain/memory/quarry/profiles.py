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

"""Load the small, packaged Quarry memory rule profile catalog."""

from __future__ import annotations

import json
from pathlib import Path


def profile_rules(
    profile: str,
) -> tuple[tuple[str, ...], dict[str, str], tuple[str, ...], dict[str, str]]:
    directory = Path(__file__).with_name("rules")
    manifest = json.loads((directory / "profiles.json").read_text(encoding="utf-8"))
    profiles = manifest["profiles"]
    if profile not in profiles:
        raise ValueError(f"Unknown memory rule profile: {profile}")
    actions: dict[str, str] = {}
    for action in ("enforce", "monitor"):
        for identifier in profiles[profile][action]:
            if identifier in actions or identifier not in manifest["rules"]:
                raise ValueError(f"Invalid memory rule profile entry: {identifier}")
            actions[identifier] = action
    memory_texts: list[str] = []
    memory_actions: dict[str, str] = {}
    flow_texts: list[str] = []
    flow_actions: dict[str, str] = {}
    for identifier, spec in manifest["rules"].items():
        if identifier not in actions:
            continue
        name = spec["file"]
        if Path(name).name != name or not name.endswith(".rule"):
            raise ValueError(f"Invalid memory rule path: {name}")
        text = (directory / name).read_text(encoding="utf-8")
        # Quarry validates the full rule when DSLScanner compiles this profile.
        rule_name = text.split('rule "', 1)[1].split('"', 1)[0]
        if spec["stage"] == "memory":
            memory_texts.append(text)
            memory_actions[rule_name] = actions[identifier]
        elif spec["stage"] == "tool":
            flow_texts.append(text)
            flow_actions[rule_name] = actions[identifier]
        else:
            raise ValueError(f"Invalid memory rule stage: {identifier}")
    return tuple(memory_texts), memory_actions, tuple(flow_texts), flow_actions
