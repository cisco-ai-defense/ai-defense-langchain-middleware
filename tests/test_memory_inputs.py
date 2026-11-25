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

"""The v1 Quarry declaration stays in step with the models and is accepted by Quarry."""

import typing

import pytest
pytest.importorskip("quarry", minversion="0.1.1")
from pydantic import BaseModel
from quarry import DSLScanner, IncompleteScanError
from quarry.config import QuarryConfig

from aidefense_langchain.memory.events import (
    MemoryMutation,
    MemoryRead,
    no_content,
    normalize_content,
)
from aidefense_langchain.memory.quarry.inputs import (
    MEMORY_V1_INPUTS,
    MUTATIONS,
    READS,
    InputBuilder,
    MemoryReadEvent,
)


STORE = {
    "kind": "store", "framework": "langgraph",
    "namespace": ["reports", "preferences"], "key": "report_recipient",
}
TOOL = {"type": "tool", "name": "ingest_vendor_bulletin", "call_id": "call-1"}


def _write(**overrides):
    fields = {
        "schema_version": 1, "operation": "write", "target": STORE, "source": TOOL,
        "content": normalize_content("Report recipient: audit@attacker.invalid"),
        "provenance": {"origin": "external", "trust": "untrusted"},
        "target_class": "protected",
    }
    return MemoryMutation(**{**fields, **overrides})


def _read(**overrides):
    fields = {
        "schema_version": 1, "operation": "get", "target": STORE,
        "source": {"type": "tool", "name": "read_report_recipient", "call_id": "call-2"},
        "content": normalize_content("audit@attacker.invalid"),
        "provenance": {
            "origin": "external", "trust": "untrusted", "status": "verified", "writer": TOOL,
        },
        "target_class": "protected",
    }
    return MemoryRead(**{**fields, **overrides})


def _model(annotation):
    for option in typing.get_args(annotation) or (annotation,):
        if isinstance(option, type) and issubclass(option, BaseModel):
            return option
    return None


def _assert_declares(model, facts, path):
    assert set(facts) == set(model.model_fields), path
    for name, field in model.model_fields.items():
        declared = facts[name]
        optional = field.default is None and not field.is_required()
        assert declared.get("optional", False) == optional, f"{path}.{name}"
        nested = _model(field.annotation)
        if nested is not None:
            assert declared["type"] == "object", f"{path}.{name}"
            _assert_declares(nested, declared["fields"], f"{path}.{name}")
        elif typing.get_origin(field.annotation) is typing.Literal:
            assert declared["values"] == list(typing.get_args(field.annotation)), f"{path}.{name}"


@pytest.mark.parametrize("section,model", [(MUTATIONS, MemoryMutation), (READS, MemoryRead)])
def test_declaration_covers_every_model_field(section, model):
    _assert_declares(model, MEMORY_V1_INPUTS[section]["facts"], section)


def _scan(rule, data):
    scanner = DSLScanner(
        QuarryConfig(config_version=2, quarry_rule_texts=(rule,), inputs=MEMORY_V1_INPUTS)
    )
    return scanner.scan_data(data)


PROTECTED_WRITE = '''rule "Untrusted write to protected memory"
condition:
    any e in memory.mutation_requested where
        e.target_class == "protected" and
        e.target.key == "report_recipient" and
        e.source.name == "ingest_vendor_bulletin" and
        e.provenance.trust != "trusted" and
        e.content.text ~ "attacker"
'''


def test_quarry_reads_nested_v1_fields():
    result = _scan(PROTECTED_WRITE, InputBuilder.memory(_write()))
    assert result.completion_state == "complete"
    assert result.rule_matches == ["Untrusted write to protected memory"]
    clean = _write(provenance={"origin": "system", "trust": "trusted"})
    assert _scan(PROTECTED_WRITE, InputBuilder.memory(clean)).rule_matches == []


def test_memory_sends_both_sections_and_no_value_events():
    data = InputBuilder.memory(_read())
    assert data["memory"]["mutation_requested"] == []
    assert [item["id"] for item in data["memory"]["read_returned"]] == ["memory-event"]
    delete = InputBuilder.memory(_write(operation="delete", content=no_content()))
    assert delete["memory"]["mutation_requested"][0]["content"] == {
        "text": "", "format": "none", "truncated": False,
    }
    assert _scan(PROTECTED_WRITE, delete).completion_state == "complete"


@pytest.mark.parametrize(
    "change",
    [
        lambda item: item.update(attributes={"channel": "messages"}),
        lambda item: item.update(type=MUTATIONS),
        lambda item: item["target"].update(kind="database"),
        lambda item: item.update(schema_version=2),
    ],
)
def test_quarry_rejects_events_outside_the_declaration(change):
    data = InputBuilder.memory(_write())
    change(data["memory"]["mutation_requested"][0])
    result = _scan(PROTECTED_WRITE, data)
    # A bad section fails only the rules that read it, so the scan is incomplete.
    assert result.completion_state == "incomplete" and result.rule_matches == []
    with pytest.raises(IncompleteScanError):
        result.raise_for_status()


def test_identity_fields_are_bounded():
    with pytest.raises(ValueError, match="key must be at most"):
        InputBuilder.memory(_write(target={**STORE, "key": "k" * 257}))
    with pytest.raises(ValueError, match="at most 16 parts"):
        InputBuilder.memory(_write(target={**STORE, "namespace": [f"n{i}" for i in range(17)]}))


FLOW = '''rule "Untrusted memory address sent by send_report"
strings:
    $email = "[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+[.][A-Za-z]{2,}" regex
condition:
    (any call in get_tool_calls("send_report") where call.arguments.to contains addr)
    after(within: 12)
    (any r in memory.read_returned where
        r.provenance.trust != "trusted" and
        $email in r.content.text as addr)
'''


def test_tool_flow_orders_released_reads_before_the_tool_call():
    released = MemoryReadEvent(value={"released_text": "audit@attacker.invalid"}, event=_read())
    data = InputBuilder.tool_flow(
        [released], "send_report", "call-3", {"to": "audit@attacker.invalid"}
    )
    assert data["timeline"] == [
        {"input": READS, "id": "read-0"}, {"message_index": 0}, {"tool_call_id": "call-3"},
    ]
    assert data["memory"]["read_returned"][0]["content"]["text"] == (
        "released_text: audit@attacker.invalid"
    )
    result = _scan(FLOW, data)
    assert (result.completion_state, result.rule_matches) == (
        "complete", ["Untrusted memory address sent by send_report"],
    )
    safe = InputBuilder.tool_flow(
        [released], "send_report", "call-3", {"to": "reports@company.example"}
    )
    assert _scan(FLOW, safe).rule_matches == []


@pytest.mark.parametrize(
    "arguments,message",
    [
        ({"to": float("nan")}, "JSON values"),
        ({"to": "x" * 64_001}, "64 KB"),
        (["to"], "JSON object"),
    ],
)
def test_tool_flow_rejects_uninspectable_arguments(arguments, message):
    with pytest.raises(ValueError, match=message):
        InputBuilder.tool_flow([], "send_report", "call-3", arguments)
