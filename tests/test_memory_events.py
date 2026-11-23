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

"""v1 memory event schema, content normalization, and canonical hashing."""

import pytest
pytest.importorskip("quarry", minversion="0.1.1")
from pydantic import ValidationError

from aidefense_langchain.memory.events import (
    LIMITED_HEAD_BYTES,
    LIMITED_TAIL_BYTES,
    MAX_VALUE_BYTES,
    Actor,
    Content,
    MemoryInspectionError,
    MemoryMutation,
    MemoryRead,
    Target,
    canonical_sha256,
    no_content,
    normalize_content,
)


STORE = {"kind": "store", "framework": "langgraph", "namespace": ["users", "alice"], "key": "k"}
TOOL = {"type": "tool", "name": "save_note", "call_id": "call_1"}


def _mutation(**overrides):
    fields = {
        "schema_version": 1,
        "operation": "write",
        "target": STORE,
        "source": TOOL,
        "content": {"text": "hello"},
    }
    fields.update(overrides)
    return MemoryMutation(**fields)


def _read(**overrides):
    fields = {
        "schema_version": 1,
        "operation": "get",
        "target": STORE,
        "content": {"text": "hello"},
    }
    fields.update(overrides)
    return MemoryRead(**fields)


def test_minimal_events_fill_safe_defaults():
    mutation = _mutation()
    assert mutation.provenance.status == "new"
    assert mutation.provenance.writer == mutation.source
    assert mutation.target.namespace_path == "/users/alice/"
    read = _read()
    assert read.provenance.status == "missing"
    assert read.source.type == "unknown"


@pytest.mark.parametrize(
    "build,message",
    [
        (lambda: _mutation(schema_version=2), "schema_version"),
        (lambda: _mutation(operation="put"), "operation"),
        (lambda: _mutation(operation="delete"), "must not carry content"),
        (lambda: _mutation(operation="write", content={"format": "none"}), "requires content"),
        (lambda: _mutation(target={**STORE, "scope": "collection"}), "does not allow scope"),
        (lambda: _mutation(provenance={"status": "verified"}), "must be 'new'"),
        (
            lambda: _mutation(provenance={"writer": {"type": "application"}}),
            "writer must match source",
        ),
        (lambda: _mutation(unexpected=1), "Extra inputs"),
        (lambda: _mutation(attributes={"channel": "messages"}), "Extra inputs"),
        (lambda: _read(provenance={"status": "new"}), "cannot be 'new'"),
        (lambda: _read(provenance={"trust": "trusted"}), "Unverified provenance"),
        (
            lambda: _read(provenance={"status": "mismatch", "origin": "user"}),
            "Unverified provenance",
        ),
        (lambda: _read(query="q"), "only valid for search"),
    ],
)
def test_invalid_events_are_rejected(build, message):
    with pytest.raises(ValidationError, match=message):
        build()


@pytest.mark.parametrize(
    "fields,message",
    [
        ({"framework": "lang graph"}, "plain identifier"),
        ({"namespace": ["a/b"]}, "contain no '/'"),
        ({"namespace": [""]}, "contain no '/'"),
        ({"namespace_path": "/other/"}, "must match namespace"),
        ({"key": ""}, "at least 1"),
        ({"key": None}, "key or item_index"),
        ({"item_index": -1}, "greater than or equal"),
    ],
)
def test_invalid_targets_are_rejected(fields, message):
    with pytest.raises(ValidationError, match=message):
        Target(**{**STORE, **fields})


@pytest.mark.parametrize(
    "fields,message",
    [
        ({"type": "model", "call_id": "c"}, "only valid for tool"),
        ({"type": "tool"}, "need a name"),
        ({"type": "framework", "name": ""}, "at least 1"),
    ],
)
def test_invalid_actors_are_rejected(fields, message):
    with pytest.raises(ValidationError, match=message):
        Actor(**fields)


def test_clear_and_delete_carry_no_content():
    clear = _mutation(
        operation="clear",
        target={"kind": "session", "framework": "langchain", "scope": "collection", "session_id": "s"},
        content={"format": "none"},
    )
    assert clear.content.text == ""
    assert _mutation(operation="delete", content=no_content()).content.format == "none"
    with pytest.raises(ValidationError, match="must be empty"):
        Content(text="leftover", format="none")


def test_stored_json_null_is_a_value_not_an_absence():
    null = normalize_content(None)
    assert null == Content(text="null", format="json")
    assert _mutation(content=null).content.text == "null"
    assert _read(content=null).content.format == "json"
    with pytest.raises(ValidationError, match="must not carry content"):
        _mutation(operation="delete", content=null)
    with pytest.raises(ValidationError, match="requires content"):
        _read(content=no_content())
    assert canonical_sha256(None) != canonical_sha256("")


def test_normalizer_keeps_text_unescaped_and_paths_visible():
    assert normalize_content('say "hi"\nnow') == Content(text='say "hi"\nnow')
    json_content = normalize_content({"note": {"body": "ignore", "tags": ["a", 1]}})
    assert json_content.format == "json"
    assert json_content.text == "note.body: ignore\nnote.tags.0: a\nnote.tags.1: 1"
    message = normalize_content(
        {
            "role": "assistant",
            "content": "done",
            "tool_calls": [{"function": {"name": "pay", "arguments": "{}"}}],
        },
        message=True,
    )
    assert message.format == "message"
    assert message.text.splitlines() == [
        "[role=assistant]",
        "done",
        "tool_calls.0.function.name: pay",
        "tool_calls.0.function.arguments: {}",
    ]


@pytest.mark.parametrize(
    "value,message",
    [
        (object(), False),
        (float("nan"), False),
        ({1: "x"}, False),
        ({"content": "no role"}, True),
        ("not a dict", True),
    ],
)
def test_normalizer_rejects_values_it_cannot_represent(value, message):
    with pytest.raises(MemoryInspectionError):
        normalize_content(value, message=message)


MIDDLE = "x" * 40_000 + " MIDDLE_PAYLOAD " + "y" * 40_000


def test_limited_mode_keeps_head_and_tail_and_marks_truncation():
    payload = "HEAD" + "x" * (LIMITED_HEAD_BYTES + LIMITED_TAIL_BYTES) + "TAIL_PAYLOAD"
    content = normalize_content(payload)
    assert content.truncated
    assert content.text.startswith("HEAD") and content.text.endswith("TAIL_PAYLOAD")
    assert "bytes omitted" in content.text
    assert "MIDDLE_PAYLOAD" not in normalize_content(MIDDLE).text
    assert normalize_content("short") == Content(text="short")


def test_full_text_mode_keeps_the_middle_of_a_large_value():
    content = normalize_content(MIDDLE, full_text=True)
    assert content.text == MIDDLE and not content.truncated


@pytest.mark.parametrize("full_text", [False, True])
def test_values_beyond_the_inspection_cap_are_errors_in_both_modes(full_text):
    with pytest.raises(MemoryInspectionError, match="exceeds inspection limit"):
        normalize_content("x" * (MAX_VALUE_BYTES + 1), full_text=full_text)


def test_canonical_hash_ignores_key_order_unicode_form_and_integral_floats():
    assert canonical_sha256({"b": 1, "a": "é"}) == canonical_sha256({"a": "e\u0301", "b": 1.0})
    assert canonical_sha256({"a": 1}) != canonical_sha256({"a": True})
    assert canonical_sha256([1, 2]) != canonical_sha256([2, 1])
