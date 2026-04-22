# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from langchain.messages import AIMessage, HumanMessage, ToolMessage

from aidefense_langchain._inspection_scope import scoped_langchain_messages


def _rows(messages):
    return [(msg.type, getattr(msg, "content", None)) for msg in messages]


def test_latest_turn_input_uses_messages_after_last_ai():
    messages = [
        HumanMessage(content="first user"),
        AIMessage(content="tool call"),
        ToolMessage(content="tool output", tool_call_id="call-1"),
    ]

    scoped = scoped_langchain_messages(
        messages,
        inspection_scope="latest_turn",
        direction="input",
    )

    assert _rows(scoped) == [("tool", "tool output")]


def test_latest_turn_output_uses_messages_after_previous_ai():
    messages = [
        HumanMessage(content="first user"),
        AIMessage(content="tool call"),
        ToolMessage(content="tool output", tool_call_id="call-1"),
        AIMessage(content="final answer"),
    ]

    scoped = scoped_langchain_messages(
        messages,
        inspection_scope="latest_turn",
        direction="output",
    )

    assert _rows(scoped) == [
        ("tool", "tool output"),
        ("ai", "final answer"),
    ]


def test_thread_scope_keeps_entire_transcript():
    messages = [
        HumanMessage(content="first user"),
        AIMessage(content="first answer"),
        HumanMessage(content="second user"),
        AIMessage(content="second answer"),
    ]

    scoped = scoped_langchain_messages(
        messages,
        inspection_scope="thread",
        direction="output",
    )

    assert _rows(scoped) == [
        ("human", "first user"),
        ("ai", "first answer"),
        ("human", "second user"),
        ("ai", "second answer"),
    ]
