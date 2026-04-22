# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Literal, Sequence


InspectionScope = Literal["latest_turn", "thread"]
VALID_INSPECTION_SCOPES: tuple[InspectionScope, ...] = ("latest_turn", "thread")


def validate_inspection_scope(inspection_scope: str) -> InspectionScope:
    if inspection_scope not in VALID_INSPECTION_SCOPES:
        allowed = ", ".join(repr(scope) for scope in VALID_INSPECTION_SCOPES)
        raise ValueError(
            f"inspection_scope must be one of {allowed}, got {inspection_scope!r}"
        )
    return inspection_scope


def scoped_langchain_messages(
    lc_messages: Sequence[Any],
    *,
    inspection_scope: InspectionScope,
    direction: str,
) -> list[Any]:
    """Select the LangChain messages that should be inspected for a hook."""
    messages = list(lc_messages)
    if inspection_scope == "thread":
        return messages

    if direction == "input":
        return _latest_input_messages(messages)
    if direction == "output":
        return _latest_output_messages(messages)
    raise ValueError(f"Unsupported direction: {direction!r}")


def _latest_input_messages(messages: Sequence[Any]) -> list[Any]:
    last_ai_index = _last_ai_index(messages)
    if last_ai_index < 0:
        return list(messages)
    return list(messages[last_ai_index + 1 :])


def _latest_output_messages(messages: Sequence[Any]) -> list[Any]:
    ai_indexes = [index for index, msg in enumerate(messages) if _is_ai_message(msg)]
    if len(ai_indexes) <= 1:
        return list(messages)
    return list(messages[ai_indexes[-2] + 1 :])


def _last_ai_index(messages: Sequence[Any]) -> int:
    for index in range(len(messages) - 1, -1, -1):
        if _is_ai_message(messages[index]):
            return index
    return -1


def _is_ai_message(message: Any) -> bool:
    return getattr(message, "type", None) == "ai"
