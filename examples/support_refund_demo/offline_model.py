# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""A deterministic stand-in for the support model, for runs without an API key.

It follows the same system prompt a real model receives: process a ticket with
``ingest_support_ticket``, issue a refund with ``lookup_refund_account`` and then
``issue_refund`` using exactly the returned account. Like a real model, it copies
the account it read from memory into the refund call.
"""

from __future__ import annotations

import re
from itertools import count
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

_ACCOUNT = re.compile(r"\bACCT-[0-9]{8}\b")
_ids = count(1)


def _call(name: str, args: dict[str, Any]) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": f"call-{next(_ids)}"}],
    )


class OfflineSupportModel(BaseChatModel):
    refund_amount: str = "$480.00"

    @property
    def _llm_type(self) -> str:
        return "offline-support-model"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "OfflineSupportModel":
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Any = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=self._next(messages))])

    def _next(self, messages: list[BaseMessage]) -> AIMessage:
        last = messages[-1]
        if isinstance(last, HumanMessage):
            request = str(last.content).lower()
            if "ticket" in request:
                return _call("ingest_support_ticket", {})
            if "refund" in request:
                return _call("lookup_refund_account", {})
            return AIMessage(content="I can process support tickets and issue refunds.")
        if isinstance(last, ToolMessage):
            result = str(last.content)
            if last.name == "lookup_refund_account" and (
                account := _ACCOUNT.search(result)
            ):
                return _call(
                    "issue_refund",
                    {"account": account.group(0), "amount": self.refund_amount},
                )
            if "blocked" in result.lower():
                return AIMessage(
                    content=f"The action was blocked, so nothing was changed or paid. ({result})"
                )
            return AIMessage(content=result)
        return AIMessage(content="Done.")
