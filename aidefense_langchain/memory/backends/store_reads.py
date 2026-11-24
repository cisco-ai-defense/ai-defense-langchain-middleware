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

"""Store read release, provenance validation, and hidden sidecar paging."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, NoReturn

from langgraph.store.base import GetOp, Item, ListNamespacesOp, Op, Result, SearchItem, SearchOp

from ..events import Content, MemoryRead, Provenance, normalize_content
from ..quarry.inputs import MemoryReadEvent
from ..quarry.scanner import ScanOutcome
from .provenance import read_provenance, sidecar_key, snapshot_json
from .store_types import current_actor, current_read_sink, decision_for

SIDECAR_NAMESPACE = ("__aidefense_memory__", "provenance", "v1")
MAX_PAGE_ROWS = 100_000


def reserved(namespace: tuple[str, ...]) -> bool:
    return namespace[:1] == SIDECAR_NAMESPACE[:1]


class StoreReadMixin:
    """Methods mixed into GuardedMemoryStore; no separate public wrapper."""

    @staticmethod
    def _store_target(namespace: tuple[str, ...], key: str) -> dict[str, Any]:
        return {"kind": "store", "framework": "langgraph", "namespace": list(namespace), "key": key}

    def _content(self, value: Any) -> Content:
        return normalize_content(value, full_text=self._scanner.full_text_scan)

    def _read_provenance(self, item: Item) -> Provenance:
        sidecar = self._backing.get(SIDECAR_NAMESPACE, sidecar_key(item.namespace, item.key))
        return read_provenance(item.value, sidecar.value if sidecar else None)

    async def _aread_provenance(self, item: Item) -> Provenance:
        sidecar = await self._backing.aget(SIDECAR_NAMESPACE, sidecar_key(item.namespace, item.key))
        return read_provenance(item.value, sidecar.value if sidecar else None)

    def _read_event(
        self, item: Item, operation: str, provenance: Provenance, query: str | None
    ) -> MemoryRead:
        return MemoryRead(
            schema_version=1,
            operation=operation,
            target=self._store_target(item.namespace, item.key),
            source=current_actor(),
            content=self._content(item.value),
            provenance=provenance,
            target_class=self._target_class(item.namespace, item.key),
            query=query,
        )

    @staticmethod
    def _copy_item(item: Item) -> Item:
        fields = dict(
            value=snapshot_json(item.value), key=item.key, namespace=item.namespace,
            created_at=item.created_at, updated_at=item.updated_at,
        )
        if isinstance(item, SearchItem):
            return SearchItem(**fields, score=item.score)
        return Item(**fields)

    def _release(self, item: Item, event: MemoryRead, outcome: ScanOutcome) -> None:
        decision = decision_for(
            event,
            "monitor" if outcome.matched else "allow",
            outcome.rules,
            outcome.scan_ms,
            monitor_incomplete=outcome.monitor_incomplete,
        )
        self._emit(replace(decision, released=True))
        sink = current_read_sink()
        if sink is not None:
            sink(MemoryReadEvent(value=snapshot_json(item.value), event=event))

    def _read_failed(self, item: Item, operation: str, exc: Exception) -> NoReturn:
        sink = current_read_sink()
        if sink is not None:
            sink(None)
        error = self._error(operation, item.namespace, item.key)
        self._emit(error.decision)
        raise error from exc

    def _inspect_read(self, item: Item, operation: str, query: str | None = None) -> Item:
        try:
            stable = self._copy_item(item)
            event = self._read_event(stable, operation, self._read_provenance(stable), query)
            outcome = self._scanner.scan(event)
        except Exception as exc:
            self._read_failed(item, operation, exc)
        self._release(stable, event, outcome)
        return stable

    async def _ainspect_read(self, item: Item, operation: str, query: str | None = None) -> Item:
        try:
            stable = self._copy_item(item)
            provenance = await self._aread_provenance(stable)
            event = self._read_event(stable, operation, provenance, query)
            outcome = await self._scanner.ascan(event)
        except Exception as exc:
            self._read_failed(item, operation, exc)
        self._release(stable, event, outcome)
        return stable

    def _visible_page(self, op: SearchOp | ListNamespacesOp) -> list[Any]:
        if op.limit < 0 or op.offset < 0 or op.limit + op.offset > MAX_PAGE_ROWS:
            raise ValueError("Store page exceeds inspection limit")
        if op.limit == 0:
            return []
        visible: list[Any] = []
        offset = 0
        while len(visible) < op.offset + op.limit:
            page = self._backing.batch([op._replace(offset=offset, limit=256)])[0]
            if not isinstance(page, list):
                raise ValueError("Store returned an invalid page")
            visible.extend(x for x in page if not reserved(x.namespace if isinstance(x, Item) else x))
            offset += len(page)
            if len(page) < 256:
                break
            if offset >= MAX_PAGE_ROWS:
                raise ValueError("Store page exceeds inspection limit")
        return visible[op.offset:op.offset + op.limit]

    async def _avisible_page(self, op: SearchOp | ListNamespacesOp) -> list[Any]:
        if op.limit < 0 or op.offset < 0 or op.limit + op.offset > MAX_PAGE_ROWS:
            raise ValueError("Store page exceeds inspection limit")
        if op.limit == 0:
            return []
        visible: list[Any] = []
        offset = 0
        while len(visible) < op.offset + op.limit:
            page = (await self._backing.abatch([op._replace(offset=offset, limit=256)]))[0]
            if not isinstance(page, list):
                raise ValueError("Store returned an invalid page")
            visible.extend(x for x in page if not reserved(x.namespace if isinstance(x, Item) else x))
            offset += len(page)
            if len(page) < 256:
                break
            if offset >= MAX_PAGE_ROWS:
                raise ValueError("Store page exceeds inspection limit")
        return visible[op.offset:op.offset + op.limit]

    def _finish_reads(self, ops: list[Op], results: list[Result]) -> list[Result]:
        output = list(results)
        for index, op in enumerate(ops):
            if isinstance(op, GetOp) and output[index] is not None:
                output[index] = self._inspect_read(output[index], "get")
            elif isinstance(op, SearchOp):
                values = self._visible_page(op) if op.namespace_prefix == () else output[index]
                output[index] = [
                    self._inspect_read(item, "search", op.query) for item in values if not reserved(item.namespace)
                ]
            elif isinstance(op, ListNamespacesOp):
                values = self._visible_page(op)
                output[index] = [ns for ns in values if not reserved(ns)]
        return output

    async def _afinish_reads(self, ops: list[Op], results: list[Result]) -> list[Result]:
        output = list(results)
        for index, op in enumerate(ops):
            if isinstance(op, GetOp) and output[index] is not None:
                output[index] = await self._ainspect_read(output[index], "get")
            elif isinstance(op, SearchOp):
                values = await self._avisible_page(op) if op.namespace_prefix == () else output[index]
                output[index] = [
                    await self._ainspect_read(item, "search", op.query) for item in values if not reserved(item.namespace)
                ]
            elif isinstance(op, ListNamespacesOp):
                values = await self._avisible_page(op)
                output[index] = [ns for ns in values if not reserved(ns)]
        return output
