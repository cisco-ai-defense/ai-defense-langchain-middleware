# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Store read release, provenance validation, and hidden sidecar paging."""

from __future__ import annotations

from typing import Any

from langgraph.store.base import GetOp, Item, ListNamespacesOp, Op, Result, SearchOp

from .provenance import MemoryOrigin, read_provenance, sidecar_key, snapshot_json
from ..quarry.scanner import MemoryReadEvent, ScanOutcome
from .store_types import MemoryDecision, current_read_sink, current_tool

SIDECAR_NAMESPACE = ("__aidefense_memory__", "provenance", "v1")
MAX_PAGE_ROWS = 100_000


def reserved(namespace: tuple[str, ...]) -> bool:
    return namespace[:1] == SIDECAR_NAMESPACE[:1]


class StoreReadMixin:
    """Methods mixed into GuardedMemoryStore; no separate public wrapper."""

    def _read_origin(self, item: Item) -> tuple[MemoryOrigin, str, str]:
        sidecar = self._backing.get(SIDECAR_NAMESPACE, sidecar_key(item.namespace, item.key))
        return read_provenance(item.value, sidecar.value if sidecar else None)

    async def _aread_origin(self, item: Item) -> tuple[MemoryOrigin, str, str]:
        sidecar = await self._backing.aget(SIDECAR_NAMESPACE, sidecar_key(item.namespace, item.key))
        return read_provenance(item.value, sidecar.value if sidecar else None)

    @staticmethod
    def _copy_item(item: Item) -> Item:
        return Item(
            value=snapshot_json(item.value), key=item.key, namespace=item.namespace,
            created_at=item.created_at, updated_at=item.updated_at,
        )

    def _release(self, item: Item, operation: str, origin: MemoryOrigin, writer: str,
                 provenance_status: str, outcome: ScanOutcome) -> None:
        self._emit(MemoryDecision(
            operation, item.namespace, item.key, writer,
            "monitor" if outcome.matched else "allow", outcome.rules, outcome.scan_ms,
            released=True, reader_tool=current_tool(), origin_kind=origin.kind,
            trust_level=origin.trust, target_class=self._target_class(item.namespace, item.key),
            provenance_status=provenance_status,
            monitor_incomplete=outcome.monitor_incomplete,
        ))
        sink = current_read_sink()
        if sink is not None:
            sink(MemoryReadEvent(
                value=item.value, namespace=item.namespace, key=item.key, writer_tool=writer,
                origin_kind=origin.kind, trust_level=origin.trust,
                target_class=self._target_class(item.namespace, item.key),
                reader_tool=current_tool(),
                provenance_status=provenance_status,
            ))

    def _inspect_read(self, item: Item, operation: str) -> Item:
        try:
            stable = self._copy_item(item)
            origin, writer, provenance_status = self._read_origin(stable)
            outcome = self._scanner.scan(
                stable.value, operation=operation, namespace=stable.namespace, key=stable.key,
                writer_tool=writer, reader_tool=current_tool(), origin_kind=origin.kind,
                trust_level=origin.trust, target_class=self._target_class(stable.namespace, stable.key),
                provenance_status=provenance_status,
            )
        except Exception as exc:
            sink = current_read_sink()
            if sink is not None:
                sink(None)
            error = self._error(operation, item.namespace, item.key)
            self._emit(error.decision)
            raise error from exc
        self._release(stable, operation, origin, writer, provenance_status, outcome)
        return stable

    async def _ainspect_read(self, item: Item, operation: str) -> Item:
        try:
            stable = self._copy_item(item)
            origin, writer, provenance_status = await self._aread_origin(stable)
            outcome = await self._scanner.ascan(
                stable.value, operation=operation, namespace=stable.namespace, key=stable.key,
                writer_tool=writer, reader_tool=current_tool(), origin_kind=origin.kind,
                trust_level=origin.trust, target_class=self._target_class(stable.namespace, stable.key),
                provenance_status=provenance_status,
            )
        except Exception as exc:
            sink = current_read_sink()
            if sink is not None:
                sink(None)
            error = self._error(operation, item.namespace, item.key)
            self._emit(error.decision)
            raise error from exc
        self._release(stable, operation, origin, writer, provenance_status, outcome)
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
                    self._inspect_read(item, "search") for item in values if not reserved(item.namespace)
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
                    await self._ainspect_read(item, "search") for item in values if not reserved(item.namespace)
                ]
            elif isinstance(op, ListNamespacesOp):
                values = await self._avisible_page(op)
                output[index] = [ns for ns in values if not reserved(ns)]
        return output
