# Copyright 2026 Cisco Systems, Inc. and its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Scan LangGraph checkpoint messages at persistence and replay boundaries."""

from __future__ import annotations

import copy
import logging
from collections.abc import AsyncIterator, Callable, Collection, Iterator, Sequence
from dataclasses import replace
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver, CheckpointTuple

from .checkpoint_inspection import _CheckpointInspection
from ..quarry.scanner import QuarryMemoryScanner
from .store_types import MemoryDecision, MemoryMode

logger = logging.getLogger("aidefense.langchain.memory")


class GuardedCheckpointSaver(_CheckpointInspection, BaseCheckpointSaver):
    """Scan message writes before persistence and restored messages on read.

    Message-channel write matches can block; read matches are observational.
    Other graph channels pass through unchanged.
    """

    def __init__(
        self,
        backing: BaseCheckpointSaver,
        scanner: QuarryMemoryScanner | None,
        *,
        mode: MemoryMode = "enforce",
        on_decision: Callable[[MemoryDecision], None] | None = None,
    ) -> None:
        if mode not in ("enforce", "monitor", "off"):
            raise ValueError("mode must be 'enforce', 'monitor', or 'off'")
        if mode != "off" and scanner is None:
            raise ValueError("scanner is required unless mode is 'off'")
        super().__init__(serde=backing.serde)
        self._backing = backing
        self._scanner = scanner
        self._mode = mode
        self._on_decision = on_decision

    def _emit(self, decision: MemoryDecision) -> None:
        if self._on_decision is None:
            return
        try:
            self._on_decision(decision)
        except Exception:
            logger.warning("Checkpoint decision callback failed")

    @property
    def config_specs(self) -> list[Any]:
        return self._backing.config_specs

    def get_next_version(self, current: Any, channel: Any) -> Any:
        return self._backing.get_next_version(current, channel)

    def with_allowlist(self, extra_allowlist: Collection[tuple[str, ...]]) -> GuardedCheckpointSaver:
        return GuardedCheckpointSaver(
            self._backing.with_allowlist(extra_allowlist),
            self._scanner,
            mode=self._mode,
            on_decision=self._on_decision,
        )

    def delete_thread(self, thread_id: str) -> None:
        self._backing.delete_thread(thread_id)

    async def adelete_thread(self, thread_id: str) -> None:
        await self._backing.adelete_thread(thread_id)

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        return self._observe(self._backing.get_tuple(config))

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        return await self._aobserve(await self._backing.aget_tuple(config))

    def list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        for item in self._backing.list(config, filter=filter, before=before, limit=limit):
            yield self._observe(item)

    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        async for item in self._backing.alist(config, filter=filter, before=before, limit=limit):
            yield await self._aobserve(item)

    def put(self, config: RunnableConfig, checkpoint: Any, metadata: Any, new_versions: Any) -> RunnableConfig:
        snapshot = copy.deepcopy(checkpoint)
        channels = snapshot.get("channel_values")
        if not isinstance(channels, dict):
            raise ValueError("Checkpoint has no channel_values")
        decisions = []
        if self._mode != "off":
            for part in self._new_items(config, list(channels.items())):
                decisions.append(self._scan_one(config, part, read=False))
        result = self._backing.put(config, snapshot, metadata, new_versions)
        for decision in decisions:
            self._emit(replace(decision, committed=True))
        return result

    async def aput(self, config: RunnableConfig, checkpoint: Any, metadata: Any, new_versions: Any) -> RunnableConfig:
        snapshot = copy.deepcopy(checkpoint)
        channels = snapshot.get("channel_values")
        if not isinstance(channels, dict):
            raise ValueError("Checkpoint has no channel_values")
        decisions = []
        if self._mode != "off":
            for part in self._new_items(config, list(channels.items())):
                decisions.append(await self._ascan_one(config, part, read=False))
        result = await self._backing.aput(config, snapshot, metadata, new_versions)
        for decision in decisions:
            self._emit(replace(decision, committed=True))
        return result

    def put_writes(
        self, config: RunnableConfig, writes: Sequence[tuple[str, Any]], task_id: str, task_path: str = ""
    ) -> None:
        snapshot = copy.deepcopy(list(writes))
        decisions = []
        if self._mode != "off":
            for part in self._items(snapshot):
                decisions.append(self._scan_one(config, part, read=False))
        self._backing.put_writes(config, snapshot, task_id, task_path)
        for decision in decisions:
            self._emit(replace(decision, committed=True))

    async def aput_writes(
        self, config: RunnableConfig, writes: Sequence[tuple[str, Any]], task_id: str, task_path: str = ""
    ) -> None:
        snapshot = copy.deepcopy(list(writes))
        decisions = []
        if self._mode != "off":
            for part in self._items(snapshot):
                decisions.append(await self._ascan_one(config, part, read=False))
        await self._backing.aput_writes(config, snapshot, task_id, task_path)
        for decision in decisions:
            self._emit(replace(decision, committed=True))
