from __future__ import annotations

import asyncio
from contextlib import suppress
from typing import Awaitable, Callable, List


class TaskRunner:
    """비동기 태스크를 추적하고 종료 시 정리하는 헬퍼."""

    def __init__(self) -> None:
        self._tasks: List[asyncio.Task[None]] = []

    def create(self, coro: Awaitable[None], name: str | None = None) -> asyncio.Task[None]:
        task = asyncio.create_task(coro, name=name)
        self._tasks.append(task)
        return task

    async def shutdown(self) -> None:
        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            with suppress(asyncio.CancelledError):
                await task
        self._tasks.clear()
