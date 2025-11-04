"""세션 로그용 간단한 인메모리 데이터 스토어 스텁."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass(slots=True)
class SessionLogEntry:
    message: str


@dataclass(slots=True)
class InMemoryDatastore:
    entries: List[SessionLogEntry] = field(default_factory=list)

    def append(self, message: str) -> None:
        self.entries.append(SessionLogEntry(message=message))

    def clear(self) -> None:
        self.entries.clear()

