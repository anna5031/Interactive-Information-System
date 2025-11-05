from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Literal


def _timestamp() -> float:
    return time.time()


@dataclass(slots=True)
class HomographyMessage:
    type: Literal["homography"] = "homography"
    timestamp: float = field(default_factory=_timestamp)
    matrix: list[list[float]] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class CommandMessage:
    type: Literal["command"] = "command"
    commandId: str = ""
    sequence: int = 0
    action: str = ""
    context: Dict[str, Any] = field(default_factory=dict)
    requiresCompletion: bool = False
    timestamp: float = field(default_factory=_timestamp)

    def as_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        if not self.requiresCompletion:
            payload.pop("requiresCompletion", None)
        return payload


@dataclass(slots=True)
class SyncMessage:
    type: Literal["sync"] = "sync"
    state: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=_timestamp)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)
