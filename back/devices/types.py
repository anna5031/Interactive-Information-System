from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class DeviceCheckResult:
    name: str
    ok: bool
    detail: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)
