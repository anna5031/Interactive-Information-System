from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .config import DevicePreferences, load_device_preferences

if TYPE_CHECKING:
    from .manager import DeviceManager
    from .types import DeviceCheckResult

__all__ = [
    "DeviceManager",
    "DevicePreferences",
    "load_device_preferences",
    "DeviceCheckResult",
]


def __getattr__(name: str) -> Any:
    if name == "DeviceManager":
        from .manager import DeviceManager as _DeviceManager

        return _DeviceManager
    if name == "DeviceCheckResult":
        from .types import DeviceCheckResult as _DeviceCheckResult

        return _DeviceCheckResult
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
