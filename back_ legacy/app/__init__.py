"""Application package bootstrap.

Lazily exposes the main application types so that modules which only rely on
lightweight helpers (e.g. `app.config`) do not eagerly import heavier optional
dependencies such as the QA audio stack.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

__all__ = ["Application", "DebugConfig", "RuntimeOverrides"]


if TYPE_CHECKING:  # pragma: no cover - for type checkers only
    from .application import Application, DebugConfig, RuntimeOverrides


def __getattr__(name: str) -> Any:
    if name in __all__:
        module = import_module(".application", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
