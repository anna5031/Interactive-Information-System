from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional


@dataclass(slots=True)
class FPSEstimator:
    """Exponential moving average FPS estimator."""

    target_fps: Optional[float] = None
    alpha: float = 0.1
    _last_timestamp: Optional[float] = None
    _fps: float = 30.0

    def mark(self) -> None:
        now = time.monotonic()
        if self._last_timestamp is None:
            self._last_timestamp = now
            return
        delta = max(1e-3, now - self._last_timestamp)
        instant = 1.0 / delta
        self._fps = (1 - self.alpha) * self._fps + self.alpha * instant
        self._last_timestamp = now

    @property
    def current(self) -> float:
        return self._fps
