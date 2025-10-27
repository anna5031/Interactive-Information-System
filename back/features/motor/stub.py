from __future__ import annotations

import math
import time
from dataclasses import dataclass

from app.events import MotorStateEvent, VisionResultEvent


@dataclass(slots=True)
class MotorStubConfig:
    max_pan_speed: float = 10.0  # degrees per update
    max_tilt_speed: float = 6.0  # degrees per update


class MotorStub:
    """탐색 결과를 기반으로 팬/틸트 각도를 계산하는 단순 스텁."""

    def __init__(self, config: MotorStubConfig):
        self._config = config
        self._pan = 0.0
        self._tilt = 0.0

    def update(self, vision: VisionResultEvent) -> MotorStateEvent:
        if vision.has_target and vision.target_position:
            target_x, target_y = vision.target_position
            self._pan = _approach(self._pan, (target_x - 0.5) * 60.0, self._config.max_pan_speed)
            self._tilt = _approach(self._tilt, (0.5 - target_y) * 40.0, self._config.max_tilt_speed)
        else:
            self._pan = math.sin(time.time() * 0.4) * 30.0
            self._tilt = math.cos(time.time() * 0.4) * 20.0

        return MotorStateEvent(
            pan=self._pan,
            tilt=self._tilt,
            has_target=vision.has_target,
            timestamp=vision.timestamp,
        )


def _approach(current: float, target: float, max_delta: float) -> float:
    delta = target - current
    if abs(delta) <= max_delta:
        return target
    return current + math.copysign(max_delta, delta)
