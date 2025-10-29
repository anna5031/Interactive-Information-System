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
        target_x: float
        target_y: float
        synthetic_target: tuple[float, float] | None = None

        if vision.has_target and vision.target_position:
            target_x, target_y = vision.target_position
        elif vision.head_position is not None:
            target_x, target_y = vision.head_position
            synthetic_target = (target_x, target_y)
        else:
            t = time.time() * 0.35
            target_x = 0.5 + 0.3 * math.sin(t)
            target_y = 0.5 + 0.24 * math.cos(t * 0.9)
            synthetic_target = (max(0.0, min(1.0, target_x)), max(0.0, min(1.0, target_y)))
            target_x, target_y = synthetic_target

        self._pan = _approach(self._pan, (target_x - 0.5) * 60.0, self._config.max_pan_speed)
        self._tilt = _approach(self._tilt, (0.5 - target_y) * 40.0, self._config.max_tilt_speed)

        head_position = vision.head_position or synthetic_target
        foot_position = vision.foot_position
        if foot_position is None and head_position is not None:
            foot_position = (
                head_position[0],
                max(0.0, min(1.0, head_position[1] + 0.12)),
            )

        direction_label = vision.direction_label
        if direction_label is None and head_position is not None:
            direction_label = _direction_from_target(head_position)

        return MotorStateEvent(
            pan=self._pan,
            tilt=self._tilt,
            has_target=vision.has_target,
            timestamp=vision.timestamp,
            head_position=head_position,
            foot_position=foot_position,
            direction_label=direction_label,
        )


def _approach(current: float, target: float, max_delta: float) -> float:
    delta = target - current
    if abs(delta) <= max_delta:
        return target
    return current + math.copysign(max_delta, delta)


def _direction_from_target(target: tuple[float, float]) -> str:
    dx = target[0] - 0.5
    dy = 0.5 - target[1]
    angle = (math.degrees(math.atan2(dy, dx)) + 360.0) % 360.0
    labels = [
        (0, "E"),
        (45, "NE"),
        (90, "N"),
        (135, "NW"),
        (180, "W"),
        (225, "SW"),
        (270, "S"),
        (315, "SE"),
    ]
    best = min(labels, key=lambda item: abs(((angle - item[0] + 180) % 360) - 180))
    return best[1]
