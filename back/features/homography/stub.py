from __future__ import annotations

import math
import time
from dataclasses import dataclass

from app.events import HomographyEvent, MotorStateEvent


@dataclass(slots=True)
class HomographyStub:
    """모터 각도를 기반으로 간단한 3x3 호모그래피 행렬을 생성."""

    def build(self, motor_state: MotorStateEvent) -> HomographyEvent:
        angle = math.radians(motor_state.pan) * 0.1 + time.time() * 0.05
        oscillation = 0.1 * math.sin(angle * 0.5)
        shear = 0.05 * math.cos(angle * 0.33)
        perspective_x = 0.001 * math.sin(angle * 0.25)
        perspective_y = 0.001 * math.cos(angle * 0.25)

        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        matrix = [
            [(1 + oscillation) * cos_a, -sin_a + shear, perspective_x],
            [sin_a + shear, (1 + oscillation) * cos_a, perspective_y],
            [0.0005 * math.cos(angle * 0.4), 0.0005 * math.sin(angle * 0.4), 1.0],
        ]
        return HomographyEvent(matrix=matrix, timestamp=motor_state.timestamp)
