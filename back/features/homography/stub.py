from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from typing import Optional, Tuple

from app.events import HomographyEvent, MotorStateEvent

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class HomographyStub:
    """모터 각도를 기반으로 간단한 3x3 호모그래피 행렬을 생성."""

    debug_logs: bool = False

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

        if self.debug_logs:
            self._log_candidate_pose(
                head=motor_state.head_position,
                foot=motor_state.foot_position,
                direction=motor_state.direction_label,
            )

        return HomographyEvent(matrix=matrix, timestamp=motor_state.timestamp)

    def _log_candidate_pose(
        self,
        head: Optional[Tuple[float, float]],
        foot: Optional[Tuple[float, float]],
        direction: Optional[str],
    ) -> None:
        head_text = _fmt_point(head)
        foot_text = _fmt_point(foot)
        logger.info(
            "Assistance candidate | head=%s foot=%s direction=%s",
            head_text,
            foot_text,
            direction or "-",
        )


def _fmt_point(point: Optional[Tuple[float, float]]) -> str:
    if point is None:
        return "-"
    return f"({point[0]:.3f}, {point[1]:.3f})"
