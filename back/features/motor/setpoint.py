from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .config import MotorGeometryConfig, MotorAxisConfig


@dataclass(slots=True)
class MotorAngles:
    tilt_deg: float
    pan_deg: float


def _clip(value: float, vmin: float, vmax: float) -> float:
    return min(max(value, vmin), vmax)


class SetpointCalculator:
    """Convert target coordinates into tilt/pan angles with limits applied."""

    def __init__(
        self,
        geometry: MotorGeometryConfig,
        pan_axis: MotorAxisConfig,
        tilt_axis: MotorAxisConfig,
    ) -> None:
        self.geometry = geometry
        self.pan_axis = pan_axis
        self.tilt_axis = tilt_axis

    def calculate_raw_angles(self, target: Sequence[float]) -> MotorAngles:
        x, y, z = (float(v) for v in target)
        r_xy = math.hypot(x, y)
        dz = z - self.geometry.tilt_axis_height_mm
        r = math.hypot(r_xy, dz)

        theta_pan = math.atan2(y, x)

        if r <= 1e-9:
            raise ValueError("Target too close to the pan/tilt origin.")

        ratio = np.clip(self.geometry.z_offset_mm / r, -1.0, 1.0)
        theta1 = math.acos(ratio)
        theta2 = math.atan2(r_xy, dz)
        theta_tilt = theta1 - theta2

        # NOTE: Legacy nudge_test 하드웨어 축 정의와 일치시키기 위해
        # tilt/pan 순서를 뒤집어 반환한다.
        return MotorAngles(
            tilt_deg=math.degrees(theta_pan),
            pan_deg=math.degrees(theta_tilt),
        )

    def apply_offsets(self, angles: MotorAngles) -> MotorAngles:
        tilt = _clip(
            angles.tilt_deg + self.tilt_axis.init_deg,
            self.tilt_axis.min_deg,
            self.tilt_axis.max_deg,
        )
        pan = _clip(
            angles.pan_deg + self.pan_axis.init_deg,
            self.pan_axis.min_deg,
            self.pan_axis.max_deg,
        )
        return MotorAngles(tilt_deg=tilt, pan_deg=pan)

    def calculate_command(self, target: Sequence[float]) -> MotorAngles:
        raw = self.calculate_raw_angles(target)
        return self.apply_offsets(raw)
