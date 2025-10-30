from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from app.events import MotorStateEvent, VisionResultEvent
from features.homography import settings as homography_settings
from features.motor import settings as motor_settings


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
        self._projector = np.array(homography_settings.PROJECTOR_POSITION_MM, dtype=float)
        self._floor_z = homography_settings.FLOOR_Z_MM
        self._projection_ahead = motor_settings.PROJECTION_AHEAD_MM
        footprint_width = homography_settings.FOOTPRINT_WIDTH_MM
        footprint_height = homography_settings.FOOTPRINT_HEIGHT_MM
        self._stage_width_mm = max(footprint_width * 2.0, 2000.0)
        self._stage_depth_mm = max(footprint_height * 2.0, 2000.0)
        self._last_distance: Optional[float] = None
        self._last_distance_timestamp: Optional[float] = None
        self._velocity_epsilon = 1e-3

    def update(self, vision: VisionResultEvent) -> MotorStateEvent:
        target_x: float
        target_y: float
        synthetic_target: Tuple[float, float] | None = None

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

        timestamp_value = vision.timestamp if vision.timestamp else time.time()

        world_target_tuple: Optional[tuple[float, float, float]] = None
        foot_world_tuple: Optional[tuple[float, float, float]] = None
        distance_to_projector: Optional[float] = None
        approach_velocity: Optional[float] = None
        is_approaching: Optional[bool] = None

        if vision.has_target:
            norm_x = target_x
            norm_y = target_y
            foot_world, world_target = self._compute_world_points(norm_x, norm_y)
            foot_world_tuple = tuple(float(v) for v in foot_world)
            world_target_tuple = tuple(float(v) for v in world_target)
            distance_to_projector = float(np.linalg.norm(foot_world - self._projector))

            if self._last_distance is not None and self._last_distance_timestamp is not None:
                delta_time = timestamp_value - self._last_distance_timestamp
                if delta_time > 1e-3:
                    delta_dist = self._last_distance - distance_to_projector
                    approach_velocity = delta_dist / delta_time
                    if abs(approach_velocity) <= self._velocity_epsilon:
                        is_approaching = False
                    else:
                        is_approaching = approach_velocity > 0.0

            self._last_distance = distance_to_projector
            self._last_distance_timestamp = timestamp_value
        else:
            self._last_distance = None
            self._last_distance_timestamp = None

        return MotorStateEvent(
            pan=self._pan,
            tilt=self._tilt,
            has_target=vision.has_target,
            timestamp=timestamp_value,
            head_position=head_position,
            foot_position=foot_position,
            direction_label=direction_label,
            target_pixel=None,
            head_pixel=None,
            foot_pixel=None,
            world_target=world_target_tuple,
            foot_world=foot_world_tuple,
            distance_to_projector=distance_to_projector,
            approach_velocity_mm_s=approach_velocity,
            is_approaching=is_approaching,
        )

    def _compute_world_points(self, norm_x: float, norm_y: float) -> tuple[np.ndarray, np.ndarray]:
        x_mm = (float(norm_x) - 0.5) * self._stage_width_mm
        y_mm = max(0.0, self._stage_depth_mm * (1.0 - float(norm_y)))
        foot = np.array(
            [
                self._projector[0] + x_mm,
                self._projector[1] + y_mm,
                self._floor_z,
            ],
            dtype=float,
        )
        vector = foot - self._projector
        vector[2] = 0.0
        norm = np.linalg.norm(vector[:2])
        if norm < 1e-6:
            direction = np.array([0.0, 1.0, 0.0])
        else:
            direction = vector / norm
        target = foot + direction * self._projection_ahead
        target[2] = self._floor_z
        return foot, target


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
