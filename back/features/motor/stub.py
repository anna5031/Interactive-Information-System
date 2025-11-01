from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace
from typing import Optional, Tuple, TYPE_CHECKING

import numpy as np

from app.events import MotorStateEvent, VisionResultEvent
from features.homography import settings as homography_settings
from features.motor import settings as motor_settings
from features.homography.mapper import PixelToWorldMapper

if TYPE_CHECKING:  # pragma: no cover
    from app.config import HomographyConfig, MotorConfig

CEILING_TARGET_OFFSET_MM = 160.0
CEILING_TARGET_TILT_DEG = 20.0
GAZE_SAMPLE_STEP_PX = 80.0


@dataclass(slots=True)
class MotorStubConfig:
    max_pan_speed: float = 10.0  # degrees per update
    max_tilt_speed: float = 6.0  # degrees per update


class MotorStub:
    """탐색 결과를 기반으로 팬/틸트 각도를 계산하는 단순 스텁."""

    def __init__(
        self,
        config: MotorStubConfig,
        *,
        motor_config: Optional["MotorConfig"] = None,
        homography_config: Optional["HomographyConfig"] = None,
        mapper: Optional[PixelToWorldMapper] = None,
    ):
        self._config = config
        self._pan = 0.0
        self._tilt = 0.0
        self._mapper = mapper

        projector = (
            homography_config.projector_position_mm
            if homography_config is not None
            else homography_settings.PROJECTOR_POSITION_MM
        )
        self._projector = np.array(projector, dtype=float)
        self._floor_z = (
            homography_config.floor_z_mm
            if homography_config is not None
            else homography_settings.FLOOR_Z_MM
        )
        self._projection_ahead = (
            motor_config.geometry.projection_ahead_mm
            if motor_config is not None
            else motor_settings.PROJECTION_AHEAD_MM
        )
        footprint_width = (
            homography_config.footprint.width_mm
            if homography_config is not None
            else homography_settings.FOOTPRINT_WIDTH_MM
        )
        footprint_height = (
            homography_config.footprint.height_mm
            if homography_config is not None
            else homography_settings.FOOTPRINT_HEIGHT_MM
        )
        self._target_plane = (
            homography_config.target_plane
            if homography_config is not None
            else homography_settings.TARGET_PLANE
        ).lower()
        self._ceiling_z = (
            homography_config.projector_position_mm[2]
            if homography_config is not None
            else homography_settings.PROJECTOR_POSITION_MM[2]
        )
        self._stage_width_mm = max(footprint_width * 2.0, 2000.0)
        self._stage_depth_mm = max(footprint_height * 2.0, 2000.0)
        if motor_config is not None:
            self._standby_pan = float(motor_config.poses.standby_pan_deg)
            self._standby_tilt = float(motor_config.poses.standby_tilt_deg)
        else:
            self._standby_pan = float(motor_settings.STANDBY_PAN_DEG)
            self._standby_tilt = float(motor_settings.STANDBY_TILT_DEG)
        self._last_distance: Optional[float] = None
        self._last_distance_timestamp: Optional[float] = None
        self._velocity_epsilon = 1e-3
        self._hold_pose = False
        self._last_state: Optional[MotorStateEvent] = None

    def update(self, vision: VisionResultEvent) -> MotorStateEvent:
        target_x: float
        target_y: float
        synthetic_target: Tuple[float, float] | None = None

        timestamp_value = vision.timestamp if vision.timestamp else time.time()

        if self._hold_pose and self._last_state is not None:
            state = replace(
                self._last_state,
                timestamp=timestamp_value,
                has_target=False,
                approach_velocity_mm_s=None,
                is_approaching=None,
            )
            self._last_state = state
            return state

        if not vision.has_target:
            self._pan = self._standby_pan
            self._tilt = self._standby_tilt
            self._last_distance = None
            self._last_distance_timestamp = None
            state = MotorStateEvent(
                pan=self._pan,
                tilt=self._tilt,
                has_target=False,
                timestamp=timestamp_value,
            )
            self._last_state = state
            return state

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

        norm_x = target_x
        norm_y = target_y

        world_target_tuple: Optional[tuple[float, float, float]] = None
        foot_world_tuple: Optional[tuple[float, float, float]] = None
        distance_to_projector: Optional[float] = None
        approach_velocity: Optional[float] = None
        is_approaching: Optional[bool] = None

        foot_world, world_target = self._derive_world_points(
            norm_x,
            norm_y,
            vision,
        )
        if foot_world is not None and world_target is not None:
            foot_world_tuple = tuple(float(v) for v in foot_world)
            world_target_tuple = tuple(float(v) for v in world_target)
            distance_to_projector = float(np.linalg.norm(foot_world - self._projector))

            if vision.has_target:
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
        else:
            self._last_distance = None
            self._last_distance_timestamp = None

        state = MotorStateEvent(
            pan=self._pan,
            tilt=self._tilt,
            has_target=vision.has_target,
            timestamp=timestamp_value,
            head_position=head_position,
            foot_position=foot_position,
            direction_label=direction_label,
            target_pixel=vision.target_pixel,
            head_pixel=vision.head_pixel,
            foot_pixel=vision.foot_pixel,
            world_target=world_target_tuple,
            foot_world=foot_world_tuple,
            distance_to_projector=distance_to_projector,
            approach_velocity_mm_s=approach_velocity,
            is_approaching=is_approaching,
        )
        self._last_state = state
        return state

    async def move_to_pose(self, *, pan_deg: float, tilt_deg: float) -> None:
        self._pan = float(
            max(motor_settings.PAN_MIN_DEG, min(motor_settings.PAN_MAX_DEG, pan_deg))
        )
        self._tilt = float(
            max(motor_settings.TILT_MIN_DEG, min(motor_settings.TILT_MAX_DEG, tilt_deg))
        )
        self._last_distance = None
        self._last_distance_timestamp = None
        if self._last_state is not None:
            self._last_state = replace(
                self._last_state,
                pan=self._pan,
                tilt=self._tilt,
                has_target=False,
                approach_velocity_mm_s=None,
                is_approaching=None,
            )
        else:
            self._last_state = MotorStateEvent(
                pan=self._pan,
                tilt=self._tilt,
                has_target=False,
                timestamp=time.time(),
            )

    async def set_hold_pose(self, hold: bool) -> None:
        self._hold_pose = hold
        self._last_distance = None
        self._last_distance_timestamp = None

    def _derive_world_points(
        self,
        norm_x: float,
        norm_y: float,
        vision: VisionResultEvent,
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if self._mapper is not None:
            pixel_source = (
                vision.foot_pixel
                or vision.target_pixel
                or vision.head_pixel
            )
            if pixel_source is not None:
                mapped = self._mapper.pixel_to_world(
                    pixel_source,
                    plane_z=self._floor_z,
                )
                if mapped is not None:
                    foot_world = np.array(mapped, dtype=float)
                    forward = self._compute_forward_direction(vision, foot_world)
                    return foot_world, self._compute_projection_target(foot_world.copy(), forward)

        foot_world, world_target = self._compute_world_points(norm_x, norm_y)
        return foot_world, world_target

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

    def _compute_forward_direction(
        self,
        vision: VisionResultEvent,
        foot_world: np.ndarray,
    ) -> Optional[np.ndarray]:
        if self._mapper is None:
            return None

        gaze = vision.gaze_vector
        foot_pixel = vision.foot_pixel
        if gaze is None or foot_pixel is None:
            return None

        sample_pixel = (
            foot_pixel[0] + gaze[0] * GAZE_SAMPLE_STEP_PX,
            foot_pixel[1] + gaze[1] * GAZE_SAMPLE_STEP_PX,
        )
        sample_world = self._mapper.pixel_to_world(
            sample_pixel,
            plane_z=self._floor_z,
        )
        if sample_world is None:
            return None

        direction = np.array(sample_world, dtype=float) - foot_world
        direction[2] = 0.0
        norm = np.linalg.norm(direction[:2])
        if norm < 1e-6:
            return None
        direction[:2] = direction[:2] / norm
        return direction

    def _compute_projection_target(
        self,
        foot_world: np.ndarray,
        forward_direction: Optional[np.ndarray],
    ) -> np.ndarray:
        if forward_direction is not None:
            direction = forward_direction.copy()
        else:
            direction = foot_world - self._projector
            direction[2] = 0.0
            norm = np.linalg.norm(direction[:2])
            if norm < 1e-6:
                direction = np.array([0.0, 1.0, 0.0], dtype=float)
            else:
                direction = direction / norm

        if self._target_plane == "ceiling":
            start = foot_world.copy()
            start[2] += CEILING_TARGET_OFFSET_MM
            theta = math.radians(CEILING_TARGET_TILT_DEG)
            horizontal_scale = math.cos(theta)
            aim_vector = np.array(
                [
                    direction[0] * horizontal_scale,
                    direction[1] * horizontal_scale,
                    math.sin(theta),
                ],
                dtype=float,
            )
            dz = self._ceiling_z - float(start[2])
            if dz <= 0.0 or abs(aim_vector[2]) < 1e-6:
                target = start
                target[2] = self._ceiling_z
                return target

            scale = dz / aim_vector[2]
            target = start + aim_vector * scale
            target[2] = self._ceiling_z
            return target

        target = foot_world.copy()
        target[:2] = target[:2] + direction[:2] * self._projection_ahead
        target[2] = self._floor_z
        return target


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
