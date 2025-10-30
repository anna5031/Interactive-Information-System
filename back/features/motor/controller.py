from __future__ import annotations

"""Real motor controller that converts vision events into hardware commands."""

import asyncio
import logging
import math
from dataclasses import dataclass
from typing import Optional, Tuple, TYPE_CHECKING

import numpy as np

from app.events import MotorStateEvent, VisionResultEvent
from features.homography.mapper import PixelToWorldMapper

from .driver import SerialMotorDriver

logger = logging.getLogger(__name__)

if TYPE_CHECKING:  # pragma: no cover
    from app.config import HomographyConfig, MotorConfig


@dataclass
class RealMotorController:
    motor_config: "MotorConfig"
    homography_config: "HomographyConfig"
    mapper: PixelToWorldMapper

    def __post_init__(self) -> None:
        self._driver = SerialMotorDriver(
            port=self.motor_config.serial.port,
            baudrate=self.motor_config.serial.baudrate,
            timeout=self.motor_config.serial.timeout,
        )
        self._lock = asyncio.Lock()
        self._driver_ready = False
        self._projector = np.array(self.homography_config.projector_position_mm, dtype=float)
        self._last_distance: Optional[float] = None
        self._last_distance_timestamp: Optional[float] = None
        self._velocity_epsilon = 1e-3
        self._last_state = MotorStateEvent(
            pan=self.motor_config.limits.pan_init_deg,
            tilt=self.motor_config.limits.tilt_init_deg,
            has_target=False,
            timestamp=0.0,
        )

    async def update(self, vision: VisionResultEvent) -> MotorStateEvent:
        async with self._lock:
            return await self._update_locked(vision)

    async def shutdown(self) -> None:
        await asyncio.to_thread(self._driver.close)
        self._driver_ready = False

    async def _update_locked(self, vision: VisionResultEvent) -> MotorStateEvent:
        await self._ensure_driver_ready()

        target_pixel = (
            vision.foot_pixel
            or vision.target_pixel
            or vision.head_pixel
        )
        if target_pixel is None:
            return self._snapshot_state(vision, has_target=False)

        foot_world = self.mapper.pixel_to_world(
            target_pixel,
            plane_z=self.homography_config.floor_z_mm,
        )
        if foot_world is None:
            logger.debug("World mapping failed for pixel %s", target_pixel)
            return self._snapshot_state(vision, has_target=False)

        foot_world_arr = np.array(foot_world, dtype=float)
        aim_world = self._compute_projection_target(foot_world_arr.copy())
        tilt_raw, pan_raw = self._compute_angles(aim_world)
        tilt_cmd = _clip(
            self.motor_config.limits.tilt_init_deg + tilt_raw,
            self.motor_config.limits.tilt_min_deg,
            self.motor_config.limits.tilt_max_deg,
        )
        pan_cmd = _clip(
            self.motor_config.limits.pan_init_deg + pan_raw,
            self.motor_config.limits.pan_min_deg,
            self.motor_config.limits.pan_max_deg,
        )

        await self._send_command(tilt_cmd, pan_cmd)

        distance = float(np.linalg.norm(foot_world_arr - self._projector))
        approach_velocity = None
        is_approaching = None
        if self._last_distance is not None and self._last_distance_timestamp is not None:
            delta_time = vision.timestamp - self._last_distance_timestamp
            if delta_time > 1e-3:
                delta_dist = self._last_distance - distance
                approach_velocity = delta_dist / delta_time
                if abs(approach_velocity) <= self._velocity_epsilon:
                    is_approaching = False
                else:
                    is_approaching = approach_velocity > 0.0

        self._last_distance = distance
        self._last_distance_timestamp = vision.timestamp

        state = MotorStateEvent(
            pan=pan_cmd,
            tilt=tilt_cmd,
            has_target=vision.has_target,
            timestamp=vision.timestamp,
            head_position=vision.head_position,
            foot_position=vision.foot_position,
            direction_label=vision.direction_label,
            target_pixel=vision.target_pixel,
            head_pixel=vision.head_pixel,
            foot_pixel=vision.foot_pixel,
            world_target=(
                float(aim_world[0]),
                float(aim_world[1]),
                float(aim_world[2]),
            ),
            foot_world=(
                float(foot_world_arr[0]),
                float(foot_world_arr[1]),
                float(foot_world_arr[2]),
            ),
            distance_to_projector=distance,
            approach_velocity_mm_s=approach_velocity,
            is_approaching=is_approaching,
        )
        self._last_state = state
        return state

    async def _ensure_driver_ready(self) -> None:
        if self._driver_ready:
            return
        await asyncio.to_thread(self._driver.open)
        if self.motor_config.ping_on_startup:
            response = await asyncio.to_thread(self._driver.ping)
            logger.info("Motor ping response: %s", response)
        self._driver_ready = True

    def _compute_projection_target(self, foot_world: np.ndarray) -> np.ndarray:
        vector = foot_world - self._projector
        vector[2] = 0.0
        norm = np.linalg.norm(vector[:2])
        if norm < 1e-6:
            vector = np.array([0.0, 1.0, 0.0])
        else:
            vector[:2] = vector[:2] / norm
        offset = vector * self.motor_config.geometry.projection_ahead_mm
        target = foot_world.copy()
        target[2] = self.homography_config.floor_z_mm
        target[:2] = target[:2] + offset[:2]
        return target

    def _compute_angles(self, target: np.ndarray) -> Tuple[float, float]:
        x, y, z = map(float, target)
        r_xy = math.hypot(x, y)
        dz = z - self.motor_config.geometry.tilt_axis_height_mm
        r = math.hypot(r_xy, dz)
        ratio = self.motor_config.geometry.projector_offset_mm / max(r, 1e-6)
        ratio = max(-1.0, min(1.0, ratio))
        theta_t = math.degrees(math.acos(ratio) - math.atan2(r_xy, dz))
        theta_p = math.degrees(math.atan2(y, x))
        return theta_t, theta_p

    async def _send_command(self, tilt_deg: float, pan_deg: float) -> None:
        # NOTE: asyncio.to_thread keeps integration simple, but if command frequency
        # grows significantly the default executor can become a bottleneck. In that
        # case switch to a dedicated ThreadPoolExecutor to control concurrency.
        await asyncio.to_thread(
            self._driver.send,
            int(round(tilt_deg)),
            int(round(pan_deg)),
        )

    def _snapshot_state(self, vision: VisionResultEvent, has_target: bool) -> MotorStateEvent:
        state = MotorStateEvent(
            pan=self._last_state.pan,
            tilt=self._last_state.tilt,
            has_target=has_target,
            timestamp=vision.timestamp,
            head_position=vision.head_position,
            foot_position=vision.foot_position,
            direction_label=vision.direction_label,
            target_pixel=vision.target_pixel,
            head_pixel=vision.head_pixel,
            foot_pixel=vision.foot_pixel,
            world_target=self._last_state.world_target,
            foot_world=self._last_state.foot_world,
            distance_to_projector=self._last_state.distance_to_projector,
            approach_velocity_mm_s=self._last_state.approach_velocity_mm_s,
            is_approaching=None if not has_target else self._last_state.is_approaching,
        )
        self._last_state = state
        return state


def _clip(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))
