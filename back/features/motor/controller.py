from __future__ import annotations

"""Real motor controller that converts vision events into hardware commands."""

import argparse
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


CEILING_TARGET_OFFSET_MM = 160.0
CEILING_TARGET_TILT_DEG = 20.0
GAZE_SAMPLE_STEP_PX = 80.0


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
        if self._lock.locked():
            logger.debug("Motor controller busy; skipping update.")
            return self._snapshot_state(vision, has_target=vision.has_target)

        async with self._lock:
            return await self._update_locked(vision)

    async def shutdown(self) -> None:
        await asyncio.to_thread(self._driver.close)
        self._driver_ready = False

    async def _update_locked(self, vision: VisionResultEvent) -> MotorStateEvent:
        await self._ensure_driver_ready()

        # Prefer feet data, then target/body pixels; skip work if nothing is detectable.
        target_pixel = (
            vision.foot_pixel
            or vision.target_pixel
            or vision.head_pixel
        )
        if target_pixel is None:
            return self._snapshot_state(vision, has_target=False)

        # Convert the selected pixel into world-space coordinates on the floor plane.
        foot_world = self.mapper.pixel_to_world(
            target_pixel,
            plane_z=self.homography_config.floor_z_mm,
        )
        if foot_world is None:
            logger.debug("World mapping failed for pixel %s", target_pixel)
            return self._snapshot_state(vision, has_target=False)

        foot_world_arr = np.array(foot_world, dtype=float)
        #forward_direction = self._compute_forward_direction(vision, foot_world_arr)

        # Offset the aim point so the projection lands slightly ahead of the user's feet.
        #aim_world = self._compute_projection_target(foot_world_arr.copy(), forward_direction)
        aim_world = foot_world_arr # aim at human foot

        # Determine the pan/tilt adjustments required to hit the aim point.
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
        # Track simple approach velocity so downstream logic can react to movement trends.
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

    def _compute_forward_direction(
        self,
        vision: VisionResultEvent,
        foot_world: np.ndarray,
    ) -> Optional[np.ndarray]:
        gaze = vision.gaze_vector
        foot_pixel = vision.foot_pixel
        if gaze is None or foot_pixel is None:
            return None

        sample_pixel = (
            foot_pixel[0] + gaze[0] * GAZE_SAMPLE_STEP_PX,
            foot_pixel[1] + gaze[1] * GAZE_SAMPLE_STEP_PX,
        )
        sample_world = self.mapper.pixel_to_world(
            sample_pixel,
            plane_z=self.homography_config.floor_z_mm,
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
            direction = foot_world - self._projector  # Direction from projector to the detected foot position.
            direction[2] = 0.0  # Flatten the vector so we only keep the horizontal (XY) components.
            norm = np.linalg.norm(direction[:2])  # Measure horizontal distance to avoid dividing by zero.
            if norm < 1e-6:
                direction = np.array([0.0, 1.0, 0.0])  # Default to projecting straight ahead if the foot is centred.
            else:
                direction[:2] = direction[:2] / norm  # Normalize the XY direction so we can scale it cleanly.
        target_plane = getattr(self.homography_config, "target_plane", "floor").lower()
        if target_plane == "ceiling":
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
            ceiling_z = float(self.homography_config.projector_position_mm[2])
            dz = ceiling_z - float(start[2])
            if dz <= 0.0 or abs(aim_vector[2]) < 1e-6:
                target = start
                target[2] = ceiling_z
                return target
            scale = dz / aim_vector[2]
            target = start + aim_vector * scale
            target[2] = ceiling_z
            return target

        offset = direction * self.motor_config.geometry.projection_ahead_mm  # Step forward by the configured distance.
        target = foot_world.copy()  # Start targeting at the foot location.
        target[2] = self.homography_config.floor_z_mm  # Force the Z coordinate to the floor plane.
        target[:2] = target[:2] + offset[:2]  # Shift the XY position forward to land slightly ahead of the feet.
        return target  # Return the final world-space aim point for the laser/projected graphic.

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

    async def _send_command(self, tilt_deg: float, pan_deg: float) -> str:
        # NOTE: asyncio.to_thread keeps integration simple, but if command frequency
        # grows significantly the default executor can become a bottleneck. In that
        # case switch to a dedicated ThreadPoolExecutor to control concurrency.
        response = await asyncio.to_thread(
            self._driver.send,
            int(round(tilt_deg)),
            int(round(pan_deg)),
        )
        return response

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


async def _cli_async(args: "argparse.Namespace") -> int:
    import sys
    from types import SimpleNamespace

    try:
        from app.config import load_config
    except Exception as exc:  # pragma: no cover - CLI convenience only
        print(f"Failed to load application config: {exc}", file=sys.stderr)
        return 1

    config = load_config()
    motor_config = config.motor

    if args.port:
        motor_config.serial.port = args.port
    if args.baudrate is not None:
        motor_config.serial.baudrate = args.baudrate
    if args.timeout is not None:
        motor_config.serial.timeout = args.timeout
    if args.skip_ping:
        motor_config.ping_on_startup = False

    dummy_mapper = SimpleNamespace(pixel_to_world=lambda *_, **__: None)
    controller = RealMotorController(
        motor_config=motor_config,
        homography_config=config.homography,
        mapper=dummy_mapper,  # type: ignore[arg-type]
    )

    try:
        await controller._ensure_driver_ready()
        response = await controller._send_command(args.tilt, args.pan)
        print(f"Motor response: {response!r}")
        return 0
    except Exception as exc:  # pragma: no cover - hardware dependent
        print(f"Motor command failed: {exc}", file=sys.stderr)
        return 1
    finally:
        await controller.shutdown()


def _cli() -> int:
    import argparse
    import asyncio

    parser = argparse.ArgumentParser(
        description="Send a pan/tilt command through RealMotorController for testing.",
    )
    parser.add_argument("--tilt", type=float, required=True, help="Tilt angle in degrees.")
    parser.add_argument("--pan", type=float, required=True, help="Pan angle in degrees.")
    parser.add_argument(
        "--port",
        help="Override serial device path (defaults to BACKEND_MOTOR_SERIAL_PORT or settings).",
    )
    parser.add_argument(
        "--baudrate",
        type=int,
        help="Override serial baudrate (defaults to BACKEND_MOTOR_SERIAL_BAUD).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        help="Override serial read timeout in seconds.",
    )
    parser.add_argument(
        "--skip-ping",
        action="store_true",
        help="Skip the startup ping before sending the command.",
    )
    args = parser.parse_args()
    return asyncio.run(_cli_async(args))


if __name__ == "__main__":  # pragma: no cover - manual motor test harness
    raise SystemExit(_cli())
