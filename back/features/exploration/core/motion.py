from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from ..config import TrackingConfig


@dataclass(slots=True)
class MotionState:
    point: Optional[np.ndarray] = None
    last_frame: Optional[int] = None
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=float))
    smoothed_velocity: np.ndarray = field(
        default_factory=lambda: np.zeros(2, dtype=float)
    )
    speed: float = 0.0
    stationary_time: float = 0.0
    is_stationary: bool = False
    angle_deg: Optional[float] = None


def _to_xy(point: Sequence[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(point, dtype=float).reshape(-1)
    if arr.size >= 2:
        return arr[:2]
    if arr.size == 1:
        return np.array([arr[0], 0.0], dtype=float)
    return np.zeros(2, dtype=float)


def update_motion_state(
    state: MotionState,
    *,
    new_point: Sequence[float] | np.ndarray,
    frame_idx: int,
    fps: float,
    config: TrackingConfig,
) -> None:
    new_position = _to_xy(new_point)
    if state.point is None:
        state.point = new_position
        state.last_frame = frame_idx
        state.velocity = np.zeros(2, dtype=float)
        state.smoothed_velocity = np.zeros(2, dtype=float)
        state.speed = 0.0
        state.stationary_time = 0.0
        state.is_stationary = False
        state.angle_deg = None
        return

    last_frame = state.last_frame if state.last_frame is not None else frame_idx - 1
    frame_gap = max(1, frame_idx - last_frame)
    fps = fps if fps > 0 else 30.0

    displacement = new_position - state.point
    instant_velocity = displacement * fps / frame_gap
    alpha = max(0.0, min(1.0, config.velocity_smoothing))
    blended_alpha = alpha**frame_gap if frame_gap > 1 else alpha
    state.smoothed_velocity = (
        blended_alpha * state.smoothed_velocity
        + (1.0 - blended_alpha) * instant_velocity
    )
    state.velocity = instant_velocity
    state.speed = float(np.linalg.norm(state.smoothed_velocity))

    if state.speed <= config.stationary_speed_threshold:
        state.stationary_time += frame_gap / fps
    else:
        state.stationary_time = 0.0

    state.is_stationary = (
        state.stationary_time >= config.stationary_duration_seconds
    )

    effective_velocity = (
        np.zeros_like(state.smoothed_velocity)
        if state.is_stationary
        else state.smoothed_velocity
    )
    effective_speed = float(np.linalg.norm(effective_velocity))
    if effective_speed >= config.angle_speed_threshold:
        state.angle_deg = (
            np.degrees(np.arctan2(-effective_velocity[1], effective_velocity[0]))
            + 360.0
        ) % 360.0
    else:
        state.angle_deg = None

    state.point = new_position
    state.last_frame = frame_idx
