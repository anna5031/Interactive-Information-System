from __future__ import annotations

"""Dataclasses and loaders for exploration pipeline configuration."""

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from . import settings


@dataclass(slots=True)
class DeviceConfig:
    device_preference: Tuple[str, ...]
    force_device: Optional[str] = None


@dataclass(slots=True)
class CameraConfig:
    source: str | int
    frame_size: Optional[Tuple[int, int]]
    fourcc: Optional[str]
    target_fps: Optional[float]
    reference_image_path: Optional[Path]


@dataclass(slots=True)
class ModelConfig:
    model_path: Path
    confidence_threshold: float
    iou_threshold: float
    keypoint_threshold: float


@dataclass(slots=True)
class TrackingConfig:
    distance_threshold: float
    max_age: int
    stationary_speed_threshold: float
    velocity_smoothing: float
    angle_speed_threshold: float
    stationary_duration_seconds: float


@dataclass(slots=True)
class AssistanceConfig:
    stationary_seconds_required: float
    cooldown_seconds: float


@dataclass(slots=True)
class ExplorationConfig:
    device: DeviceConfig
    camera: CameraConfig
    model: ModelConfig
    tracking: TrackingConfig
    assistance: AssistanceConfig
    crop_enabled: bool
    crop_ratio: float
    debug_display: bool
    debug_window_name: str


DEFAULT_EXPLORATION_CONFIG = ExplorationConfig(
    device=DeviceConfig(
        device_preference=settings.DEVICE_PREFERENCE,
        force_device=settings.FORCE_DEVICE,
    ),
    camera=CameraConfig(
        source=settings.CAMERA_SOURCE,
        frame_size=settings.CAMERA_FRAME_SIZE,
        fourcc=settings.CAMERA_FOURCC,
        target_fps=settings.CAMERA_TARGET_FPS,
        reference_image_path=Path(settings.CAMERA_REFERENCE_IMAGE)
        if settings.CAMERA_REFERENCE_IMAGE
        else None,
    ),
    model=ModelConfig(
        model_path=Path(settings.MODEL_PATH),
        confidence_threshold=settings.MODEL_CONFIDENCE_THRESHOLD,
        iou_threshold=settings.MODEL_IOU_THRESHOLD,
        keypoint_threshold=settings.MODEL_KEYPOINT_THRESHOLD,
    ),
    tracking=TrackingConfig(
        distance_threshold=settings.TRACK_DISTANCE_THRESHOLD,
        max_age=settings.TRACK_MAX_AGE,
        stationary_speed_threshold=settings.TRACK_STATIONARY_SPEED_THRESHOLD,
        velocity_smoothing=settings.TRACK_VELOCITY_SMOOTHING,
        angle_speed_threshold=settings.TRACK_ANGLE_SPEED_THRESHOLD,
        stationary_duration_seconds=settings.TRACK_STATIONARY_DURATION,
    ),
    assistance=AssistanceConfig(
        stationary_seconds_required=settings.ASSIST_STATIONARY_SECONDS,
        cooldown_seconds=settings.ASSIST_COOLDOWN_SECONDS,
    ),
    crop_enabled=settings.CROP_ENABLED,
    crop_ratio=settings.CROP_RATIO,
    debug_display=settings.DEBUG_DISPLAY,
    debug_window_name=settings.DEBUG_WINDOW_NAME,
)


def load_exploration_config() -> ExplorationConfig:
    """Return a copy of the default exploration config."""

    return copy.deepcopy(DEFAULT_EXPLORATION_CONFIG)
