from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from enum import Enum


class SpeedMetric(str, Enum):
    PIXEL = "pixel"
    WORLD = "world"


@dataclass(slots=True)
class TrackingConfig:
    distance_threshold: float = 80.0
    max_age: int = 12
    velocity_smoothing: float = 0.6
    stationary_speed_threshold: float = 10.0
    stationary_duration_seconds: float = 3.0
    angle_speed_threshold: float = 10.0
    speed_metric: SpeedMetric = SpeedMetric.PIXEL


@dataclass(slots=True)
class AssistanceConfig:
    stationary_seconds_required: float = 3.0
    cooldown_seconds: float = 5.0
    approach_timeout_seconds: float = 10.0
    approach_tolerance_mm: float = 300.0
    retreat_threshold_mm: float = 400.0
    target_origin_x_mm: float = 0.0
    target_origin_y_mm: float = 0.0
    dummy_nudge_pass_enabled: bool = False
    dummy_nudge_pass_seconds: float = 5.0


@dataclass(slots=True)
class DetectionConfig:
    confidence_threshold: float = 0.3
    iou_threshold: float = 0.7
    keypoint_confidence_threshold: float = 0.1


@dataclass(slots=True)
class MappingConfig:
    camera_calibration_file: Path = Path(
        "features/exploration/calibration/camera_calib.npz"
    )
    camera_extrinsics_file: Path = Path(
        "features/exploration/calibration/camera_extrinsics.npz"
    )
    floor_z_mm: float = 0.0
    enabled: bool = True
