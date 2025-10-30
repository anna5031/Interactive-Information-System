from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


@dataclass(slots=True)
class VisionResultEvent:
    has_target: bool
    target_position: Optional[Tuple[float, float]]
    gaze_vector: Optional[Tuple[float, float]]
    confidence: float
    timestamp: float
    frame_width: Optional[int] = None
    frame_height: Optional[int] = None
    target_pixel: Optional[Tuple[float, float]] = None
    needs_assistance: bool = False
    head_position: Optional[Tuple[float, float]] = None
    foot_position: Optional[Tuple[float, float]] = None
    head_pixel: Optional[Tuple[float, float]] = None
    foot_pixel: Optional[Tuple[float, float]] = None
    direction_label: Optional[str] = None
    stationary_duration: float = 0.0


@dataclass(slots=True)
class MotorStateEvent:
    pan: float
    tilt: float
    has_target: bool
    timestamp: float
    head_position: Optional[Tuple[float, float]] = None
    foot_position: Optional[Tuple[float, float]] = None
    direction_label: Optional[str] = None
    target_pixel: Optional[Tuple[float, float]] = None
    head_pixel: Optional[Tuple[float, float]] = None
    foot_pixel: Optional[Tuple[float, float]] = None
    world_target: Optional[Tuple[float, float, float]] = None
    foot_world: Optional[Tuple[float, float, float]] = None
    distance_to_projector: Optional[float] = None
    approach_velocity_mm_s: Optional[float] = None
    is_approaching: Optional[bool] = None


@dataclass(slots=True)
class HomographyEvent:
    matrix: list[list[float]]
    timestamp: float


@dataclass(slots=True)
class CommandEvent:
    action: str
    context: Dict[str, Any]
    requires_completion: bool
    message: Dict[str, Any]
