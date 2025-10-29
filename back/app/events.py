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
    needs_assistance: bool = False
    head_position: Optional[Tuple[float, float]] = None
    foot_position: Optional[Tuple[float, float]] = None
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
