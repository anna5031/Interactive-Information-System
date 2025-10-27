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


@dataclass(slots=True)
class MotorStateEvent:
    pan: float
    tilt: float
    has_target: bool
    timestamp: float


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

