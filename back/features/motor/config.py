from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import yaml


@dataclass(slots=True)
class MotorAxisConfig:
    min_deg: float
    max_deg: float
    init_deg: float


@dataclass(slots=True)
class MotorGeometryConfig:
    z_offset_mm: float
    tilt_axis_height_mm: float


@dataclass(slots=True)
class SerialConfig:
    port: str
    baudrate: int
    timeout: float


@dataclass(slots=True)
class QAProjectionConfig:
    ceiling_normal: Tuple[float, float, float]
    displacement_mm: float
    screen_width_mm: float
    screen_height_ratio: float
    roll_deg: float


@dataclass(slots=True)
class ProjectorConfig:
    width_px: int
    height_px: int
    horizontal_fov_deg: float
    beam_offset_mm: Tuple[float, float, float]
    pan_tilt_origin_height_mm: float
    beam_wall_displacement_m: float
    input_image_width_px: int
    input_image_height_px: int


@dataclass(slots=True)
class MotorSettings:
    beam_geometry: MotorGeometryConfig
    serial: SerialConfig
    motor_pan: MotorAxisConfig
    motor_tilt: MotorAxisConfig
    qa_projection: QAProjectionConfig
    projector: ProjectorConfig


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _as_tuple(values) -> Tuple[float, float, float]:
    if isinstance(values, str):
        return tuple(json.loads(values))
    return tuple(float(v) for v in values)


def load_motor_settings(path: Path | None = None) -> MotorSettings:
    cfg_path = path or Path("config/motor_settings.yaml")
    data = _load_yaml(cfg_path)

    beam = data.get("beam_geometry", {})
    motor = data.get("motor", {})
    serial = data.get("serial", {})
    qa = data.get("qa_projection", {})
    projector = data.get("projector", {})

    return MotorSettings(
        beam_geometry=MotorGeometryConfig(
            z_offset_mm=float(beam.get("z_offset_mm", 0.0)),
            tilt_axis_height_mm=float(beam.get("tilt_axis_height_mm", 0.0)),
        ),
        serial=SerialConfig(
            port=str(serial.get("port", "COM1")),
            baudrate=int(serial.get("baudrate", 115200)),
            timeout=float(serial.get("timeout", 1.0)),
        ),
        motor_pan=_load_axis(motor.get("pan", {})),
        motor_tilt=_load_axis(motor.get("tilt", {})),
        qa_projection=QAProjectionConfig(
            ceiling_normal=_as_tuple(qa.get("ceiling_normal", (0.0, 0.0, 1.0))),
            displacement_mm=float(qa.get("displacement_mm", 0.0)),
            screen_width_mm=float(qa.get("screen_width_mm", 0.0)),
            screen_height_ratio=float(qa.get("screen_height_ratio", 0.5625)),
            roll_deg=float(qa.get("roll_deg", 0.0)),
        ),
        projector=ProjectorConfig(
            width_px=int(projector.get("width_px", 3840)),
            height_px=int(projector.get("height_px", 2160)),
            horizontal_fov_deg=float(projector.get("horizontal_fov_deg", 45.0)),
            beam_offset_mm=_as_tuple(projector.get("beam_offset_mm", (0.0, 0.0, 0.0))),
            pan_tilt_origin_height_mm=float(
                projector.get("pan_tilt_origin_height_mm", 0.0)
            ),
            beam_wall_displacement_m=float(
                projector.get("beam_wall_displacement_m", 0.0)
            ),
            input_image_width_px=int(projector.get("input_image_width_px", 1920)),
            input_image_height_px=int(projector.get("input_image_height_px", 1080)),
        ),
    )


def _load_axis(data: Dict[str, Any]) -> MotorAxisConfig:
    return MotorAxisConfig(
        min_deg=float(data.get("min_deg", 0.0)),
        max_deg=float(data.get("max_deg", 0.0)),
        init_deg=float(data.get("init_deg", 0.0)),
    )
