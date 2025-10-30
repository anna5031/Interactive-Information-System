from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

from features.homography import settings as homography_settings
from features.motor import settings as motor_settings


@dataclass(slots=True)
class WebSocketConfig:
    host: str = "0.0.0.0"
    port: int = 8765


@dataclass(slots=True)
class MotorSerialConfig:
    port: str
    baudrate: int
    timeout: float


@dataclass(slots=True)
class MotorLimitsConfig:
    pan_min_deg: float
    pan_max_deg: float
    pan_init_deg: float
    tilt_min_deg: float
    tilt_max_deg: float
    tilt_init_deg: float


@dataclass(slots=True)
class MotorGeometryConfig:
    tilt_axis_height_mm: float
    projector_offset_mm: float
    projection_ahead_mm: float


@dataclass(slots=True)
class MotorConfig:
    backend: str
    serial: MotorSerialConfig
    limits: MotorLimitsConfig
    geometry: MotorGeometryConfig
    ping_on_startup: bool
    command_retry_delay: float


@dataclass(slots=True)
class HomographyFilesConfig:
    calibration_dir: Path
    camera_calibration_file: Path
    camera_extrinsics_file: Path


@dataclass(slots=True)
class HomographyFootprintConfig:
    width_mm: float
    height_mm: float
    foot_offset_mm: float


@dataclass(slots=True)
class HomographyConfig:
    backend: str
    target_plane: str
    floor_z_mm: float
    projector_position_mm: Tuple[float, float, float]
    files: HomographyFilesConfig
    footprint: HomographyFootprintConfig
    smoothing_alpha: float


@dataclass(slots=True)
class EngagementConfig:
    distance_threshold_mm: float
    approach_timeout_seconds: float
    approach_delta_min_mm: float
    qa_auto_delay_seconds: float
    skip_to_qa_auto: bool
    lost_target_grace_seconds: float


@dataclass(slots=True)
class AppConfig:
    websocket: WebSocketConfig
    homography_interval: float
    vision_interval: float
    command_resend_interval: float
    detection_hold_seconds: float
    motor: MotorConfig
    homography: HomographyConfig
    engagement: EngagementConfig


def _get_float_env(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return float(default)
    try:
        return float(value)
    except ValueError:
        return float(default)


def load_config() -> AppConfig:
    """Load configuration from environment variables."""
    ws_host = os.getenv("BACKEND_WS_HOST", "0.0.0.0")
    ws_port = int(os.getenv("BACKEND_WS_PORT", "8765"))
    homography_interval = _get_float_env("BACKEND_HOMOGRAPHY_INTERVAL", 0.1)
    vision_interval = _get_float_env("BACKEND_VISION_INTERVAL", 0.5)
    command_resend_interval = _get_float_env("BACKEND_COMMAND_RESEND_INTERVAL", 1.0)
    detection_hold_seconds = _get_float_env("BACKEND_DETECTION_HOLD_SECONDS", 0.0)

    motor_backend = os.getenv("BACKEND_MOTOR_BACKEND", "serial").lower()
    serial_port = os.getenv("BACKEND_MOTOR_SERIAL_PORT", motor_settings.SERIAL_PORT)
    serial_baudrate = int(
        os.getenv("BACKEND_MOTOR_SERIAL_BAUD", str(motor_settings.SERIAL_BAUDRATE))
    )
    serial_timeout = _get_float_env(
        "BACKEND_MOTOR_SERIAL_TIMEOUT", motor_settings.SERIAL_TIMEOUT
    )
    pan_min_deg = _get_float_env("BACKEND_MOTOR_PAN_MIN", motor_settings.PAN_MIN_DEG)
    pan_max_deg = _get_float_env("BACKEND_MOTOR_PAN_MAX", motor_settings.PAN_MAX_DEG)
    pan_init_deg = _get_float_env("BACKEND_MOTOR_PAN_INIT", motor_settings.PAN_INIT_DEG)
    tilt_min_deg = _get_float_env("BACKEND_MOTOR_TILT_MIN", motor_settings.TILT_MIN_DEG)
    tilt_max_deg = _get_float_env("BACKEND_MOTOR_TILT_MAX", motor_settings.TILT_MAX_DEG)
    tilt_init_deg = _get_float_env(
        "BACKEND_MOTOR_TILT_INIT", motor_settings.TILT_INIT_DEG
    )
    tilt_axis_height_mm = _get_float_env(
        "BACKEND_MOTOR_TILT_AXIS_HEIGHT", motor_settings.TILT_AXIS_HEIGHT_MM
    )
    projector_offset_mm = _get_float_env(
        "BACKEND_MOTOR_PROJECTOR_OFFSET", motor_settings.PROJECTOR_OFFSET_MM
    )
    projection_ahead_mm = _get_float_env(
        "BACKEND_MOTOR_PROJECTION_AHEAD", motor_settings.PROJECTION_AHEAD_MM
    )
    ping_on_startup = os.getenv(
        "BACKEND_MOTOR_PING_ON_STARTUP", "true"
    ).lower() not in {"false", "0", "no"}
    command_retry_delay = _get_float_env(
        "BACKEND_MOTOR_COMMAND_RETRY_DELAY", motor_settings.COMMAND_RETRY_DELAY_S
    )

    calib_dir = Path(
        os.getenv(
            "BACKEND_CALIB_DIR", str(homography_settings.CALIBRATION_DIR)
        )
    )
    camera_calib_file = Path(
        os.getenv(
            "BACKEND_CAMERA_CALIB_FILE",
            str(calib_dir / homography_settings.CAMERA_CALIBRATION_FILE.name),
        )
    )
    camera_extrinsics_file = Path(
        os.getenv(
            "BACKEND_CAMERA_EXTRINSICS_FILE",
            str(calib_dir / homography_settings.CAMERA_EXTRINSICS_FILE.name),
        )
    )
    homography_backend = os.getenv("BACKEND_HOMOGRAPHY_BACKEND", "calculator").lower()
    target_plane = os.getenv("BACKEND_TARGET_PLANE", homography_settings.TARGET_PLANE)
    floor_z_mm = _get_float_env("BACKEND_FLOOR_Z_MM", homography_settings.FLOOR_Z_MM)

    proj_pos = homography_settings.PROJECTOR_POSITION_MM
    proj_pos_x = _get_float_env("BACKEND_PROJECTOR_POS_X", proj_pos[0])
    proj_pos_y = _get_float_env("BACKEND_PROJECTOR_POS_Y", proj_pos[1])
    proj_pos_z = _get_float_env("BACKEND_PROJECTOR_POS_Z", proj_pos[2])

    footprint_width = _get_float_env(
        "BACKEND_FOOTPRINT_WIDTH_MM", homography_settings.FOOTPRINT_WIDTH_MM
    )
    footprint_height = _get_float_env(
        "BACKEND_FOOTPRINT_HEIGHT_MM", homography_settings.FOOTPRINT_HEIGHT_MM
    )
    foot_offset_mm = _get_float_env(
        "BACKEND_FOOT_FORWARD_OFFSET_MM", homography_settings.FOOT_OFFSET_MM
    )
    smoothing_alpha = _get_float_env(
        "BACKEND_HOMOGRAPHY_SMOOTHING_ALPHA", homography_settings.SMOOTHING_ALPHA
    )

    engagement_distance_mm = _get_float_env(
        "BACKEND_ENGAGEMENT_DISTANCE_MM", 800.0
    )
    engagement_timeout_s = _get_float_env(
        "BACKEND_ENGAGEMENT_APPROACH_TIMEOUT_S", 10.0
    )
    engagement_delta_mm = _get_float_env(
        "BACKEND_ENGAGEMENT_APPROACH_DELTA_MM", 100.0
    )
    engagement_qa_delay_s = _get_float_env(
        "BACKEND_ENGAGEMENT_QA_AUTO_DELAY_S", 5.0
    )
    engagement_lost_grace_s = _get_float_env(
        "BACKEND_ENGAGEMENT_LOST_TARGET_GRACE_S", 2.0
    )
    skip_to_qa_auto = os.getenv(
        "BACKEND_SKIP_TO_QA_AUTO", "false"
    ).lower() not in {"false", "0", "no"}

    motor_config = MotorConfig(
        backend=motor_backend,
        serial=MotorSerialConfig(
            port=serial_port,
            baudrate=serial_baudrate,
            timeout=serial_timeout,
        ),
        limits=MotorLimitsConfig(
            pan_min_deg=pan_min_deg,
            pan_max_deg=pan_max_deg,
            pan_init_deg=pan_init_deg,
            tilt_min_deg=tilt_min_deg,
            tilt_max_deg=tilt_max_deg,
            tilt_init_deg=tilt_init_deg,
        ),
        geometry=MotorGeometryConfig(
            tilt_axis_height_mm=tilt_axis_height_mm,
            projector_offset_mm=projector_offset_mm,
            projection_ahead_mm=projection_ahead_mm,
        ),
        ping_on_startup=ping_on_startup,
        command_retry_delay=command_retry_delay,
    )

    homography_config = HomographyConfig(
        backend=homography_backend,
        target_plane=target_plane,
        floor_z_mm=floor_z_mm,
        projector_position_mm=(proj_pos_x, proj_pos_y, proj_pos_z),
        files=HomographyFilesConfig(
            calibration_dir=calib_dir,
            camera_calibration_file=camera_calib_file,
            camera_extrinsics_file=camera_extrinsics_file,
        ),
        footprint=HomographyFootprintConfig(
            width_mm=footprint_width,
            height_mm=footprint_height,
            foot_offset_mm=foot_offset_mm,
        ),
        smoothing_alpha=smoothing_alpha,
    )

    engagement_config = EngagementConfig(
        distance_threshold_mm=engagement_distance_mm,
        approach_timeout_seconds=engagement_timeout_s,
        approach_delta_min_mm=engagement_delta_mm,
        qa_auto_delay_seconds=engagement_qa_delay_s,
        skip_to_qa_auto=skip_to_qa_auto,
        lost_target_grace_seconds=engagement_lost_grace_s,
    )

    return AppConfig(
        websocket=WebSocketConfig(host=ws_host, port=ws_port),
        homography_interval=homography_interval,
        vision_interval=vision_interval,
        command_resend_interval=command_resend_interval,
        detection_hold_seconds=detection_hold_seconds,
        motor=motor_config,
        homography=homography_config,
        engagement=engagement_config,
    )
