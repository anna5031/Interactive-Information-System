from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(slots=True)
class WebSocketConfig:
    host: str = "0.0.0.0"
    port: int = 8765


@dataclass(slots=True)
class AppConfig:
    websocket: WebSocketConfig
    homography_interval: float
    vision_interval: float
    command_resend_interval: float
    detection_hold_seconds: float


def load_config() -> AppConfig:
    """환경 변수 기반 설정 로딩."""
    ws_host = os.getenv("BACKEND_WS_HOST", "0.0.0.0")
    ws_port = int(os.getenv("BACKEND_WS_PORT", "8765"))
    homography_interval = float(os.getenv("BACKEND_HOMOGRAPHY_INTERVAL", "0.1"))
    vision_interval = float(os.getenv("BACKEND_VISION_INTERVAL", "0.5"))
    command_resend_interval = float(
        os.getenv("BACKEND_COMMAND_RESEND_INTERVAL", "1.0")
    )
    detection_hold_seconds = float(os.getenv("BACKEND_DETECTION_HOLD_SECONDS", "5.0"))
    return AppConfig(
        websocket=WebSocketConfig(host=ws_host, port=ws_port),
        homography_interval=homography_interval,
        vision_interval=vision_interval,
        command_resend_interval=command_resend_interval,
        detection_hold_seconds=detection_hold_seconds,
    )
