from __future__ import annotations

"""Video capture utilities for the exploration pipeline."""

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from .config import CameraConfig

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class CaptureMetadata:
    width: int
    height: int
    fourcc: str
    fps: float


class CameraCapture:
    """Thin wrapper around OpenCV VideoCapture."""

    def __init__(self, config: CameraConfig):
        self.config = config
        self._capture: Optional[cv2.VideoCapture] = None
        self._metadata: Optional[CaptureMetadata] = None

    def open(self) -> None:
        logger.info("Opening camera source: %s", self.config.source)
        capture = cv2.VideoCapture(self.config.source)
        if not capture.isOpened():
            raise RuntimeError(f"Unable to open camera source: {self.config.source}")

        if self.config.fourcc and len(self.config.fourcc) == 4:
            capture.set(
                cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.config.fourcc)
            )

        if self.config.frame_size:
            width, height = self.config.frame_size
            if width > 0:
                capture.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
            if height > 0:
                capture.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))

        if self.config.target_fps:
            capture.set(cv2.CAP_PROP_FPS, float(self.config.target_fps))

        metadata = CaptureMetadata(
            width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            or (self.config.frame_size[0] if self.config.frame_size else 0),
            height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            or (self.config.frame_size[1] if self.config.frame_size else 0),
            fourcc=_decode_fourcc(capture.get(cv2.CAP_PROP_FOURCC)),
            fps=float(capture.get(cv2.CAP_PROP_FPS)) or 0.0,
        )

        logger.info(
            "Camera configured: %sx%s fourcc=%s fps=%.2f",
            metadata.width,
            metadata.height,
            metadata.fourcc or "unknown",
            metadata.fps,
        )

        self._capture = capture
        self._metadata = metadata

    def read(self) -> Optional[np.ndarray]:
        if self._capture is None:
            raise RuntimeError("Camera is not opened.")
        success, frame = self._capture.read()
        if not success or frame is None:
            return None
        return frame

    def close(self) -> None:
        if self._capture is not None:
            logger.info("Releasing camera.")
            self._capture.release()
            self._capture = None

    @property
    def metadata(self) -> CaptureMetadata:
        if self._metadata is None:
            raise RuntimeError("Camera metadata requested before opening.")
        return self._metadata

    def __enter__(self) -> "CameraCapture":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def resize_frame(frame: np.ndarray, target_size: Optional[Tuple[int, int]]) -> np.ndarray:
    if not target_size:
        return frame
    width, height = target_size
    if width <= 0 or height <= 0:
        return frame
    interpolation = cv2.INTER_AREA if width < frame.shape[1] else cv2.INTER_LINEAR
    return cv2.resize(frame, (width, height), interpolation=interpolation)


def center_crop(frame: np.ndarray, ratio: float) -> Tuple[np.ndarray, Tuple[int, int]]:
    ratio = max(0.0, min(1.0, ratio))
    if ratio <= 0 or ratio >= 1:
        return frame, (0, 0)

    h, w = frame.shape[:2]
    crop_w = int(w * ratio)
    crop_h = int(h * ratio)
    x1 = (w - crop_w) // 2
    y1 = (h - crop_h) // 2
    x2 = x1 + crop_w
    y2 = y1 + crop_h
    cropped = frame[y1:y2, x1:x2]
    return cropped, (x1, y1)


def _decode_fourcc(value: float) -> str:
    try:
        value_int = int(value)
    except (TypeError, ValueError):
        return ""
    chars = [chr((value_int >> (8 * i)) & 0xFF) for i in range(4)]
    decoded = "".join(chars).strip()
    return decoded
