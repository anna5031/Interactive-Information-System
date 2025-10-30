from __future__ import annotations

"""Homography matrix calculator based on motor state and calibration."""

import logging
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

import cv2 as cv
import numpy as np

from app.events import HomographyEvent, MotorStateEvent
from features.homography.calibration import CalibrationBundle

logger = logging.getLogger(__name__)

if TYPE_CHECKING:  # pragma: no cover
    from app.config import HomographyConfig


@dataclass
class HomographyCalculator:
    calibration: CalibrationBundle
    config: "HomographyConfig"

    def __post_init__(self) -> None:
        self._rvec, _ = cv.Rodrigues(self.calibration.extrinsics.rotation)
        self._tvec = self.calibration.extrinsics.translation.astype(float)
        self._K = self.calibration.intrinsics.matrix.astype(float)
        self._dist = self.calibration.intrinsics.distortion.astype(float)
        self._last_matrix: Optional[np.ndarray] = None

    def build(self, motor_state: MotorStateEvent) -> HomographyEvent:
        matrix = self._compute_matrix(motor_state)
        if matrix is None:
            if self._last_matrix is not None:
                matrix = self._last_matrix
            else:
                matrix = np.identity(3, dtype=float)
        else:
            alpha = self.config.smoothing_alpha
            if self._last_matrix is not None and 0.0 < alpha < 1.0:
                matrix = alpha * matrix + (1.0 - alpha) * self._last_matrix
            self._last_matrix = matrix

        return HomographyEvent(
            matrix=matrix.tolist(),
            timestamp=motor_state.timestamp,
        )

    def _compute_matrix(
        self,
        motor_state: MotorStateEvent,
    ) -> Optional[np.ndarray]:
        if motor_state.world_target is None or motor_state.foot_world is None:
            logger.debug("Motor state lacks world coordinates for homography.")
            return None

        foot_world = np.array(motor_state.foot_world, dtype=float)
        target_world = np.array(motor_state.world_target, dtype=float)

        forward = target_world - foot_world
        forward[2] = 0.0
        norm = np.linalg.norm(forward)
        if norm < 1e-6:
            forward = np.array([0.0, 1.0, 0.0])
        else:
            forward = forward / norm

        right = np.array([-forward[1], forward[0], 0.0])
        right_norm = np.linalg.norm(right)
        if right_norm < 1e-6:
            right = np.array([1.0, 0.0, 0.0])
        else:
            right = right / right_norm

        base = foot_world + forward * self.config.footprint.foot_offset_mm
        half_width = self.config.footprint.width_mm * 0.5
        depth = self.config.footprint.height_mm

        corners_world = np.array(
            [
                base - right * half_width,
                base + right * half_width,
                base + right * half_width + forward * depth,
                base - right * half_width + forward * depth,
            ],
            dtype=np.float32,
        )

        image_points, _ = cv.projectPoints(
            corners_world,
            self._rvec,
            self._tvec,
            self._K,
            self._dist,
        )
        image_points = image_points.reshape(-1, 2).astype(np.float32)

        src = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        )

        H = cv.getPerspectiveTransform(src, image_points)
        if H is None:
            return None
        if abs(H[2, 2]) > 1e-9:
            H = H / H[2, 2]
        return H
