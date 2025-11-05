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
        self._projector = np.array(self.config.projector_position_mm, dtype=float)
        self._plane_normal = self._resolve_plane_normal(self.config.target_plane)
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
        target_plane = (self.config.target_plane or "").lower()
        if target_plane == "ceiling":
            matrix = self._compute_projector_plane_matrix(motor_state)
            if matrix is not None:
                return matrix
            logger.debug("Ceiling homography fallback to camera projection due to missing data.")

        return self._compute_floor_matrix(motor_state)

    def _compute_floor_matrix(
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

    def _compute_projector_plane_matrix(
        self,
        motor_state: MotorStateEvent,
    ) -> Optional[np.ndarray]:
        if motor_state.world_target is None:
            logger.debug("Ceiling homography requires world_target coordinates.")
            return None

        target_world = np.array(motor_state.world_target, dtype=float)
        beam_vector = target_world - self._projector
        beam_norm = np.linalg.norm(beam_vector)
        if beam_norm < 1e-6:
            logger.debug("Beam vector too small for ceiling homography.")
            return None

        beam_dir = beam_vector / beam_norm
        plane_normal = self._plane_normal

        # Project the beam direction onto the target plane to derive orientation axes.
        forward = beam_vector - np.dot(beam_vector, plane_normal) * plane_normal
        forward_norm = np.linalg.norm(forward)
        if forward_norm < 1e-6:
            forward = self._orthogonal_unit(plane_normal)
        else:
            forward = forward / forward_norm

        right = np.cross(plane_normal, forward)
        right_norm = np.linalg.norm(right)
        if right_norm < 1e-6:
            right = self._orthogonal_unit(plane_normal)
        else:
            right = right / right_norm

        width = self.config.footprint.width_mm
        height = self.config.footprint.height_mm

        src = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        )

        corners_world: list[np.ndarray] = []
        for u, v in src:
            offset_right = (u - 0.5) * width
            offset_forward = (v - 0.5) * height
            corner = target_world + right * offset_right + forward * offset_forward
            corners_world.append(corner)

        try:
            projector_right, projector_up = self._projector_axes(beam_dir)
        except ValueError:
            logger.debug("Unable to resolve projector axes for ceiling homography.")
            return None

        dst: list[list[float]] = []
        for corner in corners_world:
            rel = corner - self._projector
            depth = float(np.dot(rel, beam_dir))
            if abs(depth) < 1e-6:
                logger.debug("Corner projects behind projector; skipping homography.")
                return None
            x = float(np.dot(rel, projector_right) / depth)
            y = float(np.dot(rel, projector_up) / depth)
            dst.append([x, y])

        dst_points = np.array(dst, dtype=np.float32)

        H = cv.getPerspectiveTransform(src, dst_points)
        if H is None:
            return None
        if abs(H[2, 2]) > 1e-9:
            H = H / H[2, 2]
        return H

    @staticmethod
    def _resolve_plane_normal(target_plane: str | None) -> np.ndarray:
        plane = (target_plane or "").strip().lower()
        if plane == "ceiling":
            normal = np.array([0.0, 0.0, -1.0], dtype=float)
        elif plane == "floor":
            normal = np.array([0.0, 0.0, 1.0], dtype=float)
        else:
            # Default to upward normal; callers can override through configuration later.
            normal = np.array([0.0, 0.0, 1.0], dtype=float)

        norm = np.linalg.norm(normal)
        if norm < 1e-6:
            return np.array([0.0, 0.0, 1.0], dtype=float)
        return normal / norm

    @staticmethod
    def _orthogonal_unit(reference: np.ndarray) -> np.ndarray:
        candidates = (
            np.array([1.0, 0.0, 0.0], dtype=float),
            np.array([0.0, 1.0, 0.0], dtype=float),
            np.array([0.0, 0.0, 1.0], dtype=float),
        )
        for candidate in candidates:
            vector = np.cross(reference, candidate)
            norm = np.linalg.norm(vector)
            if norm >= 1e-6:
                return vector / norm
        return np.array([1.0, 0.0, 0.0], dtype=float)

    def _projector_axes(self, beam_dir: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        up_reference = np.array([0.0, 0.0, 1.0], dtype=float)
        right = np.cross(up_reference, beam_dir)
        if np.linalg.norm(right) < 1e-6:
            right = np.cross(self._orthogonal_unit(beam_dir), beam_dir)
        right_norm = np.linalg.norm(right)
        if right_norm < 1e-6:
            raise ValueError("Cannot resolve projector axes for beam direction.")
        right = right / right_norm
        up = np.cross(beam_dir, right)
        up_norm = np.linalg.norm(up)
        if up_norm < 1e-6:
            raise ValueError("Cannot resolve projector up axis.")
        up = up / up_norm
        return right, up
