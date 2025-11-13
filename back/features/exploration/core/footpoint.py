from __future__ import annotations

"""Foot point estimation and smoothing utilities."""

import logging
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from ..geometry import PixelToWorldMapper
from .tracking import Track

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class FootPointEstimatorConfig:
    alpha_stationary: float = 0.2
    alpha_walking: float = 0.45
    max_jump: float = 120.0
    walking_horizontal_threshold: float = 80.0
    walking_vertical_threshold: float = 45.0


class FootPointEstimator:
    def __init__(
        self,
        config: Optional[FootPointEstimatorConfig] = None,
        *,
        pixel_mapper: Optional[PixelToWorldMapper] = None,
        floor_z: float = 0.0,
    ) -> None:
        self.config = config or FootPointEstimatorConfig()
        self._pixel_mapper = pixel_mapper
        self._floor_z = floor_z

    def update_track(
        self,
        track: Track,
        detection: Dict[str, np.ndarray],
    ) -> np.ndarray:
        anchors = detection.get("anchors") or {}
        bbox = detection.get("bbox")
        fallback = detection.get("centroid")

        ankle_left = anchors.get("ankle_left")
        ankle_right = anchors.get("ankle_right")
        hip_center = anchors.get("hip_center")
        knee_center = anchors.get("knee_center")
        axis_point = anchors.get("axis_point")

        walking = False
        if ankle_left is not None and ankle_right is not None:
            horizontal = abs(float(ankle_left[0]) - float(ankle_right[0]))
            vertical = abs(float(ankle_left[1]) - float(ankle_right[1]))
            if (
                horizontal > self.config.walking_horizontal_threshold
                or vertical > self.config.walking_vertical_threshold
            ):
                walking = True

        track.walking = walking

        foot_primary: Optional[np.ndarray] = None
        if ankle_left is not None and ankle_right is not None:
            candidate_left = ankle_left.copy()
            candidate_right = ankle_right.copy()
            chosen = (
                candidate_left
                if candidate_left[1] >= candidate_right[1]
                else candidate_right
            )
            if track.foot_point is not None:
                dist_left = np.linalg.norm(candidate_left - track.foot_point)
                dist_right = np.linalg.norm(candidate_right - track.foot_point)
                if dist_left < dist_right * 0.8:
                    chosen = candidate_left
                elif dist_right < dist_left * 0.8:
                    chosen = candidate_right
            foot_primary = chosen
        elif ankle_left is not None:
            foot_primary = ankle_left.copy()
        elif ankle_right is not None:
            foot_primary = ankle_right.copy()
        elif knee_center is not None and hip_center is not None:
            direction = knee_center - hip_center
            length = np.linalg.norm(direction)
            if length > 1e-3:
                foot_primary = knee_center + direction
            else:
                foot_primary = knee_center.copy()
        elif hip_center is not None and bbox is not None:
            x1, _, x2, y2 = bbox
            foot_primary = np.array([(x1 + x2) * 0.5, y2], dtype=float)
        elif bbox is not None:
            x1, _, x2, y2 = bbox
            foot_primary = np.array([(x1 + x2) * 0.5, y2], dtype=float)

        if axis_point is None and hip_center is not None and bbox is not None:
            x1, _, x2, y2 = bbox
            axis_point = np.array([(x1 + x2) * 0.5, y2], dtype=float)

        if foot_primary is not None and axis_point is not None:
            combined = 0.7 * foot_primary + 0.3 * axis_point
        elif foot_primary is not None:
            combined = foot_primary
        elif axis_point is not None:
            combined = axis_point
        elif fallback is not None:
            combined = np.asarray(fallback, dtype=float)
        elif track.foot_point is not None:
            combined = track.foot_point.copy()
        else:
            combined = np.zeros(2, dtype=float)

        if track.foot_point is None:
            smoothed = combined
        else:
            delta = combined - track.foot_point
            distance = float(np.linalg.norm(delta))
            if distance > self.config.max_jump and distance > 1e-6:
                delta *= self.config.max_jump / distance
            alpha = (
                self.config.alpha_walking if walking else self.config.alpha_stationary
            )
            smoothed = (1 - alpha) * track.foot_point + alpha * (
                track.foot_point + delta
            )

        track.foot_point = smoothed
        track.foot_history.append(smoothed.copy())

        return smoothed

    def map_to_world(self, foot_point: np.ndarray) -> Optional[np.ndarray]:
        if self._pixel_mapper is None:
            return None
        try:
            world = self._pixel_mapper.pixel_to_world(
                (float(foot_point[0]), float(foot_point[1])),
                plane_z=self._floor_z,
            )
        except Exception as exc:
            logger.exception("픽셀→월드 좌표 변환 실패: %s", exc)
            return None
        if world is None:
            return None
        return np.asarray(world, dtype=float)
