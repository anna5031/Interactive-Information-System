from __future__ import annotations

"""Camera pixel → world coordinate mapper."""

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2 as cv
import numpy as np

from .calibration import CalibrationBundle


@dataclass
class PixelToWorldMapper:
    calibration: CalibrationBundle

    def __post_init__(self) -> None:
        intrinsics = self.calibration.intrinsics
        extrinsics = self.calibration.extrinsics

        self._K = intrinsics.matrix.astype(float)
        self._dist = intrinsics.distortion.astype(float)
        self._Rcw = extrinsics.rotation.astype(float)
        self._tcw = extrinsics.translation.astype(float)
        self._Rwc = self._Rcw.T
        self._Cw = -self._Rwc @ self._tcw

    def pixel_to_world(
        self,
        pixel: Tuple[float, float],
        plane_z: float = 0.0,
    ) -> Optional[Tuple[float, float, float]]:
        """Map a pixel coordinate to the world Z=plane_z plane."""
        pts = np.array([[[pixel[0], pixel[1]]]], dtype=np.float32)
        x_n, y_n = cv.undistortPoints(pts, self._K, self._dist, P=None)[0, 0]
        ray_c = np.array([[x_n], [y_n], [1.0]], dtype=np.float64)

        d_w = self._Rwc @ ray_c
        dz = float(d_w[2, 0])
        if abs(dz) < 1e-8:
            return None

        lam = (plane_z - float(self._Cw[2, 0])) / dz
        Xw = self._Cw + lam * d_w
        return (
            float(Xw[0, 0]),
            float(Xw[1, 0]),
            float(plane_z),
        )
