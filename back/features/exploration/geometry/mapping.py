from __future__ import annotations

"""Pixel → world 좌표 변환 유틸리티."""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import cv2  # type: ignore
import numpy as np

from ..config import MappingConfig

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class CameraIntrinsics:
    matrix: np.ndarray
    distortion: np.ndarray


@dataclass(slots=True)
class CameraExtrinsics:
    rotation: np.ndarray
    translation: np.ndarray


@dataclass(slots=True)
class CalibrationBundle:
    intrinsics: CameraIntrinsics
    extrinsics: CameraExtrinsics


def load_camera_intrinsics(path: Path) -> CameraIntrinsics:
    data = np.load(path)
    return CameraIntrinsics(matrix=data["K"], distortion=data["dist"])


def load_camera_extrinsics(path: Path) -> CameraExtrinsics:
    data = np.load(path)
    return CameraExtrinsics(rotation=data["R"], translation=data["t"])


def load_calibration_bundle(
    intrinsics_path: Path,
    extrinsics_path: Path,
) -> CalibrationBundle:
    return CalibrationBundle(
        intrinsics=load_camera_intrinsics(intrinsics_path),
        extrinsics=load_camera_extrinsics(extrinsics_path),
    )


@dataclass(slots=True)
class PixelToWorldMapper:
    calibration: CalibrationBundle
    _K: np.ndarray = field(init=False, repr=False)
    _dist: np.ndarray = field(init=False, repr=False)
    _Rcw: np.ndarray = field(init=False, repr=False)
    _tcw: np.ndarray = field(init=False, repr=False)
    _Rwc: np.ndarray = field(init=False, repr=False)
    _Cw: np.ndarray = field(init=False, repr=False)

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
        pts = np.array([[[pixel[0], pixel[1]]]], dtype=np.float32)
        x_n, y_n = cv2.undistortPoints(pts, self._K, self._dist, P=None)[0, 0]
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


def load_pixel_to_world_mapper(config: MappingConfig) -> Optional[PixelToWorldMapper]:
    if not config.enabled:
        logger.info("픽셀→월드 매퍼 비활성화됨")
        return None

    intrinsics_path = Path(config.camera_calibration_file)
    extrinsics_path = Path(config.camera_extrinsics_file)

    if not intrinsics_path.exists():
        logger.warning("카메라 캘리브레이션 파일을 찾을 수 없습니다: %s", intrinsics_path)
        return None
    if not extrinsics_path.exists():
        logger.warning("카메라 외부 파라미터 파일을 찾을 수 없습니다: %s", extrinsics_path)
        return None

    try:
        bundle = load_calibration_bundle(intrinsics_path, extrinsics_path)
    except Exception as exc:
        logger.exception("캘리브레이션 파일 로딩 실패: %s", exc)
        return None

    return PixelToWorldMapper(bundle)
