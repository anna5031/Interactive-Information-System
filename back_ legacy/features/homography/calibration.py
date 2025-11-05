from __future__ import annotations

"""Helpers to load camera calibration assets from disk."""

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np


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
