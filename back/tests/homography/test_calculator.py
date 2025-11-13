from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from features.homography.calculator import HomographyCalculator, HomographyConfig, ScreenConfig


def test_calculate_returns_matrix() -> None:
    screen = ScreenConfig(
        ceiling_normal=(0.0, 0.0, 1.0),
        displacement_m=0.4,
        screen_width_m=0.354,
        screen_height_m=0.354 * 9.0 / 16.0,
        roll_deg=0.0,
    )
    config = HomographyConfig(
        projector_resolution=(3840, 2160),
        horizontal_fov_deg=45.0,
        beam_offset_m=(0.05, 0.0, 0.08),
        origin_height_m=1.6,
        beam_wall_displacement_m=0.4,
        input_resolution=(1920, 1080),
    )
    calculator = HomographyCalculator(screen, config)
    H = calculator.calculate(pan_deg=0.0, tilt_deg=15.0)
    assert H.shape == (3, 3)
    assert np.isfinite(H).all()


def _demo() -> None:
    screen = ScreenConfig(
        ceiling_normal=(0.0, 0.0, 1.0),
        displacement_m=0.4,
        screen_width_m=0.354,
        screen_height_m=0.354 * 9.0 / 16.0,
        roll_deg=0.0,
    )
    config = HomographyConfig(
        projector_resolution=(3840, 2160),
        horizontal_fov_deg=45.0,
        beam_offset_m=(0.05, 0.0, 0.08),
        origin_height_m=1.6,
        beam_wall_displacement_m=0.4,
        input_resolution=(1920, 1080),
    )
    calculator = HomographyCalculator(screen, config)
    H = calculator.calculate(pan_deg=5.0, tilt_deg=15.0)
    np.set_printoptions(precision=4, suppress=True)
    print("[Homography Demo]")
    print(" pan=5°, tilt=15°")
    print(" homography matrix:\n", H)


if __name__ == "__main__":
    _demo()
