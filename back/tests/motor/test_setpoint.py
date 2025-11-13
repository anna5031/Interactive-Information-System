from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from features.motor.config import MotorAxisConfig, MotorGeometryConfig
from features.motor.setpoint import MotorAngles, SetpointCalculator


def _build_calculator() -> SetpointCalculator:
    geometry = MotorGeometryConfig(z_offset_mm=150.0, tilt_axis_height_mm=1200.0)
    pan_axis = MotorAxisConfig(min_deg=0.0, max_deg=180.0, init_deg=60.0)
    tilt_axis = MotorAxisConfig(min_deg=40.0, max_deg=170.0, init_deg=110.0)
    return SetpointCalculator(geometry, pan_axis, tilt_axis)


@pytest.fixture()
def calculator() -> SetpointCalculator:
    return _build_calculator()


def test_calculate_raw_angles_symmetry(calculator: SetpointCalculator) -> None:
    raw = calculator.calculate_raw_angles((1000.0, 0.0, 0.0))
    symmetrical = calculator.calculate_raw_angles((1000.0, 0.0, 0.0))
    assert pytest.approx(raw.tilt_deg, rel=1e-5) == symmetrical.tilt_deg
    assert pytest.approx(raw.pan_deg, rel=1e-5) == symmetrical.pan_deg


def test_pan_changes_with_y(calculator: SetpointCalculator) -> None:
    left = calculator.calculate_raw_angles((1000.0, 500.0, 0.0))
    right = calculator.calculate_raw_angles((1000.0, -500.0, 0.0))
    assert left.pan_deg == pytest.approx(-right.pan_deg)


def test_command_angles_are_clamped(calculator: SetpointCalculator) -> None:
    # extremely large tilt should still clamp to limits
    raw = MotorAngles(tilt_deg=400.0, pan_deg=-400.0)
    command = calculator.apply_offsets(raw)
    assert command.tilt_deg <= calculator.tilt_axis.max_deg
    assert command.pan_deg >= calculator.pan_axis.min_deg


def _demo() -> None:
    calculator = _build_calculator()
    target = (1200.0, 200.0, 0.0)
    raw = calculator.calculate_raw_angles(target)
    command = calculator.apply_offsets(raw)
    print("[Setpoint Demo]")
    print(" target:", target)
    print(f" raw   -> tilt={raw.tilt_deg:.2f}, pan={raw.pan_deg:.2f}")
    print(
        f" cmd   -> tilt={command.tilt_deg:.2f} (with offset), "
        f"pan={command.pan_deg:.2f} (with offset)"
    )


if __name__ == "__main__":
    _demo()
