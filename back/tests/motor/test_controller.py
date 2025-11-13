from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from features.motor.config import (
    MotorAxisConfig,
    MotorGeometryConfig,
    MotorSettings,
    SerialConfig,
    QAProjectionConfig,
    ProjectorConfig,
)
from features.motor.controller import MotorController
from features.motor.driver import DummyMotorDriver
from features.motor.setpoint import SetpointCalculator


def _make_settings() -> MotorSettings:
    return MotorSettings(
        beam_geometry=MotorGeometryConfig(z_offset_mm=150.0, tilt_axis_height_mm=1200.0),
        serial=SerialConfig(port="COM1", baudrate=115200, timeout=1.0),
        motor_pan=MotorAxisConfig(min_deg=0.0, max_deg=180.0, init_deg=60.0),
        motor_tilt=MotorAxisConfig(min_deg=40.0, max_deg=170.0, init_deg=110.0),
        qa_projection=QAProjectionConfig(
            ceiling_normal=(0.0, 0.0, 1.0),
            displacement_mm=400.0,
            screen_width_mm=354.0,
            screen_height_ratio=0.5625,
            roll_deg=0.0,
        ),
        projector=ProjectorConfig(
            width_px=3840,
            height_px=2160,
            horizontal_fov_deg=45.0,
            beam_offset_mm=(50.0, 0.0, 80.0),
            pan_tilt_origin_height_mm=1600.0,
            beam_wall_displacement_m=0.4,
            input_image_width_px=1920,
            input_image_height_px=1080,
        ),
    )


def test_controller_uses_driver_history() -> None:
    driver = DummyMotorDriver()
    settings = _make_settings()
    calculator = SetpointCalculator(settings.beam_geometry, settings.motor_pan, settings.motor_tilt)
    controller = MotorController(settings=settings, driver=driver, calculator=calculator)

    controller.point_to((1200.0, 200.0, 0.0))
    assert driver.history, "driver should have received at least one command"


def _demo() -> None:
    driver = DummyMotorDriver()
    settings = _make_settings()
    controller = MotorController(
        settings=settings,
        driver=driver,
        calculator=SetpointCalculator(
            settings.beam_geometry, settings.motor_pan, settings.motor_tilt
        ),
    )
    target = (1000.0, -300.0, 0.0)
    command = controller.point_to(target)
    print("[MotorController Demo]")
    print(" target:", target)
    print(f" command -> tilt={command.tilt_deg:.2f}, pan={command.pan_deg:.2f}")
    print(" dummy driver history:", driver.history)


if __name__ == "__main__":
    _demo()
