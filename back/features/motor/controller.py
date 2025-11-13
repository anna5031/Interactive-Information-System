from __future__ import annotations

from typing import Sequence

from .config import MotorSettings
from .driver import MotorDriver
from .setpoint import MotorAngles, SetpointCalculator


class MotorController:
    def __init__(
        self,
        settings: MotorSettings,
        driver: MotorDriver,
        calculator: SetpointCalculator | None = None,
    ) -> None:
        self.settings = settings
        self.driver = driver
        self.calculator = calculator or SetpointCalculator(
            settings.beam_geometry, settings.motor_pan, settings.motor_tilt
        )

    def point_to(self, target: Sequence[float]) -> MotorAngles:
        raw = self.calculator.calculate_raw_angles(target)
        command = self.calculator.apply_offsets(raw)
        self.driver.set_angles(command.tilt_deg, command.pan_deg)
        return command

    def ping(self) -> bool:
        return self.driver.ping()

    def shutdown(self) -> None:
        self.driver.close()
