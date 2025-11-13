from __future__ import annotations

from typing import Sequence

from .config import MotorSettings
from .driver import MotorDriver
from .setpoint import MotorAnglePair, SetpointCalculator


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

    def point_to(self, target: Sequence[float]) -> MotorAnglePair:
        pair = self.calculator.calculate_pair(target)
        command = pair.command
        self.driver.set_angles(command.tilt_deg, command.pan_deg)
        return pair

    def ping(self) -> bool:
        return self.driver.ping()

    def shutdown(self) -> None:
        self.driver.close()
