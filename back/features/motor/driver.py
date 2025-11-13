from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Protocol

logger = logging.getLogger(__name__)


class MotorDriver(Protocol):
    def set_angles(self, tilt_deg: float, pan_deg: float) -> None:
        ...

    def ping(self) -> bool:
        ...

    def close(self) -> None:
        ...


@dataclass(slots=True)
class DummyMotorDriver(MotorDriver):
    """Driver used in development environments without hardware."""

    history: list[tuple[float, float]]

    def __init__(self) -> None:
        self.history = []

    def set_angles(self, tilt_deg: float, pan_deg: float) -> None:
        logger.info("Dummy motor: tilt=%.2f pan=%.2f", tilt_deg, pan_deg)
        self.history.append((tilt_deg, pan_deg))

    def ping(self) -> bool:
        return True

    def close(self) -> None:
        logger.info("Dummy motor closed.")


class SerialMotorDriver(MotorDriver):
    """Thin wrapper around a SimpleSerial-like interface."""

    def __init__(self, port: str, baudrate: int, timeout: float) -> None:
        try:
            import serial  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("pyserial is required for SerialMotorDriver.") from exc

        self._serial_module = serial
        self._port = port
        self._baudrate = baudrate
        self._timeout = timeout
        self._connection = self._open()

    def _open(self):
        connection = self._serial_module.Serial(
            self._port, self._baudrate, timeout=self._timeout
        )
        time.sleep(2.0)
        return connection

    def set_angles(self, tilt_deg: float, pan_deg: float) -> None:
        packet = f"T:{int(round(tilt_deg))},P:{int(round(pan_deg))}\n"
        self._connection.write(packet.encode("utf-8"))

    def ping(self) -> bool:
        try:
            self._connection.write(b"PING\n")
            response = self._connection.readline().decode("utf-8").strip()
            return bool(response)
        except Exception:
            return False

    def close(self) -> None:
        if self._connection and self._connection.is_open:
            self._connection.close()
