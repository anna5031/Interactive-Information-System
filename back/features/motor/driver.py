from __future__ import annotations

"""Serial driver wrapper for the projector motor."""

import time
from dataclasses import dataclass
from typing import Any, Protocol

try:
    import serial  # type: ignore
except ImportError as exc:  # pragma: no cover - optional dependency
    serial = None  # type: ignore
    _SERIAL_IMPORT_ERROR = exc
else:
    _SERIAL_IMPORT_ERROR = None


class MotorDriver(Protocol):
    def ping(self) -> bool | str:
        ...

    def send(self, tilt_deg: int, pan_deg: int) -> str:
        ...

    def close(self) -> None:
        ...


@dataclass(slots=True)
class SerialMotorDriver:
    port: str
    baudrate: int
    timeout: float

    _serial: Any = None

    def open(self) -> None:
        if self._serial is not None and getattr(self._serial, "is_open", False):
            return
        if serial is None:
            raise RuntimeError("pyserial is required for SerialMotorDriver") from _SERIAL_IMPORT_ERROR
        self._serial = serial.Serial(self.port, self.baudrate, timeout=self.timeout)
        time.sleep(2.0)

    def ping(self) -> bool | str:
        if self._serial is None:
            return False
        try:
            self._serial.write(b"PING\n")
            line = self._serial.readline().decode("utf-8", errors="ignore").strip()
            return line
        except Exception:
            return False

    def send(self, tilt_deg: int, pan_deg: int) -> str:
        if self._serial is None:
            raise RuntimeError("Serial port is not open.")
        packet = f"T:{int(tilt_deg)},P:{int(pan_deg)}\n"
        self._serial.write(packet.encode("utf-8"))
        return self._serial.readline().decode("utf-8", errors="ignore").strip()

    def close(self) -> None:
        if self._serial is not None and getattr(self._serial, "is_open", False):
            self._serial.close()
        self._serial = None
