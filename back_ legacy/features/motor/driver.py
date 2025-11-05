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


def _default_serial_config() -> tuple[str, int, float]:
    default_port = "/dev/ttyACM0"
    default_baudrate = 115200
    default_timeout = 1.0

    if __package__:
        try:  # pragma: no cover - defensive import for CLI convenience
            from . import settings as motor_settings  # type: ignore
        except Exception:
            pass
        else:
            default_port = getattr(motor_settings, "SERIAL_PORT", default_port)
            default_baudrate = getattr(motor_settings, "SERIAL_BAUDRATE", default_baudrate)
            default_timeout = getattr(motor_settings, "SERIAL_TIMEOUT", default_timeout)

    return default_port, default_baudrate, default_timeout


def _cli() -> int:
    import argparse
    import os
    import sys

    port_default, baud_default, timeout_default = _default_serial_config()

    parser = argparse.ArgumentParser(
        description="Send a PING command to the motor controller and print the response.",
    )
    parser.add_argument(
        "--port",
        default=os.environ.get("MOTOR_SERIAL_PORT", port_default),
        help=f"Serial device path (default: %(default)s)",
    )
    parser.add_argument(
        "--baudrate",
        type=int,
        default=int(os.environ.get("MOTOR_SERIAL_BAUDRATE", baud_default)),
        help=f"Serial baudrate (default: %(default)s)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=float(os.environ.get("MOTOR_SERIAL_TIMEOUT", timeout_default)),
        help=f"Read timeout in seconds (default: %(default)s)",
    )
    args = parser.parse_args()

    driver = SerialMotorDriver(args.port, args.baudrate, args.timeout)
    try:
        driver.open()
    except RuntimeError as exc:
        print(f"Failed to open serial port: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # pragma: no cover - serial errors are hardware dependent
        errno = getattr(exc, "errno", None)
        if errno == 13:
            print(
                f"Permission denied opening {args.port}. "
                "Add your user to the dialout group or run with elevated privileges.",
                file=sys.stderr,
            )
        else:
            print(f"Unexpected error opening {args.port}: {exc}", file=sys.stderr)
        return 1

    try:
        response = driver.ping()
        print(f"Ping response: {response!r}")
        return 0
    finally:
        driver.close()


if __name__ == "__main__":  # pragma: no cover - exercised manually
    raise SystemExit(_cli())
