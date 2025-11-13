import logging
from typing import List

from serial.tools import list_ports  # type: ignore

from .types import DeviceCheckResult

logger = logging.getLogger(__name__)

ARDUINO_VENDOR_IDS = {0x2341, 0x2A03, 0x239A}


def check_arduino() -> DeviceCheckResult:
    name = "아두이노"
    candidates: List[str] = []

    ports = list(list_ports.comports())
    for port in ports:
        description = port.description or ""
        if "Arduino" in description or "tty.usb" in port.device or port.vid in ARDUINO_VENDOR_IDS:
            candidates.append(f"{port.device} ({description})")

    if candidates:
        return DeviceCheckResult(name=name, ok=True, detail=", ".join(candidates))

    return DeviceCheckResult(name=name, ok=False, detail="연결된 장치를 찾지 못했습니다.")
