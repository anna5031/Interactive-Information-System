import logging

import sounddevice as sd  # type: ignore

from .types import DeviceCheckResult

logger = logging.getLogger(__name__)


def check_speaker() -> DeviceCheckResult:
    name = "스피커"

    try:
        devices = sd.query_devices()
    except Exception as exc:  # pragma: no cover - hardware dependent
        logger.warning("스피커 장치 조회 실패: %s", exc)
        devices = []

    available = [
        {"index": idx, "name": dev["name"]}
        for idx, dev in enumerate(devices)
        if dev.get("max_output_channels", 0) > 0
    ]
    if available:
        detail = ", ".join(f"{item['index']}: {item['name']}" for item in available)
        return DeviceCheckResult(
            name=name,
            ok=True,
            detail=detail,
            meta={"devices": available},
        )

    return DeviceCheckResult(name=name, ok=False, detail="출력 장치를 찾을 수 없습니다.")
