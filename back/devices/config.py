import importlib
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Union


@dataclass
class DevicePreferences:
    microphone_priority_names: List[str] = field(default_factory=list)
    speaker_priority_names: List[str] = field(default_factory=list)
    camera_source: Optional[Union[str, int]] = None
    camera_frame_size: Optional[tuple[int, int]] = None


def load_device_preferences() -> DevicePreferences:
    try:
        module = importlib.import_module("config.device_preferences")
    except ModuleNotFoundError:
        return DevicePreferences()

    microphone_names = _ensure_str_list(
        getattr(module, "MICROPHONE_PRIORITY_NAMES", [])
    )
    speaker_names = _ensure_str_list(getattr(module, "SPEAKER_PRIORITY_NAMES", []))

    camera_source = getattr(module, "CAMERA_SOURCE", None)
    if camera_source is not None and not isinstance(camera_source, (str, int)):
        raise ValueError("CAMERA_SOURCE는 문자열 또는 정수여야 합니다.")

    camera_frame_size = _ensure_frame_size(
        getattr(module, "CAMERA_FRAME_SIZE", None)
    )

    return DevicePreferences(
        microphone_priority_names=microphone_names,
        speaker_priority_names=speaker_names,
        camera_source=camera_source,
        camera_frame_size=camera_frame_size,
    )


def _ensure_str_list(value: Iterable) -> List[str]:
    if value is None:
        return []
    result = [str(item) for item in value if str(item).strip()]
    return result


def _ensure_frame_size(value: object) -> Optional[tuple[int, int]]:
    if value is None:
        return None
    if isinstance(value, tuple) and len(value) == 2:
        width, height = value
    elif isinstance(value, list) and len(value) == 2:
        width, height = value
    else:
        raise ValueError("CAMERA_FRAME_SIZE는 (width, height) 튜플이어야 합니다.")
    try:
        width_int = int(width)
        height_int = int(height)
    except (TypeError, ValueError) as exc:
        raise ValueError("CAMERA_FRAME_SIZE는 숫자 두 개여야 합니다.") from exc
    if width_int <= 0 or height_int <= 0:
        raise ValueError("CAMERA_FRAME_SIZE 폭/높이는 양수여야 합니다.")
    return (width_int, height_int)
