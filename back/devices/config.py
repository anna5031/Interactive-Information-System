import importlib
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Union


@dataclass
class DevicePreferences:
    microphone_priority_names: List[str] = field(default_factory=list)
    speaker_priority_names: List[str] = field(default_factory=list)
    camera_source: Optional[Union[str, int]] = None


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

    return DevicePreferences(
        microphone_priority_names=microphone_names,
        speaker_priority_names=speaker_names,
        camera_source=camera_source,
    )


def _ensure_str_list(value: Iterable) -> List[str]:
    if value is None:
        return []
    result = [str(item) for item in value if str(item).strip()]
    return result
