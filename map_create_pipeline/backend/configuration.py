from __future__ import annotations

import json
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict


BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent
CONFIG_DIR = BACKEND_DIR / "config"
CONFIG_PATH = CONFIG_DIR / "settings.json"

DEFAULT_SETTINGS: Dict[str, Any] = {
    "inference": {
        "model_root": "data/models",
        "default_device": "auto",
        "models": {
            "room": {"filename": "rf_detr_room.pth", "threshold": 0.35},
            "door_stairs_elevator": {"filename": "rf_detr_stairs_elevator.pth", "threshold": 0.35},
            "wall": {"filename": "rf_detr_wall.pth", "threshold": 0.35},
        },
    }
}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


@lru_cache(maxsize=1)
def load_settings() -> Dict[str, Any]:
    settings = deepcopy(DEFAULT_SETTINGS)
    if CONFIG_PATH.exists():
        try:
            with CONFIG_PATH.open("r", encoding="utf-8") as stream:
                user_settings = json.load(stream)
            if isinstance(user_settings, dict):
                settings = _deep_merge(settings, user_settings)
        except json.JSONDecodeError as exc:
            raise ValueError(f"설정 파일을 파싱할 수 없습니다: {CONFIG_PATH}") from exc
    return settings


def get_inference_settings() -> Dict[str, Any]:
    settings = load_settings()
    inference = settings.get("inference")
    if not isinstance(inference, dict):
        raise ValueError("inference 설정이 잘못되었습니다.")
    return inference


__all__ = ["PROJECT_ROOT", "get_inference_settings"]
