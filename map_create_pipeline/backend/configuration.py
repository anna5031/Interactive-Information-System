from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict


BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent

DEFAULT_SETTINGS: Dict[str, Any] = {
    "inference": {
        # RF-DETR 가중치가 저장된 기본 경로
        "model_root": "models",
        # 모든 모델이 사용할 기본 디바이스(auto/cuda/mps/cpu)
        "default_device": "auto",
        # 추론 결과를 반환하기 전에 벽-박스 필터를 적용할지 여부
        "apply_wall_filters_during_inference": False,
        # 방/계단/엘리베이터 박스에 완전히 포함된 벽 선분 제거 여부
        "filter_walls_inside_obstacles": True,
        # 방 박스와 겹치는 부분만 잘라내는 옵션
        "clip_walls_overlapping_rooms": True,
        # RF-DETR 모델별 파일 이름 및 임계값
        "models": {
            "room": {"filename": "rf_detr_room.pth", "threshold": 0.25},
            "door_stairs_elevator": {"filename": "rf_detr_stairs_elevator.pth", "threshold": 0.14},
        },
    },
    "processing": {
        # 2단계 그래프 생성 시 중간 마스크/스켈레톤 이미지를 저장하던 옵션 (임시 비활성화)
        # "save_step_two_debug_images": True,
        # 디버그 이미지를 저장할 하위 디렉터리 이름 (임시 비활성화)
        # "step_two_debug_subdir": "graph_debug",
        # 1단계 결과를 history 디렉터리에 백업할지 여부
        "save_step_one_history": False,
        # 방/벽을 임의로 확장하거나 틈을 메우는 선행 처리 활성화 여부
        "enable_wall_expansion": False,
        # 벽/박스 자동 보정 임계값
        "auto_correction": {
            "max_box_gap": 0.04,
            "min_overlap_ratio": 0.3,
            "box_gap_iterations": 2,
            "wall_box_snap_distance": 0.02,
            "wall_wall_snap_distance": 0.015,
        },
    },
    "navigation": {
        # 복도 내 수평 이동 속도 (m/s)
        "walking_speed_mps": 1.3,
        # 계단으로 한 층을 이동하는 시간 (초)
        "stairs_seconds_per_floor": 7.0,
        # 엘리베이터로 한 층을 이동하는 시간 (초)
        "elevator_seconds_per_floor": 5.0,
    },
}


def load_settings() -> Dict[str, Any]:
    """기본 설정을 복사해 호출자에게 반환한다."""
    return deepcopy(DEFAULT_SETTINGS)


def get_inference_settings() -> Dict[str, Any]:
    settings = load_settings()
    inference = settings.get("inference")
    if not isinstance(inference, dict):
        raise ValueError("inference 설정이 잘못되었습니다.")
    return inference


def get_processing_settings() -> Dict[str, Any]:
    settings = load_settings()
    processing = settings.get("processing")
    if not isinstance(processing, dict):
        raise ValueError("processing 설정이 잘못되었습니다.")
    return processing


def get_auto_correction_settings() -> Dict[str, Any]:
    processing = get_processing_settings()
    auto_config = processing.get("auto_correction") or {}
    if not isinstance(auto_config, dict):
        raise ValueError("auto_correction 설정이 잘못되었습니다.")
    return auto_config


def get_navigation_settings() -> Dict[str, Any]:
    settings = load_settings()
    navigation = settings.get("navigation")
    if not isinstance(navigation, dict):
        raise ValueError("navigation 설정이 잘못되었습니다.")
    return navigation


__all__ = [
    "PROJECT_ROOT",
    "get_inference_settings",
    "get_processing_settings",
    "get_auto_correction_settings",
    "get_navigation_settings",
]
