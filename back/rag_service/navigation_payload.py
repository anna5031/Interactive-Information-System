from __future__ import annotations

from typing import Any, Dict, List


def build_navigation_payload(destination_room: str) -> Dict[str, Any]:
    """임시 네비게이션 데이터. 실제 구현에서는 경로 탐색 결과를 반환."""
    destination = destination_room or "정보 없음"
    steps: List[Dict[str, Any]] = [
        {"step": 1, "instruction": "현재 위치에서 정면으로 20m 이동합니다.", "estimated_distance_m": 20},
        {"step": 2, "instruction": "두 번째 교차로에서 왼쪽으로 이동합니다.", "estimated_distance_m": 10},
        {"step": 3, "instruction": f"{destination} 앞까지 직진합니다.", "estimated_distance_m": 15},
    ]

    return {
        "destination": destination,
        "eta_minutes": 3,
        "steps": steps,
        "meta": {
            "confidence": 0.92,
            "generated_at": "dummy-placeholder",
        },
    }


__all__ = ["build_navigation_payload"]
