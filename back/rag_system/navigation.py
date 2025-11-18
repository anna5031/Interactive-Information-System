from __future__ import annotations

import random
from typing import Dict, List


def find_route(origin: str | None, destination: str | None) -> Dict[str, object]:
    if not destination:
        return {
            "success": False,
            "message": "목적지를 파악할 수 없어 경로 안내를 건너뜁니다.",
            "steps": [],
        }
    sample_time = random.randint(2, 7)
    path: List[str] = [origin or "현재 위치", destination]
    return {
        "success": True,
        "message": f"{path[0]}에서 {destination}까지 약 {sample_time}분이 소요됩니다.",
        "origin": path[0],
        "destination": destination,
        "steps": [
            f"{path[0]}에서 출발합니다.",
            "안내 표지판을 따라 이동하세요.",
            f"{destination}에 도착하면 안내 문구를 확인하세요.",
        ],
    }
