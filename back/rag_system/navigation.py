from __future__ import annotations

import random
from typing import Dict


def find_route(destination: str | None) -> Dict[str, str]:
    if not destination:
        return {"success": False, "message": "목적지를 파악할 수 없어 경로 안내를 건너뜁니다."}
    sample_time = random.randint(2, 7)
    return {
        "success": True,
        "message": f"{destination}까지 약 {sample_time}분이 소요됩니다.",
        "steps": [
            "엘리베이터를 타고 해당 층으로 이동하세요.",
            f"표지판을 따라 {destination} 방향으로 직진하세요.",
        ],
    }
