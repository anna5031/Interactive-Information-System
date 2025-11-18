from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from .config import INDOOR_MAP_PATH


def load_indoor_map(path: Path | None = None) -> Dict[str, Any]:
    target = Path(path or INDOOR_MAP_PATH)
    if not target.exists():
        raise FileNotFoundError(f"실내 지도 JSON을 찾을 수 없습니다: {target}")
    data = json.loads(target.read_text(encoding="utf-8"))
    return data
