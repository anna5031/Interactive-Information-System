from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

FLOORPLAN_INDEX_FILENAME = "floorplan_index.json"


def _index_path(user_root: Path) -> Path:
    return user_root / FLOORPLAN_INDEX_FILENAME


def load_index(user_root: Path) -> Dict[str, Dict[str, Any]]:
    """사용자 루트에 저장된 Floorplan 인덱스를 읽어 dict(request_id -> entry)로 반환한다."""
    path = _index_path(user_root)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, OSError):
        return {}
    if not isinstance(payload, dict):
        return {}
    normalized: Dict[str, Dict[str, Any]] = {}
    for key, value in payload.items():
        if isinstance(key, str) and isinstance(value, dict):
            normalized[key] = value
    return normalized


def _write_index(user_root: Path, data: Dict[str, Dict[str, Any]]) -> None:
    path = _index_path(user_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def update_entry(user_root: Path, request_id: str, entry: Dict[str, Any]) -> None:
    """특정 요청 ID의 인덱스 항목을 추가/갱신한다."""
    if not request_id:
        return
    index = load_index(user_root)
    index[request_id] = entry
    _write_index(user_root, index)


def remove_entry(user_root: Path, request_id: str) -> None:
    """요청 ID에 해당하는 인덱스 항목을 제거한다."""
    if not request_id:
        return
    path = _index_path(user_root)
    if not path.exists():
        return
    index = load_index(user_root)
    if request_id in index:
        index.pop(request_id, None)
        if index:
            _write_index(user_root, index)
        else:
            try:
                path.unlink()
            except OSError:
                _write_index(user_root, index)
