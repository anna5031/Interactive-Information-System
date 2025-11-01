from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _sanitize_extra(entries: Optional[Iterable[dict]]) -> List[Dict[str, str]]:
    sanitized: List[Dict[str, str]] = []
    if not entries:
        return sanitized
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        key = (entry.get("key") or "").strip()
        value = (entry.get("value") or "").strip()
        if key or value:
            sanitized.append({"key": key, "value": value})
    return sanitized


def _sanitize_room_payload(room: dict) -> Dict[str, Optional[str]]:
    node_id = str(room.get("nodeId") or room.get("node_id") or "").strip()
    graph_node_id = room.get("graphNodeId") or room.get("graph_node_id") or node_id
    payload = {
        "nodeId": node_id,
        "graphNodeId": graph_node_id.strip() if isinstance(graph_node_id, str) else graph_node_id,
        "name": (room.get("name") or "").strip(),
        "number": (room.get("number") or "").strip(),
        "extra": _sanitize_extra(room.get("extra")),
    }
    return payload


def _sanitize_door_payload(door: dict) -> Dict[str, Optional[str]]:
    node_id = str(door.get("nodeId") or door.get("node_id") or "").strip()
    graph_node_id = door.get("graphNodeId") or door.get("graph_node_id") or node_id
    payload = {
        "nodeId": node_id,
        "graphNodeId": graph_node_id.strip() if isinstance(graph_node_id, str) else graph_node_id,
        "type": (door.get("type") or "").strip(),
        "customType": (door.get("customType") or door.get("custom_type") or "").strip(),
        "extra": _sanitize_extra(door.get("extra")),
    }
    return payload


class StepTwoRepository:
    """파일 기반 Step 2 결과 저장소."""

    def __init__(self, storage_root: Path):
        self.storage_root = _ensure_directory(storage_root)

    def _record_path(self, step_one_id: str, request_id: Optional[str] = None) -> Path:
        safe_step_one_id = step_one_id.strip()
        if request_id:
            request_dir = self.storage_root / request_id.strip()
            if request_dir.exists():
                return request_dir / f"step_two_{safe_step_one_id}.json"
        return self.storage_root / f"{safe_step_one_id}.json"

    def _normalize_record(self, step_one_id: str, record: Optional[dict]) -> Optional[dict]:
        if record is None:
            return None

        normalized = dict(record)
        normalized.pop("base", None)
        normalized.pop("details", None)
        normalized.setdefault("id", step_one_id)

        rooms = normalized.get("rooms")
        doors = normalized.get("doors")

        if rooms is None or doors is None:
            source = None
            if isinstance(record.get("rooms"), list) and isinstance(record.get("doors"), list):
                source = {"rooms": record.get("rooms"), "doors": record.get("doors"), "updatedAt": record.get("updatedAt")}
            elif isinstance(record.get("details"), dict):
                source = record.get("details")
            elif isinstance(record.get("base"), dict):
                source = record.get("base")
            if source:
                rooms = source.get("rooms", [])
                doors = source.get("doors", [])
                updated_at = source.get("updatedAt")
                if updated_at:
                    normalized["updatedAt"] = updated_at
        normalized["rooms"] = rooms or []
        normalized["doors"] = doors or []

        normalized.setdefault("createdAt", _utc_now_iso())
        normalized.setdefault("updatedAt", normalized["createdAt"])
        return normalized

    def _read_file(self, path: Path, fallback_id: Optional[str] = None) -> Optional[dict]:
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
        derived_id = fallback_id or data.get("id")
        if not derived_id:
            stem = path.stem
            if stem.startswith("step_two_"):
                derived_id = stem.replace("step_two_", "")
            else:
                derived_id = stem
        return self._normalize_record(derived_id, data)

    def _load(self, step_one_id: str) -> Optional[dict]:
        candidates: List[Path] = []

        candidates.append(self._record_path(step_one_id))

        for request_dir in self.storage_root.iterdir():
            if request_dir.is_dir():
                candidates.append(request_dir / f"step_two_{step_one_id}.json")

        legacy_dir = self.storage_root / "step_two_results"
        if legacy_dir.exists():
            candidates.append(legacy_dir / f"{step_one_id}.json")

        for path in candidates:
            record = self._read_file(path, fallback_id=step_one_id)
            if record is not None:
                return record
        return None

    def _write(self, step_one_id: str, record: dict, request_id: Optional[str] = None) -> dict:
        normalized = self._normalize_record(step_one_id, record)
        if normalized is None:
            raise ValueError("Failed to normalize step two record")
        path = self._record_path(step_one_id, request_id=request_id or normalized.get("requestId"))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(normalized, ensure_ascii=False, indent=2), encoding="utf-8")
        return normalized

    def get(self, step_one_id: str) -> Optional[dict]:
        return self._load(step_one_id)

    def list_all(self) -> List[dict]:
        records_map: Dict[str, dict] = {}

        for file_path in sorted(self.storage_root.glob("step_one_*.json")):
            record = self._read_file(file_path)
            if record and record.get("id"):
                records_map[record["id"]] = record

        for request_dir in sorted(self.storage_root.iterdir()):
            if not request_dir.is_dir():
                continue
            for file_path in sorted(request_dir.glob("step_two_*.json")):
                record = self._read_file(file_path)
                if record and record.get("id"):
                    records_map[record["id"]] = record

        legacy_dir = self.storage_root / "step_two_results"
        if legacy_dir.exists():
            for file_path in sorted(legacy_dir.glob("*.json")):
                record = self._read_file(file_path)
                if record and record.get("id"):
                    records_map[record["id"]] = record

        return list(records_map.values())

    def _ensure_record(self, step_one_id: str, record: Optional[dict]) -> dict:
        normalized = self._normalize_record(step_one_id, record)
        if normalized is None:
            now = _utc_now_iso()
            normalized = {
                "id": step_one_id,
                "createdAt": now,
                "updatedAt": now,
                "rooms": [],
                "doors": [],
            }
        else:
            normalized.setdefault("rooms", [])
            normalized.setdefault("doors", [])
        return normalized

    def save(self, step_one_id: str, payload: dict) -> dict:
        record = self._ensure_record(step_one_id, self._load(step_one_id))
        now = _utc_now_iso()
        if payload.get("requestId"):
            record["requestId"] = payload["requestId"]

        rooms = [_sanitize_room_payload(room) for room in payload.get("rooms", [])]
        doors = [_sanitize_door_payload(door) for door in payload.get("doors", [])]

        record["rooms"] = rooms
        record["doors"] = doors
        record["updatedAt"] = now

        return self._write(step_one_id, record, request_id=record.get("requestId"))
