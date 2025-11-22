from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from services.user_storage import UserScopedStorage

NEW_ROOM_INFO_PREFIX = "room_door_info_"


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


class StepThreeRepository:
    """파일 기반 Step 3 결과 저장소."""

    def __init__(self, storage_root: Path, user_storage: Optional[UserScopedStorage] = None):
        self.storage_root = _ensure_directory(storage_root)
        self.user_storage = user_storage or UserScopedStorage(self.storage_root)

    def _resolve_user_root(self, user_id: Optional[str], *, create: bool = True) -> Path:
        resolved = self.user_storage.resolve(user_id, create=create)
        return resolved.root

    def _record_path(self, user_root: Path, request_id: str, explicit_request_id: Optional[str] = None) -> Path:
        safe_request_id = (explicit_request_id or request_id or "").strip()
        if safe_request_id:
            request_dir = user_root / safe_request_id
            filename = f"{NEW_ROOM_INFO_PREFIX}{safe_request_id}.json"
            return request_dir / filename
        return user_root / f"{request_id}.json"

    def _normalize_record(self, request_id: str, record: Optional[dict]) -> Optional[dict]:
        if record is None:
            return None

        normalized = dict(record)
        normalized.pop("base", None)
        normalized.pop("details", None)
        request_id_value = (normalized.get("requestId") or normalized.get("request_id") or "").strip()
        normalized["id"] = request_id_value or request_id

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
        if normalized.get("floorLabel") is None and normalized.get("floor_label") is not None:
            normalized["floorLabel"] = normalized.get("floor_label")
        if normalized.get("floorValue") is None and normalized.get("floor_value") is not None:
            normalized["floorValue"] = normalized.get("floor_value")

        normalized.setdefault("createdAt", _utc_now_iso())
        normalized.setdefault("updatedAt", normalized["createdAt"])
        normalized.setdefault("floorLabel", normalized.get("floorLabel"))
        normalized.setdefault("floorValue", normalized.get("floorValue"))
        return normalized

    def _read_file(self, path: Path, fallback_id: Optional[str] = None) -> Optional[dict]:
        if not path.exists() or not path.is_file():
            return None
        try:
            raw_text = path.read_text(encoding="utf-8")
            data = json.loads(raw_text)
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None
        derived_id = fallback_id or data.get("id")
        if not derived_id:
            stem = path.stem
            derived_id = stem
        return self._normalize_record(derived_id, data)

    def _load(self, user_root: Path, request_id: str) -> Optional[dict]:
        candidates: List[Path] = []

        candidates.append(self._record_path(user_root, request_id))

        for request_dir in user_root.iterdir():
            if not request_dir.is_dir() or request_dir.name in {"history", "deleted"}:
                continue
            request_id = request_dir.name
            candidates.append(request_dir / f"{NEW_ROOM_INFO_PREFIX}{request_id}.json")

        history_root = user_root / "history"
        if history_root.exists():
            for request_dir in sorted(history_root.iterdir(), reverse=True):
                if not request_dir.is_dir():
                    continue
                for archive_dir in sorted(request_dir.iterdir(), reverse=True):
                    if not archive_dir.is_dir():
                        continue
                    candidates.append(archive_dir / f"{NEW_ROOM_INFO_PREFIX}{request_dir.name}.json")

        for legacy_name in ("step_three_results", "step_two_results"):
            legacy_dir = user_root / legacy_name
            if legacy_dir.exists():
                candidates.append(legacy_dir / f"{request_id}.json")

        for path in candidates:
            record = self._read_file(path, fallback_id=request_id)
            if record is None:
                continue
            target_id = request_id.strip()
            record_id = (record.get("id") or "").strip()
            request_id = (record.get("requestId") or record.get("request_id") or "").strip()
            if record_id == target_id or request_id == target_id:
                return record
        return None

    def _write(self, user_root: Path, request_id: str, record: dict, explicit_request_id: Optional[str] = None) -> dict:
        normalized = self._normalize_record(request_id, record)
        if normalized is None:
            raise ValueError("Failed to normalize step three record")
        path = self._record_path(user_root, request_id, explicit_request_id or normalized.get("requestId"))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(normalized, ensure_ascii=False, indent=2), encoding="utf-8")
        return normalized

    def get(self, request_id: str, *, user_id: Optional[str] = None) -> Optional[dict]:
        user_root = self._resolve_user_root(user_id, create=False)
        if not user_root.exists():
            return None
        return self._load(user_root, request_id)

    def list_all(self, *, user_id: Optional[str] = None) -> List[dict]:
        records_map: Dict[str, dict] = {}

        user_root = self._resolve_user_root(user_id, create=False)
        if not user_root.exists():
            return []

        for file_path in sorted(user_root.glob("step_one_*.json")):
            record = self._read_file(file_path)
            if record and record.get("id"):
                records_map[record["id"]] = record

        for request_dir in sorted(user_root.iterdir()):
            if not request_dir.is_dir() or request_dir.name in {"history", "deleted"}:
                continue
            for pattern in (f"{NEW_ROOM_INFO_PREFIX}*.json",):
                for file_path in sorted(request_dir.glob(pattern)):
                    record = self._read_file(file_path)
                    if record and record.get("id"):
                        records_map[record["id"]] = record

        history_root = user_root / "history"
        if history_root.exists():
            for request_dir in sorted(history_root.iterdir(), reverse=True):
                if not request_dir.is_dir():
                    continue
                    # for archive_dir in sorted(request_dir.iterdir(), reverse=True):
                    #     if not archive_dir.is_dir():
                    #         continue
                    #     for pattern in (
                    #         f"{NEW_ROOM_INFO_PREFIX}*.json",
                    #         f"{LEGACY_STEP_THREE_PREFIX}*.json",
                    #         f"{LEGACY_STEP_TWO_PREFIX}*.json",
                    #     ):
                    #         for file_path in sorted(archive_dir.glob(pattern)):
                    #             record = self._read_file(file_path)
                    #         if record and record.get("id") and record["id"] not in records_map:
                    #             records_map[record["id"]] = record

        for legacy_name in ("step_three_results", "step_two_results"):
            legacy_dir = user_root / legacy_name
            if legacy_dir.exists():
                for file_path in sorted(legacy_dir.glob("*.json")):
                    record = self._read_file(file_path)
                    if record and record.get("id"):
                        records_map[record["id"]] = record

        return list(records_map.values())

    def _ensure_record(self, request_id: str, record: Optional[dict]) -> dict:
        normalized = self._normalize_record(request_id, record)
        if normalized is None:
            now = _utc_now_iso()
            normalized = {
                "id": request_id,
                "createdAt": now,
                "updatedAt": now,
                "rooms": [],
                "doors": [],
                "floorLabel": None,
                "floorValue": None,
            }
        else:
            normalized.setdefault("rooms", [])
            normalized.setdefault("doors", [])
            normalized.setdefault("floorLabel", normalized.get("floorLabel"))
            normalized.setdefault("floorValue", normalized.get("floorValue"))
        return normalized

    def save(self, request_id: str, payload: dict, *, user_id: Optional[str] = None) -> dict:
        user_root = self._resolve_user_root(user_id, create=True)
        resolved_request_id = (payload.get("requestId") or request_id or "").strip()
        record = self._ensure_record(resolved_request_id, self._load(user_root, resolved_request_id))
        now = _utc_now_iso()
        if payload.get("requestId"):
            record["requestId"] = payload["requestId"]
        if payload.get("floorLabel") is not None:
            value = payload.get("floorLabel") or ""
            record["floorLabel"] = value.strip()
        if payload.get("floorValue") is not None:
            value = payload.get("floorValue") or ""
            record["floorValue"] = value.strip()

        rooms = [_sanitize_room_payload(room) for room in payload.get("rooms", [])]
        doors = [_sanitize_door_payload(door) for door in payload.get("doors", [])]

        record["rooms"] = rooms
        record["doors"] = doors
        record["updatedAt"] = now
        record["id"] = (record.get("requestId") or resolved_request_id).strip()

        return self._write(user_root, resolved_request_id, record, explicit_request_id=record.get("requestId"))
