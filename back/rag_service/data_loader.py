from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from uuid import uuid4

@dataclass(slots=True)
class KnowledgeEntry:
    """RAG용 지식 객체. 원시 JSON 항목을 보존하면서 주요 메타데이터를 유지."""

    id: str
    entry_type: str
    metadata: Dict[str, Any]
    content: str
    raw: Dict[str, Any] = field(repr=False)


@dataclass(slots=True)
class DataBundle:
    """로드된 데이터 묶음. 타입별 리스트를 유지하면서도 일괄 접근 가능."""

    professors: List[KnowledgeEntry]
    lectures: List[KnowledgeEntry]
    seminar_rooms: List[KnowledgeEntry]

    def all_entries(self) -> List[KnowledgeEntry]:
        return [*self.professors, *self.lectures, *self.seminar_rooms]


class DataLoader:
    """JSON 데이터를 로드하여 KnowledgeEntry 객체로 변환."""

    def __init__(self, data_path: Path):
        self.data_path = Path(data_path)

    def load(self) -> DataBundle:
        raw_items = self._read_json()
        professors: List[KnowledgeEntry] = []
        lectures: List[KnowledgeEntry] = []
        seminar_rooms: List[KnowledgeEntry] = []

        for index, item in enumerate(raw_items, start=1):
            entry = build_entry_from_raw(item, fallback_index=index)

            if entry.entry_type == "professor":
                professors.append(entry)
            elif entry.entry_type == "lecture":
                lectures.append(entry)
            else:
                seminar_rooms.append(entry)

        return DataBundle(
            professors=professors,
            lectures=lectures,
            seminar_rooms=seminar_rooms,
        )

    def _read_json(self) -> List[Dict[str, Any]]:
        if not self.data_path.exists():
            raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {self.data_path}")

        with self.data_path.open("r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSON 파싱 오류: {exc}") from exc

        if not isinstance(data, list):
            raise ValueError("JSON 구조가 리스트가 아닙니다.")
        return data


def _format_value(key: str, value: Any) -> List[str]:
    if value in (None, "", []):
        return []

    if isinstance(value, dict):
        lines = [f"{key}:"]
        for sub_key, sub_value in value.items():
            if sub_value in (None, ""):
                continue
            lines.append(f"  - {sub_key}: {sub_value}")
        return lines

    if isinstance(value, list):
        cleaned = [str(item) for item in value if item not in (None, "")]
        return [f"{key}: {', '.join(cleaned)}"] if cleaned else []

    return [f"{key}: {value}"]


def build_entry_from_raw(item: Dict[str, Any], *, fallback_index: int = 0) -> KnowledgeEntry:
    entry_type = str(item.get("type", "")).strip().lower() or "general"
    metadata = _extract_metadata(entry_type, item)
    content = _serialize_content(item)
    entry_id = _build_entry_id(entry_type, metadata, fallback_index=fallback_index)

    return KnowledgeEntry(
        id=entry_id,
        entry_type=entry_type,
        metadata=metadata,
        content=content,
        raw=item,
    )


def _extract_metadata(entry_type: str, item: Dict[str, Any]) -> Dict[str, Any]:
    metadata_fields: Dict[str, Optional[Any]] = {
        "building": item.get("building"),
        "room": item.get("room"),
        "prof_name": item.get("prof_name"),
        "lecture_name": item.get("lecture_name"),
        "lecture_time": item.get("lecture_time"),
        "name": item.get("name"),
    }

    if entry_type == "professor" and not metadata_fields.get("prof_name"):
        metadata_fields["prof_name"] = item.get("name")
    elif entry_type == "lecture" and not metadata_fields.get("lecture_name"):
        metadata_fields["lecture_name"] = item.get("name")
    elif entry_type == "seminar_room" and not metadata_fields.get("name"):
        metadata_fields["name"] = item.get("room")

    metadata_fields["raw_type"] = item.get("type")
    return {key: value for key, value in metadata_fields.items() if value not in (None, "", [])}


def _serialize_content(item: Dict[str, Any]) -> str:
    content_lines: List[str] = []
    for key, value in item.items():
        if key == "type":
            continue
        content_lines.extend(_format_value(key, value))
    return "\n".join(content_lines).strip()


def _build_entry_id(
    entry_type: str,
    metadata: Dict[str, Any],
    *,
    fallback_index: int,
) -> str:
    candidates: Iterable[str] = (
        metadata.get("prof_name", ""),
        metadata.get("lecture_name", ""),
        metadata.get("name", ""),
        metadata.get("room", ""),
        metadata.get("building", ""),
    )
    fallback_index_value = 0
    try:
        fallback_index_value = int(fallback_index)
    except (TypeError, ValueError):
        fallback_index_value = 0

    entry_type_slug = _slugify(entry_type) or "entry"
    base = next((candidate for candidate in candidates if candidate), None)
    base_slug = _slugify(str(base)) if base else ""

    slug_parts: List[str] = [entry_type_slug]
    if base_slug and base_slug != entry_type_slug:
        slug_parts.append(base_slug)

    if len(slug_parts) == 1:
        suffix = f"{fallback_index_value:04d}" if fallback_index_value > 0 else uuid4().hex[:8]
        slug_parts.append(suffix)

    candidate_slug = _slugify("-".join(slug_parts))
    if not candidate_slug or candidate_slug == entry_type_slug:
        suffix = f"{fallback_index_value:04d}" if fallback_index_value > 0 else uuid4().hex[:8]
        candidate_slug = _slugify(f"{entry_type_slug}-{suffix}")

    return candidate_slug or _random_entry_id(entry_type)


def _slugify(value: str) -> str:
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    value = re.sub(r"-{2,}", "-", value).strip("-")
    return value


def _random_entry_id(entry_type: str) -> str:
    return f"{entry_type}-{uuid4().hex[:8]}"
