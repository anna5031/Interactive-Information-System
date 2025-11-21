from __future__ import annotations

import base64
import binascii
import json
import shutil
import mimetypes
import re
import tempfile
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import quote_plus
from uuid import uuid4
from collections import defaultdict

import cv2
import networkx as nx
import numpy as np
from shapely.geometry import mapping
from shapely.geometry.base import BaseGeometry

from configuration import get_processing_settings
from services.floorplan_index import remove_entry as remove_floorplan_index_entry, update_entry as update_floorplan_index_entry
from services.step_three_repository import NEW_ROOM_INFO_PREFIX
from services.user_storage import UserScopedStorage
from processing.corridor_pipeline import (
    CorridorPipeline,
    CorridorPipelineConfig,
    FreeSpaceMaskBundle,
    StageOneArtifacts,
)
from processing.object_detection_out_to_graph import (
    DEFAULT_CLASS_NAMES,
    FloorPlanParser,
    load_annotation_bundle_from_texts,
)

PROCESSING_SETTINGS = get_processing_settings()
STEP_TWO_DEBUG_ENABLED = bool(PROCESSING_SETTINGS.get("save_step_two_debug_images"))
STEP_TWO_DEBUG_SUBDIR = str(PROCESSING_SETTINGS.get("step_two_debug_subdir") or "graph_debug")
STEP_ONE_HISTORY_ENABLED = bool(PROCESSING_SETTINGS.get("save_step_one_history"))
WALL_EXPANSION_ENABLED = bool(PROCESSING_SETTINGS.get("enable_wall_expansion"))
CROSS_FLOOR_WEIGHT_MARGIN = 1.0
CROSS_FLOOR_STORAGE_FILENAME = "cross_floor_connections.json"
CROSS_FLOOR_GENERATED_FLAG = "generated_cross_floor"

_DATA_URL_PATTERN = re.compile(r"^data:(?P<mime>[\w.+/-]+);base64,(?P<data>.+)$")


@dataclass
class ResultDirectoryContext:
    user_id: str
    user_root: Path
    request_id: str
    result_dir: Path
    existing_dir: Optional[Path]
    existing_metadata: Optional[dict]
    previous_image_filename: Optional[str]
    previous_image_mime: Optional[str]


@dataclass(frozen=True)
class CrossFloorConnector:
    request_id: str
    node_id: str


@dataclass
class CrossFloorConnectorInfo:
    connector_type: str
    label: str


@dataclass(frozen=True)
class CrossFloorLinkEndpoint:
    request_id: str
    node_id: str
    label: Optional[str] = None

    def key(self) -> Tuple[str, str]:
        return (self.request_id, self.node_id)

    def normalized_label(self) -> str:
        if self.label:
            return self.label
        return self.node_id


@dataclass(frozen=True)
class CrossFloorLinkRecord:
    connector_type: str
    endpoint_a: CrossFloorLinkEndpoint
    endpoint_b: CrossFloorLinkEndpoint

    def sorted_endpoints(self) -> Tuple[CrossFloorLinkEndpoint, CrossFloorLinkEndpoint]:
        ordered = sorted((self.endpoint_a, self.endpoint_b), key=lambda endpoint: endpoint.key())
        return ordered[0], ordered[1]

    def key(self) -> Tuple[str, str, str, str, str]:
        endpoint_a, endpoint_b = self.sorted_endpoints()
        return (
            self.connector_type,
            endpoint_a.request_id,
            endpoint_a.node_id,
            endpoint_b.request_id,
            endpoint_b.node_id,
        )


@dataclass
class StoredGraphRecord:
    request_id: str
    result_dir: Path
    graph_path: Path
    metadata_path: Path
    graph: dict
    metadata: dict
    node_type_map: Dict[str, str]


def _normalize_connector_type_label(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if not normalized:
        return None
    if "stair" in normalized:
        return "stair"
    if "elev" in normalized:
        return "elevator"
    return None


def _format_endpoint_label(endpoint: CrossFloorLinkEndpoint) -> str:
    return f"{endpoint.request_id}:{endpoint.node_id}"


def _per_endpoint_key(endpoint: CrossFloorLinkEndpoint, other: CrossFloorLinkEndpoint) -> Tuple[str, str, str]:
    return (endpoint.request_id, endpoint.node_id, other.request_id)


def _decode_image_data_url(data_url: str) -> Tuple[bytes, str]:
    """data URL 문자열을 디코딩해 (바이트, MIME 타입)을 반환한다."""
    if not data_url:
        raise ValueError("empty data url")
    match = _DATA_URL_PATTERN.match(data_url.strip())
    if not match:
        raise ValueError("invalid data url format")
    mime_type = match.group("mime") or "application/octet-stream"
    try:
        image_bytes = base64.b64decode(match.group("data"), validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError("invalid base64 image payload") from exc
    return image_bytes, mime_type


def _build_data_url(mime_type: str, payload: bytes) -> str:
    """이미지 바이트를 data URL 형식 문자열로 직렬화한다."""
    base64_payload = base64.b64encode(payload).decode("ascii")
    safe_mime = mime_type or "application/octet-stream"
    return f"data:{safe_mime};base64,{base64_payload}"


def _to_serializable(value):
    """numpy/shapely 객체 등을 JSON 직렬화가 가능한 값으로 변환한다."""
    if isinstance(value, BaseGeometry):
        return mapping(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): _to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_serializable(v) for v in value]
    return value


def _serialize_objects(objects: Dict[str, List[dict]]) -> Dict[str, List[dict]]:
    """객체 딕셔너리를 JSON-friendly 구조로 깊은 복사한다."""
    serialized: Dict[str, List[dict]] = {}
    for key, items in objects.items():
        bucket: List[dict] = []
        for item in items:
            serialized_item = {attr_key: _to_serializable(attr_value) for attr_key, attr_value in item.items()}
            bucket.append(serialized_item)
        serialized[key] = bucket
    return serialized


def _serialize_graph(graph: nx.Graph) -> Dict[str, List[dict]]:
    """NetworkX 그래프를 nodes/edges 리스트로 풀어 직렬화한다."""
    nodes: List[dict] = []
    for node_id, attrs in graph.nodes(data=True):
        payload: dict = {"id": str(node_id)}
        if "type" in attrs:
            payload["type"] = attrs["type"]
        if "pos" in attrs:
            payload["pos"] = _to_serializable(attrs["pos"])
        remaining = {k: v for k, v in attrs.items() if k not in {"type", "pos"}}
        if remaining:
            payload["attributes"] = _to_serializable(remaining)
        nodes.append(payload)

    edges: List[dict] = []
    for source, target, attrs in graph.edges(data=True):
        edge_payload: dict = {
            "source": str(source),
            "target": str(target),
        }
        attr_copy = dict(attrs)
        weight = attr_copy.pop("weight", None)
        if weight is not None:
            edge_payload["weight"] = _to_serializable(weight)
        if attr_copy:
            edge_payload["attributes"] = _to_serializable(attr_copy)
        edges.append(edge_payload)

    return {"nodes": nodes, "edges": edges}


def _coerce_position(value) -> Optional[List[float]]:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        coords: List[float] = []
        for item in value:
            try:
                coords.append(float(item))
            except (TypeError, ValueError):
                return None
        if not coords:
            return None
        return coords
    return None


def _normalize_graph_payload(graph: Optional[Dict[str, Any]]) -> Dict[str, List[dict]]:
    payload = graph or {}
    normalized_nodes: List[dict] = []
    for node in payload.get("nodes", []):
        if not isinstance(node, dict):
            continue
        node_id = str(node.get("id") or "").strip()
        if not node_id:
            continue
        entry: Dict[str, Any] = {"id": node_id}
        node_type = node.get("type")
        if node_type is not None:
            entry["type"] = str(node_type)
        pos_value = _coerce_position(node.get("pos"))
        if pos_value is not None:
            entry["pos"] = pos_value
        attributes = node.get("attributes")
        if isinstance(attributes, dict) and attributes:
            entry["attributes"] = attributes
        normalized_nodes.append(entry)

    normalized_edges: List[dict] = []
    for edge in payload.get("edges", []):
        if not isinstance(edge, dict):
            continue
        source = str(edge.get("source") or "").strip()
        target = str(edge.get("target") or "").strip()
        if not source or not target:
            continue
        entry: Dict[str, Any] = {"source": source, "target": target}
        weight = edge.get("weight")
        if weight is not None:
            try:
                entry["weight"] = float(weight)
            except (TypeError, ValueError):
                pass
        attributes = edge.get("attributes")
        if isinstance(attributes, dict) and attributes:
            entry["attributes"] = attributes
        normalized_edges.append(entry)

    return {"nodes": normalized_nodes, "edges": normalized_edges}


def _build_graph_context_payload(
    *,
    request_id: Optional[str],
    floor_label: Optional[str],
    floor_value: Optional[str],
    scale_reference: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """navigation_graph.json에 함께 저장할 부가 정보를 생성한다."""
    payload: Dict[str, Any] = {}

    def _clean(value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        if not isinstance(value, str):
            value = str(value)
        trimmed = value.strip()
        return trimmed or None

    resolved_request_id = _clean(request_id)
    resolved_floor_label = _clean(floor_label)
    resolved_floor_value = _clean(floor_value)
    normalized_scale_reference = scale_reference or None

    if resolved_request_id:
        payload["requestId"] = resolved_request_id
    if resolved_floor_label:
        payload["floorLabel"] = resolved_floor_label
    if resolved_floor_value:
        payload["floorValue"] = resolved_floor_value
    if normalized_scale_reference:
        payload["scaleReference"] = normalized_scale_reference
    return payload


def _normalize_scale_reference_payload(value: Optional[Dict[str, Any]]) -> Optional[Dict[str, float]]:
    if not value or not isinstance(value, dict):
        return None

    def _normalize_unit(raw: Any) -> Optional[float]:
        try:
            numeric = float(raw)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(numeric):
            return None
        if numeric < 0:
            numeric = 0.0
        elif numeric > 1:
            numeric = 1.0
        return numeric

    coords = []
    for key in ("x1", "y1", "x2", "y2"):
        normalized = _normalize_unit(value.get(key))
        if normalized is None:
            return None
        coords.append(normalized)

    length_raw = value.get("length_meters")
    if length_raw is None:
        length_raw = value.get("lengthMeters")
    try:
        length_value = float(length_raw)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(length_value) or length_value <= 0:
        return None

    if coords[0] == coords[2] and coords[1] == coords[3]:
        return None

    return {
        "x1": coords[0],
        "y1": coords[1],
        "x2": coords[2],
        "y2": coords[3],
        "length_meters": length_value,
    }



def _parse_object_detection_text(object_detection_text: str) -> Tuple[List[dict], List[str]]:
    """YOLO 포맷 텍스트를 파싱해 정규화된 dict 리스트와 식별자 포함 텍스트를 돌려준다."""
    annotations: List[dict] = []
    sanitized_lines: List[str] = []
    if not object_detection_text:
        return annotations, sanitized_lines

    for index, raw_line in enumerate(object_detection_text.splitlines()):
        tokens = raw_line.strip().split()
        if len(tokens) < 5:
            continue

        try:
            class_id = int(float(tokens[0]))
            cx = float(tokens[1])
            cy = float(tokens[2])
            width = float(tokens[3])
            height = float(tokens[4])
        except ValueError:
            continue

        remaining = tokens[5:] if len(tokens) > 5 else []
        confidence: Optional[float] = None
        identifier: Optional[str] = None

        if remaining:
            first = remaining[0]
            try:
                confidence = float(first)
                remaining = remaining[1:]
            except ValueError:
                confidence = None
            if remaining:
                identifier = remaining[0]

        if identifier is None:
            identifier = f"{class_id}-box-{index}"

        sanitized_tokens = [
            tokens[0],
            tokens[1],
            tokens[2],
            tokens[3],
            tokens[4],
            identifier,
        ]
        sanitized_lines.append(" ".join(sanitized_tokens))

        annotation = {
            "class_id": class_id,
            "x_center": float(np.clip(cx, 0.0, 1.0)),
            "y_center": float(np.clip(cy, 0.0, 1.0)),
            "width": float(np.clip(width, 0.0, 1.0)),
            "height": float(np.clip(height, 0.0, 1.0)),
            "identifier": identifier,
        }
        if confidence is not None:
            annotation["confidence"] = float(np.clip(confidence, 0.0, 1.0))

        annotations.append(annotation)

    return annotations, sanitized_lines


def _load_objects_from_texts(
    object_detection_text: str,
    wall_text: str,
    door_text: str,
    image_width: int,
    image_height: int,
    *,
    enable_wall_expansion: Optional[bool] = None,
) -> Tuple[Dict[str, List[dict]], List[dict]]:
    """rooms/walls/doors 텍스트를 임시 파일로 저장해 파서가 로드하도록 한다."""
    if enable_wall_expansion is None:
        enable_wall_expansion = WALL_EXPANSION_ENABLED
    annotations, sanitized_lines = _parse_object_detection_text(object_detection_text)
    sanitized_object_detection_text = "\n".join(sanitized_lines)
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        room_path = tmp_path / "rooms.txt"
        wall_path = tmp_path / "walls.txt"
        door_path = tmp_path / "doors.txt"

        room_path.write_text(sanitized_object_detection_text, encoding="utf-8")
        wall_path.write_text(wall_text or "", encoding="utf-8")
        door_path.write_text(door_text or "", encoding="utf-8")

        objects = load_annotation_bundle_from_texts(
            room_path,
            wall_path,
            door_path,
            image_width=image_width,
            image_height=image_height,
            enable_wall_expansion=enable_wall_expansion,
        )
    return objects, annotations

class FloorPlanProcessingService:
    """ObjectDetection/라인 텍스트를 그래프/객체 JSON으로 변환하고 디스크에 저장하는 서비스."""

    STEP_ONE_ARTIFACTS: Tuple[str, ...] = (
        "rooms.txt",
        "walls.txt",
        "walls_raw.txt",
        "doors.txt",
        "floorplan_objects.json",
        "navigation_graph.json",
        "input_annotations.json",
        "metadata.json",
    )
    GRAPH_FILENAME_PREFIX = "navigation_graph_"

    def __init__(
        self,
        storage_root: Path,
        legacy_root: Optional[Path] = None,
        user_storage: Optional[UserScopedStorage] = None,
    ):
        """결과물을 기록할 루트 디렉터리를 준비한다."""
        self.storage_root = storage_root
        self.storage_root.mkdir(parents=True, exist_ok=True)
        self.legacy_root = legacy_root
        self.user_storage = user_storage or UserScopedStorage(self.storage_root)
        self.history_dirname: Optional[str] = "history" if STEP_ONE_HISTORY_ENABLED else None
        self.deleted_dirname: str = "deleted"

    def _load_existing_metadata(self, result_dir: Path) -> Optional[dict]:
        metadata_path = result_dir / "metadata.json"
        if not metadata_path.exists():
            return None
        try:
            return json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None

    def _archive_previous_result(
        self,
        *,
        user_root: Path,
        request_id: str,
        result_dir: Path,
        metadata: Optional[dict],
    ) -> None:
        if self.history_dirname is None or not result_dir.exists():
            return
        archive_token = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid4().hex[:6]}"
        archive_dir = user_root / self.history_dirname / request_id / archive_token
        archive_dir.mkdir(parents=True, exist_ok=True)
        artifact_names: List[str] = list(self.STEP_ONE_ARTIFACTS)
        stored_image = metadata.get("stored_image_path") if metadata else None
        if stored_image:
            artifact_names.append(stored_image)
        for graph_file in sorted(result_dir.glob(f"{self.GRAPH_FILENAME_PREFIX}*.json")):
            artifact_names.append(graph_file.name)
        for name in artifact_names:
            src = result_dir / name
            if not src.exists():
                continue
            dest = archive_dir / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)

    def _get_existing_result_dir(self, user_root: Path, request_id: str) -> Optional[Path]:
        primary = user_root / request_id
        if primary.exists():
            return primary
        if self.legacy_root:
            legacy = self.legacy_root / request_id
            if legacy.exists():
                return legacy
        return None

    @classmethod
    def _graph_filename(cls, request_id: Optional[str]) -> str:
        safe_request_id = (request_id or "").strip()
        if not safe_request_id:
            return "navigation_graph.json"
        return f"{cls.GRAPH_FILENAME_PREFIX}{safe_request_id}.json"

    def _resolve_user_root(self, user_id: Optional[str], *, create: bool = True) -> Tuple[str, Path]:
        resolved = self.user_storage.resolve(user_id, create=create)
        return resolved.user_id, resolved.root

    def _build_image_url(self, request_id: str, user_id: str) -> str:
        token = quote_plus(user_id)
        return f"/api/floorplans/{request_id}/image?userId={token}"

    @staticmethod
    def _relative_to_user_root(user_root: Path, target_path: Path) -> str:
        try:
            return str(target_path.relative_to(user_root))
        except ValueError:
            return str(target_path)

    def _update_floorplan_index(
        self,
        *,
        user_root: Path,
        request_id: str,
        result_dir: Path,
        metadata: Dict[str, Any],
        graph_path: Path,
    ) -> None:
        if not request_id:
            return
        entry = {
            "requestId": request_id,
            "floorLabel": metadata.get("floor_label") or metadata.get("floorLabel"),
            "floorValue": metadata.get("floor_value") or metadata.get("floorValue"),
            "scaleReference": metadata.get("scale_reference") or metadata.get("scaleReference"),
            "imageSize": metadata.get("image_size") or metadata.get("imageSize"),
            "graphPath": self._relative_to_user_root(user_root, graph_path),
            "roomInfoPath": self._relative_to_user_root(
                user_root, result_dir / f"{NEW_ROOM_INFO_PREFIX}{request_id}.json"
            ),
            "metadataPath": self._relative_to_user_root(user_root, result_dir / "metadata.json"),
            "updatedAt": datetime.now().isoformat(timespec="seconds"),
        }
        update_floorplan_index_entry(user_root, request_id, entry)

    def _resolve_graph_path(self, result_dir: Path, request_id: Optional[str]) -> Path:
        preferred = result_dir / self._graph_filename(request_id)
        if preferred.exists():
            return preferred
        for candidate in sorted(result_dir.glob(f"{self.GRAPH_FILENAME_PREFIX}*.json")):
            if candidate.exists():
                return candidate
        return preferred

    def _iter_active_result_dirs(self, user_root: Path) -> List[Path]:
        if not user_root.exists():
            return []
        result_dirs: List[Path] = []
        for entry in sorted(user_root.iterdir()):
            if not entry.is_dir():
                continue
            if entry.name == self.deleted_dirname:
                continue
            if self.history_dirname and entry.name == self.history_dirname:
                continue
            result_dirs.append(entry)
        return result_dirs

    def _load_graph_record(self, result_dir: Path) -> Optional[StoredGraphRecord]:
        metadata_path = result_dir / "metadata.json"
        if not metadata_path.exists():
            return None
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            metadata = {}
        request_id = metadata.get("request_id") or metadata.get("requestId") or result_dir.name
        graph_path = self._resolve_graph_path(result_dir, request_id)
        if not graph_path.exists():
            return None
        try:
            graph = json.loads(graph_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
        node_type_map: Dict[str, str] = {}
        for node in graph.get("nodes", []):
            if not isinstance(node, dict):
                continue
            node_id = str(node.get("id") or "").strip()
            if not node_id:
                continue
            node_type = str(node.get("type") or "").strip().lower()
            if node_type:
                node_type_map[node_id] = node_type
        return StoredGraphRecord(
            request_id=request_id,
            result_dir=result_dir,
            graph_path=graph_path,
            metadata_path=metadata_path,
            graph=graph,
            metadata=metadata,
            node_type_map=node_type_map,
        )

    def _load_all_graph_records(self, user_root: Path) -> List[StoredGraphRecord]:
        records: List[StoredGraphRecord] = []
        for result_dir in self._iter_active_result_dirs(user_root):
            record = self._load_graph_record(result_dir)
            if record is not None:
                records.append(record)
        return records

    def _cross_floor_storage_path(self, user_root: Path) -> Path:
        return user_root / CROSS_FLOOR_STORAGE_FILENAME

    def _load_cross_floor_links(self, user_root: Path) -> List[CrossFloorLinkRecord]:
        path = self._cross_floor_storage_path(user_root)
        if not path.exists():
            return []
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return []
        connections = []
        for raw in payload.get("connections", []):
            connector_type = _normalize_connector_type_label(raw.get("connector_type") or raw.get("connectorType"))
            endpoints = raw.get("endpoints") or raw.get("nodes")
            if not connector_type or not isinstance(endpoints, list) or len(endpoints) != 2:
                continue
            parsed_endpoints: List[CrossFloorLinkEndpoint] = []
            for endpoint in endpoints:
                if not isinstance(endpoint, dict):
                    break
                request_id = str(
                    endpoint.get("request_id") or endpoint.get("requestId") or endpoint.get("floorId") or ""
                ).strip()
                node_id = str(endpoint.get("node_id") or endpoint.get("nodeId") or endpoint.get("node") or "").strip()
                if not request_id or not node_id:
                    break
                label = endpoint.get("label") or endpoint.get("name") or None
                parsed_endpoints.append(CrossFloorLinkEndpoint(request_id=request_id, node_id=node_id, label=label))
            if len(parsed_endpoints) != 2:
                continue
            connections.append(
                CrossFloorLinkRecord(connector_type=connector_type, endpoint_a=parsed_endpoints[0], endpoint_b=parsed_endpoints[1])
            )
        return self._enforce_single_link_per_floor(connections)

    def _save_cross_floor_links(self, user_root: Path, links: List[CrossFloorLinkRecord]) -> None:
        path = self._cross_floor_storage_path(user_root)
        payload = {
            "connections": [
                {
                    "connectorType": link.connector_type,
                    "endpoints": [
                        {"requestId": link.endpoint_a.request_id, "nodeId": link.endpoint_a.node_id, "label": link.endpoint_a.label},
                        {"requestId": link.endpoint_b.request_id, "nodeId": link.endpoint_b.node_id, "label": link.endpoint_b.label},
                    ],
                }
                for link in links
            ]
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _validate_cross_floor_link_uniqueness(self, links: List[CrossFloorLinkRecord]) -> None:
        endpoint_map: Dict[Tuple[str, str, str], CrossFloorLinkRecord] = {}
        for link in links:
            endpoints = (link.endpoint_a, link.endpoint_b)
            for endpoint, other in ((endpoints[0], endpoints[1]), (endpoints[1], endpoints[0])):
                key = _per_endpoint_key(endpoint, other)
                existing = endpoint_map.get(key)
                if existing and existing is not link:
                    raise ValueError(
                        f"교차층 노드 {_format_endpoint_label(endpoint)}는 이미 "
                        f"{_format_endpoint_label(other)} (요청 {other.request_id}) 층으로 연결되어 있습니다."
                    )
                endpoint_map[key] = link

    def _enforce_single_link_per_floor(
        self, links: List[CrossFloorLinkRecord]
    ) -> List[CrossFloorLinkRecord]:
        per_endpoint_target: Dict[Tuple[str, str, str], CrossFloorLinkRecord] = {}
        link_key_map: Dict[CrossFloorLinkRecord, List[Tuple[str, str, str]]] = defaultdict(list)
        ordered: List[CrossFloorLinkRecord] = []

        def _remove_link(target: CrossFloorLinkRecord) -> None:
            if target in ordered:
                ordered.remove(target)
            for stored_key in link_key_map.pop(target, []):
                per_endpoint_target.pop(stored_key, None)

        for link in links:
            endpoints = (link.endpoint_a, link.endpoint_b)
            replaced = True
            while replaced:
                replaced = False
                for endpoint, other in ((endpoints[0], endpoints[1]), (endpoints[1], endpoints[0])):
                    key = _per_endpoint_key(endpoint, other)
                    existing = per_endpoint_target.get(key)
                    if existing and existing is not link:
                        _remove_link(existing)
                        replaced = True
                        break
            ordered.append(link)
            for endpoint, other in ((endpoints[0], endpoints[1]), (endpoints[1], endpoints[0])):
                key = _per_endpoint_key(endpoint, other)
                per_endpoint_target[key] = link
                link_key_map.setdefault(link, []).append(key)
        return ordered

    def _resolve_connector_kind(
        self,
        record_map: Dict[str, StoredGraphRecord],
        request_id: str,
        node_id: str,
    ) -> str:
        record = record_map.get(request_id)
        if record is None:
            raise ValueError(f"요청 {request_id}에 대한 도면 데이터를 찾을 수 없습니다.")
        node_type_raw = record.node_type_map.get(node_id)
        if node_type_raw is None:
            raise ValueError(f"요청 {request_id}에 노드 {node_id}가 존재하지 않습니다.")
        normalized = _normalize_connector_type_label(node_type_raw)
        if normalized is None:
            raise ValueError(f"노드 {node_id} (요청 {request_id})은(는) 계단/엘리베이터 노드가 아닙니다.")
        return normalized

    def _update_cross_floor_links_for_request(
        self,
        user_root: Path,
        request_id: str,
        new_links: List[CrossFloorLinkRecord],
    ) -> None:
        new_links = self._enforce_single_link_per_floor(new_links)
        existing_links = self._load_cross_floor_links(user_root)
        existing_map: Dict[Tuple[str, str, str, str, str], CrossFloorLinkRecord] = {
            link.key(): link for link in existing_links
        }
        for key, link in list(existing_map.items()):
            if link.endpoint_a.request_id == request_id or link.endpoint_b.request_id == request_id:
                del existing_map[key]
        for link in new_links:
            existing_map[link.key()] = link
        ordered_links = sorted(
            existing_map.values(),
            key=lambda link: (
                link.connector_type,
                link.sorted_endpoints()[0].request_id,
                link.sorted_endpoints()[0].node_id,
                link.sorted_endpoints()[1].request_id,
                link.sorted_endpoints()[1].node_id,
            ),
        )
        if ordered_links:
            ordered_links = self._enforce_single_link_per_floor(ordered_links)
            self._validate_cross_floor_link_uniqueness(ordered_links)
            self._save_cross_floor_links(user_root, ordered_links)
        else:
            path = self._cross_floor_storage_path(user_root)
            if path.exists():
                path.unlink()

    def _extract_user_defined_cross_floor_links(
        self, request_id: str, graph_payload: Dict[str, Any]
    ) -> List[CrossFloorLinkRecord]:
        links: List[CrossFloorLinkRecord] = []
        for edge in graph_payload.get("edges", []):
            if not isinstance(edge, dict):
                continue
            attributes = edge.get("attributes")
            if not isinstance(attributes, dict):
                continue
            if not attributes.get("is_cross_floor"):
                continue
            if attributes.get(CROSS_FLOOR_GENERATED_FLAG):
                continue
            connector_type = _normalize_connector_type_label(
                attributes.get("connector_type") or attributes.get("connectorType")
            )
            if not connector_type:
                continue
            source_request_id = str(
                attributes.get("source_request_id") or attributes.get("sourceRequestId") or request_id or ""
            ).strip()
            source_node_id = str(
                attributes.get("source_node_id") or attributes.get("sourceNodeId") or edge.get("source") or ""
            ).strip()
            target_request_id = str(attributes.get("target_request_id") or attributes.get("targetRequestId") or "").strip()
            target_node_id = str(attributes.get("target_node_id") or attributes.get("targetNodeId") or "").strip()
            if not source_request_id or not source_node_id or not target_request_id or not target_node_id:
                continue
            source_label = attributes.get("source_node_label") or attributes.get("sourceNodeLabel") or source_node_id
            target_label = attributes.get("target_node_label") or attributes.get("targetNodeLabel") or target_node_id
            endpoint_a = CrossFloorLinkEndpoint(
                request_id=source_request_id,
                node_id=source_node_id,
                label=str(source_label) if source_label is not None else source_node_id,
            )
            endpoint_b = CrossFloorLinkEndpoint(
                request_id=target_request_id,
                node_id=target_node_id,
                label=str(target_label) if target_label is not None else target_node_id,
            )
            if endpoint_a.request_id == endpoint_b.request_id and endpoint_a.node_id == endpoint_b.node_id:
                continue
            links.append(CrossFloorLinkRecord(connector_type=connector_type, endpoint_a=endpoint_a, endpoint_b=endpoint_b))
        return links

    def _collect_cross_floor_links_from_records(self, records: List[StoredGraphRecord]) -> List[CrossFloorLinkRecord]:
        seen: set = set()
        links: List[CrossFloorLinkRecord] = []
        for record in records:
            for link in self._extract_user_defined_cross_floor_links(record.request_id, record.graph):
                key = link.key()
                if key in seen:
                    continue
                seen.add(key)
                links.append(link)
        return self._enforce_single_link_per_floor(links)

    @staticmethod
    def _is_cross_floor_edge(edge: dict) -> bool:
        if not isinstance(edge, dict):
            return False
        attributes = edge.get("attributes")
        if not isinstance(attributes, dict):
            return False
        return bool(attributes.get("is_cross_floor"))

    def _compute_max_non_cross_edge_weight(self, records: List[StoredGraphRecord]) -> float:
        max_weight = 0.0
        for record in records:
            for edge in record.graph.get("edges", []):
                if self._is_cross_floor_edge(edge):
                    continue
                weight = edge.get("weight")
                if weight is None:
                    continue
                try:
                    weight_value = float(weight)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(weight_value):
                    max_weight = max(max_weight, weight_value)
        return max_weight

    def _collect_cross_floor_adjacency_from_links(
        self,
        records: List[StoredGraphRecord],
        links: List[CrossFloorLinkRecord],
    ) -> Tuple[
        Dict[str, Dict[CrossFloorConnector, set]],
        Dict[CrossFloorConnector, CrossFloorConnectorInfo],
    ]:
        adjacency: Dict[str, Dict[CrossFloorConnector, set]] = defaultdict(lambda: defaultdict(set))
        connector_info: Dict[CrossFloorConnector, CrossFloorConnectorInfo] = {}
        record_map = {record.request_id: record for record in records}
        for link in links:
            endpoint_a, endpoint_b = link.sorted_endpoints()
            if endpoint_a.request_id not in record_map or endpoint_b.request_id not in record_map:
                continue
            if endpoint_a.request_id == endpoint_b.request_id and endpoint_a.node_id == endpoint_b.node_id:
                continue
            connector_type = _normalize_connector_type_label(link.connector_type)
            if connector_type is None:
                raise ValueError("교차층 연결 타입은 stair 또는 elevator 여야 합니다.")
            source_kind = self._resolve_connector_kind(record_map, endpoint_a.request_id, endpoint_a.node_id)
            target_kind = self._resolve_connector_kind(record_map, endpoint_b.request_id, endpoint_b.node_id)
            if source_kind != connector_type or target_kind != connector_type:
                raise ValueError(
                    f"교차층 연결 { _format_endpoint_label(endpoint_a) } ↔ { _format_endpoint_label(endpoint_b) } 의 노드 타입이 "
                    "연결 유형(stair/elevator)과 일치하지 않습니다."
                )
            if source_kind != target_kind:
                raise ValueError(
                    f"교차층 연결 { _format_endpoint_label(endpoint_a) } ↔ { _format_endpoint_label(endpoint_b) } 은(는) "
                    "서로 다른 유형의 노드를 연결할 수 없습니다."
                )
            source_key = CrossFloorConnector(request_id=endpoint_a.request_id, node_id=endpoint_a.node_id)
            target_key = CrossFloorConnector(request_id=endpoint_b.request_id, node_id=endpoint_b.node_id)
            source_info = connector_info.get(source_key)
            if source_info and source_info.connector_type != connector_type:
                continue
            target_info = connector_info.get(target_key)
            if target_info and target_info.connector_type != connector_type:
                continue
            if source_key not in connector_info:
                connector_info[source_key] = CrossFloorConnectorInfo(
                    connector_type=connector_type,
                    label=endpoint_a.normalized_label(),
                )
            if target_key not in connector_info:
                connector_info[target_key] = CrossFloorConnectorInfo(
                    connector_type=connector_type,
                    label=endpoint_b.normalized_label(),
                )
            adjacency[connector_type][source_key].add(target_key)
            adjacency[connector_type][target_key].add(source_key)
        return adjacency, connector_info

    def _collect_cross_floor_adjacency_from_records(
        self, records: List[StoredGraphRecord]
    ) -> Tuple[
        Dict[str, Dict[CrossFloorConnector, set]],
        Dict[CrossFloorConnector, CrossFloorConnectorInfo],
    ]:
        adjacency: Dict[str, Dict[CrossFloorConnector, set]] = defaultdict(lambda: defaultdict(set))
        connector_info: Dict[CrossFloorConnector, CrossFloorConnectorInfo] = {}
        record_map = {record.request_id: record for record in records}
        for record in records:
            request_id = record.request_id
            for edge in record.graph.get("edges", []):
                if not self._is_cross_floor_edge(edge):
                    continue
                attributes = edge.get("attributes") or {}
                if attributes.get(CROSS_FLOOR_GENERATED_FLAG):
                    continue
                connector_type = _normalize_connector_type_label(attributes.get("connector_type") or attributes.get("connectorType"))
                if not connector_type:
                    continue
                source_node_id = str(
                    attributes.get("source_node_id")
                    or attributes.get("sourceNodeId")
                    or edge.get("source")
                    or ""
                ).strip()
                target_request_id = str(
                    attributes.get("target_request_id") or attributes.get("targetRequestId") or ""
                ).strip()
                target_node_id = str(attributes.get("target_node_id") or attributes.get("targetNodeId") or "").strip()
                if not source_node_id or not target_request_id or not target_node_id:
                    continue
                if target_request_id not in record_map:
                    continue
                source_kind = self._resolve_connector_kind(record_map, request_id, source_node_id)
                target_kind = self._resolve_connector_kind(record_map, target_request_id, target_node_id)
                if source_kind != connector_type or target_kind != connector_type:
                    raise ValueError(
                        f"교차층 연결 {request_id}:{source_node_id} ↔ {target_request_id}:{target_node_id} 의 노드 타입이 "
                        "연결 유형(stair/elevator)과 일치하지 않습니다."
                    )
                source_key = CrossFloorConnector(request_id=request_id, node_id=source_node_id)
                target_key = CrossFloorConnector(request_id=target_request_id, node_id=target_node_id)
                source_label = str(
                    attributes.get("source_node_label") or attributes.get("sourceNodeLabel") or source_node_id
                )
                target_label = str(
                    attributes.get("target_node_label") or attributes.get("targetNodeLabel") or target_node_id
                )
                existing_source = connector_info.get(source_key)
                if existing_source and existing_source.connector_type != connector_type:
                    continue
                existing_target = connector_info.get(target_key)
                if existing_target and existing_target.connector_type != connector_type:
                    continue
                if source_key not in connector_info:
                    connector_info[source_key] = CrossFloorConnectorInfo(connector_type=connector_type, label=source_label)
                if target_key not in connector_info:
                    connector_info[target_key] = CrossFloorConnectorInfo(connector_type=connector_type, label=target_label)
                adjacency[connector_type][source_key].add(target_key)
                adjacency[connector_type][target_key].add(source_key)
        return adjacency, connector_info

    def _build_cross_floor_components(
        self, adjacency: Dict[str, Dict[CrossFloorConnector, set]]
    ) -> Dict[str, List[set]]:
        components: Dict[str, List[set]] = {}
        for connector_type, neighbor_map in adjacency.items():
            seen: set = set()
            resolved_components: List[set] = []
            for node in neighbor_map:
                if node in seen:
                    continue
                stack = [node]
                component: set = set()
                while stack:
                    current = stack.pop()
                    if current in seen:
                        continue
                    seen.add(current)
                    component.add(current)
                    for neighbor in neighbor_map.get(current, []):
                        if neighbor not in seen:
                            stack.append(neighbor)
                if len(component) >= 2:
                    resolved_components.append(component)
            if resolved_components:
                components[connector_type] = resolved_components
        return components

    def _build_base_cross_floor_edges(
        self,
        links: List[CrossFloorLinkRecord],
        cross_floor_weight: float,
        available_requests: set,
    ) -> Tuple[Dict[str, List[dict]], set]:
        base_edges: Dict[str, List[dict]] = defaultdict(list)
        pair_keys: set = set()
        for link in links:
            endpoint_a, endpoint_b = link.sorted_endpoints()
            if endpoint_a.request_id not in available_requests or endpoint_b.request_id not in available_requests:
                continue
            for source, target in ((endpoint_a, endpoint_b), (endpoint_b, endpoint_a)):
                pair_keys.add((source.request_id, source.node_id, target.request_id, target.node_id))
                attributes = {
                    "is_cross_floor": True,
                    "connector_type": link.connector_type,
                    "source_request_id": source.request_id,
                    "source_node_id": source.node_id,
                    "source_node_label": source.normalized_label(),
                    "target_request_id": target.request_id,
                    "target_node_id": target.node_id,
                    "target_node_label": target.normalized_label(),
                    "is_base_cross_floor": True,
                }
                base_edges[source.request_id].append(
                    {
                        "source": source.node_id,
                        "target": f"cross_{target.request_id}_{target.node_id}",
                        "weight": float(cross_floor_weight),
                        "attributes": attributes,
                    }
                )
        return base_edges, pair_keys

    def _build_cross_floor_edge_map(
        self,
        components: Dict[str, List[set]],
        connector_info: Dict[CrossFloorConnector, CrossFloorConnectorInfo],
        cross_floor_weight: float,
        links: List[CrossFloorLinkRecord],
        records: List[StoredGraphRecord],
    ) -> Dict[str, List[dict]]:
        available_requests = {record.request_id for record in records}
        edges_by_request, base_pair_keys = self._build_base_cross_floor_edges(
            links, cross_floor_weight, available_requests
        )
        for connector_type, component_list in components.items():
            for component in component_list:
                sorted_nodes = sorted(component, key=lambda item: (item.request_id, item.node_id))
                for source in sorted_nodes:
                    source_info = connector_info.get(source)
                    if source_info is None:
                        continue
                    for target in sorted_nodes:
                        if source == target:
                            continue
                        if (source.request_id, source.node_id, target.request_id, target.node_id) in base_pair_keys:
                            continue
                        target_info = connector_info.get(target)
                        if target_info is None:
                            continue
                        attributes = {
                            "is_cross_floor": True,
                            "connector_type": connector_type,
                            "source_request_id": source.request_id,
                            "source_node_id": source.node_id,
                            "source_node_label": source_info.label,
                            "target_request_id": target.request_id,
                            "target_node_id": target.node_id,
                            "target_node_label": target_info.label,
                            CROSS_FLOOR_GENERATED_FLAG: True,
                        }
                        edges_by_request[source.request_id].append(
                            {
                                "source": source.node_id,
                                "target": f"cross_{target.request_id}_{target.node_id}",
                                "weight": float(cross_floor_weight),
                                "attributes": attributes,
                            }
                        )
        for edge_list in edges_by_request.values():
            edge_list.sort(
                key=lambda edge: (
                    edge.get("source") or "",
                    edge.get("attributes", {}).get("target_request_id") or "",
                    edge.get("attributes", {}).get("target_node_id") or "",
                    edge.get("attributes", {}).get("is_base_cross_floor") is not True,
                )
            )
        return edges_by_request

    def _apply_cross_floor_edge_update(self, record: StoredGraphRecord, cross_edges: List[dict]) -> bool:
        existing_edges = list(record.graph.get("edges", []))
        non_cross_edges = [edge for edge in existing_edges if not self._is_cross_floor_edge(edge)]
        next_edges = non_cross_edges + cross_edges
        if len(existing_edges) == len(next_edges) and all(a == b for a, b in zip(existing_edges, next_edges)):
            return False
        record.graph["edges"] = next_edges
        record.graph_path.write_text(json.dumps(record.graph, ensure_ascii=False, indent=2), encoding="utf-8")
        metadata = record.metadata or {}
        metadata.setdefault("request_id", record.request_id)
        metadata["graph_summary"] = {
            "nodes": len(record.graph.get("nodes", [])),
            "edges": len(next_edges),
        }
        metadata["updated_at"] = datetime.now().isoformat(timespec="seconds")
        record.metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        record.metadata = metadata
        return True

    def _synchronize_cross_floor_connections(self, user_root: Path) -> None:
        records = self._load_all_graph_records(user_root)
        if not records:
            return
        links = self._load_cross_floor_links(user_root)
        if links:
            self._validate_cross_floor_link_uniqueness(links)
            adjacency, connector_info = self._collect_cross_floor_adjacency_from_links(records, links)
        else:
            links = self._collect_cross_floor_links_from_records(records)
            if links:
                self._validate_cross_floor_link_uniqueness(links)
                self._save_cross_floor_links(user_root, links)
            adjacency, connector_info = self._collect_cross_floor_adjacency_from_records(records)
        if not connector_info:
            # 교차층 정보가 없으면 기존 엣지를 제거한다.
            for record in records:
                self._apply_cross_floor_edge_update(record, [])
            return
        components = self._build_cross_floor_components(adjacency)
        base_weight = self._compute_max_non_cross_edge_weight(records)
        cross_weight = (base_weight if math.isfinite(base_weight) else 0.0) + CROSS_FLOOR_WEIGHT_MARGIN
        cross_edges_by_request = self._build_cross_floor_edge_map(
            components,
            connector_info,
            cross_weight,
            links,
            records,
        )
        for record in records:
            cross_edges = cross_edges_by_request.get(record.request_id, [])
            self._apply_cross_floor_edge_update(record, cross_edges)

    def _initialize_result_dir(self, user_id: Optional[str], request_id: Optional[str]) -> ResultDirectoryContext:
        """요청 ID에 해당하는 결과 디렉터리를 생성하고 이전 버전을 보관한다."""
        resolved_user_id, user_root = self._resolve_user_root(user_id, create=True)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        provided_request_id = (request_id or "").strip()
        active_request_id = provided_request_id or f"{timestamp}-{uuid4().hex[:6]}"
        result_dir = user_root / active_request_id
        existing_dir = self._get_existing_result_dir(user_root, active_request_id)
        existing_metadata: Optional[dict] = None
        previous_image_filename: Optional[str] = None
        previous_image_mime: Optional[str] = None
        if existing_dir is not None:
            existing_metadata = self._load_existing_metadata(existing_dir)
            if existing_metadata:
                previous_image_filename = existing_metadata.get("stored_image_path")
                previous_image_mime = existing_metadata.get("stored_image_mime")
            self._archive_previous_result(
                user_root=user_root,
                request_id=active_request_id,
                result_dir=existing_dir,
                metadata=existing_metadata,
            )

        result_dir.mkdir(parents=True, exist_ok=True)

        return ResultDirectoryContext(
            user_id=resolved_user_id,
            user_root=user_root,
            request_id=active_request_id,
            result_dir=result_dir,
            existing_dir=existing_dir,
            existing_metadata=existing_metadata,
            previous_image_filename=previous_image_filename,
            previous_image_mime=previous_image_mime,
        )

    def _store_image_payload(
        self,
        *,
        image_data_url: Optional[str],
        result_dir: Path,
        existing_dir: Optional[Path],
        previous_image_filename: Optional[str],
        previous_image_mime: Optional[str],
    ) -> Tuple[Optional[str], Optional[str]]:
        """이미지 data URL을 디스크에 저장하거나 기존 이미지를 재사용한다."""
        stored_image_filename: Optional[str] = None
        stored_image_mime: Optional[str] = None
        if image_data_url:
            try:
                image_bytes, image_mime = _decode_image_data_url(image_data_url)
                extension = mimetypes.guess_extension(image_mime) or ".png"
                stored_image_filename = f"source_image{extension}"
                image_path = result_dir / stored_image_filename
                image_path.write_bytes(image_bytes)
                stored_image_mime = image_mime
            except ValueError as exc:
                print(f"Failed to decode incoming image data: {exc}", flush=True)
                stored_image_filename = None
                stored_image_mime = None

        if not stored_image_filename and previous_image_filename:
            source_dir = existing_dir or result_dir
            legacy_image_path = source_dir / previous_image_filename
            if legacy_image_path.exists():
                destination_path = result_dir / previous_image_filename
                try:
                    if legacy_image_path.resolve() != destination_path.resolve():
                        shutil.copy2(legacy_image_path, destination_path)
                except OSError:
                    pass
                else:
                    stored_image_filename = previous_image_filename
                    stored_image_mime = previous_image_mime

        return stored_image_filename, stored_image_mime

    @staticmethod
    def _decode_mask_data_url(mask_url: str, width: int, height: int) -> np.ndarray:
        """data URL PNG를 불(bool) 마스크 배열로 변환한다."""
        if not mask_url:
            raise ValueError("mask data url is empty")
        image_bytes, _ = _decode_image_data_url(mask_url)
        byte_array = np.frombuffer(image_bytes, dtype=np.uint8)
        decoded = cv2.imdecode(byte_array, cv2.IMREAD_GRAYSCALE)
        if decoded is None:
            raise ValueError("PNG 마스크 디코딩에 실패했습니다.")
        if decoded.shape[0] != height or decoded.shape[1] != width:
            decoded = cv2.resize(decoded, (width, height), interpolation=cv2.INTER_NEAREST)
        return decoded > 0

    @staticmethod
    def _encode_bitmask(mask: np.ndarray) -> Dict[str, Any]:
        """불/바이너리 마스크를 비트팩+Base64 텍스트로 직렬화한다."""
        bool_mask = np.ascontiguousarray(mask.astype(bool))
        flat = bool_mask.reshape(-1)
        packed = np.packbits(flat, bitorder="little")
        return {
            "encoding": "bitpack-base64",
            "shape": [int(bool_mask.shape[0]), int(bool_mask.shape[1])],
            "length": int(flat.size),
            "data": base64.b64encode(packed.tobytes()).decode("ascii"),
        }

    @staticmethod
    def _decode_bitmask(payload: Dict[str, Any], width: int, height: int) -> np.ndarray:
        """비트팩+Base64 텍스트를 불 마스크 배열로 복원한다."""
        if not payload or "data" not in payload:
            raise ValueError("bitmask payload is missing data")
        encoding = payload.get("encoding") or payload.get("format")
        if encoding and encoding not in ("bitpack-base64", "bitmask-base64"):
            raise ValueError(f"unsupported bitmask encoding: {encoding}")
        raw_shape = payload.get("shape") or []
        if isinstance(raw_shape, (list, tuple)) and len(raw_shape) == 2:
            target_shape = (int(raw_shape[0]), int(raw_shape[1]))
        else:
            target_shape = (height, width)
        expected_length = int(payload.get("length") or (target_shape[0] * target_shape[1]))
        try:
            decoded_bytes = base64.b64decode(payload["data"])
        except (ValueError, binascii.Error) as exc:  # type: ignore[name-defined]
            raise ValueError("invalid base64 bitmask payload") from exc
        bit_array = np.unpackbits(np.frombuffer(decoded_bytes, dtype=np.uint8), bitorder="little")
        if bit_array.size < expected_length:
            raise ValueError("bitmask payload is shorter than expected")
        trimmed = bit_array[:expected_length]
        reshaped = trimmed.reshape(target_shape)
        if reshaped.shape != (height, width):
            reshaped = cv2.resize(reshaped.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST) > 0
            return reshaped
        return reshaped.astype(bool)

    def _decode_preview_bundle(self, payload: Dict[str, Any], width: int, height: int) -> StageOneArtifacts:
        artifact_payload = payload.get("artifact_bundle") or payload.get("artifactBundle") or {}
        masks_payload = artifact_payload.get("masks") if isinstance(artifact_payload, dict) else {}

        def _decode_artifact_mask(key: str) -> Optional[np.ndarray]:
            if not isinstance(masks_payload, dict):
                return None
            candidate = masks_payload.get(key) or masks_payload.get(f"{key[0].lower()}{key[1:]}")
            if not candidate:
                return None
            try:
                return self._decode_bitmask(candidate, width, height)
            except ValueError:
                return None

        def _decode_mask_with_fallback(key: str, artifact_keys: Sequence[str]) -> np.ndarray:
            for artifact_key in artifact_keys:
                artifact_mask = _decode_artifact_mask(artifact_key)
                if artifact_mask is not None:
                    return artifact_mask
            camel = "".join([token.capitalize() if index > 0 else token for index, token in enumerate(key.split("_"))])
            return self._decode_mask_data_url(payload.get(key) or payload.get(camel), width, height)

        free_space_mask = _decode_mask_with_fallback("free_space_mask", ("freeSpace",))
        door_mask = _decode_mask_with_fallback("door_mask", ("door",))
        room_mask = _decode_mask_with_fallback("room_mask", ("room",))
        wall_mask = _decode_mask_with_fallback("wall_mask", ("wall",))

        skeleton_payload = artifact_payload.get("skeleton") if isinstance(artifact_payload, dict) else None
        skeleton = None
        if skeleton_payload:
            try:
                skeleton = self._decode_bitmask(skeleton_payload, width, height)
            except ValueError:
                skeleton = None

        door_midpoints = None
        if isinstance(artifact_payload, dict):
            door_midpoints = artifact_payload.get("door_midpoints") or artifact_payload.get("doorMidpoints")

        return StageOneArtifacts(
            free_space=free_space_mask,
            door_mask=door_mask,
            room_mask=room_mask,
            wall_mask=wall_mask,
            skeleton=skeleton,
            door_midpoints=door_midpoints,
        )

    def _load_preview_bundle_from_metadata(
        self,
        metadata: Optional[Dict[str, Any]],
        width: int,
        height: int,
    ) -> Tuple[Optional[StageOneArtifacts], Optional[Dict[str, Any]]]:
        """metadata.json 안에 직렬화된 미리보기 묶음이 있다면 역직렬화해 돌려준다."""
        if not metadata:
            return None, None
        preview_meta = metadata.get("preview")
        if not isinstance(preview_meta, dict):
            return None, None
        try:
            bundle = self._decode_preview_bundle(preview_meta, width, height)
        except ValueError as exc:
            print(f"경고: 저장된 미리보기 데이터를 재사용할 수 없습니다: {exc}")
            return None, None
        config_payload = preview_meta.get("config")
        return bundle, config_payload

    def _serialize_preview_bundle(
        self,
        bundle: FreeSpaceMaskBundle,
        image_width: int,
        image_height: int,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "imageSize": {"width": image_width, "height": image_height},
            "freeSpaceRatio": float(np.mean(bundle.free_space)),
        }
        artifact_payload = self._serialize_stage_one_artifacts(bundle, image_width, image_height)
        if artifact_payload is not None:
            payload["artifactBundle"] = artifact_payload
        return payload

    def _serialize_stage_one_artifacts(
        self,
        bundle: FreeSpaceMaskBundle,
        image_width: int,
        image_height: int,
    ) -> Optional[Dict[str, Any]]:
        if not isinstance(bundle, StageOneArtifacts):
            return None
        artifact_payload: Dict[str, Any] = {
            "version": 1,
            "width": int(image_width),
            "height": int(image_height),
            "masks": {
                "freeSpace": self._encode_bitmask(bundle.free_space),
                "door": self._encode_bitmask(bundle.door_mask),
                "room": self._encode_bitmask(bundle.room_mask),
                "wall": self._encode_bitmask(bundle.wall_mask),
            },
        }
        if bundle.skeleton is not None:
            artifact_payload["skeleton"] = self._encode_bitmask(bundle.skeleton)
        door_midpoints_payload: Optional[List[Dict[str, Any]]] = None
        if bundle.door_midpoints:
            door_midpoints_payload = []
            for entry in bundle.door_midpoints:
                door_id = entry.get("door_id") or entry.get("doorId")
                if door_id is None:
                    continue
                try:
                    door_id_int = int(door_id)
                except (TypeError, ValueError):
                    continue
                midpoint_items: List[Dict[str, Any]] = []
                for info in entry.get("midpoints") or []:
                    rc = info.get("midpoint_rc") or info.get("midpointRc")
                    xy = info.get("midpoint_xy") or info.get("midpointXy")
                    rooms = info.get("room_ids") or info.get("roomIds")
                    payload_item: Dict[str, Any] = {}
                    if isinstance(rc, (list, tuple)) and len(rc) == 2:
                        payload_item["midpointRc"] = [float(rc[0]), float(rc[1])]
                    if isinstance(xy, (list, tuple)) and len(xy) == 2:
                        payload_item["midpointXy"] = [float(xy[0]), float(xy[1])]
                    if rooms:
                        payload_item["roomIds"] = [int(val) for val in rooms if isinstance(val, (int, float))]
                    if payload_item:
                        midpoint_items.append(payload_item)
                door_payload: Dict[str, Any] = {"doorId": door_id_int, "midpoints": midpoint_items}
                door_midpoints_payload.append(door_payload)
        if door_midpoints_payload:
            artifact_payload["doorMidpoints"] = door_midpoints_payload
        return artifact_payload

    @staticmethod
    def _serialize_pipeline_config(config: CorridorPipelineConfig) -> Dict[str, Any]:
        return {
            "doorProbeDistance": int(config.door_probe_distance),
            "morphOpenKernel": [int(config.morph_open_kernel[0]), int(config.morph_open_kernel[1])],
            "morphOpenIterations": int(config.morph_open_iterations),
            "doorTouchKernel": [int(config.door_touch_kernel[0]), int(config.door_touch_kernel[1])],
        }

    @staticmethod
    def _build_corridor_config(config_payload: Optional[Dict[str, Any]]) -> CorridorPipelineConfig:
        if not config_payload:
            return CorridorPipelineConfig()
        base = CorridorPipelineConfig()
        door_probe = config_payload.get("doorProbeDistance") or config_payload.get("door_probe_distance")
        morph_kernel = config_payload.get("morphOpenKernel") or config_payload.get("morph_open_kernel")
        morph_iters = config_payload.get("morphOpenIterations") or config_payload.get("morph_open_iterations")
        door_touch = config_payload.get("doorTouchKernel") or config_payload.get("door_touch_kernel")
        kernel_tuple = base.morph_open_kernel
        if isinstance(morph_kernel, (list, tuple)) and len(morph_kernel) == 2:
            kernel_tuple = (max(1, int(morph_kernel[0])), max(1, int(morph_kernel[1])))
        door_touch_tuple = base.door_touch_kernel
        if isinstance(door_touch, (list, tuple)) and len(door_touch) == 2:
            door_touch_tuple = (max(1, int(door_touch[0])), max(1, int(door_touch[1])))
        return CorridorPipelineConfig(
            door_probe_distance=max(1, int(door_probe)) if door_probe is not None else base.door_probe_distance,
            morph_open_kernel=kernel_tuple,
            morph_open_iterations=max(1, int(morph_iters)) if morph_iters is not None else base.morph_open_iterations,
            door_touch_kernel=door_touch_tuple,
        )

    @staticmethod
    @staticmethod
    def _extract_preview_image_bytes(payload: Dict[str, Any]) -> Optional[Dict[str, bytes]]:
        """data URL 묶음에서 PNG 바이트를 추출한다."""
        if not payload:
            return None

        def _resolve(key: str) -> Optional[str]:
            return payload.get(key) or payload.get("".join([part.capitalize() if index > 0 else part for index, part in enumerate(key.split("_"))]))

        keys = {
            "freeSpace": _resolve("free_space_mask"),
            "door": _resolve("door_mask"),
            "room": _resolve("room_mask"),
            "wall": _resolve("wall_mask"),
        }
        extracted: Dict[str, bytes] = {}
        for name, data_url in keys.items():
            if not data_url:
                continue
            try:
                image_bytes, _ = _decode_image_data_url(data_url)
            except ValueError:
                continue
            extracted[name] = image_bytes
        return extracted or None

    @staticmethod
    def _save_preview_images(preview: FreeSpaceMaskBundle, output_dir: Path, raw_images: Optional[Dict[str, bytes]] = None) -> Dict[str, str]:
        """(비활성화) 미리보기 마스크를 파일로 저장하던 로직은 현재 사용하지 않는다."""
        # output_dir.mkdir(parents=True, exist_ok=True)
        # def _write(mask: np.ndarray, name: str, raw_key: str) -> str:
        #     path = output_dir / name
        #     if raw_images and raw_key in raw_images:
        #         path.write_bytes(raw_images[raw_key])
        #     else:
        #         cv2.imwrite(str(path), mask.astype(np.uint8) * 255)
        #     return str(path)
        #
        # return {
        #     "freeSpace": _write(preview.free_space, "free_space.png", "freeSpace"),
        #     "door": _write(preview.door_mask, "door_mask.png", "door"),
        #     "room": _write(preview.room_mask, "room_mask.png", "room"),
        #     "wall": _write(preview.wall_mask, "wall_mask.png", "wall"),
        # }
        return {}

    @staticmethod
    def _validate_corridor_graph(serialized_graph: dict) -> None:
        nodes = serialized_graph.get("nodes") or []
        edges = serialized_graph.get("edges") or []

        corridor_ids = {
            node.get("id")
            for node in nodes
            if node.get("type") == "corridor"
        }
        if not corridor_ids:
            raise ValueError(
                "복도 영역을 찾을 수 없습니다. 1단계에서 복도 라인을 명확히 그려 주세요."
            )

        def _is_door_like(node_type: Optional[str]) -> bool:
            if not isinstance(node_type, str):
                return False
            return node_type.startswith("door")

        door_like_ids = {
            node.get("id")
            for node in nodes
            if _is_door_like(node.get("type"))
        }

        has_corridor_door_link = any(
            (edge.get("source") in corridor_ids and edge.get("target") in door_like_ids)
            or (edge.get("target") in corridor_ids and edge.get("source") in door_like_ids)
            for edge in edges
        )

        if not has_corridor_door_link:
            raise ValueError(
                "복도 영역이 문과 연결되지 않았습니다. 1단계에서 벽과 문 위치를 다시 확인해 주세요."
            )

    def generate_free_space_preview(
        self,
        image_width: int,
        image_height: int,
        object_detection_text: Optional[str] = None,
        wall_text: Optional[str] = None,
        door_text: Optional[str] = None,
        *,
        config: Optional[CorridorPipelineConfig] = None,
    ) -> dict:
        """객체 텍스트를 기반으로 자유 공간 마스크 미리보기를 생성한다."""
        object_detection_text = object_detection_text or ""
        wall_text = wall_text or ""
        door_text = door_text or ""

        parsed_objects, _ = _load_objects_from_texts(
            object_detection_text,
            wall_text,
            door_text,
            image_width=image_width,
            image_height=image_height,
        )

        pipeline = CorridorPipeline(
            parsed_objects,
            width=image_width,
            height=image_height,
            config=config,
        )
        artifacts = pipeline.generate_stage_one_artifacts()

        active_config = pipeline.config
        preview_payload = self._serialize_preview_bundle(artifacts, image_width, image_height)
        preview_payload["config"] = {
            "doorProbeDistance": int(active_config.door_probe_distance),
            "morphOpenKernel": list(active_config.morph_open_kernel),
            "morphOpenIterations": int(active_config.morph_open_iterations),
            "doorTouchKernel": list(active_config.door_touch_kernel),
        }
        return preview_payload

    def process(
        self,
        image_width: int,
        image_height: int,
        class_names: Optional[List[str]] = None,
        source_image_path: Optional[str] = None,
        object_detection_text: Optional[str] = None,
        wall_text: Optional[str] = None,
        wall_base_text: Optional[str] = None,
        door_text: Optional[str] = None,
        floor_label: Optional[str] = None,
        floor_value: Optional[str] = None,
        scale_reference: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        image_data_url: Optional[str] = None,
        request_id: Optional[str] = None,
        skip_graph: bool = False,
        free_space_preview: Optional[Dict[str, Any]] = None,
    ) -> dict:
        """파서/그래프 생성기를 실행해 객체·그래프·메타데이터를 생성한다."""
        class_names = class_names or DEFAULT_CLASS_NAMES

        parser = FloorPlanParser()
        normalized_scale_reference = _normalize_scale_reference_payload(scale_reference)

        print("Processing using object-detection/wall/door texts", flush=True)
        object_detection_text = object_detection_text or ""
        wall_text = wall_text or ""
        wall_base_text = wall_base_text or wall_text
        door_text = door_text or ""
        parsed_objects, annotations_list = _load_objects_from_texts(
            object_detection_text=object_detection_text,
            wall_text=wall_text,
            door_text=door_text,
            image_width=image_width,
            image_height=image_height,
        )

        parser.objects = parsed_objects
        parser._extend_doors_along_walls()
        parser.annotate_room_door_connections()
        parsed_objects = parser.objects

        serialized_objects = _serialize_objects(parsed_objects)

        serialized_graph: Dict[str, List[dict]]
        graph_summary: Optional[Dict[str, int]]

        context = self._initialize_result_dir(user_id, request_id)
        active_request_id = context.request_id
        result_dir = context.result_dir
        existing_dir = context.existing_dir
        existing_metadata = context.existing_metadata
        previous_image_filename = context.previous_image_filename
        previous_image_mime = context.previous_image_mime

        if normalized_scale_reference is None and existing_metadata:
            normalized_scale_reference = _normalize_scale_reference_payload(
                (existing_metadata.get("scale_reference") if isinstance(existing_metadata, dict) else None)
                or (existing_metadata.get("scaleReference") if isinstance(existing_metadata, dict) else None)
            )
        scale_reference_payload = dict(normalized_scale_reference) if normalized_scale_reference else None

        debug_dir: Optional[Path] = None
        # 디버그용 그래프 이미지를 저장하지 않도록 비활성화
        # if not skip_graph and STEP_TWO_DEBUG_ENABLED:
        #     debug_dir_name = STEP_TWO_DEBUG_SUBDIR.strip() or "graph_debug"
        #     debug_dir = result_dir / debug_dir_name

        preview_input = free_space_preview if isinstance(free_space_preview, dict) else None
        preview_config_payload: Optional[Dict[str, Any]] = None
        preview_bundle: Optional[StageOneArtifacts] = None
        preview_raw_images: Optional[Dict[str, bytes]] = None
        if preview_input:
            preview_config_payload = preview_input.get("config")
            preview_raw_images = self._extract_preview_image_bytes(preview_input)
            try:
                preview_bundle = self._decode_preview_bundle(preview_input, image_width, image_height)
            except ValueError as exc:
                print(f"경고: 전달된 미리보기 데이터를 사용할 수 없습니다: {exc}")
                preview_bundle = None

        if preview_bundle is None:
            preview_payload = self.generate_free_space_preview(
                image_width=image_width,
                image_height=image_height,
                object_detection_text=object_detection_text,
                wall_text=wall_text,
                door_text=door_text,
                config=None,
            )
            preview_config_payload = preview_payload.get("config")
            preview_raw_images = self._extract_preview_image_bytes(preview_payload)
            preview_bundle = self._decode_preview_bundle(preview_payload, image_width, image_height)

        pipeline_config = self._build_corridor_config(preview_config_payload)

        if skip_graph:
            serialized_graph = {"nodes": [], "edges": []}
            graph_summary = None
        else:
            nav_builder = CorridorPipeline(
                parsed_objects,
                width=image_width,
                height=image_height,
                debug_dir=debug_dir,
                config=pipeline_config,
                precomputed_stage_one=preview_bundle,
            )
            graph = nav_builder.build()
            serialized_graph = _serialize_graph(graph)
            self._validate_corridor_graph(serialized_graph)
            graph_summary = {
                "nodes": graph.number_of_nodes(),
                "edges": graph.number_of_edges(),
            }

        graph_context = _build_graph_context_payload(
            request_id=active_request_id,
            floor_label=floor_label,
            floor_value=floor_value,
            scale_reference=scale_reference_payload,
        )
        if graph_context:
            serialized_graph = {**serialized_graph, **graph_context}

        stored_image_filename, stored_image_mime = self._store_image_payload(
            image_data_url=image_data_url,
            result_dir=result_dir,
            existing_dir=existing_dir,
            previous_image_filename=previous_image_filename,
            previous_image_mime=previous_image_mime,
        )

        previous_floor_label = existing_metadata.get("floor_label") if existing_metadata else None
        previous_floor_value = existing_metadata.get("floor_value") if existing_metadata else None
        floor_label = (floor_label or previous_floor_label or "").strip() or None
        floor_value = (floor_value or previous_floor_value or "").strip() or None

        preview_metadata: Optional[Dict[str, Any]] = None
        if preview_bundle is not None:
            preview_metadata = self._serialize_preview_bundle(preview_bundle, image_width, image_height)
            preview_metadata["config"] = preview_config_payload or self._serialize_pipeline_config(pipeline_config)
            # preview_files = self._save_preview_images(preview_bundle, result_dir / "preview", preview_raw_images)
            # preview_metadata["files"] = preview_files

        annotations_payload = []
        for item in annotations_list:
            payload = {
                "class_id": int(item["class_id"]),
                "x_center": float(item["x_center"]),
                "y_center": float(item["y_center"]),
                "width": float(item["width"]),
                "height": float(item["height"]),
                "identifier": item.get("identifier"),
            }
            confidence = item.get("confidence")
            if confidence is not None:
                payload["confidence"] = float(confidence)
            annotations_payload.append(payload)

        input_payload = {
            "image_size": {"width": image_width, "height": image_height},
            "class_names": class_names,
            "annotations": annotations_payload,
            "texts": {
                "object_detection": object_detection_text,
                "wall": wall_text,
                "wall_base": wall_base_text,
                "door": door_text,
            },
        }
        if source_image_path:
            input_payload["source_image_path"] = source_image_path

        input_path = result_dir / "input_annotations.json"
        objects_path = result_dir / "floorplan_objects.json"
        graph_path = result_dir / self._graph_filename(active_request_id)
        metadata_path = result_dir / "metadata.json"
        object_detection_text_path = result_dir / "rooms.txt"
        wall_filtered_text_path = result_dir / "walls.txt"
        wall_raw_text_path = result_dir / "walls_raw.txt"
        door_text_path = result_dir / "doors.txt"

        object_detection_text_path.write_text(input_payload["texts"]["object_detection"], encoding="utf-8")
        wall_filtered_text_path.write_text(input_payload["texts"]["wall"], encoding="utf-8")
        wall_raw_text_path.write_text(input_payload["texts"]["wall_base"], encoding="utf-8")
        door_text_path.write_text(input_payload["texts"]["door"], encoding="utf-8")

        with input_path.open("w", encoding="utf-8") as f:
            json.dump(input_payload, f, ensure_ascii=False, indent=2)
        with objects_path.open("w", encoding="utf-8") as f:
            json.dump(serialized_objects, f, ensure_ascii=False, indent=2)
        with graph_path.open("w", encoding="utf-8") as f:
            json.dump(serialized_graph, f, ensure_ascii=False, indent=2)

        metadata = {
            "request_id": active_request_id,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "image_size": {"width": image_width, "height": image_height},
            "source_image_path": source_image_path,
            "graph_summary": graph_summary,
            "user_id": context.user_id,
        }
        if floor_label:
            metadata["floor_label"] = floor_label
        if floor_value:
            metadata["floor_value"] = floor_value
        if scale_reference_payload:
            metadata["scale_reference"] = dict(scale_reference_payload)
            metadata["scaleReference"] = dict(scale_reference_payload)
        # if debug_dir is not None:
        #     metadata["graph_debug_dir"] = str(debug_dir)
        if stored_image_filename:
            metadata["stored_image_path"] = stored_image_filename
            if stored_image_mime:
                metadata["stored_image_mime"] = stored_image_mime
            metadata["image_url"] = self._build_image_url(active_request_id, context.user_id)
        if preview_metadata is not None:
            metadata["preview"] = preview_metadata
        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        self._update_floorplan_index(
            user_root=context.user_root,
            request_id=active_request_id,
            result_dir=result_dir,
            metadata=metadata,
            graph_path=graph_path,
        )

        return {
            "request_id": active_request_id,
            "created_at": metadata["created_at"],
            "image_size": metadata["image_size"],
            "class_names": class_names,
            "objects": serialized_objects,
            "graph": serialized_graph,
            "saved_files": {
                "input_annotations": str(input_path),
                "objects": str(objects_path),
                "graph": str(graph_path),
                "metadata": str(metadata_path),
                "object_detection_text": str(object_detection_text_path),
                "wall_text": str(wall_filtered_text_path),
                "wall_base_text": str(wall_raw_text_path),
                "door_text": str(door_text_path),
            },
            "input_annotations": annotations_payload,
            "metadata": {
                **metadata,
                "image_data_url": image_data_url,
            },
        }

    def save_step_one(
        self,
        *,
        image_width: int,
        image_height: int,
        class_names: Optional[List[str]] = None,
        source_image_path: Optional[str] = None,
        object_detection_text: str,
        wall_text: str,
        wall_base_text: Optional[str],
        door_text: str,
        floor_label: Optional[str] = None,
        floor_value: Optional[str] = None,
        scale_reference: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        image_data_url: Optional[str] = None,
        request_id: Optional[str] = None,
        free_space_preview: Optional[Dict[str, Any]] = None,
    ) -> dict:
        """그래프 생성을 건너뛰고 텍스트/미리보기/메타데이터만 저장한다."""
        class_names = class_names or DEFAULT_CLASS_NAMES
        wall_base_text = wall_base_text or wall_text or ""

        _, annotations_list = _load_objects_from_texts(
            object_detection_text=object_detection_text or "",
            wall_text=wall_text or "",
            door_text=door_text or "",
            image_width=image_width,
            image_height=image_height,
        )

        context = self._initialize_result_dir(user_id, request_id)
        normalized_scale_reference = _normalize_scale_reference_payload(scale_reference)
        active_request_id = context.request_id
        result_dir = context.result_dir
        existing_dir = context.existing_dir
        existing_metadata = context.existing_metadata
        previous_image_filename = context.previous_image_filename
        previous_image_mime = context.previous_image_mime

        if normalized_scale_reference is None and isinstance(existing_metadata, dict):
            normalized_scale_reference = _normalize_scale_reference_payload(
                existing_metadata.get("scale_reference") or existing_metadata.get("scaleReference")
            )
        scale_reference_payload = dict(normalized_scale_reference) if normalized_scale_reference else None

        stored_image_filename, stored_image_mime = self._store_image_payload(
            image_data_url=image_data_url,
            result_dir=result_dir,
            existing_dir=existing_dir,
            previous_image_filename=previous_image_filename,
            previous_image_mime=previous_image_mime,
        )

        preview_input = free_space_preview if isinstance(free_space_preview, dict) else None
        preview_config_payload: Optional[Dict[str, Any]] = None
        preview_bundle: Optional[StageOneArtifacts] = None
        if preview_input:
            preview_config_payload = preview_input.get("config")
            try:
                preview_bundle = self._decode_preview_bundle(preview_input, image_width, image_height)
            except ValueError as exc:
                raise ValueError(f"전달된 복도 미리보기 데이터를 해석할 수 없습니다: {exc}") from exc

        if preview_bundle is None and existing_metadata is not None:
            preview_bundle, preview_config_payload = self._load_preview_bundle_from_metadata(
                existing_metadata, image_width, image_height
            )

        if preview_bundle is None:
            raise ValueError("free_space_preview 데이터가 필요합니다.")

        preview_metadata = self._serialize_preview_bundle(preview_bundle, image_width, image_height)
        if preview_config_payload:
            preview_metadata["config"] = preview_config_payload
        elif "config" not in preview_metadata:
            preview_metadata["config"] = self._serialize_pipeline_config(CorridorPipelineConfig())

        annotations_payload = []
        for item in annotations_list:
            payload = {
                "class_id": int(item["class_id"]),
                "x_center": float(item["x_center"]),
                "y_center": float(item["y_center"]),
                "width": float(item["width"]),
                "height": float(item["height"]),
                "identifier": item.get("identifier"),
            }
            confidence = item.get("confidence")
            if confidence is not None:
                payload["confidence"] = float(confidence)
            annotations_payload.append(payload)

        input_payload = {
            "image_size": {"width": image_width, "height": image_height},
            "class_names": class_names,
            "annotations": annotations_payload,
            "texts": {
                "object_detection": object_detection_text or "",
                "wall": wall_text or "",
                "wall_base": wall_base_text or "",
                "door": door_text or "",
            },
        }
        if source_image_path:
            input_payload["source_image_path"] = source_image_path

        object_detection_text_path = result_dir / "rooms.txt"
        wall_filtered_text_path = result_dir / "walls.txt"
        wall_raw_text_path = result_dir / "walls_raw.txt"
        door_text_path = result_dir / "doors.txt"
        input_path = result_dir / "input_annotations.json"

        object_detection_text_path.write_text(input_payload["texts"]["object_detection"], encoding="utf-8")
        wall_filtered_text_path.write_text(input_payload["texts"]["wall"], encoding="utf-8")
        wall_raw_text_path.write_text(input_payload["texts"]["wall_base"], encoding="utf-8")
        door_text_path.write_text(input_payload["texts"]["door"], encoding="utf-8")

        with input_path.open("w", encoding="utf-8") as handle:
            json.dump(input_payload, handle, ensure_ascii=False, indent=2)

        metadata_path = result_dir / "metadata.json"
        metadata = {
            "request_id": active_request_id,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "image_size": {"width": image_width, "height": image_height},
            "source_image_path": source_image_path,
            "graph_summary": None,
            "preview": preview_metadata,
            "user_id": context.user_id,
        }
        if floor_label:
            metadata["floor_label"] = floor_label
        if floor_value:
            metadata["floor_value"] = floor_value
        if scale_reference_payload:
            metadata["scale_reference"] = dict(scale_reference_payload)
            metadata["scaleReference"] = dict(scale_reference_payload)
        if stored_image_filename:
            metadata["stored_image_path"] = stored_image_filename
            if stored_image_mime:
                metadata["stored_image_mime"] = stored_image_mime
            metadata["image_url"] = self._build_image_url(active_request_id, context.user_id)
        with metadata_path.open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, ensure_ascii=False, indent=2)

        lines_count = len([line for line in (wall_text or "").splitlines() if line.strip()])
        doors_count = len([line for line in (door_text or "").splitlines() if line.strip()])

        return {
            "request_id": active_request_id,
            "created_at": metadata["created_at"],
            "image_size": metadata["image_size"],
            "class_names": class_names,
            "annotation_counts": {
                "boxes": len(annotations_payload),
                "walls": lines_count,
                "doors": doors_count,
            },
            "preview": preview_metadata,
            "metadata": {
                **metadata,
                "image_data_url": image_data_url,
            },
        }

    def prepare_graph(
        self,
        request_id: str,
        *,
        user_id: Optional[str] = None,
        free_space_preview: Optional[Dict[str, Any]] = None,
    ) -> dict:
        """저장된 Step1 텍스트를 기반으로 그래프를 다시 생성한다."""
        resolved_user_id, user_root = self._resolve_user_root(user_id, create=False)
        result_dir = user_root / request_id
        if not result_dir.exists():
            raise FileNotFoundError(f"저장된 결과를 찾을 수 없습니다: {request_id}")

        def _safe_read(path: Path) -> str:
            if not path.exists():
                raise FileNotFoundError(f"{path.name} 파일이 존재하지 않습니다.")
            return path.read_text(encoding="utf-8")

        object_detection_text = _safe_read(result_dir / "rooms.txt")
        wall_text = _safe_read(result_dir / "walls.txt")
        wall_base_text = _safe_read(result_dir / "walls_raw.txt")
        door_text = _safe_read(result_dir / "doors.txt")

        metadata = self._load_existing_metadata(result_dir) or {}
        input_path = result_dir / "input_annotations.json"
        try:
            input_payload = json.loads(input_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            input_payload = {}

        image_size = metadata.get("image_size") or input_payload.get("image_size") or {}
        image_width = int(image_size.get("width") or 0)
        image_height = int(image_size.get("height") or 0)
        if image_width <= 0 or image_height <= 0:
            raise ValueError("이미지 크기 정보를 찾을 수 없습니다.")

        class_names = input_payload.get("class_names") or metadata.get("class_names") or DEFAULT_CLASS_NAMES

        preview_payload = free_space_preview or metadata.get("preview")
        scale_reference_value = metadata.get("scale_reference") or metadata.get("scaleReference")
        normalized_scale_reference = _normalize_scale_reference_payload(scale_reference_value)
        if not normalized_scale_reference:
            raise ValueError(
                "저장된 결과에 기준선 정보(scale_reference)가 없어 그래프를 다시 생성할 수 없습니다. 업로드 단계에서 기준선을 다시 지정해 주세요."
            )

        return self.process(
            image_width=image_width,
            image_height=image_height,
            class_names=class_names,
            source_image_path=metadata.get("source_image_path"),
            object_detection_text=object_detection_text,
            wall_text=wall_text,
            wall_base_text=wall_base_text,
            door_text=door_text,
            image_data_url=None,
            user_id=user_id,
            request_id=request_id,
            skip_graph=False,
            free_space_preview=preview_payload,
            scale_reference=normalized_scale_reference,
        )

    def get_result(self, request_id: str, *, user_id: Optional[str] = None) -> dict:
        """저장된 결과 디렉터리에서 객체/그래프/메타데이터를 읽어 반환한다."""
        resolved_user_id, user_root = self._resolve_user_root(user_id, create=False)
        result_dir = user_root / request_id
        if not result_dir.exists():
            raise FileNotFoundError(f"저장된 결과를 찾을 수 없습니다: {request_id}")

        def load_json(path: Path) -> dict:
            with path.open("r", encoding="utf-8") as stream:
                return json.load(stream)

        objects_path = result_dir / "floorplan_objects.json"
        graph_path = self._resolve_graph_path(result_dir, request_id)
        metadata_path = result_dir / "metadata.json"
        input_path = result_dir / "input_annotations.json"
        object_detection_text_path = result_dir / "rooms.txt"
        wall_text_path = result_dir / "walls.txt"
        wall_raw_text_path = result_dir / "walls_raw.txt"
        door_text_path = result_dir / "doors.txt"

        if not graph_path.exists():
            raise FileNotFoundError(f"그래프 파일을 찾을 수 없습니다: {graph_path}")

        objects = load_json(objects_path)
        graph = load_json(graph_path)
        metadata = load_json(metadata_path)

        graph_context = _build_graph_context_payload(
            request_id=metadata.get("request_id") or metadata.get("requestId") or request_id,
            floor_label=metadata.get("floor_label") or metadata.get("floorLabel"),
            floor_value=metadata.get("floor_value") or metadata.get("floorValue"),
            scale_reference=metadata.get("scale_reference") or metadata.get("scaleReference"),
        )
        graph_updated = False
        for key, value in graph_context.items():
            if graph.get(key) != value:
                graph[key] = value
                graph_updated = True
        if graph_updated:
            graph_path.write_text(json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8")
        input_payload = load_json(input_path)

        saved_files = {
            "input_annotations": str(input_path),
            "objects": str(objects_path),
            "graph": str(graph_path),
            "metadata": str(metadata_path),
            "object_detection_text": str(object_detection_text_path),
            "wall_text": str(wall_text_path),
            "wall_base_text": str(wall_raw_text_path),
            "door_text": str(door_text_path),
        }

        request_timestamp = metadata.get("created_at") or datetime.now().isoformat(timespec="seconds")
        image_size = metadata.get("image_size") or input_payload.get("image_size") or {"width": 0, "height": 0}
        class_names = input_payload.get("class_names") or DEFAULT_CLASS_NAMES
        annotations_payload = input_payload.get("annotations", [])

        texts = {
            "object_detection": object_detection_text_path.read_text(encoding="utf-8") if object_detection_text_path.exists() else "",
            "wall": wall_text_path.read_text(encoding="utf-8") if wall_text_path.exists() else "",
            "wall_base": wall_raw_text_path.read_text(encoding="utf-8") if wall_raw_text_path.exists() else "",
            "door": door_text_path.read_text(encoding="utf-8") if door_text_path.exists() else "",
        }

        metadata_changed = False
        if not metadata.get("user_id"):
            metadata["user_id"] = resolved_user_id
            metadata_changed = True

        expected_image_url = metadata.get("image_url") or self._build_image_url(request_id, resolved_user_id)
        if metadata.get("image_url") != expected_image_url:
            metadata["image_url"] = expected_image_url
            metadata_changed = True

        if metadata_changed:
            metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

        metadata_with_texts = dict(metadata)
        metadata_with_texts["texts"] = texts
        if "image_data_url" not in metadata_with_texts:
            stored_image_filename = metadata.get("stored_image_path")
            if stored_image_filename:
                image_path = result_dir / stored_image_filename
                if image_path.exists():
                    mime_type = metadata.get("stored_image_mime") or mimetypes.guess_type(image_path.name)[0] or "image/png"
                    metadata_with_texts["image_data_url"] = _build_data_url(mime_type, image_path.read_bytes())
        metadata_with_texts.setdefault("image_url", expected_image_url)
        metadata_with_texts.setdefault("user_id", metadata.get("user_id") or resolved_user_id)

        return {
            "request_id": metadata.get("request_id", request_id),
            "created_at": request_timestamp,
            "image_size": image_size,
            "class_names": class_names,
            "objects": objects,
            "graph": graph,
            "saved_files": saved_files,
            "input_annotations": annotations_payload,
            "metadata": metadata_with_texts,
        }

    def get_graph_data(self, request_id: str, *, user_id: Optional[str] = None) -> dict:
        result = self.get_result(request_id, user_id=user_id)
        return {
            "request_id": result["request_id"],
            "graph": result["graph"],
            "objects": result.get("objects"),
            "metadata": result.get("metadata"),
        }

    def save_graph(
        self,
        request_id: str,
        graph_payload: Optional[Dict[str, Any]],
        *,
        user_id: Optional[str] = None,
    ) -> dict:
        resolved_user_id, user_root = self._resolve_user_root(user_id, create=False)
        result_dir = user_root / request_id
        if not result_dir.exists():
            raise FileNotFoundError(f"저장된 결과를 찾을 수 없습니다: {request_id}")

        normalized_graph = _normalize_graph_payload(graph_payload or {})

        metadata_path = result_dir / "metadata.json"
        metadata: Dict[str, Any] = {}
        if metadata_path.exists():
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                metadata = {}

        resolved_request_id = metadata.get("request_id") or metadata.get("requestId") or request_id
        graph_path = self._resolve_graph_path(result_dir, resolved_request_id)
        if not graph_path.exists():
            raise FileNotFoundError(f"그래프 파일을 찾을 수 없습니다: {graph_path}")
        floor_label = metadata.get("floor_label") or metadata.get("floorLabel")
        floor_value = metadata.get("floor_value") or metadata.get("floorValue")
        scale_reference = metadata.get("scale_reference") or metadata.get("scaleReference")
        graph_context = _build_graph_context_payload(
            request_id=resolved_request_id,
            floor_label=floor_label,
            floor_value=floor_value,
            scale_reference=scale_reference,
        )
        if graph_context:
            normalized_graph.update(graph_context)

        graph_path.write_text(json.dumps(normalized_graph, ensure_ascii=False, indent=2), encoding="utf-8")

        metadata.setdefault("request_id", resolved_request_id)
        metadata["graph_summary"] = {
            "nodes": len(normalized_graph.get("nodes", [])),
            "edges": len(normalized_graph.get("edges", [])),
        }
        metadata["updated_at"] = datetime.now().isoformat(timespec="seconds")
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

        self._update_floorplan_index(
            user_root=user_root,
            request_id=resolved_request_id,
            result_dir=result_dir,
            metadata=metadata,
            graph_path=graph_path,
        )

        user_defined_links = self._extract_user_defined_cross_floor_links(resolved_request_id, normalized_graph)
        self._update_cross_floor_links_for_request(user_root, resolved_request_id, user_defined_links)

        self._synchronize_cross_floor_connections(user_root)

        return self.get_graph_data(request_id, user_id=resolved_user_id)

    def list_results(self, *, user_id: Optional[str] = None) -> List[dict]:
        """저장된 모든 요청에 대해 요약 정보를 모아 정렬된 리스트로 반환한다."""
        summaries: List[dict] = []

        def _safe_read_text(path: Path) -> str:
            if not path.exists():
                return ""
            try:
                return path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                return ""

        resolved_user_id, user_root = self._resolve_user_root(user_id, create=False)
        if not user_root.exists():
            return []

        for result_dir in sorted(user_root.iterdir()):
            if not result_dir.is_dir():
                continue
            if result_dir.name == self.deleted_dirname:
                continue
            if self.history_dirname and result_dir.name == self.history_dirname:
                continue

            metadata_path = result_dir / "metadata.json"
            input_path = result_dir / "input_annotations.json"
            if not metadata_path.exists():
                continue

            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue

            try:
                input_payload = json.loads(input_path.read_text(encoding="utf-8"))
            except (FileNotFoundError, json.JSONDecodeError):
                input_payload = {}

            texts_payload = input_payload.get("texts") or {}
            object_detection_text = texts_payload.get("object_detection") or _safe_read_text(result_dir / "rooms.txt")
            wall_text = texts_payload.get("wall") or _safe_read_text(result_dir / "walls.txt")
            wall_base_text = texts_payload.get("wall_base") or _safe_read_text(result_dir / "walls_raw.txt")
            door_text = texts_payload.get("door") or _safe_read_text(result_dir / "doors.txt")

            annotations = input_payload.get("annotations") or []
            boxes_count = len(annotations)
            walls_count = len([line for line in wall_text.splitlines() if line.strip()])
            doors_count = len([line for line in door_text.splitlines() if line.strip()])

            created_at_raw = metadata.get("created_at")
            created_at: Optional[datetime] = None
            if isinstance(created_at_raw, str):
                try:
                    created_at = datetime.fromisoformat(created_at_raw)
                except ValueError:
                    created_at = None
            if created_at is None:
                created_at = datetime.fromtimestamp(result_dir.stat().st_mtime)

            image_size = metadata.get("image_size") or input_payload.get("image_size") or {"width": 0, "height": 0}
            class_names = input_payload.get("class_names") or DEFAULT_CLASS_NAMES

            image_data_url = metadata.get("image_data_url")
            image_url = metadata.get("image_url")
            if not image_data_url:
                stored_image_filename = metadata.get("stored_image_path")
                if stored_image_filename:
                    image_path = result_dir / stored_image_filename
                    if image_path.exists():
                        mime_type = metadata.get("stored_image_mime") or mimetypes.guess_type(image_path.name)[0] or "image/png"
                        try:
                            image_data_url = _build_data_url(mime_type, image_path.read_bytes())
                        except OSError:
                            image_data_url = None
                        if not image_url:
                            resolved_request_id = metadata.get("request_id") or result_dir.name
                            image_url = self._build_image_url(resolved_request_id, resolved_user_id)
            if not image_url:
                resolved_request_id = metadata.get("request_id") or result_dir.name
                image_url = self._build_image_url(resolved_request_id, resolved_user_id)

            summaries.append(
                {
                    "request_id": metadata.get("request_id") or result_dir.name,
                    "created_at": created_at.isoformat(),
                    "image_size": image_size,
                    "class_names": class_names,
                    "source_image_path": metadata.get("source_image_path"),
                    "graph_summary": metadata.get("graph_summary"),
                    "annotation_counts": {
                        "boxes": boxes_count,
                        "walls": walls_count,
                        "doors": doors_count,
                    },
                    "object_detection_text": object_detection_text,
                    "wall_text": wall_text,
                    "wall_base_text": wall_base_text,
                    "door_text": door_text,
                    "image_url": image_url,
                    "image_data_url": image_data_url,
                    "floor_label": metadata.get("floor_label"),
                    "floor_value": metadata.get("floor_value"),
                }
            )

        summaries.sort(key=lambda item: item.get("created_at") or "", reverse=True)
        return summaries

    def delete_result(
        self,
        request_id: str,
        *,
        user_id: Optional[str] = None,
        step_one_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """요청 ID와 연결된 데이터를 deleted 폴더로 이동한다."""
        safe_request_id = (request_id or "").strip()
        if not safe_request_id:
            raise ValueError("삭제할 request_id가 필요합니다.")

        _, user_root = self._resolve_user_root(user_id, create=False)
        if not user_root.exists():
            raise FileNotFoundError("삭제할 결과를 찾을 수 없습니다.")

        deleted_root = user_root / self.deleted_dirname
        deleted_root.mkdir(parents=True, exist_ok=True)
        archive_dir_name = f"{safe_request_id}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        archive_root = deleted_root / archive_dir_name
        archive_root.mkdir(parents=True, exist_ok=True)

        moved_entries: List[str] = []

        def _move_path(path: Path) -> None:
            if not path.exists():
                return
            try:
                relative = path.relative_to(user_root)
            except ValueError:
                return
            destination = archive_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(path), str(destination))
            moved_entries.append(str(relative))

        result_dir = user_root / safe_request_id
        _move_path(result_dir)

        safe_step_one_candidates = []
        if step_one_id:
            safe_step_one_candidates.append(step_one_id.strip())
        safe_step_one_candidates.append(f"step_one_{safe_request_id}")

        for candidate in safe_step_one_candidates:
            safe_candidate = (candidate or "").strip()
            if not safe_candidate:
                continue
            _move_path(user_root / f"{safe_candidate}.json")
            for legacy_name in ("step_three_results", "step_two_results"):
                _move_path(user_root / legacy_name / f"{safe_candidate}.json")

        if self.history_dirname:
            _move_path(user_root / self.history_dirname / safe_request_id)

        if not moved_entries:
            try:
                archive_root.rmdir()
            except OSError:
                pass
            raise FileNotFoundError("삭제할 결과를 찾을 수 없습니다.")

        remove_floorplan_index_entry(user_root, safe_request_id)

        return {
            "requestId": safe_request_id,
            "archivePath": str(archive_root.relative_to(user_root)),
            "movedEntries": moved_entries,
        }

    def get_image_info(self, request_id: str, *, user_id: Optional[str] = None) -> Optional[Tuple[Path, str]]:
        """저장된 요청의 원본 이미지 경로와 MIME 타입을 돌려준다."""
        _, user_root = self._resolve_user_root(user_id, create=False)
        result_dir = user_root / request_id
        metadata_path = result_dir / "metadata.json"
        if not metadata_path.exists():
            return None
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
        stored_image = metadata.get("stored_image_path")
        if not stored_image:
            return None
        image_path = result_dir / stored_image
        if not image_path.exists():
            return None
        mime_type = metadata.get("stored_image_mime") or mimetypes.guess_type(image_path.name)[0] or "image/png"
        return image_path, mime_type
