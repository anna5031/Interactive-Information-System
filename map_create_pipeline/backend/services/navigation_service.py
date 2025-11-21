from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx

from configuration import get_navigation_settings
from services.floorplan_index import load_index as load_floorplan_index
from services.step_three_repository import NEW_ROOM_INFO_PREFIX


@dataclass
class FloorGraphData:
    request_id: str
    graph: dict
    floor_label: Optional[str]
    floor_value: Optional[str]
    room_map: Dict[str, dict]
    meters_per_pixel: Optional[float]


class IndoorNavigationService:
    """여러 층 그래프를 통합해 실내 경로 탐색을 수행하는 서비스."""

    def __init__(self, storage_root: Path):
        self.storage_root = Path(storage_root)
        navigation_settings = get_navigation_settings()
        self.walking_speed_mps = self._resolve_positive_float(navigation_settings.get("walking_speed_mps"), fallback=1.0)
        self.stairs_seconds_per_floor = self._resolve_positive_float(
            navigation_settings.get("stairs_seconds_per_floor"), fallback=7.0
        )
        self.elevator_seconds_per_floor = self._resolve_positive_float(
            navigation_settings.get("elevator_seconds_per_floor"), fallback=5.0
        )

    def _resolve_building_root(self, building_id: str) -> Path:
        root = self.storage_root / str(building_id)
        if not root.exists() or not root.is_dir():
            raise FileNotFoundError(f"건물 ID {building_id}에 대한 데이터 디렉터리를 찾을 수 없습니다.")
        return root

    def _iter_floor_dirs(self, building_root: Path) -> List[Path]:
        floor_dirs: List[Path] = []
        for entry in sorted(building_root.iterdir()):
            if not entry.is_dir():
                continue
            if entry.name in {"history", "deleted"}:
                continue
            floor_dirs.append(entry)
        return floor_dirs

    @staticmethod
    def _safe_read_json(path: Path) -> Optional[dict]:
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None

    def _find_graph_path(self, floor_dir: Path, request_id: str) -> Optional[Path]:
        specific = floor_dir / f"navigation_graph_{request_id}.json"
        if specific.exists():
            return specific
        candidates = sorted(floor_dir.glob("navigation_graph_*.json"))
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        legacy = floor_dir / "navigation_graph.json"
        if legacy.exists():
            return legacy
        return None

    def _room_info_path(self, floor_dir: Path, request_id: str) -> Path:
        return floor_dir / f"{NEW_ROOM_INFO_PREFIX}{request_id}.json"

    def _build_room_map_from_payload(self, payload: Optional[dict]) -> Dict[str, dict]:
        if not payload:
            return {}
        mapping: Dict[str, dict] = {}

        def _ingest(items: Optional[List[dict]], category: str) -> None:
            if not items:
                return
            for item in items:
                node_id = str(item.get("graphNodeId") or item.get("nodeId") or "").strip()
                if not node_id:
                    continue
                mapping[node_id] = {
                    "category": category,
                    "name": (item.get("name") or "").strip(),
                    "number": (item.get("number") or "").strip(),
                    "extra": item.get("extra") or [],
                }

        _ingest(payload.get("rooms"), "room")
        _ingest(payload.get("doors"), "door")
        return mapping

    def _load_room_map_from_path(self, path: Optional[Path]) -> Dict[str, dict]:
        payload = self._safe_read_json(path) if path else None
        return self._build_room_map_from_payload(payload)

    def _load_room_map(self, floor_dir: Path, request_id: str) -> Dict[str, dict]:
        return self._load_room_map_from_path(self._room_info_path(floor_dir, request_id))

    @staticmethod
    def _resolve_index_path(root: Path, relative_path: Optional[str], fallback: Optional[Path]) -> Optional[Path]:
        if relative_path:
            candidate = Path(relative_path)
            if not candidate.is_absolute():
                candidate = root / candidate
            if candidate.exists():
                return candidate
        return fallback

    def _load_floor(self, floor_dir: Path) -> Optional[FloorGraphData]:
        request_id = floor_dir.name
        graph_path = self._find_graph_path(floor_dir, request_id)
        if not graph_path:
            return None
        graph_data = self._safe_read_json(graph_path)
        if not graph_data:
            return None
        room_map = self._load_room_map(floor_dir, request_id)
        floor_label = graph_data.get("floorLabel") or graph_data.get("floor_label")
        floor_value = graph_data.get("floorValue") or graph_data.get("floor_value")
        metadata_path = floor_dir / "metadata.json"
        metadata_payload = self._safe_read_json(metadata_path) or {}
        image_size = metadata_payload.get("image_size") or metadata_payload.get("imageSize") or graph_data.get("image_size")
        scale_reference_payload = (
            metadata_payload.get("scale_reference")
            or metadata_payload.get("scaleReference")
            or graph_data.get("scale_reference")
            or graph_data.get("scaleReference")
        )
        meters_per_pixel = self._meters_per_pixel_from_scale_reference(scale_reference_payload, image_size)
        if meters_per_pixel is None:
            fallback_value = (
                graph_data.get("metersPerPixel")
                or graph_data.get("meters_per_pixel")
                or metadata_payload.get("meters_per_pixel")
                or metadata_payload.get("metersPerPixel")
            )
            try:
                numeric = float(fallback_value)
            except (TypeError, ValueError):
                numeric = None
            if numeric is not None and math.isfinite(numeric) and numeric > 0:
                meters_per_pixel = numeric
            else:
                meters_per_pixel = None
        return FloorGraphData(
            request_id=request_id,
            graph=graph_data,
            floor_label=floor_label,
            floor_value=floor_value,
            room_map=room_map,
            meters_per_pixel=meters_per_pixel,
        )

    def _load_floor_from_index_entry(self, building_root: Path, request_id: str, entry: dict) -> Optional[FloorGraphData]:
        floor_dir = building_root / request_id
        default_graph_path = self._find_graph_path(floor_dir, request_id) if floor_dir.exists() else None
        graph_path = self._resolve_index_path(building_root, entry.get("graphPath"), default_graph_path)
        if not graph_path:
            return None
        graph_data = self._safe_read_json(graph_path)
        if not graph_data:
            return None
        room_info_default = self._room_info_path(floor_dir, request_id) if floor_dir.exists() else None
        room_map = self._load_room_map_from_path(
            self._resolve_index_path(building_root, entry.get("roomInfoPath"), room_info_default)
        )
        metadata_default = floor_dir / "metadata.json" if floor_dir.exists() else None
        metadata_path = self._resolve_index_path(building_root, entry.get("metadataPath"), metadata_default)
        metadata_payload = self._safe_read_json(metadata_path) or {}
        image_size = (
            metadata_payload.get("image_size")
            or metadata_payload.get("imageSize")
            or entry.get("imageSize")
            or graph_data.get("image_size")
        )
        scale_reference_payload = (
            metadata_payload.get("scale_reference")
            or metadata_payload.get("scaleReference")
            or graph_data.get("scale_reference")
            or graph_data.get("scaleReference")
            or entry.get("scaleReference")
        )
        floor_label = (
            metadata_payload.get("floor_label")
            or metadata_payload.get("floorLabel")
            or entry.get("floorLabel")
            or graph_data.get("floorLabel")
            or graph_data.get("floor_label")
        )
        floor_value = (
            metadata_payload.get("floor_value")
            or metadata_payload.get("floorValue")
            or entry.get("floorValue")
            or graph_data.get("floorValue")
            or graph_data.get("floor_value")
        )
        meters_per_pixel = self._meters_per_pixel_from_scale_reference(scale_reference_payload, image_size)
        if meters_per_pixel is None:
            fallback_value = (
                graph_data.get("metersPerPixel")
                or graph_data.get("meters_per_pixel")
                or metadata_payload.get("meters_per_pixel")
                or metadata_payload.get("metersPerPixel")
            )
            try:
                numeric = float(fallback_value)
            except (TypeError, ValueError):
                numeric = None
            if numeric is not None and math.isfinite(numeric) and numeric > 0:
                meters_per_pixel = numeric
            else:
                meters_per_pixel = None
        return FloorGraphData(
            request_id=request_id,
            graph=graph_data,
            floor_label=floor_label,
            floor_value=floor_value,
            room_map=room_map,
            meters_per_pixel=meters_per_pixel,
        )

    def _load_building_floors(self, building_id: str) -> List[FloorGraphData]:
        building_root = self._resolve_building_root(building_id)
        floors: List[FloorGraphData] = []
        processed: Set[str] = set()
        index_entries = load_floorplan_index(building_root)
        if index_entries:
            for request_id, entry in index_entries.items():
                if not isinstance(entry, dict):
                    continue
                floor = self._load_floor_from_index_entry(building_root, request_id, entry)
                if floor:
                    floors.append(floor)
                    processed.add(request_id)
        for floor_dir in self._iter_floor_dirs(building_root):
            if floor_dir.name in processed:
                continue
            floor = self._load_floor(floor_dir)
            if floor:
                floors.append(floor)
        if not floors:
            raise FileNotFoundError(f"건물 ID {building_id}에 등록된 층 그래프가 없습니다.")
        return floors

    @staticmethod
    def _safe_weight(value: Optional[float]) -> float:
        try:
            resolved = float(value)
        except (TypeError, ValueError):
            return 1.0
        if not math.isfinite(resolved):
            return 1.0
        return resolved

    @staticmethod
    def _resolve_positive_float(value: Optional[float], fallback: float) -> float:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return fallback
        if not math.isfinite(numeric) or numeric <= 0:
            return fallback
        return numeric

    @staticmethod
    def _normalize_scale_reference(payload: Optional[dict]) -> Optional[dict]:
        if not payload or not isinstance(payload, dict):
            return None

        def _unit(raw: Any) -> Optional[float]:
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
            normalized = _unit(payload.get(key))
            if normalized is None:
                return None
            coords.append(normalized)
        length_raw = payload.get("length_meters") or payload.get("lengthMeters")
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

    @classmethod
    def _meters_per_pixel_from_scale_reference(cls, scale_reference: Optional[dict], image_size: Optional[dict]) -> Optional[float]:
        normalized = cls._normalize_scale_reference(scale_reference)
        if not normalized or not image_size or not isinstance(image_size, dict):
            return None
        try:
            width = float(image_size.get("width"))
            height = float(image_size.get("height"))
        except (TypeError, ValueError):
            return None
        if not math.isfinite(width) or not math.isfinite(height) or width <= 0 or height <= 0:
            return None
        dx = (normalized["x2"] - normalized["x1"]) * width
        dy = (normalized["y2"] - normalized["y1"]) * height
        pixel_length = math.hypot(dx, dy)
        if not math.isfinite(pixel_length) or pixel_length <= 0:
            return None
        return normalized["length_meters"] / pixel_length

    def _build_combined_graph(
        self, floors: List[FloorGraphData]
    ) -> Tuple[nx.Graph, Dict[Tuple[str, str], str], Dict[str, List[str]]]:
        graph = nx.Graph()
        node_index: Dict[Tuple[str, str], str] = {}
        id_only_index: Dict[str, List[str]] = defaultdict(list)

        for floor in floors:
            floor_label = floor.floor_label or floor.graph.get("floorLabel") or floor.graph.get("floor_label")
            floor_value = floor.floor_value or floor.graph.get("floorValue") or floor.graph.get("floor_value")
            for node in floor.graph.get("nodes", []):
                node_id = str(node.get("id") or "").strip()
                if not node_id:
                    continue
                composite_id = f"{floor.request_id}:{node_id}"
                if composite_id in graph.nodes:
                    continue
                node_attrs = {
                    "request_id": floor.request_id,
                    "node_id": node_id,
                    "type": node.get("type"),
                    "pos": node.get("pos"),
                    "floor_label": floor_label,
                    "floor_value": floor_value,
                    "meters_per_pixel": floor.meters_per_pixel,
                }
                detail = floor.room_map.get(node_id)
                if detail:
                    node_attrs.update(
                        {
                            "category": detail.get("category"),
                            "name": detail.get("name"),
                            "number": detail.get("number"),
                            "extra": detail.get("extra"),
                        }
                    )
                graph.add_node(composite_id, **node_attrs)
                node_index[(floor.request_id, node_id)] = composite_id
                id_only_index[node_id].append(composite_id)

        standard_edges: set = set()
        cross_edges: set = set()

        for floor in floors:
            request_id = floor.request_id
            for edge in floor.graph.get("edges", []):
                attributes = edge.get("attributes") if isinstance(edge.get("attributes"), dict) else {}
                is_cross_floor = bool(attributes.get("is_cross_floor"))
                weight = self._safe_weight(edge.get("weight"))
                if is_cross_floor:
                    source_node_id = str(
                        attributes.get("source_node_id") or attributes.get("sourceNodeId") or edge.get("source") or ""
                    ).strip()
                    target_request_id = str(
                        attributes.get("target_request_id") or attributes.get("targetRequestId") or ""
                    ).strip()
                    target_node_id = str(
                        attributes.get("target_node_id") or attributes.get("targetNodeId") or ""
                    ).strip()
                    if not source_node_id or not target_request_id or not target_node_id:
                        continue
                    source_key = node_index.get((request_id, source_node_id))
                    target_key = node_index.get((target_request_id, target_node_id))
                    if not source_key or not target_key:
                        continue
                    edge_key = tuple(sorted([source_key, target_key]))
                    if edge_key in cross_edges:
                        continue
                    cross_edges.add(edge_key)
                    edge_attrs = {"weight": weight, "is_cross_floor": True}
                    if attributes:
                        edge_attrs["edge_attributes"] = attributes
                    graph.add_edge(source_key, target_key, **edge_attrs)
                else:
                    source_node_id = str(edge.get("source") or "").strip()
                    target_node_id = str(edge.get("target") or "").strip()
                    if not source_node_id or not target_node_id:
                        continue
                    source_key = node_index.get((request_id, source_node_id))
                    target_key = node_index.get((request_id, target_node_id))
                    if not source_key or not target_key:
                        continue
                    edge_key = tuple(sorted([source_key, target_key]))
                    if edge_key in standard_edges:
                        existing = graph.get_edge_data(edge_key[0], edge_key[1]) or {}
                        if weight < existing.get("weight", weight):
                            graph[edge_key[0]][edge_key[1]]["weight"] = weight
                            if attributes:
                                graph[edge_key[0]][edge_key[1]]["edge_attributes"] = attributes
                        continue
                    standard_edges.add(edge_key)
                    edge_attrs = {"weight": weight, "is_cross_floor": False}
                    if attributes:
                        edge_attrs["edge_attributes"] = attributes
                    graph.add_edge(edge_key[0], edge_key[1], **edge_attrs)

        return graph, node_index, id_only_index

    def _resolve_node_key(
        self,
        reference: str,
        node_index: Dict[Tuple[str, str], str],
        id_only_index: Dict[str, List[str]],
    ) -> str:
        trimmed = (reference or "").strip()
        if not trimmed:
            raise ValueError("노드 식별자가 비어 있습니다.")
        if ":" in trimmed:
            floor_id, node_id = trimmed.split(":", 1)
            key = node_index.get((floor_id.strip(), node_id.strip()))
            if not key:
                raise ValueError(f"{trimmed}에 해당하는 노드를 찾을 수 없습니다.")
            return key
        candidates = id_only_index.get(trimmed)
        if not candidates:
            raise ValueError(f"노드 ID {trimmed} 를 찾을 수 없습니다.")
        if len(candidates) > 1:
            floors = ", ".join(sorted(candidates))
            raise ValueError(f"노드 ID {trimmed} 가 여러 층에 존재합니다. floorId:nodeId 형식으로 지정하세요. (후보: {floors})")
        return candidates[0]

    @staticmethod
    def _resolve_meters_per_pixel_attr(value: Optional[float]) -> Optional[float]:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(numeric) or numeric <= 0:
            return None
        return numeric

    def _meters_per_pixel_between(self, source_attrs: dict, target_attrs: dict) -> float:
        meters_per_pixel = self._resolve_meters_per_pixel_attr(source_attrs.get("meters_per_pixel"))
        if meters_per_pixel is None:
            meters_per_pixel = self._resolve_meters_per_pixel_attr(target_attrs.get("meters_per_pixel"))
        if meters_per_pixel is None:
            raise ValueError("그래프에 유효한 축척 정보(meters_per_pixel)가 없어 이동 시간을 계산할 수 없습니다.")
        return meters_per_pixel

    def _compute_horizontal_duration(self, weight: float, source_attrs: dict, target_attrs: dict) -> float:
        meters_per_pixel = self._meters_per_pixel_between(source_attrs, target_attrs)
        distance_m = max(0.0, float(weight)) * meters_per_pixel
        if self.walking_speed_mps <= 0:
            return distance_m
        return distance_m / self.walking_speed_mps

    @staticmethod
    def _parse_floor_value(value: Optional[str]) -> Optional[int]:
        if value is None:
            return None
        token = str(value).strip().upper()
        if not token:
            return None
        token = token.replace("층", "")
        if token.startswith("B"):
            digits = token[1:] or "1"
            try:
                return -int(digits)
            except ValueError:
                return None
        if token.startswith("F"):
            digits = token[1:] or "1"
            try:
                return int(digits)
            except ValueError:
                return None
        if token.endswith("F"):
            digits = token[:-1] or "1"
            try:
                return int(digits)
            except ValueError:
                return None
        try:
            return int(token)
        except ValueError:
            return None

    def _connector_seconds_per_floor(self, connector_type: Optional[str]) -> float:
        normalized = (connector_type or "").strip().lower()
        if normalized == "elevator":
            return self.elevator_seconds_per_floor
        return self.stairs_seconds_per_floor

    def _compute_vertical_duration(self, source_attrs: dict, target_attrs: dict, connector_type: Optional[str]) -> float:
        floor_a = self._parse_floor_value(source_attrs.get("floor_value"))
        floor_b = self._parse_floor_value(target_attrs.get("floor_value"))
        if floor_a is None or floor_b is None:
            floor_diff = 1
        else:
            floor_diff = abs(floor_a - floor_b)
            if floor_a < 0 < floor_b or floor_b < 0 < floor_a:
                floor_diff = max(1, floor_diff - 1)
            if floor_diff == 0:
                floor_diff = 1
        seconds_per_floor = self._connector_seconds_per_floor(connector_type)
        return floor_diff * seconds_per_floor

    def _build_path_payload(self, graph: nx.Graph, path: List[str]) -> dict:
        if not path:
            return {
                "nodes": [],
                "edges": [],
                "cross_floor_segments": [],
                "weight_summary": {"normal_total": 0.0, "cross_floor_total": 0.0, "overall": 0.0},
                "duration_summary": {"horizontal_total": 0.0, "vertical_total": 0.0, "overall": 0.0},
            }

        def _node_entry(node_key: str) -> dict:
            attrs = graph.nodes[node_key]
            return {
                "request_id": attrs.get("request_id"),
                "node_id": attrs.get("node_id"),
                "floor_label": attrs.get("floor_label"),
                "floor_value": attrs.get("floor_value"),
                "type": attrs.get("type"),
                "category": attrs.get("category"),
                "name": attrs.get("name"),
                "number": attrs.get("number"),
            }

        door_endpoint_type = "door_endpoints"
        significant_indices: List[int] = []
        for idx, node_key in enumerate(path):
            node_type = (graph.nodes[node_key].get("type") or "").strip().lower()
            if idx == 0 or idx == len(path) - 1 or node_type != door_endpoint_type:
                if not significant_indices or significant_indices[-1] != idx:
                    significant_indices.append(idx)
        if significant_indices[-1] != len(path) - 1:
            significant_indices.append(len(path) - 1)

        nodes_payload = [_node_entry(path[idx]) for idx in significant_indices]

        edges_payload: List[dict] = []
        cross_floor_segments: List[dict] = []
        normal_total = 0.0
        cross_total = 0.0
        horizontal_duration = 0.0
        vertical_duration = 0.0

        for start_idx, end_idx in zip(significant_indices, significant_indices[1:]):
            segment_length = end_idx - start_idx
            total_weight = 0.0
            segment_duration = 0.0
            is_cross = False
            edge_attributes = {}
            for idx in range(start_idx, end_idx):
                source = path[idx]
                target = path[idx + 1]
                edge_data = graph.get_edge_data(source, target) or {}
                weight = float(edge_data.get("weight", 0.0))
                total_weight += weight
                if edge_data.get("is_cross_floor"):
                    is_cross = True
                    connector_type = (edge_data.get("edge_attributes") or {}).get("connector_type")
                    segment_duration += self._compute_vertical_duration(
                        graph.nodes[source], graph.nodes[target], connector_type
                    )
                else:
                    segment_duration += self._compute_horizontal_duration(weight, graph.nodes[source], graph.nodes[target])
                if segment_length == 1:
                    edge_attributes = edge_data.get("edge_attributes") or {}
            if segment_length > 1:
                edge_attributes = {"aggregated_segment": True, "segment_edges": segment_length}
            payload = {
                "from": graph.nodes[path[start_idx]].get("node_id"),
                "from_request_id": graph.nodes[path[start_idx]].get("request_id"),
                "to": graph.nodes[path[end_idx]].get("node_id"),
                "to_request_id": graph.nodes[path[end_idx]].get("request_id"),
                "weight": total_weight,
                "is_cross_floor": is_cross,
                "edge_attributes": edge_attributes,
                "duration_seconds": segment_duration,
            }
            edges_payload.append(payload)
            if is_cross:
                cross_total += total_weight
                vertical_duration += segment_duration
                cross_floor_segments.append(payload)
            else:
                normal_total += total_weight
                horizontal_duration += segment_duration

        return {
            "nodes": nodes_payload,
            "edges": edges_payload,
            "cross_floor_segments": cross_floor_segments,
            "weight_summary": {
                "normal_total": normal_total,
                "cross_floor_total": cross_total,
                "overall": normal_total + cross_total,
            },
            "duration_summary": {
                "horizontal_total": horizontal_duration,
                "vertical_total": vertical_duration,
                "overall": horizontal_duration + vertical_duration,
            },
        }

    def find_shortest_path(self, building_id: str, start: str, destination: str) -> dict:
        floors = self._load_building_floors(building_id)
        graph, node_index, id_only_index = self._build_combined_graph(floors)
        start_key = self._resolve_node_key(start, node_index, id_only_index)
        destination_key = self._resolve_node_key(destination, node_index, id_only_index)
        if start_key == destination_key:
            return {
                "nodes": [
                    {
                        "request_id": graph.nodes[start_key].get("request_id"),
                        "node_id": graph.nodes[start_key].get("node_id"),
                        "floor_label": graph.nodes[start_key].get("floor_label"),
                        "floor_value": graph.nodes[start_key].get("floor_value"),
                        "type": graph.nodes[start_key].get("type"),
                        "category": graph.nodes[start_key].get("category"),
                        "name": graph.nodes[start_key].get("name"),
                        "number": graph.nodes[start_key].get("number"),
                    }
                ],
                "edges": [],
                "cross_floor_segments": [],
                "weight_summary": {"normal_total": 0.0, "cross_floor_total": 0.0, "overall": 0.0},
            }
        try:
            path = nx.shortest_path(graph, start_key, destination_key, weight="weight")
        except nx.NetworkXNoPath as exc:
            raise ValueError("출발지와 도착지를 연결하는 경로를 찾을 수 없습니다.") from exc
        return self._build_path_payload(graph, path)
