from __future__ import annotations

import base64
import binascii
import json
import mimetypes
import re
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from uuid import uuid4

import networkx as nx
import numpy as np
from shapely.geometry import mapping
from shapely.geometry.base import BaseGeometry

from processing.yolo_out_to_graph import (
    DEFAULT_CLASS_NAMES,
    FloorPlanNavigationGraph,
    FloorPlanParser,
    load_annotation_bundle_from_texts,
)

_DATA_URL_PATTERN = re.compile(r"^data:(?P<mime>[\w.+/-]+);base64,(?P<data>.+)$")


def _decode_image_data_url(data_url: str) -> Tuple[bytes, str]:
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
    base64_payload = base64.b64encode(payload).decode("ascii")
    safe_mime = mime_type or "application/octet-stream"
    return f"data:{safe_mime};base64,{base64_payload}"


def _to_serializable(value):
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
    serialized: Dict[str, List[dict]] = {}
    for key, items in objects.items():
        bucket: List[dict] = []
        for item in items:
            serialized_item = {attr_key: _to_serializable(attr_value) for attr_key, attr_value in item.items()}
            bucket.append(serialized_item)
        serialized[key] = bucket
    return serialized


def _serialize_graph(graph: nx.Graph) -> Dict[str, List[dict]]:
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


def _parse_yolo_text(yolo_text: str) -> Tuple[List[dict], List[str]]:
    annotations: List[dict] = []
    sanitized_lines: List[str] = []
    if not yolo_text:
        return annotations, sanitized_lines

    for index, raw_line in enumerate(yolo_text.splitlines()):
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
    yolo_text: str,
    wall_text: str,
    door_text: str,
    image_width: int,
    image_height: int,
) -> Tuple[Dict[str, List[dict]], List[dict]]:
    annotations, sanitized_lines = _parse_yolo_text(yolo_text)
    sanitized_yolo_text = "\n".join(sanitized_lines)
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        room_path = tmp_path / "rooms.txt"
        wall_path = tmp_path / "walls.txt"
        door_path = tmp_path / "doors.txt"

        room_path.write_text(sanitized_yolo_text, encoding="utf-8")
        wall_path.write_text(wall_text or "", encoding="utf-8")
        door_path.write_text(door_text or "", encoding="utf-8")

        objects = load_annotation_bundle_from_texts(
            room_path,
            wall_path,
            door_path,
            image_width=image_width,
            image_height=image_height,
        )
    return objects, annotations

class FloorPlanProcessingService:
    def __init__(self, storage_root: Path):
        self.storage_root = storage_root
        self.storage_root.mkdir(parents=True, exist_ok=True)

    def process(
        self,
        image_width: int,
        image_height: int,
        class_names: Optional[List[str]] = None,
        source_image_path: Optional[str] = None,
        yolo_text: Optional[str] = None,
        wall_text: Optional[str] = None,
        door_text: Optional[str] = None,
        image_data_url: Optional[str] = None,
    ) -> dict:
        class_names = class_names or DEFAULT_CLASS_NAMES

        parser = FloorPlanParser()

        print("Processing using yolo/wall/door texts", flush=True)
        yolo_text = yolo_text or ""
        wall_text = wall_text or ""
        door_text = door_text or ""
        parsed_objects, annotations_list = _load_objects_from_texts(
            yolo_text=yolo_text,
            wall_text=wall_text,
            door_text=door_text,
            image_width=image_width,
            image_height=image_height,
        )

        parser.objects = parsed_objects
        parser._extend_doors_along_walls()
        parser.annotate_room_door_connections()
        parsed_objects = parser.objects

        # nav_builder = FloorPlanNavigationGraph(parsed_objects, width=image_width, height=image_height, debug_dir=Path("../data/tmp"))
        nav_builder = FloorPlanNavigationGraph(parsed_objects, width=image_width, height=image_height)

        graph = nav_builder.build()

        serialized_objects = _serialize_objects(parsed_objects)
        serialized_graph = _serialize_graph(graph)

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        request_id = f"{timestamp}-{uuid4().hex[:6]}"
        result_dir = self.storage_root / request_id
        result_dir.mkdir(parents=True, exist_ok=True)

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
                "yolo": yolo_text,
                "wall": wall_text,
                "door": door_text,
            },
        }
        if source_image_path:
            input_payload["source_image_path"] = source_image_path

        input_path = result_dir / "input_annotations.json"
        objects_path = result_dir / "floorplan_objects.json"
        graph_path = result_dir / "navigation_graph.json"
        metadata_path = result_dir / "metadata.json"
        yolo_text_path = result_dir / "rooms.txt"
        wall_text_path = result_dir / "walls.txt"
        door_text_path = result_dir / "doors.txt"

        yolo_text_path.write_text(input_payload["texts"]["yolo"], encoding="utf-8")
        wall_text_path.write_text(input_payload["texts"]["wall"], encoding="utf-8")
        door_text_path.write_text(input_payload["texts"]["door"], encoding="utf-8")

        with input_path.open("w", encoding="utf-8") as f:
            json.dump(input_payload, f, ensure_ascii=False, indent=2)
        with objects_path.open("w", encoding="utf-8") as f:
            json.dump(serialized_objects, f, ensure_ascii=False, indent=2)
        with graph_path.open("w", encoding="utf-8") as f:
            json.dump(serialized_graph, f, ensure_ascii=False, indent=2)

        metadata = {
            "request_id": request_id,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "image_size": {"width": image_width, "height": image_height},
            "source_image_path": source_image_path,
            "graph_summary": {
                "nodes": graph.number_of_nodes(),
                "edges": graph.number_of_edges(),
            },
        }
        if stored_image_filename:
            metadata["stored_image_path"] = stored_image_filename
            if stored_image_mime:
                metadata["stored_image_mime"] = stored_image_mime
            metadata["image_url"] = f"/api/floorplans/{request_id}/image"
        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        return {
            "request_id": request_id,
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
                "yolo_text": str(yolo_text_path),
                "wall_text": str(wall_text_path),
                "door_text": str(door_text_path),
            },
            "input_annotations": annotations_payload,
            "metadata": {
                **metadata,
                "image_data_url": image_data_url,
            },
        }

    def get_result(self, request_id: str) -> dict:
        result_dir = self.storage_root / request_id
        if not result_dir.exists():
            raise FileNotFoundError(f"저장된 결과를 찾을 수 없습니다: {request_id}")

        def load_json(path: Path) -> dict:
            with path.open("r", encoding="utf-8") as stream:
                return json.load(stream)

        objects_path = result_dir / "floorplan_objects.json"
        graph_path = result_dir / "navigation_graph.json"
        metadata_path = result_dir / "metadata.json"
        input_path = result_dir / "input_annotations.json"
        yolo_text_path = result_dir / "rooms.txt"
        wall_text_path = result_dir / "walls.txt"
        door_text_path = result_dir / "doors.txt"

        objects = load_json(objects_path)
        graph = load_json(graph_path)
        metadata = load_json(metadata_path)
        input_payload = load_json(input_path)

        saved_files = {
            "input_annotations": str(input_path),
            "objects": str(objects_path),
            "graph": str(graph_path),
            "metadata": str(metadata_path),
            "yolo_text": str(yolo_text_path),
            "wall_text": str(wall_text_path),
            "door_text": str(door_text_path),
        }

        request_timestamp = metadata.get("created_at") or datetime.now().isoformat(timespec="seconds")
        image_size = metadata.get("image_size") or input_payload.get("image_size") or {"width": 0, "height": 0}
        class_names = input_payload.get("class_names") or DEFAULT_CLASS_NAMES
        annotations_payload = input_payload.get("annotations", [])

        texts = {
            "yolo": yolo_text_path.read_text(encoding="utf-8") if yolo_text_path.exists() else "",
            "wall": wall_text_path.read_text(encoding="utf-8") if wall_text_path.exists() else "",
            "door": door_text_path.read_text(encoding="utf-8") if door_text_path.exists() else "",
        }

        metadata_with_texts = dict(metadata)
        metadata_with_texts["texts"] = texts
        if "image_data_url" not in metadata_with_texts:
            stored_image_filename = metadata.get("stored_image_path")
            if stored_image_filename:
                image_path = result_dir / stored_image_filename
                if image_path.exists():
                    mime_type = metadata.get("stored_image_mime") or mimetypes.guess_type(image_path.name)[0] or "image/png"
                    metadata_with_texts["image_data_url"] = _build_data_url(mime_type, image_path.read_bytes())
        metadata_with_texts.setdefault("image_url", metadata.get("image_url") or f"/api/floorplans/{request_id}/image")

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

    def list_results(self) -> List[dict]:
        summaries: List[dict] = []

        def _safe_read_text(path: Path) -> str:
            if path.exists():
                return path.read_text(encoding="utf-8")
            return ""

        for result_dir in sorted(self.storage_root.iterdir()):
            if not result_dir.is_dir():
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
            yolo_text = texts_payload.get("yolo") or _safe_read_text(result_dir / "rooms.txt")
            wall_text = texts_payload.get("wall") or _safe_read_text(result_dir / "walls.txt")
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
                            image_url = f"/api/floorplans/{metadata.get('request_id') or result_dir.name}/image"

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
                    "yolo_text": yolo_text,
                    "wall_text": wall_text,
                    "door_text": door_text,
                    "image_url": image_url,
                    "image_data_url": image_data_url,
                }
            )

        summaries.sort(key=lambda item: item.get("created_at") or "", reverse=True)
        return summaries

    def get_image_info(self, request_id: str) -> Optional[Tuple[Path, str]]:
        result_dir = self.storage_root / request_id
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
