from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import networkx as nx
from shapely.geometry import shape

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from processing.yolo_out_to_graph import FloorPlanVisualizer

def load_objects(objects_path: Path) -> Dict[str, List[Dict[str, Any]]]:
    raw = json.loads(objects_path.read_text(encoding="utf-8"))
    parsed: Dict[str, List[Dict[str, Any]]] = {}

    for key, items in raw.items():
        converted: List[Dict[str, Any]] = []
        for item in items or []:
            transformed = dict(item)

            polygon_raw = transformed.get("polygon")
            if polygon_raw:
                try:
                    transformed["polygon"] = shape(polygon_raw)
                except Exception:
                    transformed["polygon"] = None

            centroid_raw = transformed.get("centroid")
            if centroid_raw:
                try:
                    transformed["centroid"] = shape(centroid_raw)
                except Exception:
                    transformed["centroid"] = None

            converted.append(transformed)
        parsed[key] = converted
    return parsed


def load_graph(graph_path: Path) -> nx.Graph:
    data = json.loads(graph_path.read_text(encoding="utf-8"))
    G = nx.Graph()

    for node in data.get("nodes", []):
        node_id = node.get("id")
        if node_id is None:
            continue
        attrs = {}
        if "type" in node:
            attrs["type"] = node["type"]
        if "pos" in node:
            attrs["pos"] = tuple(node["pos"])
        extra = node.get("attributes")
        if isinstance(extra, dict):
            attrs.update(extra)
        G.add_node(node_id, **attrs)

    for edge in data.get("edges", []):
        source = edge.get("source")
        target = edge.get("target")
        if source is None or target is None:
            continue
        attrs = {}
        if "weight" in edge:
            attrs["weight"] = edge["weight"]
        extra = edge.get("attributes")
        if isinstance(extra, dict):
            attrs.update(extra)
        G.add_edge(source, target, **attrs)

    return G


def main() -> None:
    parser = argparse.ArgumentParser(description="저장된 floorplan 자료를 시각화합니다.")
    parser.add_argument(
        "--bundle-dir",
        type=Path,
        required=True,
        help="map_create_pipeline/backend/data/<request_id> 경로",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="시각화 이미지를 저장할 경로 (미지정 시 bundle-dir 아래 PNG 생성)",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="창을 띄우지 않고 파일만 저장합니다.",
    )
    args = parser.parse_args()
    bundle_dir: Path = args.bundle_dir.resolve()

    objects_path = bundle_dir / "floorplan_objects.json"
    graph_path = bundle_dir / "navigation_graph.json"

    if not objects_path.exists():
        raise FileNotFoundError(f"objects 파일이 없습니다: {objects_path}")
    if not graph_path.exists():
        raise FileNotFoundError(f"graph 파일이 없습니다: {graph_path}")

    parsed_objects = load_objects(objects_path)
    graph = load_graph(graph_path)

    output_path: Optional[Path] = args.output.resolve() if args.output else None
    if output_path is None:
        output_path = bundle_dir / "visualized_floorplan.png"

    visualizer = FloorPlanVisualizer(graph, parsed_objects)
    visualizer.show(
        path=None,
        output_path=output_path,
        show_window=not args.no_show,
        annotate_connections=True,
    )
    print(f"시각화 결과가 {output_path} 에 저장되었습니다.")


if __name__ == "__main__":
    main()
