import argparse
from calendar import c
import heapq
from collections import defaultdict
from dataclasses import dataclass
import json
import pickle
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple
from datetime import datetime
from itertools import combinations

import cv2
import matplotlib.pyplot as plt

import networkx as nx
from itertools import combinations

import numpy as np
from matplotlib.patches import Polygon as MplPolygon
from scipy.ndimage import binary_fill_holes, distance_transform_edt, label, center_of_mass
from scipy.signal import convolve2d
from scipy.spatial import KDTree
from shapely.geometry import LineString, Polygon, Point
from shapely.prepared import prep
from skimage.draw import polygon as draw_polygon
from skimage.morphology import medial_axis, skeletonize

if __package__ is None or __package__ == "":
    CURRENT_DIR = Path(__file__).resolve().parent
    PARENT_DIR = CURRENT_DIR.parent
    for candidate in (CURRENT_DIR, PARENT_DIR):
        candidate_str = str(candidate)
        if candidate_str not in sys.path:
            sys.path.append(candidate_str)

try:
    from .util.find_slanted_line_102 import skeleton_to_segments  # type: ignore
except ImportError:  # pragma: no cover
    from util.find_slanted_line_102 import skeleton_to_segments


@dataclass
class Detection:
    bbox: np.ndarray  # [x_min, y_min, x_max, y_max]
    confidence: float
    class_id: int


DEFAULT_CLASS_NAMES = ['room', 'stairs', 'wall', 'elevator', 'door']
OBJECT_KEY_MAP = {
    'room': 'rooms',
    'rooms': 'rooms',
    'door': 'doors',
    'doors': 'doors',
    'stair': 'stairs',
    'stairs': 'stairs',
    'elevator': 'elevators',
    'elevators': 'elevators',
    'wall': 'walls',
    'walls': 'walls',
}

EXPORT_BUNDLE_FILENAME = 'floorplan_bundle.pkl'
EXPORT_MANIFEST_FILENAME = 'floorplan_manifest.json'
EXPORT_IMAGE_FILENAME = 'annotated_floorplan.png'


def load_class_names(path: Optional[Path]) -> List[str]:
    if path is None:
        return DEFAULT_CLASS_NAMES
    if not path.exists():
        raise FileNotFoundError(f"클래스 이름 파일을 찾을 수 없습니다: {path}")
    with path.open('r', encoding='utf-8') as f:
        names = [line.strip() for line in f.readlines() if line.strip()]
    if not names:
        raise ValueError("클래스 이름 파일이 비어있습니다.")
    return names


def read_yolo_label_file(label_path: Path, image_path: Path) -> List[Detection]:
    if not label_path.exists():
        raise FileNotFoundError(f"YOLO 라벨 파일을 찾을 수 없습니다: {label_path}")
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"이미지를 불러올 수 없습니다: {image_path}")
    height, width = image.shape[:2]

    detections: List[Detection] = []
    with label_path.open('r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            class_id = int(float(parts[0]))
            x_center = float(parts[1]) * width
            y_center = float(parts[2]) * height
            box_w = float(parts[3]) * width
            box_h = float(parts[4]) * height
            confidence = float(parts[5]) if len(parts) >= 6 else 1.0

            x_min = max(0.0, x_center - box_w / 2.0)
            y_min = max(0.0, y_center - box_h / 2.0)
            x_max = min(width, x_center + box_w / 2.0)
            y_max = min(height, y_center + box_h / 2.0)
            detections.append(Detection(np.array([x_min, y_min, x_max, y_max]), confidence, class_id))
    return detections


def load_ultralytics_predictions(
    predictions_dir: Path,
    image_path: Path,
) -> List[Detection]:
    label_path = predictions_dir / 'labels' / f"{image_path.stem}.txt"
    return read_yolo_label_file(label_path, image_path)


def parse_exclude_ids(raw: Optional[Sequence[str]]) -> Dict[str, Set[int]]:
    mapping: Dict[str, Set[int]] = {}
    if not raw:
        return mapping

    for token in raw:
        if ':' not in token:
            raise ValueError(f"잘못된 exclude-ids 형식: {token}")
        label_part, ids_part = token.split(':', 1)
        key = OBJECT_KEY_MAP.get(label_part.strip().lower())
        if key is None:
            raise ValueError(f"지원하지 않는 클래스 이름: {label_part}")
        id_strings = [s.strip() for s in ids_part.split(',') if s.strip()]
        if not id_strings:
            continue
        try:
            id_values = {int(value) for value in id_strings}
        except ValueError as exc:
            raise ValueError(f"exclude-ids 숫자 변환 실패: {ids_part}") from exc
        mapping.setdefault(key, set()).update(id_values)

    return mapping


def load_coco_ground_truth(
    coco_path: Path,
    image_root: Optional[Path],
    image_id: Optional[int],
    file_name: Optional[str],
) -> tuple[List[Detection], List[str], Path]:
    if not coco_path.exists():
        raise FileNotFoundError(f"COCO json 파일을 찾을 수 없습니다: {coco_path}")

    with coco_path.open('r', encoding='utf-8') as f:
        coco = json.load(f)

    categories = coco.get('categories', [])
    if not categories:
        raise ValueError('COCO 파일에 category 정보가 없습니다.')

    categories = sorted(categories, key=lambda c: c['id'])
    class_names = [cat['name'] for cat in categories]
    category_to_index = {cat['id']: idx for idx, cat in enumerate(categories)}

    images = coco.get('images', [])
    if not images:
        raise ValueError('COCO 파일에 image 정보가 없습니다.')

    selected_image = None
    if image_id is not None:
        selected_image = next((img for img in images if img['id'] == image_id), None)
        if selected_image is None:
            raise ValueError(f"image_id {image_id} 를 COCO 파일에서 찾을 수 없습니다.")
    elif file_name is not None:
        selected_image = next((img for img in images if img['file_name'] == file_name), None)
        if selected_image is None:
            raise ValueError(f"file_name '{file_name}' 을 COCO 파일에서 찾을 수 없습니다.")
    else:
        selected_image = images[0]

    image_rel_path = Path(selected_image['file_name'])
    if image_root is not None:
        image_path = image_root / image_rel_path
    elif image_rel_path.is_absolute():
        image_path = image_rel_path
    else:
        raise ValueError("상대 경로 이미지가 발견되었습니다. --image-root 를 지정해주세요.")

    detections: List[Detection] = []
    image_id_val = selected_image['id']
    width = selected_image['width']
    height = selected_image['height']

    for ann in coco.get('annotations', []):
        if ann.get('image_id') != image_id_val:
            continue
        cat_id = ann.get('category_id')
        if cat_id not in category_to_index:
            continue
        x, y, w, h = ann.get('bbox', [0, 0, 0, 0])
        bbox = np.array([x, y, x + w, y + h], dtype=float)
        bbox[0] = max(0.0, min(bbox[0], width))
        bbox[1] = max(0.0, min(bbox[1], height))
        bbox[2] = max(0.0, min(bbox[2], width))
        bbox[3] = max(0.0, min(bbox[3], height))

        detections.append(Detection(bbox=bbox, confidence=1.0, class_id=category_to_index[cat_id]))

    if not detections:
        raise ValueError('선택된 이미지에 대한 어노테이션이 없습니다.')

    return detections, class_names, image_path


def format_yolo_results(detections: Iterable[Detection], class_names: Sequence[str]):
    formatted_results = []
    for det in detections:
        x_min, y_min, x_max, y_max = det.bbox
        corners = [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]
        label = class_names[int(det.class_id)]

        formatted_results.append({
            'label': label,
            'corners': corners,
            'confidence': det.confidence,
        })
    return formatted_results


def merge_nearby_points(points: np.ndarray, threshold: float = 3.0) -> np.ndarray:
    """주어진 좌표 배열에서 threshold 이하 거리에 있는 점들을 하나로 병합한다."""
    if points.size == 0:
        return points.astype(int)
    if len(points) == 1:
        return points.astype(int)

    coords = points.astype(float)
    tree = KDTree(coords)
    parents = np.arange(len(coords))

    def _find(i: int) -> int:
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    def _union(a: int, b: int) -> None:
        root_a, root_b = _find(a), _find(b)
        if root_a == root_b:
            return
        parents[root_b] = root_a

    for idx, coord in enumerate(coords):
        neighbors = tree.query_ball_point(coord, threshold)
        for neighbor_idx in neighbors:
            if neighbor_idx <= idx:
                continue
            _union(idx, neighbor_idx)

    clusters: Dict[int, List[np.ndarray]] = defaultdict(list)
    for idx, coord in enumerate(coords):
        root = _find(idx)
        clusters[root].append(coord)

    merged_points = []
    for cluster_coords in clusters.values():
        averaged = np.rint(np.mean(cluster_coords, axis=0)).astype(int)
        merged_points.append(averaged)

    return np.unique(np.array(merged_points, dtype=int), axis=0)


def clamp_unit_interval(value: float) -> float:
    if not np.isfinite(value):
        return 0.0
    if value <= 0.0:
        return 0.0
    if value >= 1.0:
        return 1.0
    return float(value)


def parse_room_box_file(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"방/계단/엘리베이터 텍스트 파일을 찾을 수 없습니다: {path}")

    annotations: List[Dict[str, Any]] = []
    with path.open('r', encoding='utf-8') as f:
        for raw_line in f:
            tokens = raw_line.strip().split()
            if len(tokens) < 5:
                continue

            label_token, cx_token, cy_token, w_token, h_token, provided_id = tokens
            try:
                cx = float(cx_token)
                cy = float(cy_token)
                width = float(w_token)
                height = float(h_token)
            except ValueError:
                continue

            x_min = clamp_unit_interval(cx - width / 2.0)
            y_min = clamp_unit_interval(cy - height / 2.0)
            x_max = clamp_unit_interval(cx + width / 2.0)
            y_max = clamp_unit_interval(cy + height / 2.0)

            annotations.append({
                'id': provided_id,
                'label_id': label_token,
                'x_min': min(x_min, x_max),
                'y_min': min(y_min, y_max),
                'x_max': max(x_min, x_max),
                'y_max': max(y_min, y_max),
            })

    return annotations


def parse_wall_segment_file(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"벽 선분 텍스트 파일을 찾을 수 없습니다: {path}")

    segments: List[Dict[str, Any]] = []
    with path.open('r', encoding='utf-8') as f:
        for raw_line in f:
            tokens = raw_line.strip().split()
            if len(tokens) < 4:
                continue

            numeric_tokens: List[float] = []
            for token in tokens[:4]:
                try:
                    numeric_tokens.append(float(token))
                except ValueError:
                    numeric_tokens = []
                    break
            if len(numeric_tokens) != 4:
                continue

            x1, y1, x2, y2 = numeric_tokens

            provided_id = tokens[4]

            segments.append({
                'id': provided_id,
                'x1': clamp_unit_interval(x1),
                'y1': clamp_unit_interval(y1),
                'x2': clamp_unit_interval(x2),
                'y2': clamp_unit_interval(y2),
            })

    return segments


def parse_door_point_file(
    path: Path,
    boxes: List[Dict[str, Any]],
    walls: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"문 포인트 텍스트 파일을 찾을 수 없습니다: {path}")

    boxes_by_id = {box['id']: box for box in boxes}
    walls_by_id = {wall['id']: wall for wall in walls}

    def _parse_anchor(tokens: List[str]) -> Optional[Dict[str, Any]]:
        if not tokens:
            return None
        anchor_type = tokens[0].lower()
        payload_tokens = tokens[1:]
        if anchor_type == 'line':
            line_id: Optional[str] = None
            t_value: Optional[float] = None

            remaining = payload_tokens
            if len(remaining) >= 3:
                _, id_token, t_token = remaining[:3]
                line_id = id_token
                try:
                    t_value = clamp_unit_interval(float(t_token))
                except ValueError:
                    t_value = None
            elif len(remaining) >= 2:
                id_token, t_token = remaining[:2]
                line_id = id_token
                try:
                    t_value = clamp_unit_interval(float(t_token))
                except ValueError:
                    t_value = None
            elif remaining:
                line_id = remaining[0]

            if line_id is None or line_id not in walls_by_id:
                return None

            anchor_payload: Dict[str, Any] = {
                'type': 'line',
                'id': line_id,
            }
            if t_value is not None:
                anchor_payload['t'] = t_value
            return anchor_payload

        if anchor_type == 'box':
            box_id: Optional[str] = None
            edge: Optional[str] = None
            t_value: Optional[float] = None

            remaining = payload_tokens
            if len(remaining) >= 4:
                _, id_token, edge_token, t_token = remaining[:4]
                box_id = id_token
                edge = edge_token.lower()
                try:
                    t_value = clamp_unit_interval(float(t_token))
                except ValueError:
                    t_value = None
            elif len(remaining) >= 3:
                id_token, edge_token, t_token = remaining[:3]
                box_id = id_token
                edge = edge_token.lower()
                try:
                    t_value = clamp_unit_interval(float(t_token))
                except ValueError:
                    t_value = None
            elif len(remaining) >= 2:
                id_token, edge_token = remaining[:2]
                box_id = id_token
                edge = edge_token.lower()
            elif remaining:
                box_id = remaining[0]

            if box_id is None or box_id not in boxes_by_id:
                return None

            anchor_payload = {
                'type': 'box',
                'id': box_id,
            }
            if edge in {'top', 'bottom', 'left', 'right'}:
                anchor_payload['edge'] = edge
            if t_value is not None:
                anchor_payload['t'] = t_value
            return anchor_payload

        return None

    door_points: List[Dict[str, Any]] = []
    with path.open('r', encoding='utf-8') as f:
        for line_index, raw_line in enumerate(f):
            tokens = raw_line.strip().split()
            if not tokens:
                continue

            cursor = 0
            label_id = '0'
            first_token = tokens[cursor]
            if first_token.lower() in {'door'} or first_token == '0':
                label_id = first_token
                cursor += 1

            if cursor + 1 >= len(tokens):
                continue

            try:
                x_value = float(tokens[cursor])
                y_value = float(tokens[cursor + 1])
            except ValueError:
                continue

            cursor += 2
            anchor = None
            if cursor < len(tokens):
                anchor = _parse_anchor(tokens[cursor:])
            door_points.append({
                'id': f"door-{line_index}",
                'label_id': label_id,
                'x': clamp_unit_interval(x_value),
                'y': clamp_unit_interval(y_value),
                'anchor': anchor,
            })

    return door_points


WALL_SEGMENT_THICKNESS_PX = 2.0
WALL_SEGMENT_EXTENSION_PX = 3.0
DOOR_LENGTH_MIN_PX = 5.0
DOOR_DEPTH_MIN_PX = 7.0
TMP_ROOM_EDGE_THICKNESS_PX = 3.0
TMP_ROOM_EDGE_EXTENSION_PX = 3.0
TMP_ROOM_GAP_FILL_MARGIN_PX = 1.0
TMP_ROOM_GAP_MAX_PX = 20.0
TMP_ROOM_MIN_OVERLAP_PX = 3.0

def _clip_point(point: np.ndarray, width: int, height: int) -> List[float]:
    x = float(np.clip(point[0], 0.0, max(width - 1, 0)))
    y = float(np.clip(point[1], 0.0, max(height - 1, 0)))
    return [x, y]


def _segment_to_rectangle(
    start: np.ndarray,
    end: np.ndarray,
    thickness: float,
    extension: float,
    width: int,
    height: int,
) -> Optional[Tuple[List[List[float]], np.ndarray, np.ndarray, float, np.ndarray, np.ndarray, np.ndarray, float]]:
    raw_start = np.array(start, dtype=float)
    raw_end = np.array(end, dtype=float)
    vec = raw_end - raw_start
    raw_length = float(np.linalg.norm(vec))
    if raw_length <= 1e-6:
        return None

    direction = vec / raw_length
    start_ext = raw_start.copy()
    end_ext = raw_end.copy()
    if extension > 0.0:
        start_ext = start_ext - direction * extension
        end_ext = end_ext + direction * extension

    normal = np.array([-direction[1], direction[0]])
    half_thickness = max(thickness / 2.0, 0.1)

    corners = [
        _clip_point(start_ext + normal * half_thickness, width, height),
        _clip_point(end_ext + normal * half_thickness, width, height),
        _clip_point(end_ext - normal * half_thickness, width, height),
        _clip_point(start_ext - normal * half_thickness, width, height),
    ]

    ext_length = float(np.linalg.norm(end_ext - start_ext))

    return corners, start_ext, end_ext, ext_length, direction, raw_start, raw_end, raw_length


def load_annotation_bundle_from_texts(
    room_path: Path,
    wall_path: Path,
    door_path: Path,
    image_width: int,
    image_height: int,
) -> Dict[str, List[Dict[str, Any]]]:
    boxes = parse_room_box_file(room_path)
    walls = parse_wall_segment_file(wall_path)
    doors = parse_door_point_file(door_path, boxes, walls)

    objects: Dict[str, List[Dict[str, Any]]] = {
        'rooms': [],
        'doors': [],
        'stairs': [],
        'elevators': [],
        'walls': [],
        'wall_segments': [],
    }

    box_lookup: Dict[str, Dict[str, Any]] = {}
    box_edges: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]] = {}
    wall_lookup: Dict[str, Dict[str, Any]] = {}
    wall_segments_for_search: List[Dict[str, Any]] = []
    wall_segment_lines: List[LineString] = []
    pending_room_edges: List[Dict[str, Any]] = []
    room_polygons_for_gap: Dict[str, Polygon] = {}

    for segment in walls:
        start = np.array([segment['x1'] * image_width, segment['y1'] * image_height], dtype=float)
        end = np.array([segment['x2'] * image_width, segment['y2'] * image_height], dtype=float)
        if np.linalg.norm(end - start) <= 1e-6:
            continue
        line = LineString([(float(start[0]), float(start[1])), (float(end[0]), float(end[1]))])
        if not line.is_empty:
            wall_segment_lines.append(line)
    def _register_wall_segment(
        start: np.ndarray,
        end: np.ndarray,
        thickness: float,
        extension: float,
        source_id: str,
        category: str,
    ) -> None:
        result = _segment_to_rectangle(start, end, thickness, extension, image_width, image_height)
        if result is None:
            return
        corners, adj_start, adj_end, length_ext, direction, raw_start, raw_end, raw_length = result

        poly = Polygon(corners)
        if poly.is_empty:
            return
        if not poly.is_valid:
            poly = poly.buffer(0)
            if poly.is_empty:
                return

        wall_obj = {
            'id': len(objects['walls']),
            'corners': corners,
            'polygon': poly,
            'centroid': poly.centroid,
            'source_id': source_id,
            'category': category,
        }
        objects['walls'].append(wall_obj)

        wall_lookup[source_id] = {
            'start': adj_start,
            'end': adj_end,
            'length': length_ext,
            'direction': direction,
            'raw_start': raw_start,
            'raw_end': raw_end,
            'raw_length': raw_length,
            'polygon': poly,
        }
        wall_segments_for_search.append({
            'raw_start': raw_start,
            'raw_end': raw_end,
            'raw_length': raw_length,
            'direction': direction,
            'id': source_id,
        })
        objects['wall_segments'].append({
            'id': source_id,
            'category': category,
            'start': [float(raw_start[0]), float(raw_start[1])],
            'end': [float(raw_end[0]), float(raw_end[1])],
        })
    def _add_box_object(annotation: Dict[str, Any], key: str) -> None:
        x_min = annotation['x_min'] * image_width
        y_min = annotation['y_min'] * image_height
        x_max = annotation['x_max'] * image_width
        y_max = annotation['y_max'] * image_height

        corners = [
            _clip_point(np.array([x_min, y_min]), image_width, image_height),
            _clip_point(np.array([x_max, y_min]), image_width, image_height),
            _clip_point(np.array([x_max, y_max]), image_width, image_height),
            _clip_point(np.array([x_min, y_max]), image_width, image_height),
        ]

        poly = Polygon(corners)
        if poly.is_empty:
            return
        if not poly.is_valid:
            poly = poly.buffer(0)
            if poly.is_empty:
                return

        obj = {
            'id': len(objects[key]),
            'corners': corners,
            'polygon': poly,
            'centroid': poly.centroid,
            'source_id': annotation['id'],
        }
        objects[key].append(obj)

        box_lookup[annotation['id']] = {
            'corners': corners,
            'polygon': poly,
            'key': key,
            'object_id': obj['id'],
        }
        if key in ('rooms', 'stairs', 'elevators'):
            room_polygons_for_gap[annotation['id']] = poly
        # 미리 모서리 벡터를 저장하여 문의 방향 계산에 활용
        c0 = np.array(corners[0])
        c1 = np.array(corners[1])
        c2 = np.array(corners[2])
        c3 = np.array(corners[3])
        box_edges[annotation['id']] = {
            'top': (c0, c1),          # left → right
            'right': (c1, c2),        # top → bottom
            'bottom': (c3, c2),       # left → right (along bottom edge)
            'left': (c0, c3),         # top → bottom
        }

        edge_specs = [
            ('top', c0.copy(), c1.copy(), f"tmp1_edge_{annotation['id']}") ,
            ('right', c1.copy(), c2.copy(), f"tmp2_edge_{annotation['id']}") ,
            ('bottom', c2.copy(), c3.copy(), f"tmp3_edge_{annotation['id']}") ,
            ('left', c3.copy(), c0.copy(), f"tmp4_edge_{annotation['id']}") ,
        ]

        if key == 'rooms':
            for edge_name, start, end, source_id in edge_specs:
                orientation = 'horizontal' if edge_name in ('top', 'bottom') else 'vertical'
                if orientation == 'horizontal':
                    coord = float(start[1])
                    range_min = float(min(start[0], end[0]))
                    range_max = float(max(start[0], end[0]))
                else:
                    coord = float(start[0])
                    range_min = float(min(start[1], end[1]))
                    range_max = float(max(start[1], end[1]))
                pending_room_edges.append({
                    'room_id': annotation['id'],
                    'edge_name': edge_name,
                    'start': start,
                    'end': end,
                    'thickness': TMP_ROOM_EDGE_THICKNESS_PX,
                    'extension': TMP_ROOM_EDGE_EXTENSION_PX,
                    'source_id': source_id,
                    'category': 'tmp_room_edge',
                    'orientation': orientation,
                    'coord': coord,
                    'range_min': range_min,
                    'range_max': range_max,
                })
        else:
            for _edge_name, start, end, source_id in edge_specs:
                _register_wall_segment(start, end, TMP_ROOM_EDGE_THICKNESS_PX, TMP_ROOM_EDGE_EXTENSION_PX, source_id, 'tmp_not_room_edge')

    def _build_gap_polygon(edge_a: Dict[str, Any], edge_b: Dict[str, Any]) -> Optional[Polygon]:
        if edge_a['orientation'] != edge_b['orientation']:
            return None
        if edge_a['orientation'] == 'vertical':
            x_left = min(edge_a['coord'], edge_b['coord'])
            x_right = max(edge_a['coord'], edge_b['coord'])
            if x_right <= x_left:
                return None
            y_low = max(edge_a['range_min'], edge_b['range_min'])
            y_high = min(edge_a['range_max'], edge_b['range_max'])
            if y_high <= y_low:
                return None
            coords = [
                (x_left, y_low),
                (x_right, y_low),
                (x_right, y_high),
                (x_left, y_high),
            ]
        else:
            y_low = min(edge_a['coord'], edge_b['coord'])
            y_high = max(edge_a['coord'], edge_b['coord'])
            if y_high <= y_low:
                return None
            x_left = max(edge_a['range_min'], edge_b['range_min'])
            x_right = min(edge_a['range_max'], edge_b['range_max'])
            if x_right <= x_left:
                return None
            coords = [
                (x_left, y_low),
                (x_right, y_low),
                (x_right, y_high),
                (x_left, y_high),
            ]
        poly = Polygon(coords)
        if poly.is_empty:
            return None
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty:
            return None
        return poly

    def _finalize_room_edges() -> None:
        if not pending_room_edges:
            return

        opposite_edges = {'left': 'right', 'right': 'left', 'top': 'bottom', 'bottom': 'top'}

        # 서로 마주보는 방 벽 사이의 유격을 줄여 무료 공간 누수를 방지
        for idx, edge in enumerate(pending_room_edges):
            opp_name = opposite_edges.get(edge['edge_name'])
            if opp_name is None:
                continue
            for jdx in range(idx + 1, len(pending_room_edges)):
                other = pending_room_edges[jdx]
                if other['edge_name'] != opp_name:
                    continue
                if edge['orientation'] != other['orientation']:
                    continue
                if edge['room_id'] == other['room_id']:
                    continue

                overlap = min(edge['range_max'], other['range_max']) - max(edge['range_min'], other['range_min'])
                if overlap <= TMP_ROOM_MIN_OVERLAP_PX:
                    continue

                raw_distance = abs(edge['coord'] - other['coord'])
                current_gap = raw_distance - (edge['thickness'] / 2.0 + other['thickness'] / 2.0)
                if current_gap <= 0.0:
                    continue
                if current_gap > TMP_ROOM_GAP_MAX_PX:
                    continue

                gap_poly = _build_gap_polygon(edge, other)
                if gap_poly is None:
                    continue

                blocked = False
                for room_id, poly in room_polygons_for_gap.items():
                    if room_id in (edge['room_id'], other['room_id']):
                        continue
                    if poly.intersects(gap_poly):
                        blocked = True
                        break
                if blocked:
                    continue

                for line in wall_segment_lines:
                    if line.intersects(gap_poly):
                        blocked = True
                        break
                if blocked:
                    continue

                desired_thickness = max(
                    edge['thickness'],
                    other['thickness'],
                    raw_distance + TMP_ROOM_GAP_FILL_MARGIN_PX,
                )
                edge['thickness'] = desired_thickness
                other['thickness'] = desired_thickness

        for edge in pending_room_edges:
            _register_wall_segment(edge['start'], edge['end'], edge['thickness'], edge['extension'], edge['source_id'], edge['category'])


    label_to_key = {
        '2': 'rooms',
        '3': 'stairs',
        '1': 'elevators',
    }

    for annotation in boxes:
        key = label_to_key.get(annotation['label_id'])
        if key is None:
            continue
        _add_box_object(annotation, key)

    _finalize_room_edges()

    for segment in walls:
        start = np.array([segment['x1'] * image_width, segment['y1'] * image_height], dtype=float)
        end = np.array([segment['x2'] * image_width, segment['y2'] * image_height], dtype=float)
        _register_wall_segment(start, end, WALL_SEGMENT_THICKNESS_PX, WALL_SEGMENT_EXTENSION_PX, segment['id'], 'annotation')

    def _compute_door_geometry(door: Dict[str, Any]) -> Optional[Tuple[List[List[float]], Polygon, Point]]:
        center = np.array([door['x'] * image_width, door['y'] * image_height], dtype=float)
        anchor = door.get('anchor') or {}
        tangent = None
        base_length = None

        if anchor.get('type') == 'box':
            box_id = anchor.get('id')
            edge_name = anchor.get('edge')
            t_value = anchor.get('t')
            edges = box_edges.get(box_id)

            edge_start: Optional[np.ndarray] = None
            edge_end: Optional[np.ndarray] = None

            if edges:
                if edge_name in edges:
                    edge_start, edge_end = edges[edge_name]
                    # door_wall_id = f"door-edge-{door['id']}"
                    # _register_wall_segment(edge_start, edge_end, 3.0, 0.0, door_wall_id, 'door_edge')
                else:
                    raise ValueError(f"문의 anchor box edge '{edge_name}' 이 올바르지 않습니다.")
                seg_vec = edge_end - edge_start
                base_length = float(np.linalg.norm(seg_vec))
                if base_length > 1e-6:
                    tangent = seg_vec / base_length
                    t_float = float(t_value) if isinstance(t_value, (int, float)) else 0.0
                    t_float = float(np.clip(t_float, 0.0, 1.0))
                    center[:] = edge_start + tangent * (base_length * t_float)

        elif anchor.get('type') == 'line':
            line_id = anchor.get('id')
            line_info = wall_lookup.get(line_id)
            if line_info is None:
                raise ValueError(f"문의 anchor line id '{line_id}' 을 찾을 수 없습니다.")
            tangent = line_info['direction']
            base_length = line_info.get('raw_length', line_info.get('length'))
            t_value = anchor.get('t')
            base_length = float(base_length or 0.0)
            if isinstance(t_value, (int, float)) and base_length > 0.0:
                origin = line_info.get('raw_start', line_info.get('start'))
                t_float = float(np.clip(float(t_value), 0.0, 1.0))
                center[:] = origin + tangent * (base_length * t_float)

        if tangent is None:
            raise ValueError(f"문의 anchor 정보가 올바르지 않습니다. (anchor: {anchor.get('type')}) anchor_id: {anchor.get('id')}")
        
        # 만약 tangent가 너무 작은 경우에 쓰는건데 필요할 경우 주석해제하자.
        # tangent_norm = float(np.linalg.norm(tangent))
        # if tangent_norm <= 1e-6:
        #     tangent = np.array([1.0, 0.0])
        #     tangent_norm = 1.0
        # tangent = tangent / tangent_norm
        normal = np.array([-tangent[1], tangent[0]])

        door_length = float(min(DOOR_LENGTH_MIN_PX, base_length))
        canvas_scale = float(max(image_width, image_height))
        door_depth = float(min(canvas_scale * 0.015, DOOR_DEPTH_MIN_PX))

        half_len = door_length / 2.0
        half_depth = door_depth / 2.0

        corners = [
            _clip_point(center + tangent * half_len + normal * half_depth, image_width, image_height),
            _clip_point(center - tangent * half_len + normal * half_depth, image_width, image_height),
            _clip_point(center - tangent * half_len - normal * half_depth, image_width, image_height),
            _clip_point(center + tangent * half_len - normal * half_depth, image_width, image_height),
        ]

        poly = Polygon(corners)
        if poly.is_empty:
            return None
        if not poly.is_valid:
            poly = poly.buffer(0)
            if poly.is_empty:
                return None
        return corners, poly, poly.centroid, normal

    for door in doors:
        geom = _compute_door_geometry(door)
        if geom is None:
            continue
        corners, poly, centroid, normal = geom
        door_obj = {
            'id': len(objects['doors']),
            'corners': corners,
            'normal_direction': normal,
            'polygon': poly,
            'centroid': centroid,
            'source_id': door['id'],
            'anchor': door.get('anchor'),
        }
        objects['doors'].append(door_obj)

    return objects

class FloorPlanParser:
    def __init__(self):
        self.objects = {}

    def parse_yolo_output(
        self,
        formatted_results,
        include_classes: Optional[Sequence[str]] = None,
        exclude_classes: Optional[Sequence[str]] = None,
    ):
        include_set = {name.lower() for name in include_classes} if include_classes else None
        exclude_set = {name.lower() for name in exclude_classes} if exclude_classes else None

        self.objects = {
            'rooms': [],
            'doors': [],
            'stairs': [],
            'elevators': [],
            'walls': [],
            'wall_segments': [],
        }
        label_map = {
            'room': 'rooms',
            'door': 'doors',
            'stairs': 'stairs',
            'elevator': 'elevators',
            'wall': 'walls',
        }

        def _append_wall_edges(corners: Sequence[Sequence[float]], source_id: str, category: str) -> None:
            if not corners:
                return
            coords = [np.asarray(pt, dtype=float) for pt in corners]
            if len(coords) < 2:
                return
            for idx in range(len(coords)):
                start = coords[idx]
                end = coords[(idx + 1) % len(coords)]
                if np.allclose(start, end):
                    continue
                segment = {
                    'id': f"{source_id}-edge-{idx}",
                    'category': category,
                    'start': [float(start[0]), float(start[1])],
                    'end': [float(end[0]), float(end[1])],
                }
                self.objects['wall_segments'].append(segment)

        for det in formatted_results:
            lbl = det['label']
            if lbl not in label_map:
                continue

            lbl_key = lbl.lower()
            if include_set is not None and lbl_key not in include_set:
                continue
            if exclude_set is not None and lbl_key in exclude_set:
                continue

            key = label_map[lbl]
            poly = Polygon(det['corners'])
            self.objects[key].append({
                'id': len(self.objects[key]),
                'corners': det['corners'],
                'polygon': poly,
                'centroid': poly.centroid,
                # 'confidence': det.get('confidence', 1.0)
            })
            if key == 'walls':
                _append_wall_edges(det['corners'], f"wall-{len(self.objects[key]) - 1}", 'bbox')
        self._extend_doors_along_walls()
        self.annotate_room_door_connections()
        return self.objects

    def _extend_doors_along_walls(self) -> None:
        doors = self.objects.get('doors') or []
        if not doors:
            return

        walls = [
            wall
            for wall in self.objects.get('walls', [])
            if isinstance(wall.get('polygon'), Polygon) and not wall['polygon'].is_empty
        ]

        for door in doors:
            door_poly: Optional[Polygon] = door.get('polygon')
            if not isinstance(door_poly, Polygon) or door_poly.is_empty:
                door['parallel_edge_midpoints'] = []
                door['parallel_edge_contacts'] = {}
                continue

            if not door_poly.is_valid:
                door_poly = door_poly.buffer(0)
            if door_poly.is_empty:
                door['parallel_edge_midpoints'] = []
                door['parallel_edge_contacts'] = {}
                continue

            minx, miny, maxx, maxy = door_poly.bounds
            if maxx <= minx or maxy <= miny:
                door['parallel_edge_midpoints'] = []
                door['parallel_edge_contacts'] = {}
                continue

            best_overlap_area = 0.0
            dominant_wall: Optional[Dict[str, Any]] = None
            for wall in walls:
                wall_poly = wall.get('polygon')
                if wall_poly is None or wall_poly.is_empty:
                    continue
                if not wall_poly.is_valid:
                    wall_poly = wall_poly.buffer(0)
                    if wall_poly.is_empty:
                        continue
                if not door_poly.intersects(wall_poly):
                    continue
                intersection = door_poly.intersection(wall_poly)
                if intersection.is_empty:
                    continue
                overlap_area = intersection.area
                if overlap_area > best_overlap_area:
                    best_overlap_area = overlap_area
                    dominant_wall = wall

            def _edge_line(bounds: Sequence[float], edge_name: str) -> LineString:
                x1, y1, x2, y2 = bounds
                if edge_name == 'left':
                    return LineString([(x1, y1), (x1, y2)])
                if edge_name == 'right':
                    return LineString([(x2, y1), (x2, y2)])
                if edge_name == 'top':
                    return LineString([(x1, y1), (x2, y1)])
                if edge_name == 'bottom':
                    return LineString([(x1, y2), (x2, y2)])
                raise ValueError(f"Unsupported edge name: {edge_name}")

            dominant_bounds: Optional[Tuple[float, float, float, float]] = None
            if dominant_wall is not None:
                wall_poly = dominant_wall.get('polygon')
                if isinstance(wall_poly, Polygon) and not wall_poly.is_empty:
                    if not wall_poly.is_valid:
                        wall_poly = wall_poly.buffer(0)
                    if not wall_poly.is_empty:
                        dominant_bounds = wall_poly.bounds

            width = maxx - minx
            height = maxy - miny
            if dominant_bounds is not None:
                wall_width = dominant_bounds[2] - dominant_bounds[0]
                wall_height = dominant_bounds[3] - dominant_bounds[1]
                use_horizontal = wall_width >= wall_height
            else:
                use_horizontal = width >= height

            target_edges: Tuple[str, str]
            if use_horizontal:
                target_edges = ('top', 'bottom')
            else:
                target_edges = ('left', 'right')

            bounds = [float(minx), float(miny), float(maxx), float(maxy)]

            def _collect_edge_contacts(edge_name: str) -> Tuple[Set[str], List[Dict[str, Any]]]:
                edge_line = _edge_line((minx, miny, maxx, maxy), edge_name)
                probe_geom = edge_line.buffer(0.5, cap_style=2)
                contact_types: Set[str] = set()
                touching_walls: List[Dict[str, Any]] = []
                for key in ('walls', 'rooms', 'doors', 'stairs', 'elevators'):
                    objs = self.objects.get(key, [])
                    for candidate in objs:
                        if key == 'doors' and candidate is door:
                            continue
                        poly = candidate.get('polygon')
                        if poly is None or poly.is_empty:
                            continue
                        current_poly = poly
                        if not current_poly.is_valid:
                            current_poly = current_poly.buffer(0)
                            if current_poly.is_empty:
                                continue
                        if probe_geom.intersects(current_poly):
                            contact_types.add(key)
                            if key == 'walls':
                                touching_walls.append(candidate)
                return contact_types, touching_walls

            edge_contact_map: Dict[str, Dict[str, Any]] = {}
            for edge_name in target_edges:
                contacts, touching_walls = _collect_edge_contacts(edge_name)
                edge_contact_map[edge_name] = {
                    'contacts': sorted(contacts),
                    'touching_walls': touching_walls,
                }

            def _extend_bounds(edge_name: str, touching_wall_objs: List[Dict[str, Any]]) -> None:
                if not touching_wall_objs:
                    return
                valid_wall_bounds: List[Tuple[float, float, float, float]] = []
                for wall_obj in touching_wall_objs:
                    wall_poly = wall_obj.get('polygon')
                    if wall_poly is None or wall_poly.is_empty:
                        continue
                    current_poly = wall_poly
                    if not current_poly.is_valid:
                        current_poly = current_poly.buffer(0)
                        if current_poly.is_empty:
                            continue
                    valid_wall_bounds.append(current_poly.bounds)
                if not valid_wall_bounds:
                    return

                offset = 1.0
                if edge_name == 'left':
                    target_val = min(b[0] for b in valid_wall_bounds) - offset
                    candidate = min(bounds[0], target_val)
                    if candidate < bounds[0]:
                        bounds[0] = max(candidate, 0.0)
                elif edge_name == 'right':
                    target_val = max(b[2] for b in valid_wall_bounds) + offset
                    candidate = max(bounds[2], target_val)
                    if candidate > bounds[2]:
                        bounds[2] = candidate
                elif edge_name == 'top':
                    target_val = min(b[1] for b in valid_wall_bounds) - offset
                    candidate = min(bounds[1], target_val)
                    if candidate < bounds[1]:
                        bounds[1] = max(candidate, 0.0)
                elif edge_name == 'bottom':
                    target_val = max(b[3] for b in valid_wall_bounds) + offset
                    candidate = max(bounds[3], target_val)
                    if candidate > bounds[3]:
                        bounds[3] = candidate

            for edge_name in target_edges:
                info = edge_contact_map.get(edge_name)
                if not info:
                    continue
                if 'walls' in info['contacts']:
                    _extend_bounds(edge_name, info['touching_walls'])

            new_minx, new_miny, new_maxx, new_maxy = bounds
            if new_maxx <= new_minx or new_maxy <= new_miny:
                door['parallel_edge_midpoints'] = []
                door['parallel_edge_contacts'] = {}
                continue

            updated_corners = [
                [new_minx, new_miny],
                [new_maxx, new_miny],
                [new_maxx, new_maxy],
                [new_minx, new_maxy],
            ]
            updated_poly = Polygon(updated_corners)
            if not updated_poly.is_valid:
                updated_poly = updated_poly.buffer(0)
            if updated_poly.is_empty:
                door['parallel_edge_midpoints'] = []
                door['parallel_edge_contacts'] = {}
                continue

            door['corners'] = updated_corners
            door['polygon'] = updated_poly
            door['centroid'] = updated_poly.centroid
            door['parallel_edge_contacts'] = {
                edge: edge_contact_map[edge]['contacts']
                for edge in target_edges
                if edge in edge_contact_map
            }

            parallel_midpoints: List[Dict[str, Any]] = []
            for edge_name in target_edges:
                info = edge_contact_map.get(edge_name)
                edge_line = _edge_line((new_minx, new_miny, new_maxx, new_maxy), edge_name)
                if edge_line.length == 0:
                    continue
                midpoint_x, midpoint_y = edge_line.interpolate(0.5, normalized=True).coords[0]
                midpoint_row = int(round(midpoint_y))
                midpoint_col = int(round(midpoint_x))
                parallel_midpoints.append({
                    'edge': edge_name,
                    'midpoint_xy': (midpoint_x, midpoint_y),
                    'midpoint_rc': (midpoint_row, midpoint_col),
                    'contacts': info['contacts'] if info else [],
                    'touches_wall': bool(info and 'walls' in info['contacts']),
                })

            door['parallel_edge_midpoints'] = parallel_midpoints

    def annotate_room_door_connections(self, min_overlap_area: float = 1.0) -> None:
        """Assign bidirectional room/door connectivity metadata based on polygon overlap.

        Doors gain `connected_room_ids` that mirror the `id` values of overlapping rooms,
        while rooms expose `connected_door_ids` so downstream code can quickly check door counts.
        """
        rooms = self.objects.get('rooms', [])
        doors = self.objects.get('doors', [])

        if not rooms or not doors:
            for room in rooms:
                room['connected_door_ids'] = []
            for door in doors:
                door['connected_room_ids'] = []
            return

        def _normalise(poly: Polygon) -> Polygon:
            if poly.is_empty:
                return poly
            if not poly.is_valid:
                poly = poly.buffer(0)
            return poly

        for room in rooms:
            room_poly = _normalise(room['polygon'])
            room['polygon'] = room_poly
            room['connected_door_ids'] = []

        prepared_rooms = [(room, prep(room['polygon'])) for room in rooms if not room['polygon'].is_empty]

        for door in doors:
            door_poly = _normalise(door['polygon'])
            door['polygon'] = door_poly
            door['connected_room_ids'] = []
            if door_poly.is_empty:
                continue
            for room, prepared in prepared_rooms:
                if not prepared.intersects(door_poly):
                    continue
                overlap_area = door_poly.intersection(room['polygon']).area
                # if overlap_area <= min_overlap_area:
                #     continue
                door['connected_room_ids'].append(room['id'])
                room['connected_door_ids'].append(door['id'])

        for door in doors:
            if door['connected_room_ids']:
                unique_room_ids = sorted(set(door['connected_room_ids']))
                door['connected_room_ids'] = unique_room_ids

        for room in rooms:
            if room['connected_door_ids']:
                unique_door_ids = sorted(set(room['connected_door_ids']))
                room['connected_door_ids'] = unique_door_ids


class FloorPlanNavigationGraph:
    """
    벡터화된 마스크 생성과 A* 경로 탐색을 사용하여
    성능과 정확도를 모두 개선한 내비게이션 그래프 생성기.
    """
    def __init__(self, parsed_objects, width, height, debug_dir: Optional[Path] = None):
        self.objs = parsed_objects
        self.w, self.h = width, height
        self.G = nx.Graph()
        self.corridor_nodes_pos = {}
        self.corridor_endpoint_door_links = {}
        self.corridor_endpoint_room_links = {}
        self.door_endpoint_pixels = {}
        self.door_endpoint_rooms = {}
        self.newly_found_endpoint_coords: Set[Tuple[int, int]] = set()
        self.newly_found_corridor_nodes: Set[str] = set()
        self.door_to_corridor_nodes = defaultdict(list)
        self.corridor_kdtree = None
        self.debug_dir = debug_dir

    def build(self):
        # for i in range(len(self.objs.get('doors', []))):
        #     mask = self._create_free_space_mask_optimized(i)
        #     if self.debug_dir:
        #         self._save_mask(mask, f'free_space_mask_{i}.png')
        # i=26
        mask, door_mask, room_mask, wall_mask = self._create_free_space_mask_optimized()
        if self.debug_dir:
            self._save_mask(mask, 'free_space_mask.png')
        # sys.exit(0)
        skeleton = self._create_corridor_skeleton(mask, door_mask)
        self._remove_temporary_wall_edges()
        if self.debug_dir:
            self._save_mask(skeleton, 'skeleton.png', scale_to_uint8=True)
        # sys.exit(0)
        self._convert_skeleton_to_graph_optimized(skeleton, door_mask, room_mask, wall_mask)
        self._connect_objects_to_graph_optimized()
        self._simplify_graph_with_all_shortest_paths()

        if self.debug_dir:
            self._save_final_visualization()

        return self.G
    def _simplify_graph_with_all_shortest_paths(self) -> None:
        """
        모든 중요 지점(터미널) 쌍 간의 최단 경로를 모두 보존하는 방식으로 그래프를 단순화함.
        """
        # print("모든 최단 경로 보존 방식으로 그래프 단순화를 시작합니다...")

        # 1. 터미널 노드(반드시 연결되어야 할 중요 노드) 식별
        terminal_nodes = [
            n for n, data in self.G.nodes(data=True)
            if data.get('type') not in {'corridor'}
        ]

        if len(terminal_nodes) < 2:
            # print("터미널 노드가 부족하여 단순화를 건너뜁니다.")
            return

        # print(f"{len(terminal_nodes)}개의 터미널 노드 간 모든 최단 경로를 계산합니다...")
        
        essential_nodes = set(terminal_nodes)
        essential_edges = set()

        # 2. 모든 터미널 노드 쌍(combinations)에 대해 최단 경로 계산
        for start_node, end_node in combinations(terminal_nodes, 2):
            try:
                # Dijkstra 알고리즘으로 최단 경로를 찾음
                path = nx.dijkstra_path(self.G, source=start_node, target=end_node, weight='weight')
                
                # 3. 경로에 포함된 모든 노드와 엣지를 '필수' 요소로 추가
                essential_nodes.update(path)
                
                # 경로를 엣지 쌍으로 변환하여 추가
                for i in range(len(path) - 1):
                    # 엣지는 순서에 상관없이 저장하기 위해 정렬
                    edge = tuple(sorted((path[i], path[i+1])))
                    essential_edges.add(edge)

            except nx.NetworkXNoPath:
                # 두 터미널 사이에 경로가 없는 경우임
                # print(f"경고: 노드 {start_node}와 {end_node} 사이에 경로가 없습니다.")
                continue
        
        # 4. 필수적이지 않은 노드와 엣지 제거
        original_nodes = set(self.G.nodes())
        nodes_to_remove = original_nodes - essential_nodes
        
        # original_node_count = self.G.number_of_nodes()
        self.G.remove_nodes_from(list(nodes_to_remove))
        # simplified_node_count = self.G.number_of_nodes()
        
        # print(f"그래프 단순화 완료. 노드 수: {original_node_count} -> {simplified_node_count}")
        
        # 그래프 변경에 따른 의존적인 멤버 변수들 업데이트
        self.corridor_nodes_pos = {
            n: data['pos']
            for n, data in self.G.nodes(data=True)
            if 'pos' in data
        }
        if self.corridor_nodes_pos:
            self.corridor_kdtree = KDTree(list(self.corridor_nodes_pos.values()))
        else:
            self.corridor_kdtree = None
    def _create_free_space_mask_optimized(self):

        obstacle = np.zeros((self.h, self.w), dtype=np.uint8)
        room_mask = np.zeros((self.h, self.w), dtype=np.uint8)
        wall_mask = np.zeros((self.h, self.w), dtype=np.uint8)
        door_mask = np.zeros((self.h, self.w), dtype=np.uint8)
        door_boxes: List[np.ndarray] = []

        obstacle_categories = ("rooms", "walls", "stairs", "elevators")
        for key in obstacle_categories:
            for obj in self.objs.get(key, []):
                poly_corners = np.clip(np.array(obj['corners']), [0, 0], [self.w - 1, self.h - 1])
                rr, cc = draw_polygon(
                    np.round(poly_corners[:, 1]).astype(int),
                    np.round(poly_corners[:, 0]).astype(int),
                    (self.h, self.w),
                )
                obstacle[rr, cc] = 1
                if key == 'rooms':
                    room_mask[rr, cc] = 1
                if key == 'walls':
                    wall_mask[rr, cc] = 1

        # 문 마스크 채우기 및 장애물에서 문 영역 제외
        for obj in self.objs.get('doors', []):
            poly_corners = np.clip(np.array(obj['corners']), [0, 0], [self.w - 1, self.h - 1])
            rr, cc = draw_polygon(
                np.round(poly_corners[:, 1]).astype(int),
                np.round(poly_corners[:, 0]).astype(int),
                (self.h, self.w),
            )
            door_mask[rr, cc] = 1
            obstacle[rr, cc] = 0  # 장애물에서 문 제거
            room_mask[rr, cc] = 0   # 방에서 문 제거
            door_boxes.append(poly_corners)
        
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed_obstacle = cv2.morphologyEx(obstacle, cv2.MORPH_DILATE, kernel_close, iterations=1)
        closed_obstacle = obstacle  # 벽 두께가 충분히 두꺼운 경우 닫기 연산 생략 가능
        building_mask = binary_fill_holes(closed_obstacle.astype(bool)).astype(np.uint8)
        free_space = building_mask & (~obstacle)
        free_space[door_mask.astype(bool)] = 0 # 문 영역은 아직 복도가 아님

        # 벽으로 사방이 막혀 있고 문과 접하지 않는 영역은 폐쇄된 공간으로 간주해 제거
        structure = np.ones((3, 3), dtype=np.int8)
        labeled_components, num_components = label(free_space.astype(bool), structure=structure)
        if num_components:
            door_touch_mask = cv2.dilate(door_mask.astype(np.uint8), np.ones((3, 3), dtype=np.uint8), iterations=1).astype(bool)
            unreachable_mask = np.zeros_like(free_space, dtype=bool)

            for component_idx in range(1, num_components + 1):
                component_mask = labeled_components == component_idx
                if not np.any(component_mask):
                    continue

                if np.any(door_touch_mask[component_mask]):
                    continue

                touches_border = (
                    np.any(component_mask[0, :])
                    or np.any(component_mask[-1, :])
                    or np.any(component_mask[:, 0])
                    or np.any(component_mask[:, -1])
                )

                if touches_border:
                    continue

                unreachable_mask |= component_mask

            if np.any(unreachable_mask):
                obstacle[unreachable_mask] = 1
                wall_mask[unreachable_mask] = 1
                free_space[unreachable_mask] = 0

                building_mask = binary_fill_holes(obstacle.astype(bool)).astype(np.uint8)
                free_space = building_mask & (~obstacle)
                free_space[door_mask.astype(bool)] = 0

        MAX_PROBE_DISTANCE = 10      # 복도를 찾기 위해 탐색할 최대 거리 (픽셀)

        for corners in door_boxes:
            # 현재 문 하나의 마스크를 생성
            current_door_mask = np.zeros((self.h, self.w), dtype=np.uint8)
            rr, cc = draw_polygon(
                np.round(corners[:, 1]).astype(int),
                np.round(corners[:, 0]).astype(int),
                (self.h, self.w),
            )
            current_door_mask[rr, cc] = 1

            overlap_mask = current_door_mask & wall_mask

            if np.sum(overlap_mask) == 0:
                continue
            min_r, max_r = np.min(rr), np.max(rr)
            min_c, max_c = np.min(cc), np.max(cc)
            door_centroid = np.array([(min_c + max_c) / 2.0, (min_r + max_r) / 2.0])

            overlap_coords = np.argwhere(overlap_mask)
            overlap_min_r, overlap_min_c = overlap_coords.min(axis=0)
            overlap_max_r, overlap_max_c = overlap_coords.max(axis=0)
            overlap_centroid = np.array([(overlap_min_c + overlap_max_c) / 2.0, (overlap_min_r + overlap_max_r) / 2.0])

            exit_vector = overlap_centroid - door_centroid
            norm = np.linalg.norm(exit_vector)
            if norm < 1e-6:
                continue
            exit_vector /= norm
            exit_vector=[round(exit_vector[0]), round(exit_vector[1])]
            
            is_vertical_exit = abs(exit_vector[1]) > abs(exit_vector[0])
        
            door_center_c = int(round(door_centroid[0]))
            door_center_r = int(round(door_centroid[1]))

            if is_vertical_exit:
                # 주된 방향이 수직일 경우
                start_c = door_center_c
                if exit_vector[1] < 0: # 위로
                    start_r = min_r
                else: # 아래로
                    start_r = max_r
            else:
                # 주된 방향이 수평일 경우
                start_r = door_center_r
                if exit_vector[0] < 0: # 왼쪽으로
                    start_c = min_c
                else: # 오른쪽으로
                    start_c = max_c

            for i in range(1, MAX_PROBE_DISTANCE + 1):
                probe_c = int(start_c + exit_vector[0] * i)
                probe_r = int(start_r + exit_vector[1] * i)

                if not (0 <= probe_r < self.h and 0 <= probe_c < self.w):
                    break

                if free_space[probe_r, probe_c] == 1:
                    # print(f"복도 발견 at distance {i} pixels")
                    # 벡터 방향에 따라 문의 너비/높이를 결정
                    is_vertical_exit = abs(exit_vector[1]) > abs(exit_vector[0])
                    if is_vertical_exit: # 수직 방향으로 연결
                        pt1 = (min_c, probe_r if exit_vector[1] < 0 else max_r)
                        pt2 = (max_c, min_r if exit_vector[1] < 0 else probe_r)
                    else: # 수평 방향으로 연결
                        pt1 = (probe_c if exit_vector[0] < 0 else max_c, min_r)
                        pt2 = (min_c if exit_vector[0] < 0 else probe_c, max_r)
                    
                    # 좌표가 올바른 순서(좌상단, 우하단)가 되도록 정렬
                    final_pt1 = (min(pt1[0], pt2[0]), min(pt1[1], pt2[1]))
                    final_pt2 = (max(pt1[0], pt2[0]), max(pt1[1], pt2[1]))

                    # Rect 형식으로 함수 호출
                    cv2.rectangle(img=free_space, pt1=final_pt1,pt2=final_pt2, color=1, thickness=-1)
                    break
                
                # if obstacle[probe_r, probe_c] == 1:
                #     break # 다른 장애물을 만나면 중단
        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        free_space = cv2.morphologyEx(free_space, cv2.MORPH_OPEN, kernel_open, iterations=1)

        # 최종적으로 문 자체도 free_space에 포함
        free_space[door_mask.astype(bool)] = 1

        return free_space.astype(bool), door_mask.astype(bool), room_mask.astype(bool), wall_mask.astype(bool)
    
    def _create_corridor_skeleton(self, mask, door_mask):
        # dist = distance_transform_edt(mask)
        # thresh = dist.max() * 0.1
        # with_door_mask = (dist>thresh)| door_mask
        skel= skeletonize(mask, method='zhang')
        # return skel
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed_skel = cv2.morphologyEx(skel.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        closed_skel = skeletonize(closed_skel.astype(bool), method='zhang')
        return closed_skel.astype(bool)

    def _remove_temporary_wall_edges(self) -> None:
        tmp_categories = {'tmp_not_room_edge'}
        walls = self.objs.get('walls')
        if walls:
            filtered_walls = [
                wall
                for wall in walls
                if (wall.get('category') or '').lower() not in tmp_categories
            ]
            if len(filtered_walls) != len(walls):
                self.objs['walls'] = filtered_walls

        wall_segments = self.objs.get('wall_segments')
        if wall_segments:
            filtered_segments = [
                seg
                for seg in wall_segments
                if (seg.get('category') or '').lower() not in tmp_categories
            ]
            if len(filtered_segments) != len(wall_segments):
                self.objs['wall_segments'] = filtered_segments

    def _save_mask(self, mask: np.ndarray, filename: str, scale_to_uint8: bool = False) -> None:
        if not self.debug_dir:
            return
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.debug_dir / filename
        if scale_to_uint8:
            data = (mask.astype(np.uint8) * 255)
        else:
            data = (mask.astype(np.uint8) * 255)
        cv2.imwrite(str(output_path), data)

    def _save_final_visualization(self) -> None:
        if not self.debug_dir:
            return

        try:
            self.debug_dir.mkdir(parents=True, exist_ok=True)
            visualizer = FloorPlanVisualizer(self.G, self.objs)
            output_path = self.debug_dir / EXPORT_IMAGE_FILENAME
            visualizer.show(
                path=None,
                output_path=output_path,
                show_window=False,
                annotate_connections=True,
                save_out=True,
            )
        except Exception as exc:
            print(f"경고: 최종 그래프 시각화를 저장하지 못했습니다: {exc}")

    def _astar_path(self, skeleton, start, end):
        """ A* 알고리즘으로 스켈레톤 상의 최단 경로와 거리를 계산 """
        start, end = tuple(start), tuple(end)
        open_set = [(0, start)] # (f_score, node)
        came_from = {}
        g_score = {start: 0}

        def heuristic(a, b):
            return np.hypot(a[0] - b[0], a[1] - b[1])

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == end:
                return g_score[end] # 경로 길이 반환

            for dy, dx in [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
                neighbor = (current[0] + dy, current[1] + dx)
                
                if not (0 <= neighbor[0] < skeleton.shape[0] and 0 <= neighbor[1] < skeleton.shape[1]):
                    continue
                if not skeleton[neighbor[0], neighbor[1]]:
                    continue

                tentative_g_score = g_score[current] + np.hypot(dy, dx)
                
                if tentative_g_score < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score = tentative_g_score + heuristic(neighbor, end)
                    heapq.heappush(open_set, (f_score, neighbor))
        return None # 경로 없음

    def _convert_skeleton_to_graph_optimized(self, skel, door_mask, room_mask, wall_mask):
        """ 키포인트 감지 및 A*를 이용한 정확한 엣지 가중치 계산 """
        self.corridor_endpoint_door_links = {}
        self.corridor_endpoint_room_links = {}
        self.door_endpoint_pixels = {}
        self.door_endpoint_rooms = {}
        self.newly_found_endpoint_coords = set()
        self.newly_found_corridor_nodes = set()
        self.door_to_corridor_nodes = defaultdict(list)

        def _new_endpoint_info() -> Dict[str, Set[int]]:
            return {'door_ids': set(), 'room_ids': set()}

        def _merge_endpoint_info(dest: Dict[str, Set[int]], src: Optional[Dict[str, Set[int]]]) -> None:
            if not src:
                return
            door_ids = src.get('door_ids')
            room_ids = src.get('room_ids')
            if door_ids:
                dest['door_ids'].update(int(val) for val in door_ids)
            if room_ids:
                dest['room_ids'].update(int(val) for val in room_ids)

        def _has_endpoint_links(info: Dict[str, Set[int]]) -> bool:
            return bool(info.get('door_ids') or info.get('room_ids'))

        room_polygons_raw: List[Tuple[int, Polygon]] = []
        for room in self.objs.get('rooms', []):
            poly = room.get('polygon')
            if isinstance(poly, Polygon) and not poly.is_empty:
                room_polygons_raw.append((room['id'], poly))
        prepared_rooms_for_midpoints = [
            (room_id, prep(poly))
            for room_id, poly in room_polygons_raw
        ]

        kernel = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]])
        neighbors = convolve2d(skel.astype(np.uint8), kernel, mode='same')
        interest_points = neighbors * skel

        structural_forbidden_mask = wall_mask.astype(bool) | room_mask.astype(bool)
        door_forbidden_mask = door_mask.astype(bool)
        combined_forbidden_mask = structural_forbidden_mask | door_forbidden_mask
        junction_mask = (interest_points >= 13)
        junction_points_to_remove = junction_mask & combined_forbidden_mask
        final_junction_mask = junction_mask & (~junction_points_to_remove)

        labeled_junctions, num_labels = label(final_junction_mask)
        if num_labels > 0:
            junction_coords_raw = center_of_mass(final_junction_mask, labeled_junctions, range(1, num_labels + 1))
            junction_coords_arr = np.asarray(junction_coords_raw, dtype=float)
            if junction_coords_arr.ndim == 1:
                junction_coords_arr = junction_coords_arr.reshape(1, -1)
            junction_coords = np.round(junction_coords_arr).astype(int)
        else:
            junction_coords = np.empty((0, 2), dtype=int)

        diagonal_segments = skeleton_to_segments(skel)
        diagonal_segment_points = []
        for seg in diagonal_segments:
            if len(seg) > 0:
                diagonal_segment_points.append(seg[0])  # 시작점
                diagonal_segment_points.append(seg[-1]) # 끝점
        if diagonal_segment_points:
            diagonal_segment_points_np = np.asarray(diagonal_segment_points, dtype=int)
            if diagonal_segment_points_np.ndim == 1:
                diagonal_segment_points_np = diagonal_segment_points_np.reshape(1, -1)
        else:
            diagonal_segment_points_np = np.empty((0, 2), dtype=int)
        # print(f"중복 제거 전 교차로 개수: {np.sum(final_junction_mask)}")
        # print(f"중복 제거 후 교차로 개수: {len(junction_coords)}")

        # junction_coords = np.argwhere(final_junction_mask)

        endpoint_mask = (interest_points == 11)
        original_endpoint_coords = np.argwhere(endpoint_mask)

        diagonal_combined_points = np.vstack((original_endpoint_coords, diagonal_segment_points_np))
        diagonal_unique_combined_points = np.unique(diagonal_combined_points, axis=0)
        # --- 2. 끝점을 '안전한 끝점'과 '후퇴 필요한 끝점'으로 분리 ---
        forbidden_mask = structural_forbidden_mask

        safe_endpoints = []
        retracing_needed_endpoints = []

        for r, c in diagonal_unique_combined_points:
            if structural_forbidden_mask[r, c]:
                retracing_needed_endpoints.append((r, c))
            elif door_forbidden_mask[r, c]:
                continue
            else:
                safe_endpoints.append((r, c))
        # --- 3. 후퇴가 필요한 끝점에 대해 후퇴 수행 ---
        newly_found_endpoints = []
        skel_for_tracing = skel.astype(bool).copy()

        for r_start, c_start in retracing_needed_endpoints:
            current_r, current_c = r_start, c_start

            encountered_door_mask = False

            for _ in range(np.sum(skel)):
                if door_forbidden_mask[current_r, current_c]:
                    encountered_door_mask = True
                    break
                if not forbidden_mask[current_r, current_c]:
                    newly_found_endpoints.append((current_r, current_c))
                    break
                skel_for_tracing[current_r, current_c] = False

                found_next = False
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0:
                            continue
                        
                        next_r, next_c = current_r + dr, current_c + dc
                        
                        if not (0 <= next_r < self.h and 0 <= next_c < self.w):
                            continue
                        if skel_for_tracing[next_r, next_c]:
                            current_r, current_c = next_r, next_c
                            found_next = True
                            break
                    if found_next:
                        break
                if not found_next:
                    break
            if encountered_door_mask:
                continue
        if newly_found_endpoints:
            newly_found_endpoints_np = np.unique(np.array(newly_found_endpoints), axis=0)
        else:
            newly_found_endpoints_np = np.empty((0, 2), dtype=int)

        newly_found_coord_set: Set[Tuple[int, int]] = {
            (int(coord[0]), int(coord[1]))
            for coord in newly_found_endpoints_np.tolist()
        } if newly_found_endpoints_np.size > 0 else set()

        door_endpoint_map: Dict[Tuple[int, int], Dict[str, Set[int]]] = {}

        door_parallel_points: List[List[int]] = []
        door_parallel_info: Dict[Tuple[int, int], Dict[str, Set[int]]] = {}
        for door in self.objs.get('doors', []):
            door_id = door.get('id')
            if door_id is None:
                continue
            for info in door.get('parallel_edge_midpoints', []) or []:
                midpoint_rc = info.get('midpoint_rc')
                midpoint_xy = info.get('midpoint_xy')
                if not midpoint_rc or len(midpoint_rc) != 2:
                    continue
                row, col = int(midpoint_rc[0]), int(midpoint_rc[1])
                coord_tuple = (row, col)
                door_parallel_points.append([row, col])
                payload = door_parallel_info.setdefault(coord_tuple, _new_endpoint_info())
                payload['door_ids'].add(int(door_id))

                rooms_for_midpoint: Set[int] = set()
                if midpoint_xy and prepared_rooms_for_midpoints:
                    try:
                        midpoint_point = Point(float(midpoint_xy[0]), float(midpoint_xy[1]))
                    except (TypeError, ValueError):
                        midpoint_point = None
                    if midpoint_point is not None:
                        for room_id, prepared in prepared_rooms_for_midpoints:
                            if prepared.contains(midpoint_point) or prepared.touches(midpoint_point):
                                rooms_for_midpoint.add(int(room_id))

                if rooms_for_midpoint:
                    payload['room_ids'].update(rooms_for_midpoint)
                    info['room_ids'] = sorted(rooms_for_midpoint)
                    # print(f"Door {door_id} midpoint at {midpoint_rc} connects to rooms {info['room_ids']}")
                else:
                    info['room_ids'] = []

        if door_parallel_points:
            door_parallel_np = np.unique(np.array(door_parallel_points, dtype=int), axis=0)
            if newly_found_endpoints_np.size > 0:
                newly_found_endpoints_np = np.unique(
                    np.vstack([newly_found_endpoints_np, door_parallel_np]),
                    axis=0,
                )
            else:
                newly_found_endpoints_np = door_parallel_np

            parallel_coord_set = {
                (int(coord[0]), int(coord[1]))
                for coord in door_parallel_np.tolist()
            }
            if newly_found_coord_set:
                newly_found_coord_set |= parallel_coord_set
            else:
                newly_found_coord_set = parallel_coord_set

            for coord_tuple, payload in door_parallel_info.items():
                if coord_tuple not in parallel_coord_set:
                    continue
                dest_payload = door_endpoint_map.setdefault(coord_tuple, _new_endpoint_info())
                _merge_endpoint_info(dest_payload, payload)

        if safe_endpoints:
            safe_endpoints_np = np.array(safe_endpoints, dtype=int)
            safe_points_np = np.vstack([safe_endpoints_np, junction_coords])
        else:
            safe_points_np = junction_coords

        if safe_points_np.size > 0:
            safe_points_np = merge_nearby_points(safe_points_np, threshold=3.0)
        
        # --- 4. 장애물 내부에 위치한 키포인트 보정 및 제거 ---
        key_points_coords = safe_points_np
        skeleton_coords = np.argwhere(skel)
        if len(key_points_coords) == 0 and skeleton_coords.size > 0:
            key_points_coords = np.array([skeleton_coords[len(skeleton_coords) // 2]])

        if door_endpoint_map:
            door_pixel_map: Dict[Tuple[int, int], List[int]] = {}
            door_room_map: Dict[Tuple[int, int], List[int]] = {}
            for coord, payload in door_endpoint_map.items():
                coord_key = (int(coord[0]), int(coord[1]))
                door_ids_sorted = sorted({int(val) for val in payload.get('door_ids', set())}) if payload.get('door_ids') else []
                room_ids_sorted = sorted({int(val) for val in payload.get('room_ids', set())}) if payload.get('room_ids') else []
                if door_ids_sorted:
                    door_pixel_map[coord_key] = door_ids_sorted
                if room_ids_sorted:
                    door_room_map[coord_key] = room_ids_sorted
            self.door_endpoint_pixels = door_pixel_map
            self.door_endpoint_rooms = door_room_map
        else:
            self.door_endpoint_pixels = {}
            self.door_endpoint_rooms = {}
        self.newly_found_endpoint_coords = newly_found_coord_set

        node_map = {tuple(coord): f"corridor_{i}" for i, coord in enumerate(key_points_coords)}
        node_map_endpoints = {tuple(coord): f"door_endpoints_{i}" for i, coord in enumerate(newly_found_endpoints_np)}
        
        for coord, node_id in node_map.items():
            coord_tuple = (int(coord[0]), int(coord[1]))

            self.G.add_node(
                node_id,
                pos=(coord[1], coord[0]),
                type='corridor',
            )

            self.corridor_nodes_pos[node_id] = (coord[1], coord[0])
        for coord, node_id in node_map_endpoints.items():
            coord_tuple = (int(coord[0]), int(coord[1]))
            door_ids_list = self.door_endpoint_pixels.get(coord_tuple, [])
            room_ids_list = self.door_endpoint_rooms.get(coord_tuple, [])
            is_newly_found = coord_tuple in self.newly_found_endpoint_coords

            self.G.add_node(
                node_id,
                pos=(coord[1], coord[0]),
                type='door_endpoints',
                door_link_ids=door_ids_list,
                room_link_ids=room_ids_list,
                is_newly_found=is_newly_found,
            )

            if is_newly_found:
                self.newly_found_corridor_nodes.add(node_id)

            if door_ids_list:
                self.corridor_endpoint_door_links[node_id] = door_ids_list
                for door_id in door_ids_list:
                    self.door_to_corridor_nodes[door_id].append(node_id)

            if room_ids_list:
                self.corridor_endpoint_room_links[node_id] = room_ids_list

            self.corridor_nodes_pos[node_id] = (coord[1], coord[0])

        if len(key_points_coords) > 1:
            coords_list = [tuple(coord) for coord in key_points_coords.tolist()]
            coords_array = np.array(coords_list, dtype=float)
            tree = KDTree(coords_array)
            added_edges: Set[Tuple[int, int]] = set()

            def try_add_edge(i: int, j: int) -> bool:
                start_coord = coords_list[i]
                end_coord = coords_list[j]

                dist = self._astar_path(skel, np.array(start_coord), np.array(end_coord))
                if dist is None or dist == 0:
                    return False

                line_points = np.linspace(start_coord, end_coord, max(3, int(dist))).astype(int)
                for p in line_points[1:-1]:
                    pt = tuple(p)
                    if pt in node_map and pt not in (start_coord, end_coord):
                        return False

                start_pos = (start_coord[1], start_coord[0])
                end_pos = (end_coord[1], end_coord[0])
                if not self._is_line_clear(start_pos, end_pos):
                    return False

                self.G.add_edge(node_map[start_coord], node_map[end_coord], weight=dist)
                return True

            for i, start_coord in enumerate(coords_list):
                start_node = node_map[start_coord]
                current_degree = self.G.degree(start_node)

                neighbors_idx = tree.query_ball_point(coords_array[i], r=max(self.w, self.h))
                neighbors_idx = [idx for idx in neighbors_idx if idx != i]

                if neighbors_idx:
                    dists = np.linalg.norm(coords_array[neighbors_idx] - coords_array[i], axis=1)
                    neighbor_candidates = [idx for _, idx in sorted(zip(dists, neighbors_idx))]
                else:
                    distances_all = np.linalg.norm(coords_array - coords_array[i], axis=1)
                    neighbor_candidates = [idx for idx in np.argsort(distances_all) if idx != i]

                for j in neighbor_candidates:
                    edge_key = tuple(sorted((i, j)))
                    if edge_key in added_edges:
                        continue
                    if try_add_edge(i, j):
                        added_edges.add(edge_key)
                        current_degree += 1

                if current_degree == 0 and len(coords_list) > 1:
                    distances_all = np.linalg.norm(coords_array - coords_array[i], axis=1)
                    for j in np.argsort(distances_all):
                        if j == i:
                            continue
                        edge_key = tuple(sorted((i, j)))
                        if edge_key in added_edges:
                            continue
                        if try_add_edge(i, j):
                            added_edges.add(edge_key)
                            break

        # Remove corridor nodes that remain isolated before attaching other object nodes
        isolated_corridor_coords = [
            coord
            for coord, node_id in list(node_map.items())
            if self.G.has_node(node_id) and self.G.degree(node_id) == 0
        ]
        for coord in isolated_corridor_coords:
            node_id = node_map.pop(coord, None)
            if not node_id:
                continue
            if self.G.has_node(node_id):
                self.G.remove_node(node_id)
            self.corridor_nodes_pos.pop(node_id, None)

        mst_graph = nx.minimum_spanning_tree(self.G, weight='weight')
        self.G = mst_graph

        mst_node_coords = [coord for coord, node_id in node_map.items()]

        if mst_node_coords:
            mst_kdtree = KDTree(mst_node_coords)
            for endpoint_coord, endpoint_node_id in node_map_endpoints.items():
                
                distances, indices = mst_kdtree.query(endpoint_coord, k=len(mst_node_coords))

                # k=1인 경우 결과가 스칼라 값(np.int64 등)로 반환될 수 있으므로 1차원 배열로 변환
                distances = np.atleast_1d(distances)
                indices = np.atleast_1d(indices)

                candidates = []
                for idx in indices:
                    corridor_coord = mst_node_coords[int(idx)]
                    dist = self._astar_path(skel, np.array(endpoint_coord), np.array(corridor_coord))
                    if dist is not None and dist > 0:
                        candidates.append((dist, corridor_coord))
                        
                if not candidates:
                    continue

                sorted_candidates = sorted(candidates, key=lambda x: x[0])
                
                for dist, corridor_coord in sorted_candidates:
                    corridor_node_id = node_map[tuple(corridor_coord)]
                    start_pos = (endpoint_coord[1], endpoint_coord[0])
                    end_pos = (corridor_coord[1], corridor_coord[0])
                    
                    if self._is_line_clear(start_pos, end_pos):
                        self.G.add_edge(endpoint_node_id, corridor_node_id, weight=dist)
                        self.corridor_nodes_pos[endpoint_node_id] = start_pos
                        break
        if self.corridor_nodes_pos:
            self.corridor_kdtree = KDTree(list(self.corridor_nodes_pos.values()))

    def _is_line_clear(self, pos1, pos2):
        line = LineString([pos1, pos2])
        # R-tree와 같은 공간 인덱스를 사용하면 성능을 더 향상시킬 수 있음
        for key in ('rooms', 'walls', 'stairs', 'elevators'):
            for obj in self.objs[key]:
                poly = obj.get('polygon')
                if poly is None or poly.is_empty:
                    continue
                if not poly.intersects(line):
                    continue
                if poly.touches(line):
                    # allow tangential contacts along room/wall boundaries
                    continue
                return False
        return True
    def _is_line_clear_without_stairs_elevators(self, pos1, pos2):
        line = LineString([pos1, pos2])
        # R-tree와 같은 공간 인덱스를 사용하면 성능을 더 향상시킬 수 있음
        for key in ('rooms', 'walls'):
            for obj in self.objs[key]:
                poly = obj.get('polygon')
                if poly is None or poly.is_empty:
                    continue
                if not poly.intersects(line):
                    continue
                if poly.touches(line):
                    # allow tangential contacts along room/wall boundaries
                    continue
                return False
        return True
    def _is_line_clear_without_rooms(self, pos1, pos2):
        line = LineString([pos1, pos2])
        # R-tree와 같은 공간 인덱스를 사용하면 성능을 더 향상시킬 수 있음
        for key in ( 'walls', 'stairs', 'elevators'):
            for obj in self.objs[key]:
                poly = obj.get('polygon')
                if poly is None or poly.is_empty:
                    continue
                if not poly.intersects(line):
                    continue
                if poly.touches(line):
                    # allow tangential contacts along room/wall boundaries
                    continue
                return False
        return True

    def _connect_objects_to_graph_optimized(self):
        """ KDTree를 사용한 빠른 최근접 노드 탐색 """
        if not self.corridor_nodes_pos: return

        corridor_node_ids = list(self.corridor_nodes_pos.keys())
        object_types = ['rooms', 'doors', 'stairs', 'elevators']

        door_nodes_info: List[Tuple[str, Dict[str, Any]]] = []

        for obj_type in object_types:
            for obj in self.objs[obj_type]:
                node_id = f"{obj_type[:-1]}_{obj['id']}"
                centroid = obj['centroid']
                node_attrs = {
                    'pos': (centroid.x, centroid.y),
                    'type': obj_type[:-1],
                }

                if obj_type == 'doors':
                    node_attrs['door_id'] = obj['id']
                    node_attrs['connected_room_ids'] = obj.get('connected_room_ids', [])
                if obj_type == 'rooms':
                    node_attrs['rooms_id'] = obj['id']
                    node_attrs['connected_door_ids'] = obj.get('connected_door_ids', [])
                # if obj_type in ('rooms'):
                #     print(f"node {node_id}")
                self.G.add_node(node_id, **node_attrs)

                if obj_type == 'doors':
                    door_nodes_info.append((node_id, obj))
                    continue

                if obj_type == 'rooms':
                    # Room centroids now link via door boundary endpoints instead of directly to corridors
                    continue

                distances, indices = self.corridor_kdtree.query(
                    [centroid.x, centroid.y],
                    k=len(corridor_node_ids),
                )

                if not hasattr(indices, '__iter__'):
                    indices, distances = [indices], [distances]

                for dist, idx in zip(distances, indices):
                    candidate_node_id = corridor_node_ids[idx]
                    candidate_pos = self.corridor_nodes_pos[candidate_node_id]
                    if self._is_line_clear_without_stairs_elevators((centroid.x, centroid.y), candidate_pos):
                        self.G.add_edge(node_id, candidate_node_id, weight=dist)
                        break

        for node_id, obj in door_nodes_info:
            centroid = obj['centroid']
            door_pos = (float(centroid.x), float(centroid.y))
            door_pos_vec = np.array(door_pos, dtype=float)
            door_id = obj['id']
            connected_rooms = obj.get('connected_room_ids') or []
            corridor_targets = list(dict.fromkeys(self.door_to_corridor_nodes.get(door_id, [])))

            added_endpoint_edge = False
            for endpoint_node_id in corridor_targets:
                if not self.G.has_node(endpoint_node_id):
                    continue
                endpoint_pos = self.corridor_nodes_pos.get(endpoint_node_id)
                if endpoint_pos is None:
                    node_pos_attr = self.G.nodes[endpoint_node_id].get('pos')
                    if not node_pos_attr:
                        continue
                    endpoint_pos = (float(node_pos_attr[0]), float(node_pos_attr[1]))
                    self.corridor_nodes_pos[endpoint_node_id] = endpoint_pos
                if self.G.has_edge(node_id, endpoint_node_id):
                    continue
                endpoint_vec = np.array(endpoint_pos, dtype=float)
                weight = float(np.linalg.norm(door_pos_vec - endpoint_vec))
                self.G.add_edge(node_id, endpoint_node_id, weight=weight)
                added_endpoint_edge = True

            if added_endpoint_edge or connected_rooms:
                continue

            if len(corridor_node_ids) == 0:
                continue

            distances, indices = self.corridor_kdtree.query(
                [door_pos[0], door_pos[1]],
                k=min(5, len(corridor_node_ids)),
            )
            if not hasattr(indices, '__iter__'):
                indices, distances = [indices], [distances]

            for dist, idx in zip(distances, indices):
                candidate_id = corridor_node_ids[idx]
                candidate_pos = self.corridor_nodes_pos[candidate_id]
                if not self._is_line_clear(door_pos, candidate_pos):
                    continue
                if self.G.has_edge(node_id, candidate_id):
                    break
                self.G.add_edge(node_id, candidate_id, weight=dist)
                break

        if self.corridor_endpoint_room_links:
            for endpoint_node_id, room_ids in self.corridor_endpoint_room_links.items():
                if endpoint_node_id not in self.G:
                    continue
                endpoint_pos = self.corridor_nodes_pos.get(endpoint_node_id)
                if endpoint_pos is None:
                    node_pos_attr = self.G.nodes[endpoint_node_id].get('pos')
                    if not node_pos_attr:
                        continue
                    endpoint_pos = (float(node_pos_attr[0]), float(node_pos_attr[1]))
                endpoint_vec = np.array(endpoint_pos, dtype=float)
                # print(f"Processing door_endpoints node {endpoint_node_id} linking to rooms {room_ids}")
                for room_id in room_ids:
                    room_node_id = f"room_{room_id}"
                    if room_node_id not in self.G:
                        # print(f"  Skipping non-existent room node {room_node_id}")
                        continue
                    if self.G.has_edge(endpoint_node_id, room_node_id):
                        # print(f"  Edge already exists between {endpoint_node_id} and {room_node_id}, skipping")
                        continue
                    room_pos_attr = self.G.nodes[room_node_id].get('pos')
                    if not room_pos_attr:
                        # print(f"  Skipping room node {room_node_id} with no position")
                        continue
                    room_pos = (float(room_pos_attr[0]), float(room_pos_attr[1]))
                    room_vec = np.array(room_pos, dtype=float)
                    weight = float(np.linalg.norm(endpoint_vec - room_vec))
                    self.G.add_edge(endpoint_node_id, room_node_id, weight=weight)
                    # print(f"Connected door_endpoints node {endpoint_node_id} to room node {room_node_id} with weight {weight:.2f}")

        room_to_endpoint_nodes: Dict[int, List[str]] = defaultdict(list)
        for endpoint_node_id, room_ids in self.corridor_endpoint_room_links.items():
            if endpoint_node_id not in self.G:
                continue
            for room_id in room_ids:
                room_to_endpoint_nodes[int(room_id)].append(endpoint_node_id)

        for endpoint_nodes in room_to_endpoint_nodes.values():
            if len(endpoint_nodes) < 2:
                continue
            for i in range(len(endpoint_nodes)):
                node_a = endpoint_nodes[i]
                pos_a = self.corridor_nodes_pos.get(node_a)
                if pos_a is None:
                    pos_attr = self.G.nodes[node_a].get('pos')
                    if not pos_attr:
                        continue
                    pos_a = (float(pos_attr[0]), float(pos_attr[1]))
                    self.corridor_nodes_pos[node_a] = pos_a
                vec_a = np.array(pos_a, dtype=float)
                for j in range(i + 1, len(endpoint_nodes)):
                    node_b = endpoint_nodes[j]
                    if self.G.has_edge(node_a, node_b):
                        continue
                    pos_b = self.corridor_nodes_pos.get(node_b)
                    if pos_b is None:
                        pos_attr = self.G.nodes[node_b].get('pos')
                        if not pos_attr:
                            continue
                        pos_b = (float(pos_attr[0]), float(pos_attr[1]))
                        self.corridor_nodes_pos[node_b] = pos_b
                    if not self._is_line_clear_without_rooms(pos_a, pos_b):
                        continue
                    vec_b = np.array(pos_b, dtype=float)
                    weight = float(np.linalg.norm(vec_a - vec_b))
                    self.G.add_edge(node_a, node_b, weight=weight)

class FloorPlanVisualizer:
    def __init__(self, G, objs):
        self.G = G
        self.objs = objs

    def show(
        self,
        path=None,
        title="Floor Plan Navigation",
        output_path: Optional[Path] = None,
        show_window: bool = True,
        annotate_connections: bool = True,
        highlight_path: bool = False,
        path_color: str = 'cyan',
        path_width: float = 2.5,
        path_node_size: float = 45.0,
        path_start_color: str = 'lime',
        path_end_color: str = 'red',
        path_node_outline: str = 'black',
        node_show: bool = True,
        edge_show: bool = True,
        save_out: bool = True,
    ):
        """Draw polygons, graph nodes, and optionally door↔room labels to verify connectivity."""
        fig, axes = plt.subplots(1, 2, figsize=(18, 8), sharex=False, sharey=False)
        ax_main: plt.Axes = axes[0]
        ax_lines: plt.Axes = axes[1]
        ax_lines.set_title("Graph Edge Distribution")
        ax_main.set_title(title)
        colors = {
            'room': 'lightgray',
            'wall': 'darkgray',
            'door': 'saddlebrown',
            'stairs': 'skyblue',
            'elevator': 'indigo',
        }
        for r in self.objs['rooms']:
            ax_main.add_patch(MplPolygon(r['corners'], facecolor=colors['room'], edgecolor='black', alpha=0.7))
        for w in self.objs['walls']:
            ax_main.add_patch(MplPolygon(w['corners'], facecolor=colors['wall'], edgecolor='black'))
        for d in self.objs['doors']:
            ax_main.add_patch(MplPolygon(d['corners'], facecolor=colors['door'], edgecolor='black'))
        for s in self.objs['stairs']:
            ax_main.add_patch(MplPolygon(s['corners'], facecolor=colors['stairs'], edgecolor='black'))
        for el in self.objs['elevators']:
            ax_main.add_patch(MplPolygon(el['corners'], facecolor=colors['elevator'], edgecolor='black'))

        corridor_nodes_with_doors = {
            n
            for n, attr in self.G.nodes(data=True)
            if attr.get('type') == 'door_endpoints' and attr.get('door_link_ids')
        }

        if annotate_connections:
            for room in self.objs['rooms']:
                label = f"R{room['id']}"
                doors = room.get('connected_door_ids')
                if doors:
                    label += f"\nD{','.join(str(d_id) for d_id in doors)}"
                centroid = room.get('centroid')
                if centroid:
                    ax_main.text(centroid.x, centroid.y, label, ha='center', va='center', fontsize=3, color='black', bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.6))

            for door in self.objs['doors']:
                label = f"D{door['id']}"
                rooms = door.get('connected_room_ids')
                if rooms:
                    label += f"\nR{','.join(str(r_id) for r_id in rooms)}"
                centroid = door.get('centroid')
                if centroid:
                    ax_main.text(centroid.x, centroid.y, label, ha='center', va='center', fontsize=4, color='saddlebrown', bbox=dict(boxstyle='round,pad=0.15', fc='white', alpha=0.7))

            for node_id in corridor_nodes_with_doors:
                attrs = self.G.nodes[node_id]
                node_pos = attrs.get('pos')
                door_ids = attrs.get('door_link_ids', [])
                room_ids = attrs.get('room_link_ids', [])
                if not node_pos or not door_ids:
                    continue
                x, y = node_pos
                label = f"D{','.join(str(d_id) for d_id in door_ids)}"
                if room_ids:
                    label += f"\nR{','.join(str(r_id) for r_id in room_ids)}"
                ax_main.text(
                    x,
                    y - 6,
                    label,
                    ha='center',
                    va='top',
                    fontsize=3,
                    color='deeppink',
                    bbox=dict(boxstyle='round,pad=0.15', fc='white', alpha=0.65),
                )

        pos = nx.get_node_attributes(self.G, 'pos')
        node_colors = []
        for n in self.G.nodes():
            node_type = self.G.nodes[n].get('type')
            if node_type == 'corridor':
                node_colors.append('green')
            elif node_type == 'door_endpoints':
                node_colors.append('deeppink')
            else:
                node_colors.append('red')
        # 기본 노드/엣지
        if node_show:
            nodes_coll = nx.draw_networkx_nodes(self.G, pos, node_size=0.1, node_color=node_colors, ax=ax_main)
            if hasattr(nodes_coll, 'set_zorder'):
                nodes_coll.set_zorder(5)
        if edge_show:
            edges_coll = nx.draw_networkx_edges(self.G, pos, alpha=0.6, edge_color='orange', ax=ax_main)
            if isinstance(edges_coll, list):
                for coll in edges_coll:
                    if hasattr(coll, 'set_zorder'):
                        coll.set_zorder(4)
            elif hasattr(edges_coll, 'set_zorder'):
                edges_coll.set_zorder(4)

        if highlight_path and path:
            valid_path = [node for node in path if node in pos]
            if valid_path:
                path_nodes = nx.draw_networkx_nodes(
                    self.G,
                    pos,
                    nodelist=valid_path,
                    node_color=path_color,
                    node_size=path_node_size,
                    ax=ax_main,
                )
                if hasattr(path_nodes, 'set_zorder'):
                    path_nodes.set_zorder(6)

                if len(valid_path) >= 2:
                    path_edges = list(zip(valid_path, valid_path[1:]))
                    path_edges_coll = nx.draw_networkx_edges(
                        self.G,
                        pos,
                        edgelist=path_edges,
                        edge_color=path_color,
                        width=path_width,
                        ax=ax_main,
                    )
                    if isinstance(path_edges_coll, list):
                        for coll in path_edges_coll:
                            if hasattr(coll, 'set_zorder'):
                                coll.set_zorder(5)
                    elif hasattr(path_edges_coll, 'set_zorder'):
                        path_edges_coll.set_zorder(5)

                start_node = valid_path[0]
                end_node = valid_path[-1]
                start_pos = pos[start_node]
                end_pos = pos[end_node]
                ax_main.scatter(
                    [start_pos[0]],
                    [start_pos[1]],
                    s=path_node_size * 1.5,
                    c=path_start_color,
                    edgecolors=path_node_outline,
                    linewidths=1.0,
                    zorder=7,
                )
                ax_main.scatter(
                    [end_pos[0]],
                    [end_pos[1]],
                    s=path_node_size * 1.5,
                    c=path_end_color,
                    edgecolors=path_node_outline,
                    linewidths=1.0,
                    zorder=7,
                )

        ax_main.set_aspect('equal', adjustable='box')
        ax_main.invert_yaxis()

        for r in self.objs['rooms']:
            ax_lines.add_patch(MplPolygon(r['corners'], facecolor='none', edgecolor='silver', linewidth=0.6, linestyle='--'))
        for r in self.objs['stairs']:
            ax_lines.add_patch(MplPolygon(r['corners'], facecolor='none', edgecolor='skyblue', linewidth=0.6, linestyle='--'))
        for r in self.objs['elevators']:
            ax_lines.add_patch(MplPolygon(r['corners'], facecolor='none', edgecolor='indigo', linewidth=0.6, linestyle='--'))

        wall_segments = self.objs.get('wall_segments') or []
        if not wall_segments:
            for wall in self.objs.get('walls', []):
                corners = wall.get('corners') or []
                if len(corners) < 2:
                    continue
                for idx in range(len(corners)):
                    start = corners[idx]
                    end = corners[(idx + 1) % len(corners)]
                    if start == end:
                        continue
                    ax_lines.plot([start[0], end[0]], [start[1], end[1]], color='dimgray', linewidth=1.0, alpha=0.9)
        else:
            for seg in wall_segments:
                start = seg.get('start')
                end = seg.get('end')
                if not start or not end:
                    continue
                category = (seg.get('category') or '').lower()
                color = 'black' if category in {'annotation', 'line', 'wall'} else 'royalblue'
                linewidth = 1.2 if category in {'annotation', 'line', 'wall'} else 0.9
                ax_lines.plot([start[0], end[0]], [start[1], end[1]], color=color, linewidth=linewidth, alpha=0.95)

        for door in self.objs.get('doors', []):
            corners = door.get('corners')
            if not corners:
                continue
            ax_lines.add_patch(MplPolygon(corners, fill=False, edgecolor=colors['door'], linewidth=1.0, linestyle='-'))

        ax_lines.set_aspect('equal', adjustable='box')
        ax_lines.invert_yaxis()
        ax_lines.set_xlabel('x')
        ax_lines.set_ylabel('y')
        ax_lines.grid(False)
        if output_path is not None and save_out:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=300)
            print(f"그래프가 {output_path} 에 저장되었습니다.")

        if show_window:
            plt.show()
        else:
            plt.close(fig)


def export_floorplan_artifacts(
    export_dir: Path,
    parsed_objects: Dict[str, List[Dict[str, Any]]],
    graph: nx.Graph,
    metadata: Dict[str, Any],
) -> Dict[str, Path]:
    """Persist serialized graph/object bundle and a JSON manifest for downstream consumers."""
    export_dir.mkdir(parents=True, exist_ok=True)

    bundle_path = export_dir / EXPORT_BUNDLE_FILENAME
    payload = {
        'objects': parsed_objects,
        'graph': graph,
        'metadata': metadata,
    }
    with bundle_path.open('wb') as f:
        pickle.dump(payload, f)

    manifest_payload = dict(metadata)
    manifest_payload['bundle_filename'] = bundle_path.name
    manifest_path = export_dir / EXPORT_MANIFEST_FILENAME
    with manifest_path.open('w', encoding='utf-8') as f:
        json.dump(manifest_payload, f, ensure_ascii=False, indent=2)

    return {
        'bundle': bundle_path,
        'manifest': manifest_path,
    }

# ===================================================================
# 실행 예시 (MAIN)
# ===================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLO 출력으로부터 실내 그래프 생성")
    parser.add_argument('--predictions-dir', type=Path, help="Ultralytics 예측 결과 디렉터리 (labels/ 포함)")
    parser.add_argument('--image', type=Path, help="그래프를 생성할 대상 원본 이미지 경로")
    parser.add_argument('--class-names', type=Path, default=None, help="클래스 이름이 줄바꿈으로 저장된 텍스트 파일 경로")
    parser.add_argument('--coco-json', type=Path, help="cubicasa5k_to_coco.py 로 생성된 COCO json 경로")
    parser.add_argument('--image-id', type=int, help="COCO json 내 대상 이미지 id")
    parser.add_argument('--file-name', type=str, help="COCO json 내 대상 이미지 file_name")
    parser.add_argument('--image-root', type=Path, help="COCO file_name 이 상대경로일 경우 기준이 되는 이미지 루트 경로")
    parser.add_argument('--dataset-root', type=Path, help="YOLO dataset root (images/labels/...) 경로")
    parser.add_argument('--dataset-split', type=str, default='train', help="YOLO dataset split (train/val/test 등)")
    parser.add_argument('--room-txt', type=Path, help="방/계단/엘리베이터 박스 정보가 담긴 텍스트 파일 경로")
    parser.add_argument('--wall-txt', type=Path, help="벽 선분 정보가 담긴 텍스트 파일 경로")
    parser.add_argument('--door-txt', type=Path, help="문 포인트 정보가 담긴 텍스트 파일 경로")
    parser.add_argument('--include-classes', nargs='*', help="여기에 명시된 클래스만 사용 (예: room wall)")
    parser.add_argument('--exclude-classes', nargs='*', help="여기에 명시된 클래스는 제외")
    parser.add_argument(
        '--exclude-ids',
        nargs='*',
        help="특정 객체 ID 제외 (예: room:0,2 wall:1). ID는 각 객체 타입별로 0부터 시작",
    )
    parser.add_argument('--output', type=Path, help="주석이 포함된 그래프 이미지를 저장할 경로")
    parser.add_argument('--export-dir', type=Path, help="그래프, 객체, 주석 이미지를 번들로 저장할 디렉터리")
    parser.add_argument('--no-show', action='store_true', help="그래프 창을 띄우지 않고 저장만 수행")
    parser.add_argument('--debug-dir', type=Path, help="중간 결과(마스크/스켈레톤)를 저장할 디렉터리")
    parser.add_argument('--no-connection-labels', action='store_true', help="문-방 연결 텍스트 라벨을 숨김")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    provided_annotation_paths = [args.room_txt, args.wall_txt, args.door_txt]
    provided_annotation_count = sum(1 for path in provided_annotation_paths if path is not None)
    use_annotation_bundle = provided_annotation_count > 0

    if use_annotation_bundle and provided_annotation_count != 3:
        raise ValueError('--room-txt, --wall-txt, --door-txt 는 모두 함께 지정해야 합니다.')

    parser = FloorPlanParser()
    parsed_objects: Dict[str, List[Dict[str, Any]]]
    class_names: List[str] = DEFAULT_CLASS_NAMES
    image_path: Optional[Path] = None
    image: Optional[np.ndarray] = None

    if use_annotation_bundle:
        if args.coco_json or args.dataset_root or args.predictions_dir:
            raise ValueError('라벨링 도구 출력(--room-txt/--wall-txt/--door-txt)와 다른 입력 옵션은 동시에 사용할 수 없습니다.')
        if args.image is None:
            raise ValueError('--room-txt 모드를 사용할 때는 --image 로 원본 이미지를 지정해야 합니다.')

        image_path = args.image
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"이미지를 불러올 수 없습니다: {image_path}")
        height, width = image.shape[:2]

        class_names = load_class_names(args.class_names) if args.class_names else DEFAULT_CLASS_NAMES
        parsed_objects = load_annotation_bundle_from_texts(
            args.room_txt,
            args.wall_txt,
            args.door_txt,
            width,
            height,
        )
        parser.objects = parsed_objects
        parser._extend_doors_along_walls()
        parser.annotate_room_door_connections()
        parsed_objects = parser.objects
    else:
        if args.coco_json is not None:
            detections, class_names, image_path = load_coco_ground_truth(
                args.coco_json,
                args.image_root,
                args.image_id,
                args.file_name,
            )
        elif args.dataset_root is not None:
            if args.image is None:
                raise ValueError('--dataset-root 사용 시 --image 에 이미지 경로를 지정해야 합니다.')
            image_dir = args.dataset_root / 'images' / args.dataset_split
            label_dir = args.dataset_root / 'labels' / args.dataset_split

            image_candidate = args.image if args.image.is_absolute() else image_dir / args.image
            if not image_candidate.exists():
                raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {image_candidate}")

            try:
                rel_path = image_candidate.relative_to(image_dir)
            except ValueError as exc:
                raise ValueError('이미지는 dataset_root/images/<split>/ 아래에 있어야 합니다.') from exc

            image_path = image_candidate.resolve()
            label_path = (label_dir / rel_path.with_suffix('.txt')).resolve()
            default_class_path = args.dataset_root / 'classes.txt'
            class_names = load_class_names(args.class_names or (default_class_path if default_class_path.exists() else None))
            detections = read_yolo_label_file(label_path, image_path)
        else:
            if args.predictions_dir is None or args.image is None:
                raise ValueError('--predictions-dir 와 --image 를 모두 지정하거나, --coco-json 또는 --dataset-root 를 사용해야 합니다.')
            class_names = load_class_names(args.class_names)
            image_path = args.image
            detections = load_ultralytics_predictions(args.predictions_dir, args.image)

        if image_path is None:
            raise ValueError('이미지 경로를 결정하지 못했습니다.')

        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"이미지를 불러올 수 없습니다: {image_path}")
        height, width = image.shape[:2]

        formatted_results = format_yolo_results(detections, class_names)
        parsed_objects = parser.parse_yolo_output(
            formatted_results,
            include_classes=args.include_classes,
            exclude_classes=args.exclude_classes,
        )

    canonical_label_map = {
        'rooms': 'room',
        'doors': 'door',
        'walls': 'wall',
        'stairs': 'stairs',
        'elevators': 'elevator',
    }
    include_set = {cls.lower() for cls in args.include_classes} if args.include_classes else None
    exclude_set = {cls.lower() for cls in args.exclude_classes} if args.exclude_classes else None
    if include_set or exclude_set:
        for key, label_name in canonical_label_map.items():
            if key not in parsed_objects:
                continue
            if include_set is not None and label_name not in include_set:
                parsed_objects[key] = []
                continue
            if exclude_set is not None and label_name in exclude_set:
                parsed_objects[key] = []

    exclude_ids_map = parse_exclude_ids(args.exclude_ids)
    for key, idxs in exclude_ids_map.items():
        if key not in parsed_objects:
            continue
        filtered = [obj for obj in parsed_objects[key] if obj['id'] not in idxs]
        for new_id, obj in enumerate(filtered):
            obj['id'] = new_id
        parsed_objects[key] = filtered

    parser.objects = parsed_objects
    parser.annotate_room_door_connections()
    parsed_objects = parser.objects

    nav_graph_builder = FloorPlanNavigationGraph(
        parsed_objects,
        width=width,
        height=height,
        debug_dir=args.debug_dir,
    )
    navigation_graph = nav_graph_builder.build()

    export_dir = args.export_dir.resolve() if args.export_dir else None
    if export_dir is not None:
        export_dir.mkdir(parents=True, exist_ok=True)

    annotated_output_path = args.output
    if annotated_output_path is None and export_dir is not None:
        annotated_output_path = export_dir / EXPORT_IMAGE_FILENAME

    if annotated_output_path is not None:
        annotated_output_path = annotated_output_path.resolve()

    visualizer = FloorPlanVisualizer(navigation_graph, parsed_objects)
    visualizer.show(
        path=None,
        output_path=annotated_output_path,
        show_window=not args.no_show,
        annotate_connections=not args.no_connection_labels,
    )

    if export_dir is not None:
        if image_path is None:
            raise ValueError('이미지 경로 정보를 찾을 수 없습니다.')
        metadata: Dict[str, Any] = {
            'bundle_version': 1,
            'created_at': datetime.now().isoformat(timespec='seconds'),
            'source_image_path': str(image_path.resolve()),
            'annotated_image_path': str(annotated_output_path) if annotated_output_path else None,
            'image_size': {
                'width': width,
                'height': height,
            },
            'class_names': list(class_names),
            'include_classes': list(args.include_classes or []),
            'exclude_classes': list(args.exclude_classes or []),
            'exclude_ids': list(args.exclude_ids or []),
            'graph_summary': {
                'nodes': navigation_graph.number_of_nodes(),
                'edges': navigation_graph.number_of_edges(),
            },
        }
        artifacts = export_floorplan_artifacts(export_dir, parsed_objects, navigation_graph, metadata)
        print(f"그래프 데이터가 {artifacts['bundle']} 에 저장되었습니다.")
        print(f"메타데이터가 {artifacts['manifest']} 에 저장되었습니다.")


if __name__ == '__main__':
    main()
