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
import networkx as nx
from itertools import combinations

import numpy as np
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

try:
    from .corridor_pipeline import FloorPlanNavigationGraph  # type: ignore
except ImportError:  # pragma: no cover
    from corridor_pipeline import FloorPlanNavigationGraph  # type: ignore

try:
    from .floorplan_visualizer import FloorPlanVisualizer, EXPORT_IMAGE_FILENAME  # type: ignore
except ImportError:  # pragma: no cover
    from floorplan_visualizer import FloorPlanVisualizer, EXPORT_IMAGE_FILENAME  # type: ignore


@dataclass
class Detection:
    """단일 객체 감지 결과를 담는 자료형. 박스, 신뢰도, 클래스 ID 정보를 포함한다."""

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


def load_class_names(path: Optional[Path]) -> List[str]:
    """COCO 클래스 이름 파일을 읽어 리스트로 반환한다."""
    if path is None:
        return DEFAULT_CLASS_NAMES
    if not path.exists():
        raise FileNotFoundError(f"클래스 이름 파일을 찾을 수 없습니다: {path}")
    with path.open('r', encoding='utf-8') as f:
        names = [line.strip() for line in f.readlines() if line.strip()]
    if not names:
        raise ValueError("클래스 이름 파일이 비어있습니다.")
    return names


def read_object_detection_label_file(label_path: Path, image_path: Path) -> List[Detection]:
    """YOLO 형식의 라벨 파일을 읽어 Detection 리스트로 변환한다."""
    if not label_path.exists():
        raise FileNotFoundError(f"객체 감지 라벨 파일을 찾을 수 없습니다: {label_path}")
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
    """Ultralytics 예측 디렉터리에서 이미지 이름에 해당하는 라벨을 읽는다."""
    label_path = predictions_dir / 'labels' / f"{image_path.stem}.txt"
    return read_object_detection_label_file(label_path, image_path)


def parse_exclude_ids(raw: Optional[Sequence[str]]) -> Dict[str, Set[int]]:
    """CLI 인수로 넘어온 'label:id1,id2' 형태를 파싱해 제외할 객체 ID 집합을 만든다."""
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
    """COCO 주석 파일에서 지정된 이미지의 정답 박스와 클래스 이름을 읽어온다."""
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


def format_object_detection_results(detections: Iterable[Detection], class_names: Sequence[str]):
    """시각화/검증 용도로 감지 결과를 label+corners 형태로 직렬화한다."""
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
    """정규화 좌표를 0~1 범위에 안전하게 제한한다."""
    if not np.isfinite(value):
        return 0.0
    if value <= 0.0:
        return 0.0
    if value >= 1.0:
        return 1.0
    return float(value)


def parse_room_box_file(path: Path) -> List[Dict[str, Any]]:
    """rooms.txt를 읽어 라벨·중심점·크기를 정규화된 bbox로 변환한다."""
    if not path.exists():
        raise FileNotFoundError(f"방/계단/엘리베이터 텍스트 파일을 찾을 수 없습니다: {path}")

    annotations: List[Dict[str, Any]] = []
    with path.open('r', encoding='utf-8') as f:
        for raw_line in f:
            tokens = raw_line.strip().split()
            if len(tokens) < 6:
                continue
            if len(tokens) == 6:
                label_token, cx_token, cy_token, w_token, h_token, provided_id = tokens
            else:
                # Ultralytics export 등에서는 confidence/별칭이 추가될 수 있으므로
                # 앞의 5개 토큰을 bbox 정보로, 마지막 토큰을 ID로 사용한다.
                label_token = tokens[0]
                cx_token, cy_token, w_token, h_token = tokens[1:5]
                provided_id = tokens[-1]
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
    """wall.txt 선분을 파싱해 정규화된 시작/끝 좌표와 ID를 반환한다."""
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
    """door.txt를 읽어 문 포인트와 라인/박스 anchor 정보를 추출한다."""
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
TMP_ROOM_EDGE_THICKNESS_PX = 2.0
TMP_ROOM_EDGE_EXTENSION_PX = 0.0
TMP_ROOM_GAP_FILL_MARGIN_PX = 1.0
TMP_ROOM_GAP_MAX_PX = 20.0
TMP_ROOM_MIN_OVERLAP_PX = 3.0

def _clip_point(point: np.ndarray, width: int, height: int) -> List[float]:
    """이미지 경계를 벗어나지 않도록 좌표를 보정한다."""
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
    """선분을 중심으로 두께/연장량을 고려한 직사각형과 보조 정보를 생성한다."""
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
    *,
    enable_wall_expansion: bool = False,
) -> Dict[str, List[Dict[str, Any]]]:
    """세 개의 텍스트 파일을 읽어 그래프 구축에 필요한 rooms/doors/walls 사전을 만든다."""
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

    if enable_wall_expansion:
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
        effective_extension = extension if enable_wall_expansion else 0.0
        result = _segment_to_rectangle(start, end, thickness, effective_extension, image_width, image_height)
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
        if enable_wall_expansion and key in ('rooms', 'stairs', 'elevators'):
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

    def _finalize_room_edges(adjust_for_gaps: bool) -> None:
        if not pending_room_edges:
            return

        if adjust_for_gaps:
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
    _finalize_room_edges(enable_wall_expansion)

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
    """객체 감지 결과와 텍스트 파일을 구조화된 방/문/벽 데이터로 변환한다."""
    def __init__(self):
        """내부 objects 사전을 초기화한다."""
        self.objects = {}

    def parse_object_detection_output(
        self,
        formatted_results,
        include_classes: Optional[Sequence[str]] = None,
        exclude_classes: Optional[Sequence[str]] = None,
    ):
        """필터링된 객체 감지 결과를 받아 rooms/doors/walls 리스트를 구성한다."""
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
        """문 사각형을 벽과의 교차 비율에 맞춰 연장하고, 어느 벽과 맞닿았는지 기록한다."""
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
        """문과 방/계단/엘리베이터의 폴리곤 겹침 면적을 이용해 상호 연결 관계를 기록한다."""
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



def export_floorplan_artifacts(
    export_dir: Path,
    parsed_objects: Dict[str, List[Dict[str, Any]]],
    graph: nx.Graph,
    metadata: Dict[str, Any],
) -> Dict[str, Path]:
    """그래프/객체 번들과 매니페스트를 저장해 후속 파이프라인이 재활용할 수 있게 한다."""
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
    """CLI 실행에 필요한 경로·옵션 인수를 파싱한다."""
    parser = argparse.ArgumentParser(description="객체 감지 출력으로부터 실내 그래프 생성")
    parser.add_argument('--predictions-dir', type=Path, help="Ultralytics 예측 결과 디렉터리 (labels/ 포함)")
    parser.add_argument('--image', type=Path, help="그래프를 생성할 대상 원본 이미지 경로")
    parser.add_argument('--class-names', type=Path, default=None, help="클래스 이름이 줄바꿈으로 저장된 텍스트 파일 경로")
    parser.add_argument('--coco-json', type=Path, help="cubicasa5k_to_coco.py 로 생성된 COCO json 경로")
    parser.add_argument('--image-id', type=int, help="COCO json 내 대상 이미지 id")
    parser.add_argument('--file-name', type=str, help="COCO json 내 대상 이미지 file_name")
    parser.add_argument('--image-root', type=Path, help="COCO file_name 이 상대경로일 경우 기준이 되는 이미지 루트 경로")
    parser.add_argument('--dataset-root', type=Path, help="객체 감지 데이터셋 루트 (images/labels/...) 경로")
    parser.add_argument('--dataset-split', type=str, default='train', help="객체 감지 데이터셋 split (train/val/test 등)")
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
    parser.add_argument(
        '--enable-wall-expansion',
        action='store_true',
        help="방/벽 선분을 확장해 틈을 메우는 선행 처리를 활성화한다.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI 엔트리포인트: 감지 결과를 불러 파서를 돌리고 그래프/시각화를 출력한다."""
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
            enable_wall_expansion=args.enable_wall_expansion,
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
            detections = read_object_detection_label_file(label_path, image_path)
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

        formatted_results = format_object_detection_results(detections, class_names)
        parsed_objects = parser.parse_object_detection_output(
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
