from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple
import math


_MIN_DIMENSION = 1e-4


def _safe_float(value: Optional[float], default: float = 0.0) -> float:
    try:
        result = float(value)
        if math.isnan(result):
            return default
        return result
    except (TypeError, ValueError):
        return default


def _clamp_unit(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def _distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _project_point_onto_segment(
    point: Tuple[float, float], segment: Tuple[Tuple[float, float], Tuple[float, float]]
) -> Tuple[Tuple[float, float], float]:
    (x1, y1), (x2, y2) = segment
    vx = x2 - x1
    vy = y2 - y1
    denom = vx * vx + vy * vy
    if denom <= 0:
        return (x1, y1), 0.0
    t = ((point[0] - x1) * vx + (point[1] - y1) * vy) / denom
    t_clamped = max(0.0, min(1.0, t))
    proj_x = x1 + vx * t_clamped
    proj_y = y1 + vy * t_clamped
    return (_clamp_unit(proj_x), _clamp_unit(proj_y)), t_clamped


def _normalize_payload_items(items) -> List[Dict]:
    normalized: List[Dict] = []
    if not items:
        return normalized
    for item in items:
        if hasattr(item, "model_dump"):
            normalized.append(item.model_dump(by_alias=True))
        elif isinstance(item, dict):
            normalized.append(dict(item))
    return normalized


_CONTACT_TOLERANCE = 1e-3
_RECT_EPSILON = 1e-9


def _ranges_overlap_length(a_min: float, a_max: float, b_min: float, b_max: float) -> float:
    return max(0.0, min(a_max, b_max) - max(a_min, b_min))


def _point_in_rect(point: Tuple[float, float], rect: Tuple[float, float, float, float], eps: float = _RECT_EPSILON) -> bool:
    x, y = point
    min_x, min_y, max_x, max_y = rect
    return (min_x - eps) <= x <= (max_x + eps) and (min_y - eps) <= y <= (max_y + eps)


def _rectangles_overlap(
    a: Tuple[float, float, float, float],
    b: Tuple[float, float, float, float],
    eps: float = _RECT_EPSILON,
) -> bool:
    a_min_x, a_min_y, a_max_x, a_max_y = a
    b_min_x, b_min_y, b_max_x, b_max_y = b
    if a_max_x <= b_min_x + eps or b_max_x <= a_min_x + eps:
        return False
    if a_max_y <= b_min_y + eps or b_max_y <= a_min_y + eps:
        return False
    return True


def _line_intersects_rect(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    rect: Tuple[float, float, float, float],
) -> bool:
    min_x, min_y, max_x, max_y = rect
    if max(x1, x2) <= min_x or min(x1, x2) >= max_x or max(y1, y2) <= min_y or min(y1, y2) >= max_y:
        return False
    if _point_in_rect((x1, y1), rect) or _point_in_rect((x2, y2), rect):
        return True

    dx = x2 - x1
    dy = y2 - y1
    p_q = [
        (-dx, x1 - min_x),
        (dx, max_x - x1),
        (-dy, y1 - min_y),
        (dy, max_y - y1),
    ]
    t0 = 0.0
    t1 = 1.0
    for p, q in p_q:
        if abs(p) < 1e-12:
            if q < 0:
                return False
            continue
        r = q / p
        if p < 0:
            if r > t1:
                return False
            if r > t0:
                t0 = r
        else:
            if r < t0:
                return False
            if r < t1:
                t1 = r
    return t0 <= t1


@dataclass
class MutableBox:
    raw: Dict
    id: str
    label_id: Optional[str]
    min_x: float
    min_y: float
    max_x: float
    max_y: float

    @classmethod
    def from_payload(cls, payload: Dict) -> "MutableBox":
        x = _safe_float(payload.get("x"))
        y = _safe_float(payload.get("y"))
        width = max(_safe_float(payload.get("width")), 0.0)
        height = max(_safe_float(payload.get("height")), 0.0)
        max_x = x + width
        max_y = y + height
        min_x = min(x, max_x)
        min_y = min(y, max_y)
        max_x = max(x, max_x)
        max_y = max(y, max_y)
        return cls(
            raw=dict(payload),
            id=str(payload.get("id") or payload.get("name") or ""),
            label_id=payload.get("labelId") or payload.get("label_id"),
            min_x=_clamp_unit(min_x),
            min_y=_clamp_unit(min_y),
            max_x=_clamp_unit(max_x),
            max_y=_clamp_unit(max_y),
        )

    @property
    def width(self) -> float:
        return max(0.0, self.max_x - self.min_x)

    @property
    def height(self) -> float:
        return max(0.0, self.max_y - self.min_y)

    def clamp_bounds(self) -> None:
        self.min_x, self.max_x = sorted((_clamp_unit(self.min_x), _clamp_unit(self.max_x)))
        self.min_y, self.max_y = sorted((_clamp_unit(self.min_y), _clamp_unit(self.max_y)))
        if self.width < _MIN_DIMENSION:
            mid_x = (self.min_x + self.max_x) / 2.0
            half = _MIN_DIMENSION / 2.0
            self.min_x = _clamp_unit(mid_x - half)
            self.max_x = _clamp_unit(mid_x + half)
        if self.height < _MIN_DIMENSION:
            mid_y = (self.min_y + self.max_y) / 2.0
            half = _MIN_DIMENSION / 2.0
            self.min_y = _clamp_unit(mid_y - half)
            self.max_y = _clamp_unit(mid_y + half)

    def to_payload(self) -> Dict:
        prepared = dict(self.raw)
        prepared["x"] = float(self.min_x)
        prepared["y"] = float(self.min_y)
        width = max(_MIN_DIMENSION, self.max_x - self.min_x)
        height = max(_MIN_DIMENSION, self.max_y - self.min_y)
        if self.min_x + width > 1.0:
            width = max(_MIN_DIMENSION, 1.0 - self.min_x)
        if self.min_y + height > 1.0:
            height = max(_MIN_DIMENSION, 1.0 - self.min_y)
        prepared["width"] = float(width)
        prepared["height"] = float(height)
        return prepared

    def build_edges(self) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
        return [
            ((self.min_x, self.min_y), (self.max_x, self.min_y)),
            ((self.max_x, self.min_y), (self.max_x, self.max_y)),
            ((self.max_x, self.max_y), (self.min_x, self.max_y)),
            ((self.min_x, self.max_y), (self.min_x, self.min_y)),
        ]


@dataclass
class MutableSegment:
    raw: Dict
    id: str
    label_id: Optional[str]
    x1: float
    y1: float
    x2: float
    y2: float

    @classmethod
    def from_payload(cls, payload: Dict) -> "MutableSegment":
        return cls(
            raw=dict(payload),
            id=str(payload.get("id") or payload.get("name") or ""),
            label_id=payload.get("labelId") or payload.get("label_id"),
            x1=_clamp_unit(_safe_float(payload.get("x1"))),
            y1=_clamp_unit(_safe_float(payload.get("y1"))),
            x2=_clamp_unit(_safe_float(payload.get("x2"))),
            y2=_clamp_unit(_safe_float(payload.get("y2"))),
        )

    def get_point(self, index: int) -> Tuple[float, float]:
        if index == 0:
            return (self.x1, self.y1)
        return (self.x2, self.y2)

    def set_point(self, index: int, point: Tuple[float, float]) -> None:
        px = _clamp_unit(point[0])
        py = _clamp_unit(point[1])
        if index == 0:
            self.x1, self.y1 = px, py
        else:
            self.x2, self.y2 = px, py

    def to_payload(self) -> Dict:
        prepared = dict(self.raw)
        prepared["x1"] = float(self.x1)
        prepared["y1"] = float(self.y1)
        prepared["x2"] = float(self.x2)
        prepared["y2"] = float(self.y2)
        return prepared


def _describe_edge(box: MutableBox, edge_name: str) -> Tuple[float, float, float, str]:
    if edge_name == "left":
        return box.min_x, box.min_y, box.max_y, "vertical"
    if edge_name == "right":
        return box.max_x, box.min_y, box.max_y, "vertical"
    if edge_name == "top":
        return box.min_y, box.min_x, box.max_x, "horizontal"
    if edge_name == "bottom":
        return box.max_y, box.min_x, box.max_x, "horizontal"
    raise ValueError(f"unsupported edge: {edge_name}")


def _box_rect(box: MutableBox) -> Tuple[float, float, float, float]:
    return (box.min_x, box.min_y, box.max_x, box.max_y)


class FloorPlanAutoCorrectionService:
    def __init__(self, config: Optional[Dict[str, float]] = None) -> None:
        defaults = {
            "max_box_gap": 0.04,
            "min_overlap_ratio": 0.3,
            "box_gap_iterations": 2,
            "wall_box_snap_distance": 0.02,
            "wall_wall_snap_distance": 0.015,
        }
        self.config = {**defaults, **(config or {})}

    def auto_correct(self, payload) -> Dict:
        box_items = _normalize_payload_items(getattr(payload, "boxes", None))
        line_items = _normalize_payload_items(getattr(payload, "lines", None))
        boxes = [MutableBox.from_payload(box) for box in box_items]
        lines = [MutableSegment.from_payload(line) for line in line_items]

        stats = {
            "boxBoxAdjustments": 0,
            "wallBoxSnaps": 0,
            "wallWallSnaps": 0,
        }

        if lines:
            stats["wallWallSnaps"] = self._align_wall_endpoints_to_segments(lines)

        box_edges: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
        if boxes:
            for box in boxes:
                box_edges.extend(box.build_edges())

        if lines and box_edges:
            stats["wallBoxSnaps"] = self._snap_wall_endpoints_to_edges(lines, box_edges)

        if boxes:
            stats["boxBoxAdjustments"] = self._close_box_gaps(boxes, lines)

        return {
            "boxes": [box.to_payload() for box in boxes],
            "lines": [line.to_payload() for line in lines],
            "stats": stats,
        }

    def _close_box_gaps(self, boxes: List[MutableBox], lines: List[MutableSegment]) -> int:
        adjustments = 0
        if not boxes:
            return adjustments
        max_gap = max(0.0, float(self.config.get("max_box_gap", 0.02)))
        min_overlap_ratio = max(0.0, min(1.0, float(self.config.get("min_overlap_ratio", 0.25))))
        iterations = int(max(1, self.config.get("box_gap_iterations", 1)))
        line_refs = lines or []

        for _ in range(iterations):
            changed = False
            for i in range(len(boxes)):
                for j in range(i + 1, len(boxes)):
                    if self._close_gap_between_boxes(
                        boxes[i],
                        boxes[j],
                        max_gap,
                        min_overlap_ratio,
                        boxes,
                        line_refs,
                    ):
                        adjustments += 1
                        changed = True
            if not changed:
                break
        return adjustments

    def _close_gap_between_boxes(
        self,
        first: MutableBox,
        second: MutableBox,
        max_gap: float,
        min_overlap_ratio: float,
        boxes: List[MutableBox],
        lines: List[MutableSegment],
    ) -> bool:
        horizontal_changed = self._close_linear_gap(
            first,
            second,
            axis="x",
            max_gap=max_gap,
            min_overlap_ratio=min_overlap_ratio,
            boxes=boxes,
            lines=lines,
        )
        vertical_changed = self._close_linear_gap(
            first,
            second,
            axis="y",
            max_gap=max_gap,
            min_overlap_ratio=min_overlap_ratio,
            boxes=boxes,
            lines=lines,
        )
        return horizontal_changed or vertical_changed

    def _close_linear_gap(
        self,
        a: MutableBox,
        b: MutableBox,
        *,
        axis: str,
        max_gap: float,
        min_overlap_ratio: float,
        boxes: List[MutableBox],
        lines: List[MutableSegment],
    ) -> bool:
        lines = lines or []
        if axis == "x":
            left, right = (a, b) if a.min_x <= b.min_x else (b, a)
            gap = right.min_x - left.max_x
            overlap_min = max(left.min_y, right.min_y)
            overlap_max = min(left.max_y, right.max_y)
            overlap = overlap_max - overlap_min
            reference = min(left.height, right.height)
            if gap <= 0 or gap > max_gap or overlap <= 0 or reference <= 0:
                return False
            if overlap / reference < min_overlap_ratio:
                return False
            rect = (left.max_x, overlap_min, right.min_x, overlap_max)
            if rect[2] - rect[0] <= 0:
                return False
            if self._rectangle_has_obstacles(rect, boxes, {left.id, right.id}, lines):
                return False
            left_fixed = self._edge_is_pinned(left, "right", boxes, {right.id}, lines)
            right_fixed = self._edge_is_pinned(right, "left", boxes, {left.id}, lines)
            if left_fixed and right_fixed:
                return False
            left_move = 0.0
            right_move = 0.0
            if left_fixed:
                right_move = gap
            elif right_fixed:
                left_move = gap
            else:
                left_move = gap / 2.0
                right_move = gap - left_move
            left.max_x += left_move
            right.min_x -= right_move
            left.clamp_bounds()
            right.clamp_bounds()
            return True

        top, bottom = (a, b) if a.min_y <= b.min_y else (b, a)
        gap = bottom.min_y - top.max_y
        overlap_min = max(top.min_x, bottom.min_x)
        overlap_max = min(top.max_x, bottom.max_x)
        overlap = overlap_max - overlap_min
        reference = min(top.width, bottom.width)
        if gap <= 0 or gap > max_gap or overlap <= 0 or reference <= 0:
            return False
        if overlap / reference < min_overlap_ratio:
            return False
        rect = (overlap_min, top.max_y, overlap_max, bottom.min_y)
        if rect[3] - rect[1] <= 0:
            return False
        if self._rectangle_has_obstacles(rect, boxes, {top.id, bottom.id}, lines):
            return False
        top_fixed = self._edge_is_pinned(top, "bottom", boxes, {bottom.id}, lines)
        bottom_fixed = self._edge_is_pinned(bottom, "top", boxes, {top.id}, lines)
        if top_fixed and bottom_fixed:
            return False
        top_move = 0.0
        bottom_move = 0.0
        if top_fixed:
            bottom_move = gap
        elif bottom_fixed:
            top_move = gap
        else:
            top_move = gap / 2.0
            bottom_move = gap - top_move
        top.max_y += top_move
        bottom.min_y -= bottom_move
        top.clamp_bounds()
        bottom.clamp_bounds()
        return True

    def _rectangle_has_obstacles(
        self,
        rect: Tuple[float, float, float, float],
        boxes: List[MutableBox],
        excluded_box_ids: set,
        lines: List[MutableSegment],
    ) -> bool:
        min_x, min_y, max_x, max_y = rect
        if max_x - min_x <= 0 or max_y - min_y <= 0:
            return True
        excluded = set(excluded_box_ids or set())
        for box in boxes:
            if box.id in excluded:
                continue
            if _rectangles_overlap(rect, _box_rect(box), _CONTACT_TOLERANCE):
                return True
        for line in lines or []:
            if _line_intersects_rect(line.x1, line.y1, line.x2, line.y2, rect):
                return True
        return False

    def _edge_is_pinned(
        self,
        box: MutableBox,
        edge_name: str,
        boxes: List[MutableBox],
        exclude_box_ids: set,
        lines: List[MutableSegment],
    ) -> bool:
        excluded = set(exclude_box_ids or set())
        if self._edge_has_box_contact(box, edge_name, boxes, excluded):
            return True
        if self._edge_has_wall_contact(box, edge_name, lines):
            return True
        return False

    def _edge_has_box_contact(
        self,
        box: MutableBox,
        edge_name: str,
        boxes: List[MutableBox],
        exclude_box_ids: set,
    ) -> bool:
        coord, span_start, span_end, orientation = _describe_edge(box, edge_name)
        for other in boxes:
            if other.id == box.id or other.id in exclude_box_ids:
                continue
            overlap = 0.0
            if orientation == "vertical":
                overlap = _ranges_overlap_length(span_start, span_end, other.min_y, other.max_y)
                if overlap <= 0:
                    continue
                if abs(other.min_x - coord) <= _CONTACT_TOLERANCE or abs(other.max_x - coord) <= _CONTACT_TOLERANCE:
                    return True
            else:
                overlap = _ranges_overlap_length(span_start, span_end, other.min_x, other.max_x)
                if overlap <= 0:
                    continue
                if abs(other.min_y - coord) <= _CONTACT_TOLERANCE or abs(other.max_y - coord) <= _CONTACT_TOLERANCE:
                    return True
        return False

    def _edge_has_wall_contact(
        self,
        box: MutableBox,
        edge_name: str,
        lines: List[MutableSegment],
    ) -> bool:
        if not lines:
            return False
        coord, span_start, span_end, orientation = _describe_edge(box, edge_name)
        for line in lines:
            endpoints = ((line.x1, line.y1), (line.x2, line.y2))
            for px, py in endpoints:
                if orientation == "vertical":
                    if abs(px - coord) <= _CONTACT_TOLERANCE and (span_start - _CONTACT_TOLERANCE) <= py <= (
                        span_end + _CONTACT_TOLERANCE
                    ):
                        return True
                else:
                    if abs(py - coord) <= _CONTACT_TOLERANCE and (span_start - _CONTACT_TOLERANCE) <= px <= (
                        span_end + _CONTACT_TOLERANCE
                    ):
                        return True
        return False

    def _snap_wall_endpoints_to_edges(
        self,
        lines: List[MutableSegment],
        edges: Iterable[Tuple[Tuple[float, float], Tuple[float, float]]],
    ) -> int:
        threshold = max(0.0, float(self.config.get("wall_box_snap_distance", 0.02)))
        if threshold <= 0:
            return 0
        snaps = 0
        for line in lines:
            for endpoint_index in (0, 1):
                point = line.get_point(endpoint_index)
                best_point: Optional[Tuple[float, float]] = None
                best_distance = None
                for edge in edges:
                    candidate, _ = _project_point_onto_segment(point, edge)
                    distance = _distance(point, candidate)
                    if distance <= 0:
                        best_point = candidate
                        best_distance = 0.0
                        break
                    if best_distance is None or distance < best_distance:
                        best_distance = distance
                        best_point = candidate
                if best_point is not None and best_distance is not None and best_distance <= threshold:
                    line.set_point(endpoint_index, best_point)
                    snaps += 1
        return snaps

    def _align_wall_endpoints_to_segments(self, lines: List[MutableSegment]) -> int:
        threshold = max(0.0, float(self.config.get("wall_wall_snap_distance", 0.015)))
        if threshold <= 0:
            return 0
        snaps = 0
        for index, line in enumerate(lines):
            for endpoint_index in (0, 1):
                point = line.get_point(endpoint_index)
                best_point: Optional[Tuple[float, float]] = None
                best_distance: Optional[float] = None
                for other_index, other in enumerate(lines):
                    if index == other_index:
                        continue
                    candidate, _ = _project_point_onto_segment(point, ((other.x1, other.y1), (other.x2, other.y2)))
                    distance = _distance(point, candidate)
                    if distance <= 0:
                        best_point = candidate
                        best_distance = 0.0
                        break
                    if best_distance is None or distance < best_distance:
                        best_distance = distance
                        best_point = candidate
                if best_point is not None and best_distance is not None and best_distance <= threshold:
                    line.set_point(endpoint_index, best_point)
                    snaps += 1
        return snaps
