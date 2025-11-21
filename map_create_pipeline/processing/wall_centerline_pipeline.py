from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import cv2
except ImportError as exc:  # pragma: no cover
    cv2 = None
    _cv_error = exc
else:
    _cv_error = None

try:
    from skimage.morphology import skeletonize
except ImportError as exc:  # pragma: no cover
    skeletonize = None
    _skimage_error = exc
else:
    _skimage_error = None


@dataclass(frozen=True)
class LineSegment:
    """스켈레톤 선분 또는 병합 결과를 나타내는 자료형."""

    x1: float
    y1: float
    x2: float
    y2: float

    def as_array(self) -> np.ndarray:
        return np.array([[self.x1, self.y1], [self.x2, self.y2]], dtype=float)

    def length(self) -> float:
        return float(math.hypot(self.x2 - self.x1, self.y2 - self.y1))

    def direction(self) -> np.ndarray:
        dx = self.x2 - self.x1
        dy = self.y2 - self.y1
        norm = math.hypot(dx, dy)
        if norm <= 1e-6:
            return np.array([0.0, 0.0])
        return np.array([dx / norm, dy / norm])

    def angle_deg(self) -> float:
        angle = math.degrees(math.atan2(self.y2 - self.y1, self.x2 - self.x1))
        return (angle + 180.0) % 180.0


@dataclass
class WallPipelineDebugConfig:
    """파이프라인 중간 산출물을 파일로 남기기 위한 설정."""

    output_dir: Path
    prefix: str = "wall_pipeline"
    save_grayscale: bool = True
    save_overlays: bool = True


@dataclass
class WallPipelineParams:
    """Canny → Morphology → Thinning → Hough 파이프라인 하이퍼파라미터."""

    canny_sigma: float = 0.33
    morph_kernel_size: int = 5
    morph_iterations: int = 1
    enable_opening: bool = False
    enable_closing: bool = True

    hough_rho: float = 1.0
    hough_theta_deg: float = 1.0
    hough_threshold: int = 10
    hough_min_line_length: float = 10.0
    hough_max_line_gap: float = 5.0

    merge_angle_threshold_deg: float = 3.0
    merge_distance_threshold: float = 5.0
    merge_max_gap: float = 25.0


@dataclass
class WallPipelineResult:
    """새 파이프라인의 핵심 산출물."""

    raw_lines: List[LineSegment]
    merged_lines: List[LineSegment]
    final_lines: List[LineSegment]
    debug_assets: List[str] = field(default_factory=list)
    wall_unit: Optional[float] = None


DoorHint = Tuple[float, float, float, float]
DoorHintProvider = Callable[[], Sequence[DoorHint]]


class WallPipelineDebugWriter:
    """디버그 설정에 따라 회색조/선분/바이너리 이미지를 디스크에 기록한다."""

    def __init__(self, config: Optional[WallPipelineDebugConfig], image_array: np.ndarray):
        self.config = config
        self.outputs: List[str] = []
        if not config:
            self.base_image = None
            return
        self.base_image = self._prepare_base_image(image_array)
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

    def save_grayscale(self, suffix: str, gray_image: np.ndarray) -> None:
        if not self.config or not self.config.save_grayscale:
            return
        if gray_image.ndim == 2:
            image = gray_image
        else:
            image = cv2.cvtColor(gray_image, cv2.COLOR_RGB2GRAY)
        self._write_image(suffix, image)

    def save_binary(self, suffix: str, binary_image: np.ndarray) -> None:
        if not self.config or not self.config.save_grayscale:
            return
        self._write_image(suffix, binary_image)

    def save_lines(
        self,
        suffix: str,
        lines: Sequence[LineSegment],
        color=(0, 0, 255),
        thickness: int = 1,
    ) -> None:
        if not self.config or not self.config.save_overlays:
            return
        canvas = self.base_image.copy() if self.base_image is not None else None
        if canvas is None:
            return
        for line in lines:
            cv2.line(
                canvas,
                (int(round(line.x1)), int(round(line.y1))),
                (int(round(line.x2)), int(round(line.y2))),
                color,
                thickness,
                lineType=cv2.LINE_AA,
            )
        self._write_image(suffix, canvas)

    def _prepare_base_image(self, image_array: np.ndarray) -> np.ndarray:
        if image_array.ndim == 2:
            return cv2.cvtColor(image_array, cv2.COLOR_GRAY2BGR)
        if image_array.ndim == 3:
            channels = image_array.shape[2]
            if channels == 1:
                return cv2.cvtColor(image_array, cv2.COLOR_GRAY2BGR)
            if channels == 3:
                return cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
            if channels == 4:
                return cv2.cvtColor(image_array, cv2.COLOR_RGBA2BGR)
        raise ValueError("지원하지 않는 이미지 형태로 디버그 이미지를 생성할 수 없습니다.")

    def _write_image(self, suffix: str, image: np.ndarray) -> None:
        if not self.config:
            return
        filename = suffix if suffix.lower().endswith(".png") else f"{suffix}.png"
        safe_prefix = self.config.prefix.rstrip("_")
        path = self.config.output_dir / f"{safe_prefix}_{filename}"
        cv2.imwrite(str(path), image)
        self.outputs.append(str(path))


class WallCenterlinePipeline:
    """Canny-Thinning-Hough 기반 엣지 스켈레톤 파이프라인."""

    def __init__(self, params: Optional[WallPipelineParams] = None):
        if cv2 is None:
            raise ImportError(
                "OpenCV(cv2)를 불러오지 못했습니다. pip install opencv-python 으로 설치해 주세요."
            ) from _cv_error
        if skeletonize is None:
            raise ImportError(
                "scikit-image의 skeletonize 함수를 사용할 수 없습니다. pip install scikit-image 로 설치해 주세요."
            ) from _skimage_error
        self.params = params or WallPipelineParams()

    def run(
        self,
        image_array: np.ndarray,
        debug_config: Optional[WallPipelineDebugConfig] = None,
        door_hint_provider: Optional[DoorHintProvider] = None,
    ) -> WallPipelineResult:
        debug_writer = WallPipelineDebugWriter(debug_config, image_array)
        gray = self._to_grayscale(image_array)
        debug_writer.save_grayscale("00_grayscale.png", gray)

        thick_edges = self._auto_canny_edges(gray)
        debug_writer.save_binary("01_thick_edges.png", thick_edges)

        clean_edges = self._apply_morphology(thick_edges)
        debug_writer.save_binary("02_clean_edges.png", clean_edges)

        if door_hint_provider:
            width = int(image_array.shape[1])
            height = int(image_array.shape[0])
            door_boxes = self._resolve_door_hints(door_hint_provider, width, height)
            if door_boxes:
                clean_edges = self._bridge_door_gaps(clean_edges, door_boxes)
                debug_writer.save_binary("02b_door_bridged_edges.png", clean_edges)

        skeleton = self._thin_edges(clean_edges)
        debug_writer.save_binary("03_skeleton.png", skeleton)

        raw_lines = self._vectorize_skeleton(skeleton)
        debug_writer.save_lines("04_raw_lines.png", raw_lines, color=(50, 200, 255))

        merged_lines = self._merge_collinear_lines(raw_lines)
        debug_writer.save_lines("05_merged_lines.png", merged_lines, color=(255, 0, 255))

        final_lines = merged_lines
        debug_writer.save_lines("06_final_lines.png", final_lines, color=(0, 0, 255), thickness=3)

        return WallPipelineResult(
            raw_lines=raw_lines,
            merged_lines=merged_lines,
            final_lines=final_lines,
            debug_assets=debug_writer.outputs,
        )

    def _to_grayscale(self, image_array: np.ndarray) -> np.ndarray:
        if image_array is None:
            raise ValueError("이미지 배열이 비어 있습니다.")
        if image_array.dtype != np.uint8:
            clipped = np.clip(image_array, 0, 255)
            working = clipped.astype(np.uint8)
        else:
            working = image_array

        if working.ndim == 2:
            return working
        if working.ndim == 3:
            if working.shape[2] == 1:
                return working[:, :, 0]
            if working.shape[2] == 3:
                return cv2.cvtColor(working, cv2.COLOR_RGB2GRAY)
            if working.shape[2] == 4:
                rgb = cv2.cvtColor(working, cv2.COLOR_RGBA2RGB)
                return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        raise ValueError(f"지원하지 않는 이미지 형태입니다: {working.shape}")

    def _auto_canny_edges(self, gray: np.ndarray) -> np.ndarray:
        sigma = max(0.01, self.params.canny_sigma)
        median = float(np.median(gray))
        lower = int(max(0.0, (1.0 - sigma) * median))
        upper = int(min(255.0, (1.0 + sigma) * median))
        lower = max(0, lower - 5)
        upper = min(255, upper + 5)
        if lower >= upper:
            upper = min(255, lower + 1)
            lower = max(0, upper - 1)
        edges = cv2.Canny(gray, lower, upper, apertureSize=3, L2gradient=True)
        return edges

    def _apply_morphology(self, binary: np.ndarray) -> np.ndarray:
        params = self.params
        ksize = max(1, int(params.morph_kernel_size))
        if ksize % 2 == 0:
            ksize += 1
        kernel = np.ones((ksize, ksize), np.uint8)
        result = binary.copy()
        iterations = max(1, int(params.morph_iterations))
        if params.enable_closing:
            result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel, iterations=iterations)
        if params.enable_opening:
            result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel, iterations=iterations)
        return result

    def _thin_edges(self, binary: np.ndarray) -> np.ndarray:
        if skeletonize is None:  # pragma: no cover - __init__에서 이미 확인
            raise RuntimeError("scikit-image skeletonize 를 사용할 수 없습니다.")
        binary_bool = binary > 0
        skeleton_bool = skeletonize(binary_bool, method="zhang")
        return (skeleton_bool.astype(np.uint8) * 255)

    def _resolve_door_hints(
        self,
        provider: DoorHintProvider,
        width: int,
        height: int,
    ) -> List[DoorHint]:
        try:
            hints = list(provider() or [])
        except Exception as exc:  # pragma: no cover - 디버깅용
            print(f"[WallPipeline] door hint provider failed: {exc}", flush=True)
            return []

        sanitized: List[DoorHint] = []
        for hint in hints:
            if hint is None or len(hint) != 4:
                continue
            try:
                x1, y1, x2, y2 = (float(hint[0]), float(hint[1]), float(hint[2]), float(hint[3]))
            except (TypeError, ValueError):
                continue
            if x2 <= x1 and y2 <= y1:
                continue
            sx1 = float(np.clip(min(x1, x2), 0.0, width - 1))
            sx2 = float(np.clip(max(x1, x2), 0.0, width - 1))
            sy1 = float(np.clip(min(y1, y2), 0.0, height - 1))
            sy2 = float(np.clip(max(y1, y2), 0.0, height - 1))
            if (sx2 - sx1) < 1.0 and (sy2 - sy1) < 1.0:
                continue
            sanitized.append((sx1, sy1, sx2, sy2))
        return sanitized

    def _bridge_door_gaps(self, binary: np.ndarray, door_boxes: Sequence[DoorHint]) -> np.ndarray:
        bridged = binary.copy()
        height, width = bridged.shape[:2]
        for x1, y1, x2, y2 in door_boxes:
            x1_i = int(round(np.clip(x1, 0.0, width - 1)))
            x2_i = int(round(np.clip(x2, 0.0, width - 1)))
            y1_i = int(round(np.clip(y1, 0.0, height - 1)))
            y2_i = int(round(np.clip(y2, 0.0, height - 1)))
            span_w = max(1, abs(x2_i - x1_i))
            span_h = max(1, abs(y2_i - y1_i))
            pad = 2
            thickness = max(1, min(7, int(round(min(span_w, span_h) / 2.0)) + 1))
            if span_h >= span_w:
                cx = int(round((x1_i + x2_i) / 2.0))
                y_start = max(0, min(y1_i, y2_i) - pad)
                y_end = min(height - 1, max(y1_i, y2_i) + pad)
                cv2.line(bridged, (cx, y_start), (cx, y_end), 255, thickness=thickness, lineType=cv2.LINE_AA)
            else:
                cy = int(round((y1_i + y2_i) / 2.0))
                x_start = max(0, min(x1_i, x2_i) - pad)
                x_end = min(width - 1, max(x1_i, x2_i) + pad)
                cv2.line(bridged, (x_start, cy), (x_end, cy), 255, thickness=thickness, lineType=cv2.LINE_AA)
        return bridged

    def _vectorize_skeleton(self, skeleton: np.ndarray) -> List[LineSegment]:
        params = self.params
        theta = math.radians(max(0.1, params.hough_theta_deg))
        detections = cv2.HoughLinesP(
            skeleton,
            rho=max(1.0, params.hough_rho),
            theta=theta,
            threshold=max(1, int(params.hough_threshold)),
            minLineLength=max(1.0, params.hough_min_line_length),
            maxLineGap=max(0.0, params.hough_max_line_gap),
        )
        segments: List[LineSegment] = []
        if detections is None:
            return segments
        for entry in detections:
            x1, y1, x2, y2 = entry[0]
            segments.append(LineSegment(float(x1), float(y1), float(x2), float(y2)))
        return segments

    def _merge_collinear_lines(self, lines: Sequence[LineSegment]) -> List[LineSegment]:
        if not lines:
            return []
        params = self.params
        current = list(lines)
        merged_any = True
        while merged_any:
            merged_any = False
            next_lines: List[LineSegment] = []
            consumed = [False] * len(current)
            for i, base_line in enumerate(current):
                if consumed[i]:
                    continue
                working = base_line
                for j in range(i + 1, len(current)):
                    if consumed[j]:
                        continue
                    candidate = current[j]
                    if not self._should_merge(working, candidate, params):
                        continue
                    working = self._merge_lines(working, candidate)
                    consumed[j] = True
                    merged_any = True
                consumed[i] = True
                next_lines.append(working)
            current = next_lines
        return current

    def _should_merge(self, line_a: LineSegment, line_b: LineSegment, params: WallPipelineParams) -> bool:
        if not self._is_parallel(line_a, line_b, params.merge_angle_threshold_deg):
            return False
        if self._parallel_distance(line_a, line_b) > params.merge_distance_threshold:
            return False
        gap = self._segment_gap(line_a, line_b)
        return gap <= params.merge_max_gap

    def _is_parallel(self, line_a: LineSegment, line_b: LineSegment, threshold_deg: float) -> bool:
        diff = abs(line_a.angle_deg() - line_b.angle_deg())
        diff = min(diff, 180.0 - diff)
        return diff <= threshold_deg

    def _parallel_distance(self, line_a: LineSegment, line_b: LineSegment) -> float:
        dir_vec = line_a.direction()
        if np.linalg.norm(dir_vec) <= 1e-6:
            return float("inf")
        normal = np.array([-dir_vec[1], dir_vec[0]])
        normal_length = np.linalg.norm(normal)
        if normal_length <= 1e-6:
            return float("inf")
        normal /= normal_length
        ref_point = np.array([line_a.x1, line_a.y1])
        point_b1 = np.array([line_b.x1, line_b.y1])
        point_b2 = np.array([line_b.x2, line_b.y2])
        distance1 = abs(np.dot(point_b1 - ref_point, normal))
        distance2 = abs(np.dot(point_b2 - ref_point, normal))
        return float((distance1 + distance2) / 2.0)

    def _segment_gap(self, line_a: LineSegment, line_b: LineSegment) -> float:
        if self._segments_intersect(line_a, line_b):
            return 0.0
        a_points = line_a.as_array()
        b_points = line_b.as_array()
        distances = [
            self._point_to_segment_distance(a_points[0], b_points[0], b_points[1]),
            self._point_to_segment_distance(a_points[1], b_points[0], b_points[1]),
            self._point_to_segment_distance(b_points[0], a_points[0], a_points[1]),
            self._point_to_segment_distance(b_points[1], a_points[0], a_points[1]),
        ]
        return float(min(distances))

    def _segments_intersect(self, line_a: LineSegment, line_b: LineSegment) -> bool:
        p1, p2 = line_a.as_array()
        q1, q2 = line_b.as_array()
        o1 = self._orientation(p1, p2, q1)
        o2 = self._orientation(p1, p2, q2)
        o3 = self._orientation(q1, q2, p1)
        o4 = self._orientation(q1, q2, p2)

        if o1 * o2 < 0 and o3 * o4 < 0:
            return True
        if abs(o1) < 1e-6 and self._on_segment(p1, q1, p2):
            return True
        if abs(o2) < 1e-6 and self._on_segment(p1, q2, p2):
            return True
        if abs(o3) < 1e-6 and self._on_segment(q1, p1, q2):
            return True
        if abs(o4) < 1e-6 and self._on_segment(q1, p2, q2):
            return True
        return False

    def _orientation(self, p: np.ndarray, q: np.ndarray, r: np.ndarray) -> float:
        return float((q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]))

    def _on_segment(self, p: np.ndarray, q: np.ndarray, r: np.ndarray) -> bool:
        return (
            min(p[0], r[0]) - 1e-6 <= q[0] <= max(p[0], r[0]) + 1e-6
            and min(p[1], r[1]) - 1e-6 <= q[1] <= max(p[1], r[1]) + 1e-6
        )

    def _point_to_segment_distance(self, point: np.ndarray, seg_start: np.ndarray, seg_end: np.ndarray) -> float:
        seg = seg_end - seg_start
        seg_len_sq = float(np.dot(seg, seg))
        if seg_len_sq <= 1e-6:
            return float(np.linalg.norm(point - seg_start))
        t = float(np.dot(point - seg_start, seg) / seg_len_sq)
        t = max(0.0, min(1.0, t))
        projection = seg_start + seg * t
        return float(np.linalg.norm(point - projection))

    def _merge_lines(self, line_a: LineSegment, line_b: LineSegment) -> LineSegment:
        points = np.vstack([line_a.as_array(), line_b.as_array()])
        mean = np.mean(points, axis=0)
        centered = points - mean
        _, _, vh = np.linalg.svd(centered)
        direction = vh[0]
        direction_norm = np.linalg.norm(direction)
        if direction_norm <= 1e-6:
            return line_a
        direction /= direction_norm
        projections = points @ direction
        perp_offsets = points - np.outer(projections, direction)
        offset = np.mean(perp_offsets, axis=0)

        min_proj = np.min(projections)
        max_proj = np.max(projections)
        start = direction * min_proj + offset
        end = direction * max_proj + offset
        return LineSegment(float(start[0]), float(start[1]), float(end[0]), float(end[1]))
