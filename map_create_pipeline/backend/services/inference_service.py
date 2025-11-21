from __future__ import annotations

import io
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple
from datetime import datetime
from uuid import uuid4

import numpy as np
from PIL import Image

from configuration import PROJECT_ROOT, get_inference_settings
from processing.wall_centerline_pipeline import LineSegment, WallCenterlinePipeline, WallPipelineDebugConfig, WallPipelineResult

try:
    import supervision as sv
except ImportError:  # pragma: no cover - 런타임 의존성 누락 시 예외로 전파
    sv = None

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

try:
    from rfdetr import RFDETRMedium
except ImportError:  # pragma: no cover - 설치 안내를 위해 지연 처리
    RFDETRMedium = None


class TorchNotAvailableError(RuntimeError):
    """PyTorch 미설치 시 발생시키는 에러."""


class DependencyNotAvailableError(RuntimeError):
    """기타 필수 패키지(rfdetr, supervision) 누락 시 발생."""


@dataclass(frozen=True)
class Detection:
    """RF-DETR 모델이 반환한 단일 박스 감지 결과."""

    label_id: str
    score: float
    x1: float
    y1: float
    x2: float
    y2: float

    def clamp(self, width: int, height: int) -> "Detection":
        return Detection(
            self.label_id,
            self.score,
            max(0.0, min(float(width), self.x1)),
            max(0.0, min(float(height), self.y1)),
            max(0.0, min(float(width), self.x2)),
            max(0.0, min(float(height), self.y2)),
        )

    def to_object_detection_line(self, width: int, height: int, identifier: str) -> str:
        """객체 감지 포맷 한 줄로 직렬화해 wall/room 텍스트에 저장한다."""
        if width <= 0 or height <= 0:
            raise ValueError("이미지 크기는 0보다 커야 합니다.")

        clamped = self.clamp(width, height)
        box_width = max(clamped.x2 - clamped.x1, 0.0)
        box_height = max(clamped.y2 - clamped.y1, 0.0)
        cx = clamped.x1 + box_width / 2.0
        cy = clamped.y1 + box_height / 2.0

        norm_cx = np.clip(cx / width, 0.0, 1.0)
        norm_cy = np.clip(cy / height, 0.0, 1.0)
        norm_w = np.clip(box_width / width, 0.0, 1.0)
        norm_h = np.clip(box_height / height, 0.0, 1.0)
        confidence = np.clip(self.score, 0.0, 1.0)

        return f"{self.label_id} {norm_cx:.6f} {norm_cy:.6f} {norm_w:.6f} {norm_h:.6f} {confidence:.4f} {identifier}"


@dataclass
class ModelConfig:
    """RF-DETR 모델 하나를 초기화하기 위한 경로·클래스 매핑 설정."""

    name: str
    weight_path: Path
    class_map: Dict[int, str]
    confidence_threshold: float = 0.35
    device: Optional[str] = None


DEFAULT_MODEL_CLASS_MAPS: Dict[str, Dict[int, str]] = {
    "room": {0: "2"},
    "door_stairs_elevator": {0: "0", 1: "3", 2: "1"},
    "stairs_elevator": {0: "3", 1: "1"},
    "door": {0: "0"},
}


def resolve_device(choice: Optional[str]) -> str:
    """선호 장치를 기반으로 CUDA/MPS/CPU 중 사용 가능한 실행 장치를 선택한다."""
    if torch is None:
        return "cpu"

    normalized = (choice or "cpu").lower()
    valid = {"auto", "cuda", "mps", "cpu"}
    if normalized not in valid:
        normalized = "cpu"

    if normalized == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():  # type: ignore[attr-defined]
            return "mps"
        return "cpu"
    if normalized == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if normalized == "mps":
        has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()  # type: ignore[attr-defined]
        return "mps" if has_mps else "cpu"
    return "cpu"


def _ensure_dependencies() -> None:
    """필수 런타임 의존성(torch/rfdetr/supervision)이 설치됐는지 확인한다."""
    if torch is None:
        raise TorchNotAvailableError(
            "PyTorch가 설치되어 있지 않아 RF-DETR 추론을 수행할 수 없습니다. "
            "pip install torch torchvision 명령으로 설치한 뒤 다시 시도하세요."
        )
    if RFDETRMedium is None:
        raise DependencyNotAvailableError(
            "rfdetr 패키지를 불러오지 못했습니다. pip install rfdetr 로 설치한 뒤 다시 시도하세요."
        )
    if sv is None:
        raise DependencyNotAvailableError(
            "supervision 패키지를 불러오지 못했습니다. pip install supervision 으로 설치해 주세요."
        )


def _load_image(image_bytes: bytes) -> Tuple[Image.Image, np.ndarray]:
    """업로드된 바이트를 PIL 이미지와 numpy 배열로 동시에 변환한다."""
    buffer = io.BytesIO(image_bytes)
    image = Image.open(buffer).convert("RGB")
    array = np.asarray(image)
    return image, array


@lru_cache(maxsize=None)
def _load_rfdetr_model(weights_path: str, device: str) -> RFDETRMedium:
    """가중치 경로와 장치를 기반으로 RF-DETR 모델을 메모리에 로드한다."""
    return RFDETRMedium(pretrain_weights=weights_path, device=device)


class RFDetrModel:
    """RF-DETRMedium을 감싸고 공통 유틸리티(장치 선택 등)를 제공하는 래퍼."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self._device: Optional[str] = None

    def _load_runtime(self) -> Tuple[RFDETRMedium, str]:
        """가중치가 존재하는지 확인하고 필요 시 장치/모델을 초기화한다."""
        _ensure_dependencies()

        weight_path = Path(self.config.weight_path)
        if not weight_path.exists():
            raise FileNotFoundError(f"{self.config.name} 모델 가중치가 존재하지 않습니다: {weight_path}")

        if self._device is None:
            self._device = resolve_device(self.config.device)
        model = _load_rfdetr_model(str(weight_path), self._device)
        return model, self._device

    def predict(self, image: Image.Image) -> List[Detection]:
        """PIL 이미지를 입력 받아 Detection 리스트로 변환한다."""
        model, _ = self._load_runtime()
        detections_output = model.predict([image], threshold=self.config.confidence_threshold)

        if sv is None:  # pragma: no cover - _ensure_dependencies 가 이미 가로챔
            raise DependencyNotAvailableError("supervision 패키지가 필요합니다.")

        if isinstance(detections_output, sv.Detections):
            detections_sequence = [detections_output]
        else:
            detections_sequence = list(detections_output or [])

        if not detections_sequence:
            return []

        parsed: List[Detection] = []
        primary = detections_sequence[0]
        for xyxy, score, class_id in zip(primary.xyxy, primary.confidence, primary.class_id):
            label_id = self.config.class_map.get(int(class_id))
            if label_id is None:
                continue
            parsed.append(
                Detection(
                    label_id=label_id,
                    score=float(score),
                    x1=float(xyxy[0]),
                    y1=float(xyxy[1]),
                    x2=float(xyxy[2]),
                    y2=float(xyxy[3]),
                )
            )
        return parsed


class FloorPlanInferenceService:
    """4개의 RF-DETR 모델을 조합해 도면 객체를 검출하는 서비스."""

    OBSTACLE_LABEL_IDS = {"1", "2", "3"}  # elevator, room, stairs

    def __init__(
        self,
        model_configs: Sequence[ModelConfig],
        pipeline_debug_root: Optional[Path] = None,
        filter_walls_inside_obstacles: bool = True,
        clip_walls_overlapping_rooms: bool = True,
        apply_wall_filters_during_inference: bool = True,
    ):
        """모델 구성 리스트를 받아 RF-DETR 래퍼와 벽 파이프라인을 초기화한다."""
        self.models: Dict[str, RFDetrModel] = {}
        for config in model_configs:
            if not config:
                continue
            self.models[config.name] = RFDetrModel(config)
        self.wall_pipeline_debug_root = pipeline_debug_root
        self.filter_walls_inside_obstacles = filter_walls_inside_obstacles
        self.clip_walls_overlapping_rooms = clip_walls_overlapping_rooms
        self.apply_wall_filters_during_inference = apply_wall_filters_during_inference
        try:
            self.wall_pipeline: Optional[WallCenterlinePipeline] = WallCenterlinePipeline()
        except ImportError as exc:
            self.wall_pipeline = None
            print(f"[RF-DETR] 벽 중심선 파이프라인 초기화 실패: {exc}", flush=True)

    @staticmethod
    def _build_wall_lines_from_detections(
        detections: Iterable[Detection],
        width: int,
        height: int,
    ) -> List[str]:
        """벽 박스 감지를 중심선 포맷으로 변환해 legacy wall.txt 라인을 만든다."""
        lines: List[str] = []
        for index, detection in enumerate(detections):
            clamped = detection.clamp(width, height)
            box_width = max(clamped.x2 - clamped.x1, 1.0)
            box_height = max(clamped.y2 - clamped.y1, 1.0)
            center_x = clamped.x1 + box_width / 2.0
            center_y = clamped.y1 + box_height / 2.0

            if box_width >= box_height:
                x1 = clamped.x1
                x2 = clamped.x2
                y1 = y2 = center_y
            else:
                x1 = x2 = center_x
                y1 = clamped.y1
                y2 = clamped.y2

            norm_x1 = np.clip(x1 / width, 0.0, 1.0)
            norm_y1 = np.clip(y1 / height, 0.0, 1.0)
            norm_x2 = np.clip(x2 / width, 0.0, 1.0)
            norm_y2 = np.clip(y2 / height, 0.0, 1.0)

            line_id = f"wall-{index}"
            lines.append(f"{norm_x1:.6f} {norm_y1:.6f} {norm_x2:.6f} {norm_y2:.6f} {line_id}")
        return lines

    @staticmethod
    def _build_object_detection_text(
        detections: Iterable[Detection],
        width: int,
        height: int,
    ) -> List[str]:
        """사각형 감지 결과를 객체 감지 텍스트로 직렬화한다."""
        lines: List[str] = []
        for index, detection in enumerate(detections):
            identifier = f"{detection.label_id}-box-{index}"
            lines.append(detection.to_object_detection_line(width, height, identifier))
        return lines

    @staticmethod
    def _build_wall_lines_from_segments(
        segments: Sequence[LineSegment],
        width: int,
        height: int,
    ) -> List[str]:
        """기하 파이프라인 선분을 wall.txt 형식으로 정규화한다."""
        lines: List[str] = []
        for index, segment in enumerate(segments):
            if width <= 0 or height <= 0:
                continue
            norm_x1 = np.clip(segment.x1 / width, 0.0, 1.0)
            norm_y1 = np.clip(segment.y1 / height, 0.0, 1.0)
            norm_x2 = np.clip(segment.x2 / width, 0.0, 1.0)
            norm_y2 = np.clip(segment.y2 / height, 0.0, 1.0)
            lines.append(f"{norm_x1:.6f} {norm_y1:.6f} {norm_x2:.6f} {norm_y2:.6f} wall-center-{index}")
        return lines

    @staticmethod
    def _normalize_box_coords(x1: float, y1: float, x2: float, y2: float) -> Tuple[float, float, float, float]:
        return (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))

    @staticmethod
    def _segment_inside_box(segment: LineSegment, box: Tuple[float, float, float, float]) -> bool:
        x_min, y_min, x_max, y_max = box
        return (
            x_min <= segment.x1 <= x_max
            and x_min <= segment.x2 <= x_max
            and y_min <= segment.y1 <= y_max
            and y_min <= segment.y2 <= y_max
        )

    def _filter_segments_inside_boxes(
        self,
        segments: Sequence[LineSegment],
        boxes: Sequence[Tuple[float, float, float, float]],
    ) -> List[LineSegment]:
        if not segments or not boxes:
            return list(segments)
        filtered: List[LineSegment] = []
        for segment in segments:
            if any(self._segment_inside_box(segment, box) for box in boxes):
                continue
            filtered.append(segment)
        return filtered

    @staticmethod
    def _point_on_segment(segment: LineSegment, t: float) -> Tuple[float, float]:
        return (
            segment.x1 + (segment.x2 - segment.x1) * t,
            segment.y1 + (segment.y2 - segment.y1) * t,
        )

    def _clip_segment_outside_box(
        self,
        segment: LineSegment,
        box: Tuple[float, float, float, float],
        epsilon: float = 1e-3,
    ) -> List[LineSegment]:
        x_min, y_min, x_max, y_max = box
        dx = segment.x2 - segment.x1
        dy = segment.y2 - segment.y1
        t_enter = 0.0
        t_exit = 1.0
        eps = 1e-9
        constraints = [
            (-dx, segment.x1 - x_min),
            (dx, x_max - segment.x1),
            (-dy, segment.y1 - y_min),
            (dy, y_max - segment.y1),
        ]
        for p, q in constraints:
            if abs(p) < eps:
                if q < 0:
                    return [segment]
                continue
            r = q / p
            if p < 0:
                if r > t_exit:
                    return [segment]
                if r > t_enter:
                    t_enter = r
            else:
                if r < t_enter:
                    return [segment]
                if r < t_exit:
                    t_exit = r
        if t_enter >= t_exit:
            return [segment]
        inside_start = max(0.0, min(1.0, t_enter))
        inside_end = max(0.0, min(1.0, t_exit))
        if inside_end - inside_start <= epsilon:
            return [segment]
        if inside_start <= 0.0 and inside_end >= 1.0:
            return []

        segments_out: List[LineSegment] = []

        def _append_segment(t0: float, t1: float) -> None:
            if t1 - t0 <= epsilon:
                return
            start_x, start_y = self._point_on_segment(segment, t0)
            end_x, end_y = self._point_on_segment(segment, t1)
            segments_out.append(LineSegment(start_x, start_y, end_x, end_y))

        if inside_start > 0.0:
            _append_segment(0.0, inside_start)
        if inside_end < 1.0:
            _append_segment(inside_end, 1.0)
        return segments_out

    def _clip_segments_against_boxes(
        self,
        segments: Sequence[LineSegment],
        boxes: Sequence[Tuple[float, float, float, float]],
    ) -> List[LineSegment]:
        if not segments or not boxes:
            return list(segments)
        current: List[LineSegment] = list(segments)
        for box in boxes:
            next_segments: List[LineSegment] = []
            for segment in current:
                clipped = self._clip_segment_outside_box(segment, box)
                if clipped:
                    next_segments.extend(clipped)
            current = next_segments
            if not current:
                break
        return current

    def _apply_wall_filters(
        self,
        result: WallPipelineResult,
        detections: Sequence[Detection],
    ) -> WallPipelineResult:
        obstacle_boxes: List[Tuple[float, float, float, float]] = []
        room_boxes: List[Tuple[float, float, float, float]] = []
        for det in detections:
            box = self._normalize_box_coords(det.x1, det.y1, det.x2, det.y2)
            if det.label_id in self.OBSTACLE_LABEL_IDS:
                obstacle_boxes.append(box)
            if det.label_id == "2":  # room
                room_boxes.append(box)

        def _process_segments(segments: Sequence[LineSegment]) -> List[LineSegment]:
            updated = list(segments)
            if self.filter_walls_inside_obstacles and obstacle_boxes:
                updated = self._filter_segments_inside_boxes(updated, obstacle_boxes)
            if self.clip_walls_overlapping_rooms and room_boxes:
                updated = self._clip_segments_against_boxes(updated, room_boxes)
            return updated

        result.raw_lines = _process_segments(result.raw_lines)
        result.merged_lines = _process_segments(result.merged_lines)
        result.final_lines = _process_segments(result.final_lines)
        return result

    def _predict_all_models(self, image: Image.Image) -> List[Detection]:
        """모든 RF-DETR 모델을 실행해 감지 결과를 누적한다."""
        aggregated: List[Detection] = []
        for model in self.models.values():
            aggregated.extend(model.predict(image))
        return aggregated

    def _run_wall_pipeline(
        self,
        image_array: np.ndarray,
        debug_token: Optional[str] = None,
        door_hint_provider: Optional[Callable[[], Sequence[Tuple[float, float, float, float]]]] = None,
    ) -> Optional[WallPipelineResult]:
        """이미지 자체의 통계를 기반으로 엣지 경계선을 생성한다."""
        if self.wall_pipeline is None:
            return None
        debug_config: Optional[WallPipelineDebugConfig] = None
        if self.wall_pipeline_debug_root is not None:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            unique = uuid4().hex[:6]
            run_dir = self.wall_pipeline_debug_root / f"{timestamp}-{unique}"
            prefix = (debug_token or "wall_pipeline").replace(" ", "_")
            debug_config = WallPipelineDebugConfig(output_dir=run_dir, prefix=prefix)
        try:
            return self.wall_pipeline.run(
                image_array,
                debug_config=debug_config,
                door_hint_provider=door_hint_provider,
            )
        except Exception as exc:  # pragma: no cover - 디버깅 목적
            print(f"[RF-DETR] 벽 중심선 파이프라인 실패: {exc}", flush=True)
            return None

    def infer_from_file(self, image_bytes: bytes, filename: Optional[str] = None) -> Dict[str, object]:
        """이미지를 입력 받아 RF-DETR 감지와 벽 파이프라인 결과를 반환한다."""
        image, image_array = _load_image(image_bytes)
        width, height = image.size

        debug_label = (filename or "uploaded").rsplit(".", 1)[0]

        all_box_detections = self._predict_all_models(image)

        pipeline_result: Optional[WallPipelineResult] = None
        if self.wall_pipeline is not None:
            door_boxes = [
                (det.x1, det.y1, det.x2, det.y2)
                for det in all_box_detections
                if det.label_id == "0"
            ]
            door_hint_provider: Optional[Callable[[], Sequence[Tuple[float, float, float, float]]]] = None
            if door_boxes:
                door_hint_provider = lambda boxes=door_boxes: boxes
            pipeline_result = self._run_wall_pipeline(
                image_array,
                debug_label,
                door_hint_provider,
            )

        if (
            self.apply_wall_filters_during_inference
            and pipeline_result
            and all_box_detections
        ):
            pipeline_result = self._apply_wall_filters(pipeline_result, all_box_detections)

        wall_lines: List[str] = []
        wall_line_source = "none"
        if pipeline_result:
            candidate_sets = [
                ("pipeline_final", pipeline_result.final_lines),
                ("pipeline_merged", pipeline_result.merged_lines),
                ("pipeline_raw", pipeline_result.raw_lines),
            ]
            for source_label, candidates in candidate_sets:
                if candidates:
                    wall_lines = self._build_wall_lines_from_segments(candidates, width, height)
                    wall_line_source = source_label
                    break

        object_detection_lines = self._build_object_detection_text(all_box_detections, width, height)

        rooms_count = sum(1 for det in all_box_detections if det.label_id == "2")
        doors_count = sum(1 for det in all_box_detections if det.label_id == "0")
        stairs_count = sum(1 for det in all_box_detections if det.label_id == "3")
        elevator_count = sum(1 for det in all_box_detections if det.label_id == "1")

        pipeline_stats: Optional[dict] = None
        if pipeline_result is not None:
            pipeline_stats = {
                "raw_lines": len(pipeline_result.raw_lines),
                "merged_lines": len(pipeline_result.merged_lines),
                "final_lines": len(pipeline_result.final_lines),
                "wall_unit": pipeline_result.wall_unit,
            }
            if pipeline_result.debug_assets:
                pipeline_stats["debug_assets"] = pipeline_result.debug_assets
        if pipeline_stats is None:
            pipeline_stats = {}
        pipeline_stats["wall_source"] = wall_line_source

        return {
            "file_name": filename or "uploaded.png",
            "image_width": width,
            "image_height": height,
            "object_detection_text": "\n".join(object_detection_lines),
            "wall_text": "\n".join(wall_lines),
            "door_text": "",
            "class_names": ["room", "stairs", "wall", "elevator", "door"],
            "counts": {
                "rooms": rooms_count,
                "stairs_elevators": stairs_count + elevator_count,
                "doors": doors_count,
                "walls": len(wall_lines),
            },
            "wall_pipeline": pipeline_stats,
        }


def build_inference_service_from_config() -> FloorPlanInferenceService:
    """환경설정 파일을 읽어 RF-DETR 모델과 디버그 옵션을 초기화한다."""
    inference_cfg = get_inference_settings()

    model_root_value = inference_cfg.get("model_root", "data/models")
    model_root_path = Path(model_root_value)
    if not model_root_path.is_absolute():
        model_root_path = PROJECT_ROOT / model_root_path

    default_device = inference_cfg.get("default_device", "auto")
    models_cfg = inference_cfg.get("models", {})

    def _normalize_class_map(raw: Dict) -> Dict[int, str]:
        normalized: Dict[int, str] = {}
        for key, value in (raw or {}).items():
            try:
                index = int(key)
            except (TypeError, ValueError):
                continue
            normalized[index] = str(value)
        return normalized

    configs: List[ModelConfig] = []
    for model_name, model_entry in models_cfg.items():
        if not isinstance(model_entry, dict):
            continue

        raw_class_map = model_entry.get("class_map")
        if not raw_class_map:
            raw_class_map = DEFAULT_MODEL_CLASS_MAPS.get(model_name)
        class_map = _normalize_class_map(raw_class_map or {})
        if not class_map:
            print(f"[RF-DETR] {model_name} 모델의 클래스 매핑이 없어 건너뜁니다.", flush=True)
            continue

        path_value = model_entry.get("path")
        weight_path: Optional[Path] = None
        if path_value:
            candidate = Path(path_value)
            weight_path = candidate if candidate.is_absolute() else PROJECT_ROOT / candidate
        else:
            filename = model_entry.get("filename")
            if filename:
                weight_path = model_root_path / filename

        if weight_path is None:
            print(f"[RF-DETR] {model_name} 모델의 가중치 경로가 설정되지 않아 건너뜁니다.", flush=True)
            continue

        if not weight_path.exists():
            print(f"[RF-DETR] {model_name} 모델 가중치를 찾을 수 없어 건너뜁니다: {weight_path}", flush=True)
            continue

        threshold_value = model_entry.get("threshold")
        try:
            threshold = float(threshold_value)
        except (TypeError, ValueError):
            threshold = 0.35

        device_value = model_entry.get("device", default_device)

        configs.append(
            ModelConfig(
                name=str(model_name),
                weight_path=weight_path,
                class_map=class_map,
                confidence_threshold=threshold,
                device=device_value,
            )
        )

    debug_dir_value = inference_cfg.get("wall_pipeline_debug_dir") or os.environ.get("WALL_PIPELINE_DEBUG_DIR")
    pipeline_debug_root: Optional[Path] = None
    if debug_dir_value:
        debug_path = Path(debug_dir_value)
        pipeline_debug_root = debug_path if debug_path.is_absolute() else PROJECT_ROOT / debug_path
        pipeline_debug_root.mkdir(parents=True, exist_ok=True)

    filter_walls_inside_obstacles = bool(inference_cfg.get("filter_walls_inside_obstacles", True))
    clip_walls_overlapping_rooms = bool(inference_cfg.get("clip_walls_overlapping_rooms", True))
    apply_wall_filters_during_inference = bool(inference_cfg.get("apply_wall_filters_during_inference", False))

    return FloorPlanInferenceService(
        configs,
        pipeline_debug_root=pipeline_debug_root,
        filter_walls_inside_obstacles=filter_walls_inside_obstacles,
        clip_walls_overlapping_rooms=clip_walls_overlapping_rooms,
        apply_wall_filters_during_inference=apply_wall_filters_during_inference,
    )
