from __future__ import annotations

import io
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from configuration import PROJECT_ROOT, get_inference_settings

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
    """단일 객체 검출 결과."""

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

    def to_yolo_line(self, width: int, height: int, identifier: str) -> str:
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
    """RF-DETR 단일 모델 설정."""

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
    "wall": {0: "4"},
}


def resolve_device(choice: Optional[str]) -> str:
    """detection_model_test의 resolve_device 로직을 참고한 장치 선택."""
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
    buffer = io.BytesIO(image_bytes)
    image = Image.open(buffer).convert("RGB")
    array = np.asarray(image)
    return image, array


@lru_cache(maxsize=None)
def _load_rfdetr_model(weights_path: str, device: str) -> RFDETRMedium:
    return RFDETRMedium(pretrain_weights=weights_path, device=device)


class RFDetrModel:
    """단일 RF-DETR 모델 래퍼."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self._device: Optional[str] = None

    def _load_runtime(self) -> Tuple[RFDETRMedium, str]:
        _ensure_dependencies()

        weight_path = Path(self.config.weight_path)
        if not weight_path.exists():
            raise FileNotFoundError(f"{self.config.name} 모델 가중치가 존재하지 않습니다: {weight_path}")

        if self._device is None:
            self._device = resolve_device(self.config.device)
        model = _load_rfdetr_model(str(weight_path), self._device)
        return model, self._device

    def predict(self, image: Image.Image) -> List[Detection]:
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

    def __init__(self, model_configs: Sequence[ModelConfig]):
        self.models: Dict[str, RFDetrModel] = {}
        for config in model_configs:
            if not config:
                continue
            self.models[config.name] = RFDetrModel(config)

    @staticmethod
    def _build_wall_lines(
        detections: Iterable[Detection],
        width: int,
        height: int,
    ) -> List[str]:
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
    def _build_yolo_text(
        detections: Iterable[Detection],
        width: int,
        height: int,
    ) -> List[str]:
        lines: List[str] = []
        for index, detection in enumerate(detections):
            identifier = f"{detection.label_id}-box-{index}"
            lines.append(detection.to_yolo_line(width, height, identifier))
        return lines

    def infer_from_file(self, image_bytes: bytes, filename: Optional[str] = None) -> Dict[str, object]:
        image, _ = _load_image(image_bytes)
        width, height = image.size

        all_box_detections: List[Detection] = []
        wall_detections: List[Detection] = []

        for name, model in self.models.items():
            detections = model.predict(image)
            if name == "wall":
                wall_detections = detections
            else:
                all_box_detections.extend(detections)

        wall_lines = self._build_wall_lines(wall_detections, width, height)
        yolo_lines = self._build_yolo_text(all_box_detections, width, height)

        rooms_count = sum(1 for det in all_box_detections if det.label_id == "2")
        doors_count = sum(1 for det in all_box_detections if det.label_id == "0")
        stairs_count = sum(1 for det in all_box_detections if det.label_id == "3")
        elevator_count = sum(1 for det in all_box_detections if det.label_id == "1")

        return {
            "file_name": filename or "uploaded.png",
            "image_width": width,
            "image_height": height,
            "yolo_text": "\n".join(yolo_lines),
            "wall_text": "\n".join(wall_lines),
            "door_text": "",
            "class_names": ["room", "stairs", "wall", "elevator", "door"],
            "counts": {
                "rooms": rooms_count,
                "stairs_elevators": stairs_count + elevator_count,
                "doors": doors_count,
                "walls": len(wall_detections),
            },
        }


def build_inference_service_from_config() -> FloorPlanInferenceService:
    """
    설정 파일(config/settings.json)을 기반으로 RF-DETR 모델 구성을 생성합니다.
    """
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

    return FloorPlanInferenceService(configs)
