"""RF-DETR backend wrapper."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any, Sequence

import streamlit as st
import supervision as sv
from PIL import Image

from config import DEFAULT_RFDETR_VARIANT, RFDETR_MODEL_CLASS_MAP
from inference.base import Detection, InferenceBackend, InferenceOutput


def _normalize_variant(variant: str | None) -> str:
    normalized = (variant or DEFAULT_RFDETR_VARIANT).lower()
    if normalized not in RFDETR_MODEL_CLASS_MAP:
        raise ValueError(f"지원하지 않는 RF-DETR 모델 종류입니다: {variant!r}")
    return normalized


def _get_constructor(variant: str):
    class_name = RFDETR_MODEL_CLASS_MAP[variant]
    try:
        module = import_module("rfdetr")
    except ImportError as exc:  # pragma: no cover - surfaced via UI instead
        raise RuntimeError(
            "RF-DETR 패키지를 찾을 수 없습니다. requirements.txt의 의존성을 설치했는지 확인하세요."
        ) from exc
    try:
        return getattr(module, class_name)
    except AttributeError as exc:  # pragma: no cover - surfaced via UI instead
        raise RuntimeError(
            f"RF-DETR 패키지에서 {class_name} 클래스를 찾을 수 없습니다. 패키지 버전을 확인하세요."
        ) from exc


@st.cache_resource(show_spinner=False)
def _load_rfdetr_model(weights_path: str, device: str, variant: str) -> Any:
    """Load RF-DETR weights with caching."""
    constructor = _get_constructor(variant)
    return constructor(pretrain_weights=weights_path, device=device)


def _build_label_map(model: Any) -> dict[int, str]:
    """Generate a mapping from class IDs to labels with multiple fallbacks."""
    class_info = getattr(model, "class_names", None)
    label_map: dict[int, str] = {}
    if isinstance(class_info, dict):
        numeric_entries: list[tuple[int, str]] = []
        for key, value in class_info.items():
            try:
                numeric_key = int(key)
            except (TypeError, ValueError):
                continue
            label = str(value)
            label_map[numeric_key] = label
            numeric_entries.append((numeric_key, label))
        if numeric_entries:
            numeric_entries.sort(key=lambda item: item[0])
            for idx, (_, name) in enumerate(numeric_entries):
                label_map.setdefault(idx, name)
        return label_map
    if isinstance(class_info, (list, tuple)):
        return {idx: str(name) for idx, name in enumerate(class_info)}
    return {}


class RFDETRBackend(InferenceBackend):
    """InferenceBackend implementation for RF-DETR checkpoints."""

    def __init__(self, weights: Path, device: str, variant: str | None = None) -> None:
        self.weights_path = str(weights)
        self.device = device
        self.variant = _normalize_variant(variant)

    def predict_batch(
        self, images: Sequence[Image.Image], confidence: float
    ) -> list[InferenceOutput]:
        if not images:
            return []

        model = _load_rfdetr_model(self.weights_path, self.device, self.variant)
        label_map = _build_label_map(model)

        predictions = model.predict(list(images), threshold=confidence)
        if isinstance(predictions, sv.Detections):
            predictions = [predictions]

        outputs: list[InferenceOutput] = []
        for detections in predictions:
            if detections is None:
                detections = sv.Detections.empty()

            parsed_detections: list[Detection] = []
            for xyxy, score, class_id in zip(
                detections.xyxy, detections.confidence, detections.class_id
            ):
                parsed_detections.append(
                    Detection(
                        label=label_map.get(int(class_id), str(int(class_id))),
                        confidence=float(score),
                        box=(
                            float(xyxy[0]),
                            float(xyxy[1]),
                            float(xyxy[2]),
                            float(xyxy[3]),
                        ),
                    )
                )

            outputs.append(InferenceOutput(detections=parsed_detections))

        return outputs
