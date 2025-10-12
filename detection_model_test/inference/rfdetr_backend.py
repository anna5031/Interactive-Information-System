"""RF-DETR backend wrapper."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import streamlit as st
import supervision as sv
from PIL import Image

from inference.base import Detection, InferenceBackend, InferenceOutput


try:
    from rfdetr import RFDETRMedium
except ImportError as exc:  # pragma: no cover - surfaced via UI instead
    raise RuntimeError(
        "RFDETRMedium is not available. Ensure the rf-detr package is installed."
    ) from exc


@st.cache_resource(show_spinner=False)
def _load_rfdetr_model(weights_path: str, device: str) -> RFDETRMedium:
    """Load RF-DETR weights with caching."""
    return RFDETRMedium(pretrain_weights=weights_path, device=device)


def _build_label_map(model: RFDETRMedium) -> dict[int, str]:
    class_info = model.class_names
    if isinstance(class_info, dict):
        ordered = sorted(class_info.items(), key=lambda item: int(item[0]))
        return {idx: str(name) for idx, (_, name) in enumerate(ordered)}
    if isinstance(class_info, (list, tuple)):
        return {idx: str(name) for idx, name in enumerate(class_info)}
    return {}


class RFDETRBackend(InferenceBackend):
    """InferenceBackend implementation for RF-DETR checkpoints."""

    def __init__(self, weights: Path, device: str) -> None:
        self.weights_path = str(weights)
        self.device = device

    def predict_batch(self, images: Sequence[Image.Image], confidence: float) -> list[InferenceOutput]:
        if not images:
            return []

        model = _load_rfdetr_model(self.weights_path, self.device)
        label_map = _build_label_map(model)

        predictions = model.predict(list(images), threshold=confidence)
        if isinstance(predictions, sv.Detections):
            predictions = [predictions]

        outputs: list[InferenceOutput] = []
        for detections in predictions:
            if detections is None:
                detections = sv.Detections.empty()

            parsed_detections: list[Detection] = []
            for xyxy, score, class_id in zip(detections.xyxy, detections.confidence, detections.class_id):
                parsed_detections.append(
                    Detection(
                        label=label_map.get(int(class_id), str(int(class_id))),
                        confidence=float(score),
                        box=(float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])),
                    )
                )

            outputs.append(InferenceOutput(detections=parsed_detections))

        return outputs
