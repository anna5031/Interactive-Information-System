"""Inference backends for YOLO and RF-DETR models."""
from __future__ import annotations

from core.model_registry import BACKEND_RFDETR, BACKEND_YOLO, ModelInfo
from inference.base import InferenceBackend


def create_backend(model: ModelInfo, device: str, iou: float | None = None) -> InferenceBackend:
    """Instantiate an inference backend for the given model metadata."""
    if model.backend == BACKEND_YOLO:
        from inference.yolo_backend import YOLOBackend

        return YOLOBackend(model.path, device, iou=iou)
    if model.backend == BACKEND_RFDETR:
        from inference.rfdetr_backend import RFDETRBackend

        return RFDETRBackend(model.path, device)
    raise ValueError(f"Unsupported backend: {model.backend}")
