"""Ultralytics YOLO backend wrapper."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO

from config import DEFAULT_IOU
from inference.base import Detection, InferenceBackend, InferenceOutput


@st.cache_resource(show_spinner=False)
def _load_yolo_model(weights_path: str, device: str) -> YOLO:
    """Load a YOLO model with caching."""
    model = YOLO(weights_path)
    # Ultralytics accepts cuda index notation; default to first device.
    target_device = "cuda:0" if device == "cuda" else device
    if target_device:
        model.to(target_device)
    return model


class YOLOBackend(InferenceBackend):
    """InferenceBackend implementation for YOLO checkpoints."""

    def __init__(self, weights: Path, device: str, iou: float | None = None) -> None:
        self.weights_path = str(weights)
        self.device = "cuda:0" if device == "cuda" else device
        self.iou = iou if iou is not None else DEFAULT_IOU

    def predict_batch(self, images: Sequence[Image.Image], confidence: float) -> list[InferenceOutput]:
        if not images:
            return []

        model = _load_yolo_model(self.weights_path, self.device)
        outputs: list[InferenceOutput] = []

        for image in images:
            frame = np.array(image)
            result = model.predict(
                source=frame,
                conf=confidence,
                iou=self.iou,
                device=self.device,
                verbose=False,
                save=False,
            )[0]

            detections: list[Detection] = []
            boxes = getattr(result, "boxes", None)
            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy()
                scores = boxes.conf.cpu().numpy()
                classes = boxes.cls.cpu().numpy()
                names = result.names or {}
                for coords, score, cls_id in zip(xyxy, scores, classes):
                    label = names.get(int(cls_id), str(int(cls_id)))
                    detections.append(
                        Detection(
                            label=str(label),
                            confidence=float(score),
                            box=(float(coords[0]), float(coords[1]), float(coords[2]), float(coords[3])),
                        )
                    )

            outputs.append(InferenceOutput(detections=detections))

        return outputs
