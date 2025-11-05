from __future__ import annotations

"""YOLO pose detector wrapper."""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from ultralytics import YOLO

from .config import ModelConfig
from .constants import (
    FACE_INDICES,
    TORSO_SETS,
    KP_LEFT_EAR,
    KP_LEFT_EYE,
    KP_LEFT_HIP,
    KP_LEFT_SHOULDER,
    KP_LEFT_ANKLE,
    KP_NOSE,
    KP_RIGHT_EAR,
    KP_RIGHT_EYE,
    KP_RIGHT_HIP,
    KP_RIGHT_SHOULDER,
    KP_RIGHT_ANKLE,
)

logger = logging.getLogger(__name__)

Detection = Dict[str, np.ndarray | float | int]


@dataclass(slots=True)
class PoseDetector:
    """Wraps a YOLO pose model for inference."""

    model_config: ModelConfig
    device: str
    model: YOLO | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        path = str(self.model_config.model_path)
        logger.info("Loading YOLO pose model from %s", path)

        requested_device = self.device
        prefer_engine = path.endswith(".engine")
        device_to_use = "cuda:0" if prefer_engine else requested_device

        try:
            model = YOLO(path)
            model.to(device_to_use)
            self.device = device_to_use
            self.model = model
            if prefer_engine:
                logger.info("TensorRT engine loaded on %s", device_to_use)
        except Exception as exc:
            fallback_path = str(self.model_config.model_path.with_suffix(".pt"))
            if prefer_engine and Path(fallback_path).exists():
                logger.warning(
                    "TensorRT 엔진 로드 실패(%s). 파이토치 가중치로 대체합니다: %s",
                    exc,
                    fallback_path,
                )
                model = YOLO(fallback_path)
                model.to(requested_device)
                self.device = requested_device
                self.model = model
            else:
                raise

    def predict(self, frame: np.ndarray) -> List[Detection]:
        if self.model is None:
            raise RuntimeError("YOLO model is not initialised.")

        results = self.model.predict(
            frame,
            device=self.device,
            conf=self.model_config.confidence_threshold,
            iou=self.model_config.iou_threshold,
            verbose=False,
        )
        detections: List[Detection] = []
        if not results:
            return detections

        result = results[0]
        boxes = result.boxes
        keypoints = result.keypoints

        boxes_xyxy = boxes.xyxy.cpu().numpy() if boxes is not None else np.empty((0, 4))
        boxes_conf = boxes.conf.cpu().numpy() if boxes is not None else np.empty((0,))
        boxes_cls = (
            boxes.cls.cpu().numpy().astype(int)
            if boxes is not None and boxes.cls is not None
            else np.zeros(len(boxes_xyxy), dtype=int)
        )

        if keypoints is not None:
            keypoints_xy = keypoints.xy.cpu().numpy()
            keypoints_conf = (
                keypoints.conf.cpu().numpy()
                if keypoints.conf is not None
                else np.ones(keypoints_xy.shape[:-1], dtype=float)
            )
        else:
            keypoints_xy = np.empty((0, 17, 2))
            keypoints_conf = np.empty((0, 17))

        for det_idx in range(len(boxes_xyxy)):
            x1, y1, x2, y2 = boxes_xyxy[det_idx]
            centroid = np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=float)
            kp_xy = (
                keypoints_xy[det_idx] if det_idx < len(keypoints_xy) else np.empty((0, 2))
            )
            kp_conf = (
                keypoints_conf[det_idx]
                if det_idx < len(keypoints_conf)
                else np.empty((0,))
            )
            anchor = compute_face_anchor(kp_xy, kp_conf, centroid)
            foot_point = compute_foot_anchor(kp_xy, kp_conf, (x1 + x2) * 0.5, y2)
            detections.append(
                {
                    "bbox": np.array([x1, y1, x2, y2], dtype=float),
                    "label": int(boxes_cls[det_idx]) if det_idx < len(boxes_cls) else 0,
                    "conf": float(boxes_conf[det_idx]) if det_idx < len(boxes_conf) else 0.0,
                    "centroid": anchor,
                    "keypoints_xy": kp_xy,
                    "keypoints_conf": kp_conf,
                    "bbox_center": centroid,
                    "foot_point": foot_point,
                }
            )

        return detections


def compute_face_anchor(
    keypoints_xy: np.ndarray,
    keypoints_conf: np.ndarray,
    fallback: np.ndarray,
) -> np.ndarray:
    """Heuristic to estimate a stable anchor point for tracking."""
    def _valid_point(index: int) -> Optional[np.ndarray]:
        if (
            keypoints_xy.size == 0
            or keypoints_conf.size == 0
            or index >= keypoints_xy.shape[0]
            or index >= keypoints_conf.shape[0]
        ):
            return None
        point = keypoints_xy[index]
        conf = float(keypoints_conf[index])
        if not np.isfinite(point).all() or conf <= 0:
            return None
        return point

    candidates: List[np.ndarray] = []
    nose = _valid_point(KP_NOSE)
    left_eye = _valid_point(KP_LEFT_EYE)
    right_eye = _valid_point(KP_RIGHT_EYE)
    left_ear = _valid_point(KP_LEFT_EAR)
    right_ear = _valid_point(KP_RIGHT_EAR)

    if nose is not None:
        candidates.append(nose)
    if left_eye is not None and right_eye is not None:
        candidates.append((left_eye + right_eye) * 0.5)
    if left_ear is not None and right_ear is not None:
        candidates.append((left_ear + right_ear) * 0.5)

    if candidates:
        return np.mean(candidates, axis=0)

    for point in [nose, left_eye, right_eye, left_ear, right_ear]:
        if point is not None:
            return point

    torso_points: List[np.ndarray] = []
    for idx_a, idx_b in TORSO_SETS:
        point_a = _valid_point(idx_a)
        point_b = _valid_point(idx_b)
        if point_a is not None and point_b is not None:
            torso_points.append((point_a + point_b) * 0.5)
    if torso_points:
        return np.mean(torso_points, axis=0)

    best_idx = None
    best_conf = -np.inf
    indices = FACE_INDICES + [
        KP_LEFT_SHOULDER,
        KP_RIGHT_SHOULDER,
        KP_LEFT_HIP,
        KP_RIGHT_HIP,
    ]
    for idx in indices:
        if (
            keypoints_xy.size == 0
            or keypoints_conf.size == 0
            or idx >= keypoints_xy.shape[0]
            or idx >= keypoints_conf.shape[0]
        ):
            continue
        point = keypoints_xy[idx]
        conf = float(keypoints_conf[idx])
        if not np.isfinite(point).all():
            continue
        if conf > best_conf:
            best_conf = conf
            best_idx = idx
    if best_idx is not None and best_conf > 0:
        return keypoints_xy[best_idx]
    return fallback


def compute_foot_anchor(
    keypoints_xy: np.ndarray,
    keypoints_conf: np.ndarray,
    fallback_x: float,
    fallback_y: float,
) -> np.ndarray:
    """Approximate foot position using ankle keypoints."""
    points: List[np.ndarray] = []

    for idx in (KP_LEFT_ANKLE, KP_RIGHT_ANKLE):
        if (
            keypoints_xy.size == 0
            or keypoints_conf.size == 0
            or idx >= keypoints_xy.shape[0]
            or idx >= keypoints_conf.shape[0]
        ):
            continue
        point = keypoints_xy[idx]
        conf = float(keypoints_conf[idx])
        if not np.isfinite(point).all() or conf <= 0:
            continue
        points.append(point)

    if points:
        return np.mean(points, axis=0)

    return np.array([fallback_x, fallback_y], dtype=float)
