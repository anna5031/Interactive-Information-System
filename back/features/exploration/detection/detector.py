from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
from ultralytics.engine.results import Results  # type: ignore

from .constants import (
    FACE_INDICES,
    TORSO_SETS,
    KP_LEFT_ANKLE,
    KP_RIGHT_ANKLE,
    KP_LEFT_EAR,
    KP_RIGHT_EAR,
    KP_LEFT_EYE,
    KP_RIGHT_EYE,
    KP_LEFT_SHOULDER,
    KP_RIGHT_SHOULDER,
    KP_LEFT_HIP,
    KP_RIGHT_HIP,
    KP_LEFT_KNEE,
    KP_RIGHT_KNEE,
    KP_NOSE,
)
from ..config import DetectionConfig


@dataclass(slots=True)
class DetectionResult:
    centroid: np.ndarray
    keypoints: np.ndarray
    confidence: float


def build_detections(result: Results, config: DetectionConfig) -> List[Dict[str, np.ndarray]]:
    detections: List[Dict[str, np.ndarray]] = []
    keypoints = getattr(result, "keypoints", None)
    boxes = getattr(result, "boxes", None)

    if keypoints is None or boxes is None or len(result) == 0:
        return detections

    try:
        kp_xy = keypoints.xy.cpu().numpy()
    except Exception:  # pragma: no cover
        kp_xy = keypoints.xy

    try:
        kp_conf = keypoints.conf.cpu().numpy() if keypoints.conf is not None else None
    except Exception:  # pragma: no cover
        kp_conf = keypoints.conf if keypoints.conf is not None else None

    try:
        scores = boxes.conf.cpu().numpy()
    except Exception:  # pragma: no cover
        scores = boxes.conf

    try:
        boxes_xyxy = boxes.xyxy.cpu().numpy()
    except Exception:  # pragma: no cover
        boxes_xyxy = boxes.xyxy

    if kp_xy is None or len(kp_xy) == 0:
        return detections

    for idx, person_kp in enumerate(kp_xy):
        score = float(scores[idx]) if scores is not None and idx < len(scores) else 0.0
        if score < config.confidence_threshold:
            continue
        fallback_centroid = _bbox_center(boxes_xyxy, idx)
        conf_array = kp_conf[idx] if kp_conf is not None and idx < len(kp_conf) else None

        face_anchor = compute_face_anchor(
            person_kp,
            conf_array,
            fallback_centroid,
            config,
        )

        anchors = collect_pose_anchors(
            person_kp,
            conf_array,
            boxes_xyxy[idx] if boxes_xyxy is not None and idx < len(boxes_xyxy) else None,
            fallback_centroid,
            config,
        )

        detections.append(
            {
                "centroid": face_anchor.astype(float),
                "keypoints": person_kp.astype(float),
                "bbox": boxes_xyxy[idx].astype(float) if boxes_xyxy is not None else None,
                "confidence": float(scores[idx]) if scores is not None else 0.0,
                "anchors": anchors,
            }
        )
    return detections


def _bbox_center(boxes_xyxy: Optional[np.ndarray], idx: int) -> np.ndarray:
    if boxes_xyxy is None or idx >= len(boxes_xyxy):
        return np.array([0.0, 0.0], dtype=float)
    x1, y1, x2, y2 = boxes_xyxy[idx]
    return np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=float)


def _valid_point(
    keypoints_xy: np.ndarray,
    keypoints_conf: Optional[np.ndarray],
    index: int,
    threshold: float,
) -> Optional[np.ndarray]:
    if keypoints_xy.size == 0 or index >= keypoints_xy.shape[0]:
        return None
    point = keypoints_xy[index]
    if not np.isfinite(point).all():
        return None
    if keypoints_conf is not None:
        if index >= keypoints_conf.shape[0]:
            return None
        if float(keypoints_conf[index]) < threshold:
            return None
    return point


def compute_face_anchor(
    keypoints_xy: np.ndarray,
    keypoints_conf: Optional[np.ndarray],
    fallback: np.ndarray,
    config: DetectionConfig,
) -> np.ndarray:
    candidates: List[np.ndarray] = []

    thr = config.keypoint_confidence_threshold
    nose = _valid_point(keypoints_xy, keypoints_conf, KP_NOSE, thr)
    left_eye = _valid_point(keypoints_xy, keypoints_conf, KP_LEFT_EYE, thr)
    right_eye = _valid_point(keypoints_xy, keypoints_conf, KP_RIGHT_EYE, thr)
    left_ear = _valid_point(keypoints_xy, keypoints_conf, KP_LEFT_EAR, thr)
    right_ear = _valid_point(keypoints_xy, keypoints_conf, KP_RIGHT_EAR, thr)

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
        point_a = _valid_point(keypoints_xy, keypoints_conf, idx_a, thr)
        point_b = _valid_point(keypoints_xy, keypoints_conf, idx_b, thr)
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
        point = _valid_point(keypoints_xy, keypoints_conf, idx, thr)
        if point is None:
            continue
        conf = float(keypoints_conf[idx]) if keypoints_conf is not None else 1.0
        if conf > best_conf:
            best_conf = conf
            best_idx = idx
    if best_idx is not None and best_conf > 0:
        point = keypoints_xy[best_idx]
        if np.isfinite(point).all():
            return point
    return fallback


def collect_pose_anchors(
    keypoints_xy: np.ndarray,
    keypoints_conf: Optional[np.ndarray],
    bbox: Optional[np.ndarray],
    fallback: np.ndarray,
    config: DetectionConfig,
) -> Dict[str, Optional[np.ndarray]]:
    thr = config.keypoint_confidence_threshold
    anchors: Dict[str, Optional[np.ndarray]] = {
        "ankle_left": _valid_point(keypoints_xy, keypoints_conf, KP_LEFT_ANKLE, thr),
        "ankle_right": _valid_point(keypoints_xy, keypoints_conf, KP_RIGHT_ANKLE, thr),
        "hip_center": _mean_pair(keypoints_xy, keypoints_conf, KP_LEFT_HIP, KP_RIGHT_HIP, thr),
        "knee_center": _mean_pair(keypoints_xy, keypoints_conf, KP_LEFT_KNEE, KP_RIGHT_KNEE, thr),
        "shoulder_center": _mean_pair(keypoints_xy, keypoints_conf, KP_LEFT_SHOULDER, KP_RIGHT_SHOULDER, thr),
        "nose": _valid_point(keypoints_xy, keypoints_conf, KP_NOSE, thr),
    }
    anchors["axis_point"] = _compute_axis_projection(
        hip_center=anchors["hip_center"],
        shoulder_center=anchors["shoulder_center"],
        nose=anchors["nose"],
        bbox=bbox,
        fallback=fallback,
    )
    return anchors


def _compute_axis_projection(
    hip_center: Optional[np.ndarray],
    shoulder_center: Optional[np.ndarray],
    nose: Optional[np.ndarray],
    bbox: Optional[np.ndarray],
    fallback: np.ndarray,
) -> Optional[np.ndarray]:
    axis_origin: Optional[np.ndarray] = None
    axis_dir: Optional[np.ndarray] = None

    if hip_center is not None and shoulder_center is not None:
        axis_origin = hip_center
        axis_dir = hip_center - shoulder_center
    elif hip_center is not None and nose is not None:
        axis_origin = hip_center
        axis_dir = hip_center - nose
    elif hip_center is not None:
        axis_origin = hip_center
        axis_dir = np.array([0.0, 1.0], dtype=float)
    elif shoulder_center is not None and nose is not None:
        axis_origin = shoulder_center
        axis_dir = shoulder_center - nose
    elif bbox is not None:
        axis_origin = np.array([(bbox[0] + bbox[2]) * 0.5, bbox[1]], dtype=float)
        axis_dir = np.array([0.0, 1.0], dtype=float)
    else:
        axis_origin = fallback
        axis_dir = np.array([0.0, 1.0], dtype=float)

    if axis_origin is None or axis_dir is None:
        return None

    norm = np.linalg.norm(axis_dir)
    if norm < 1e-3:
        return None
    axis_dir = axis_dir / norm
    if axis_dir[1] < 0:
        axis_dir = -axis_dir

    if bbox is not None and abs(axis_dir[1]) > 1e-3:
        t = (bbox[3] - axis_origin[1]) / axis_dir[1]
        if t > 0:
            return axis_origin + axis_dir * t

    return np.array([axis_origin[0], fallback[1]], dtype=float)


def _mean_pair(
    keypoints_xy: np.ndarray,
    keypoints_conf: Optional[np.ndarray],
    idx_a: int,
    idx_b: int,
    threshold: float,
) -> Optional[np.ndarray]:
    point_a = _valid_point(keypoints_xy, keypoints_conf, idx_a, threshold)
    point_b = _valid_point(keypoints_xy, keypoints_conf, idx_b, threshold)
    if point_a is None or point_b is None:
        return None
    return (point_a + point_b) * 0.5
