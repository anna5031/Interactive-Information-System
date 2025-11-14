"""Utilities for loading COCO annotations and evaluating detections."""
from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Dict, Iterable, List, Sequence

from inference.base import Detection, InferenceOutput


def _normalize_label(label: str) -> str:
    return str(label).strip().lower()


@dataclass(frozen=True)
class GroundTruthBox:
    """Single labeled bounding box from COCO annotations."""

    label: str
    normalized_label: str
    box: tuple[float, float, float, float]


@dataclass(frozen=True)
class CocoGroundTruth:
    """Parsed COCO annotation content indexed by image file name."""

    annotations_by_file: Dict[str, List[GroundTruthBox]]
    category_names: Dict[int, str]


@dataclass(frozen=True)
class ImageEvaluation:
    """Evaluation metrics for a single image."""

    image_label: str
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float


@dataclass(frozen=True)
class ClassEvaluation:
    """Metrics aggregated per class label."""

    label: str
    normalized_label: str
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float


@dataclass(frozen=True)
class EvaluationSummary:
    """Aggregated evaluation metrics for a batch of images."""

    total_tp: int
    total_fp: int
    total_fn: int
    precision: float
    recall: float
    f1: float
    accuracy: float
    per_image: List[ImageEvaluation]
    per_class: List[ClassEvaluation]
    evaluated_images: List[str]
    missing_images: List[str]


def load_coco_ground_truth(raw_bytes: bytes) -> CocoGroundTruth:
    """Parse a COCO annotation JSON payload."""
    try:
        payload = json.loads(raw_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("유효한 COCO JSON 파일이 아닙니다.") from exc

    categories = payload.get("categories") or []
    category_names: Dict[int, str] = {
        int(category["id"]): str(category["name"])
        for category in categories
        if "id" in category and "name" in category
    }

    images = payload.get("images") or []
    image_id_to_name: Dict[int, str] = {}
    for record in images:
        if "id" not in record or "file_name" not in record:
            continue
        image_id_to_name[int(record["id"])] = str(record["file_name"])

    annotations = payload.get("annotations") or []
    annotations_by_file: Dict[str, List[GroundTruthBox]] = {}

    for annotation in annotations:
        image_id = annotation.get("image_id")
        category_id = annotation.get("category_id")
        bbox = annotation.get("bbox")
        if image_id is None or category_id is None or bbox is None:
            continue
        file_name = image_id_to_name.get(int(image_id))
        category_name = category_names.get(int(category_id))
        if not file_name or not category_name:
            continue
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        x, y, width, height = map(float, bbox)
        box = (x, y, x + width, y + height)
        annotations_by_file.setdefault(file_name, []).append(
            GroundTruthBox(
                label=category_name,
                normalized_label=_normalize_label(category_name),
                box=box,
            )
        )

    return CocoGroundTruth(
        annotations_by_file=annotations_by_file,
        category_names=category_names,
    )


def _iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection

    if union <= 0.0:
        return 0.0
    return intersection / union


def _safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0 else 0.0


def _evaluate_image(
    image_label: str,
    detections: Iterable[Detection],
    ground_truth: List[GroundTruthBox],
    confidence_threshold: float,
    iou_threshold: float,
) -> tuple[ImageEvaluation, Dict[str, Dict[str, float | int | str]]]:
    normalized_predictions = [
        (
            _normalize_label(detection.label),
            detection.box,
            detection.confidence,
            detection.label,
        )
        for detection in detections
        if detection.confidence >= confidence_threshold
    ]
    normalized_predictions.sort(key=lambda item: item[2], reverse=True)

    matched_gt = [False] * len(ground_truth)
    true_positives = 0
    false_positives = 0
    per_class: Dict[str, Dict[str, float | int | str]] = {}

    def _ensure_class(label_key: str, display_label: str) -> Dict[str, float | int | str]:
        entry = per_class.setdefault(
            label_key,
            {"label": display_label or label_key, "tp": 0, "fp": 0, "fn": 0},
        )
        if not entry["label"] and display_label:
            entry["label"] = display_label
        return entry

    for normalized_label, box, _, raw_label in normalized_predictions:
        best_index = -1
        best_score = 0.0
        for idx, gt in enumerate(ground_truth):
            if matched_gt[idx] or normalized_label != gt.normalized_label:
                continue
            score = _iou(box, gt.box)
            if score >= iou_threshold and score > best_score:
                best_index = idx
                best_score = score
        if best_index >= 0:
            matched_gt[best_index] = True
            true_positives += 1
            entry = _ensure_class(normalized_label, raw_label)
            entry["tp"] = int(entry["tp"]) + 1
        else:
            false_positives += 1
            entry = _ensure_class(normalized_label, raw_label)
            entry["fp"] = int(entry["fp"]) + 1

    false_negatives = matched_gt.count(False)
    for matched, gt in zip(matched_gt, ground_truth):
        if not matched:
            entry = _ensure_class(gt.normalized_label, gt.label)
            entry["fn"] = int(entry["fn"]) + 1

    precision = _safe_ratio(true_positives, true_positives + false_positives)
    recall = _safe_ratio(true_positives, true_positives + false_negatives)
    f1 = _safe_ratio(2 * precision * recall, precision + recall)

    return (
        ImageEvaluation(
            image_label=image_label,
            true_positives=true_positives,
            false_positives=false_positives,
            false_negatives=false_negatives,
            precision=precision,
            recall=recall,
            f1=f1,
        ),
        per_class,
    )


def evaluate_batch_predictions(
    image_labels: Sequence[str],
    outputs: Sequence[InferenceOutput],
    ground_truth: CocoGroundTruth,
    confidence_threshold: float,
    iou_threshold: float,
) -> EvaluationSummary:
    per_image: List[ImageEvaluation] = []
    evaluated_images: List[str] = []
    missing_images: List[str] = []
    per_class_totals: Dict[str, Dict[str, float | int | str]] = {}

    total_tp = total_fp = total_fn = 0

    for label, output in zip(image_labels, outputs):
        gt_boxes = ground_truth.annotations_by_file.get(label)
        if not gt_boxes:
            missing_images.append(label)
            continue
        evaluated_images.append(label)
        result, per_class = _evaluate_image(label, output.detections, gt_boxes, confidence_threshold, iou_threshold)
        per_image.append(result)
        total_tp += result.true_positives
        total_fp += result.false_positives
        total_fn += result.false_negatives
        for norm_label, stats in per_class.items():
            entry = per_class_totals.setdefault(
                norm_label,
                {"label": stats.get("label") or norm_label, "tp": 0, "fp": 0, "fn": 0},
            )
            entry["tp"] = int(entry["tp"]) + int(stats.get("tp", 0))
            entry["fp"] = int(entry["fp"]) + int(stats.get("fp", 0))
            entry["fn"] = int(entry["fn"]) + int(stats.get("fn", 0))
            if not entry["label"] and stats.get("label"):
                entry["label"] = stats.get("label")

    precision = _safe_ratio(total_tp, total_tp + total_fp)
    recall = _safe_ratio(total_tp, total_tp + total_fn)
    f1 = _safe_ratio(2 * precision * recall, precision + recall)
    accuracy = _safe_ratio(total_tp, total_tp + total_fp + total_fn)

    class_summaries: List[ClassEvaluation] = []
    for norm_label, stats in sorted(per_class_totals.items(), key=lambda item: str(item[1]["label"]).lower()):
        tp = int(stats.get("tp", 0))
        fp = int(stats.get("fp", 0))
        fn = int(stats.get("fn", 0))
        class_precision = _safe_ratio(tp, tp + fp)
        class_recall = _safe_ratio(tp, tp + fn)
        class_f1 = _safe_ratio(2 * class_precision * class_recall, class_precision + class_recall)
        class_summaries.append(
            ClassEvaluation(
                label=str(stats.get("label") or norm_label),
                normalized_label=norm_label,
                true_positives=tp,
                false_positives=fp,
                false_negatives=fn,
                precision=class_precision,
                recall=class_recall,
                f1=class_f1,
            )
        )

    return EvaluationSummary(
        total_tp=total_tp,
        total_fp=total_fp,
        total_fn=total_fn,
        precision=precision,
        recall=recall,
        f1=f1,
        accuracy=accuracy,
        per_image=per_image,
        per_class=class_summaries,
        evaluated_images=evaluated_images,
        missing_images=missing_images,
    )
