from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import hashlib
import io

import streamlit as st
from PIL import Image, UnidentifiedImageError

from config import CONFIDENCE_MIN, DEFAULT_IOU, TEST_IMAGE_DIR, format_confidence
from core.model_registry import BACKEND_RFDETR, ModelInfo, get_model, list_available_models
from inference import create_backend
from inference.base import InferenceOutput, resolve_device
from utils.image_loader import list_test_images, load_image
from ui.components import (
    render_device_status,
    render_evaluation_results,
    render_image_section,
    render_sidebar,
)
from utils.coco_eval import CocoGroundTruth, evaluate_batch_predictions, load_coco_ground_truth


st.set_page_config(page_title="YOLO · RF-DETR 대조", layout="wide")


@dataclass(frozen=True)
class PredictionKey:
    model_id: str
    device: str
    image_ids: tuple[str, ...]
    iou: float | None
    rfdetr_variant: str | None


PREDICTION_CACHE_KEY = "prediction_cache"


def _get_selected_model(models: list[ModelInfo]) -> Optional[ModelInfo]:
    if not models:
        return None
    stored_id = st.session_state.get("selected_model_id")
    if isinstance(stored_id, str):
        match = get_model(stored_id, models)
        if match:
            return match
    return None


def _cache_lookup(key: PredictionKey) -> list[InferenceOutput] | None:
    cache: dict[PredictionKey, list[InferenceOutput]] = st.session_state.setdefault(PREDICTION_CACHE_KEY, {})
    return cache.get(key)


def _cache_store(key: PredictionKey, outputs: list[InferenceOutput]) -> None:
    cache: dict[PredictionKey, list[InferenceOutput]] = st.session_state.setdefault(PREDICTION_CACHE_KEY, {})
    cache[key] = outputs


def _load_coco_labels(uploaded_file) -> CocoGroundTruth | None:
    if uploaded_file is None:
        return None
    cache: dict[str, object] = st.session_state.setdefault("coco_label_cache", {})
    data_bytes = uploaded_file.getvalue()
    digest = hashlib.sha256(data_bytes).hexdigest()
    if cache.get("digest") == digest:
        return cache.get("data")  # type: ignore[return-value]
    try:
        parsed = load_coco_ground_truth(data_bytes)
    except ValueError as exc:
        st.error(f"COCO 라벨을 불러오는 중 오류가 발생했습니다: {exc}")
        return None
    cache["digest"] = digest
    cache["data"] = parsed
    cache["name"] = uploaded_file.name
    return parsed


def _prepare_uploaded_images(files: list) -> tuple[list[str], list[str], list[Image.Image]]:
    image_ids: list[str] = []
    labels: list[str] = []
    images: list[Image.Image] = []

    for index, uploaded in enumerate(files, start=1):
        name = uploaded.name or f"upload_{index}"
        data = uploaded.getvalue()
        if not data:
            continue
        try:
            image = Image.open(io.BytesIO(data)).convert("RGB")
        except UnidentifiedImageError:
            st.warning(f"이미지 파일이 아닙니다: {name}")
            continue
        image_ids.append(f"upload:{hashlib.sha256(data).hexdigest()}")
        labels.append(name)
        images.append(image)
    return image_ids, labels, images


def _prepare_folder_images() -> tuple[list[str], list[str], list[Image.Image]]:
    paths = list_test_images(TEST_IMAGE_DIR)
    image_ids = [f"path:{path.resolve()}" for path in paths]
    labels = [path.name for path in paths]
    images = [load_image(path) for path in paths]
    return image_ids, labels, images


def _resolve_predictions(
    model: ModelInfo,
    device: str,
    image_ids: list[str],
    pil_images: list[Image.Image],
    iou_threshold: float | None,
    rfdetr_variant: str | None,
) -> list[InferenceOutput]:
    key = PredictionKey(model.id, device, tuple(image_ids), iou_threshold, rfdetr_variant)
    cached = _cache_lookup(key)
    if cached is not None:
        return cached

    backend = create_backend(model, device, iou=iou_threshold, rfdetr_variant=rfdetr_variant)
    with st.spinner("추론 실행 중..."):
        outputs = backend.predict_batch(pil_images, CONFIDENCE_MIN)

    _cache_store(key, outputs)
    return outputs


def main() -> None:
    st.title("모델 추론 비교 · YOLO & RF-DETR")
    st.caption("test_image 안의 이미지를 한 번에 확인하고 confidence와 장치를 조절하세요.")

    models = list_available_models()
    selected_model = _get_selected_model(models)
    default_device = st.session_state.get("device_choice", "auto")

    (
        selected_model,
        confidence,
        iou_threshold,
        device_choice,
        rfdetr_variant,
        annotation_file,
    ) = render_sidebar(models, selected_model, default_device)
    resolved_device = resolve_device(device_choice)
    uploaded_files = render_device_status(resolved_device)

    if not models:
        st.warning("사용 가능한 모델이 없습니다. `model` 폴더를 확인하세요.")
        return

    if uploaded_files:
        image_ids, image_labels, pil_images = _prepare_uploaded_images(uploaded_files)
        if not pil_images:
            st.warning("업로드한 파일에서 사용할 이미지를 찾을 수 없습니다.")
            return
        dataset_caption = "업로드한 이미지로 추론합니다."
    else:
        image_ids, image_labels, pil_images = _prepare_folder_images()
        if not pil_images:
            st.warning("`test_image` 폴더에서 이미지를 찾을 수 없습니다.")
            return
        dataset_caption = "`test_image` 폴더의 이미지를 사용합니다."

    image_count = len(pil_images)

    if selected_model is None:
        st.info("사이드바에서 사용할 모델을 선택하세요.")
        return

    st.write(
        f"**이미지 {image_count}개** · Confidence {format_confidence(confidence)} · Device {resolved_device.upper()}"
    )
    st.caption(dataset_caption)

    try:
        outputs = _resolve_predictions(
            selected_model,
            resolved_device,
            image_ids,
            pil_images,
            iou_threshold=iou_threshold,
            rfdetr_variant=rfdetr_variant,
        )
    except Exception as exc:  # pragma: no cover - surfaced via Streamlit
        st.error(f"추론 실행 중 오류가 발생했습니다: {exc}")
        return

    if selected_model.backend == BACKEND_RFDETR and annotation_file is not None:
        coco_labels = _load_coco_labels(annotation_file)
        if coco_labels is not None:
            eval_iou = iou_threshold if iou_threshold is not None else DEFAULT_IOU
            summary = evaluate_batch_predictions(
                image_labels,
                outputs,
                coco_labels,
                confidence_threshold=confidence,
                iou_threshold=eval_iou,
            )
            render_evaluation_results(summary)
    for label, original_image, result in zip(image_labels, pil_images, outputs):
        render_image_section(
            image_label=label,
            original_image=original_image,
            detections=result.detections,
            confidence_threshold=confidence,
            iou_threshold=iou_threshold,
        )


if __name__ == "__main__":
    main()
