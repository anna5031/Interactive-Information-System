"""Reusable UI components for the Streamlit app."""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import pandas as pd
import streamlit as st
from PIL import Image

from config import (
    IMAGE_EXTENSIONS,
    CONFIDENCE_MAX,
    CONFIDENCE_MIN,
    CONFIDENCE_STEP,
    DEFAULT_CONFIDENCE,
    DEFAULT_IOU,
    DEVICE_OPTIONS,
    IOU_MAX,
    IOU_MIN,
    IOU_STEP,
    format_confidence,
)
from core.model_registry import BACKEND_RFDETR, BACKEND_YOLO, ModelInfo
from inference.base import Detection, describe_runtime
from utils.visualization import annotate_image


RFDETR_PRIORITY = ("best_total", "best_ema", "best_regular")


def _priority_index(name: str) -> int:
    for idx, keyword in enumerate(RFDETR_PRIORITY):
        if keyword in name:
            return idx
    return len(RFDETR_PRIORITY)


def _family_priority(models: Sequence[ModelInfo]) -> int:
    return min(
        (_priority_index(model.path.name) for model in models),
        default=len(RFDETR_PRIORITY),
    )


def render_sidebar(
    models: Sequence[ModelInfo],
    selected_model: ModelInfo | None,
    default_device: str,
) -> tuple[ModelInfo | None, float, float | None, str]:
    """Render sidebar controls and return selected values."""
    st.sidebar.header("모델 설정")
    if not models:
        st.sidebar.warning("사용 가능한 모델이 없습니다. `model` 폴더를 확인하세요.")
        return None, DEFAULT_CONFIDENCE, None, default_device

    yolo_models = [model for model in models if model.backend == BACKEND_YOLO]
    rfdetr_models = [model for model in models if model.backend == BACKEND_RFDETR]

    backend_options: list[str] = []
    if yolo_models:
        backend_options.append(BACKEND_YOLO)
    if rfdetr_models:
        backend_options.append(BACKEND_RFDETR)

    if not backend_options:
        st.sidebar.warning("선택할 수 있는 모델이 없습니다.")
        return None, DEFAULT_CONFIDENCE, None, default_device

    backend_labels = {BACKEND_YOLO: "YOLO", BACKEND_RFDETR: "RF-DETR"}
    default_backend = backend_options[0]
    if selected_model and selected_model.backend in backend_options:
        default_backend = selected_model.backend

    backend_choice = st.sidebar.radio(
        "모델 종류",
        options=backend_options,
        index=backend_options.index(default_backend),
        format_func=lambda value: backend_labels.get(value, value.upper()),
        key="model_backend",
        horizontal=True,
    )

    chosen_model: ModelInfo | None = None

    if backend_choice == BACKEND_YOLO:
        ordered_models = sorted(yolo_models, key=lambda item: item.name.lower())
        options = [None] + ordered_models
        default_option = (
            selected_model
            if selected_model and selected_model.backend == BACKEND_YOLO
            else None
        )
        selected_option = st.sidebar.selectbox(
            "YOLO 모델",
            options=options,
            index=options.index(default_option) if default_option in options else 0,
            format_func=lambda item: "모델을 선택하세요" if item is None else item.name,
            key="yolo_model_select",
        )
        if isinstance(selected_option, ModelInfo):
            chosen_model = selected_option

    elif backend_choice == BACKEND_RFDETR:
        families: dict[str, list[ModelInfo]] = defaultdict(list)
        for model in rfdetr_models:
            families[model.family or "기타"].append(model)

        if not families:
            st.sidebar.warning("RF-DETR 체크포인트를 찾을 수 없습니다.")
            return None, DEFAULT_CONFIDENCE, None, default_device

        sorted_families = sorted(
            families.items(),
            key=lambda item: (_family_priority(item[1]), item[0].lower()),
        )
        family_names = [name for name, _ in sorted_families]

        family_options = [None] + family_names
        default_family = None
        if selected_model and selected_model.backend == BACKEND_RFDETR:
            default_family = selected_model.family or "기타"
        family_choice = st.sidebar.selectbox(
            "폴더 선택",
            options=family_options,
            index=(
                family_options.index(default_family)
                if default_family in family_options
                else 0
            ),
            format_func=lambda item: "폴더를 선택하세요" if item is None else item,
            key="rfdetr_family_select",
        )

        if family_choice and family_choice in families:
            candidates = sorted(
                families[family_choice],
                key=lambda item: (_priority_index(item.path.name), item.name.lower()),
            )
            if not candidates:
                st.sidebar.warning("선택한 폴더에서 체크포인트를 찾을 수 없습니다.")
            else:
                if (
                    selected_model
                    and selected_model.backend == BACKEND_RFDETR
                    and selected_model in candidates
                ):
                    default_candidate = selected_model
                else:
                    default_candidate = min(
                        candidates,
                        key=lambda item: (
                            _priority_index(item.path.name),
                            item.name.lower(),
                        ),
                    )
                selected_candidate = st.sidebar.selectbox(
                    "체크포인트 선택",
                    options=candidates,
                    index=candidates.index(default_candidate),
                    format_func=lambda item: item.path.name,
                    key="rfdetr_model_select",
                )
                chosen_model = selected_candidate

    confidence = st.sidebar.slider(
        "Confidence Threshold",
        min_value=float(CONFIDENCE_MIN),
        max_value=float(CONFIDENCE_MAX),
        value=float(DEFAULT_CONFIDENCE),
        step=float(CONFIDENCE_STEP),
        format="%.2f",
        key="confidence_slider",
    )

    iou_value: float | None = None
    if backend_choice == BACKEND_YOLO:
        iou_value = st.sidebar.slider(
            "IoU Threshold (YOLO)",
            min_value=float(IOU_MIN),
            max_value=float(IOU_MAX),
            value=float(DEFAULT_IOU),
            step=float(IOU_STEP),
            format="%.2f",
            key="iou_slider",
        )

    device_choice = st.sidebar.radio(
        "연산 장치",
        options=DEVICE_OPTIONS,
        index=(
            DEVICE_OPTIONS.index(default_device)
            if default_device in DEVICE_OPTIONS
            else 0
        ),
        horizontal=True,
        format_func=lambda value: value.upper() if value != "auto" else "AUTO",
        key="device_choice",
    )

    if chosen_model:
        st.session_state["selected_model_id"] = chosen_model.id
    else:
        st.session_state.pop("selected_model_id", None)

    return chosen_model, confidence, iou_value, device_choice


def render_device_status(device: str) -> list:
    """Show the resolved device in the sidebar and provide image upload."""
    st.sidebar.caption(f"현재 장치: {describe_runtime(device)}")
    upload_types = [ext.lstrip(".") for ext in IMAGE_EXTENSIONS]
    uploaded = st.sidebar.file_uploader(
        "추론할 이미지 업로드",
        type=upload_types,
        accept_multiple_files=True,
        help="업로드하면 해당 이미지들만 추론합니다.",
        key="image_uploader",
    )
    return uploaded or []


def _detections_dataframe(detections: Sequence[Detection]) -> pd.DataFrame:
    rows = [
        {
            "Label": detection.label,
            "Confidence": format_confidence(detection.confidence),
            "x1": f"{detection.box[0]:.1f}",
            "y1": f"{detection.box[1]:.1f}",
            "x2": f"{detection.box[2]:.1f}",
            "y2": f"{detection.box[3]:.1f}",
        }
        for detection in detections
    ]
    return pd.DataFrame(rows)


def render_image_section(
    image_label: str,
    original_image: Image.Image,
    detections: Sequence[Detection],
    confidence_threshold: float,
    iou_threshold: float | None,
) -> None:
    """Render original, annotated, and detection table for a single image."""
    filtered = [
        detection
        for detection in detections
        if detection.confidence >= confidence_threshold
    ]
    annotated_image = annotate_image(original_image, filtered)

    st.markdown(f"#### {image_label}")
    original_col, annotated_col = st.columns(2, gap="large")
    with original_col:
        st.image(original_image, caption="원본 이미지", width="stretch")
    with annotated_col:
        st.image(annotated_image, caption="추론 결과", width="stretch")

    expander_label = f"탐지 결과 ({len(filtered)}개)"
    if iou_threshold is not None:
        expander_label += f" · IoU {iou_threshold:.2f}"

    with st.expander(expander_label, expanded=False):
        if filtered:
            df = _detections_dataframe(filtered)
            st.dataframe(df, hide_index=True, width="stretch")
        else:
            st.caption("표시할 탐지 결과가 없습니다.")
