import logging
from typing import Optional, Union

import cv2  # type: ignore

from .types import DeviceCheckResult

logger = logging.getLogger(__name__)


CameraSource = Optional[Union[int, str]]
CameraFrameSize = Optional[tuple[int, int]]


def check_camera(
    source: CameraSource = None,
    frame_size: CameraFrameSize = None,
) -> DeviceCheckResult:
    name = "카메라"
    _suppress_opencv_logs()

    selected_source = source if source is not None else 0
    cap = cv2.VideoCapture(selected_source, cv2.CAP_ANY)
    if cap is not None and cap.isOpened():
        try:
            detail, meta = _build_success_detail(cap, selected_source, frame_size)
        except RuntimeError as exc:
            cap.release()
            return DeviceCheckResult(
                name=name,
                ok=False,
                detail=f"카메라 해상도 설정 실패: {exc}",
            )
        cap.release()
        return DeviceCheckResult(
            name=name,
            ok=True,
            detail=detail,
            meta=meta,
        )

    return DeviceCheckResult(
        name=name,
        ok=False,
        detail=f"카메라 소스를 열 수 없습니다: {selected_source}",
    )


def _suppress_opencv_logs() -> None:
    try:
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
    except AttributeError:
        logger.debug("OpenCV 로그 레벨 제어를 지원하지 않는 버전입니다.")


def _build_success_detail(
    cap: cv2.VideoCapture,
    source: CameraSource,
    frame_size: CameraFrameSize,
) -> tuple[str, dict]:
    detail = f"사용할 카메라 소스: {source}"
    meta: dict = {"source": source}

    if frame_size is None:
        return detail, meta

    width, height = frame_size
    applied_width, applied_height = _apply_frame_size(cap, width, height)
    if applied_width and applied_height:
        detail += f", frame_size={applied_width}x{applied_height}"
        meta["frame_size"] = (applied_width, applied_height)
    else:
        detail += (
            f", frame_size 요청값 적용 불가 (requested={width}x{height})"
        )
    return detail, meta


def _apply_frame_size(
    cap: cv2.VideoCapture,
    width: int,
    height: int,
) -> tuple[int, int]:
    if width <= 0 or height <= 0:
        logger.warning(
            "잘못된 카메라 해상도 요청: width=%s, height=%s (무시합니다).",
            width,
            height,
        )
        return 0, 0

    set_w = cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
    set_h = cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
    applied_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    applied_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    logger.info(
        "카메라 해상도 설정 요청 (requested=%dx%d) → 실제 장치 보고값 (actual=%dx%d)",
        width,
        height,
        applied_width,
        applied_height,
    )

    if not (set_w and set_h):
        logger.warning("카메라 드라이버가 해상도 설정을 거부했습니다.")
    if applied_width != width or applied_height != height:
        logger.warning(
            "카메라 해상도 적용 불일치: requested=%dx%d, actual=%dx%d",
            width,
            height,
            applied_width,
            applied_height,
        )

    return applied_width, applied_height
