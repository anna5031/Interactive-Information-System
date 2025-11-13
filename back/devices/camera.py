import logging
from typing import Optional, Union

import cv2  # type: ignore

from .types import DeviceCheckResult

logger = logging.getLogger(__name__)


CameraSource = Optional[Union[int, str]]


def check_camera(source: CameraSource = None) -> DeviceCheckResult:
    name = "카메라"
    _suppress_opencv_logs()

    selected_source = source if source is not None else 0
    cap = cv2.VideoCapture(selected_source, cv2.CAP_ANY)
    if cap is not None and cap.isOpened():
        cap.release()
        return DeviceCheckResult(
            name=name,
            ok=True,
            detail=f"사용할 카메라 소스: {selected_source}",
            meta={"source": selected_source},
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
