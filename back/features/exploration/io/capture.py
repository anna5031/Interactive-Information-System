from __future__ import annotations

"""Camera capture management for the exploration pipeline."""

import asyncio
import logging
from typing import Optional

import cv2  # type: ignore
import numpy as np

logger = logging.getLogger(__name__)


class CameraCapture:
    def __init__(
        self,
        source: Optional[int | str] = None,
        frame_size: Optional[tuple[int, int]] = None,
    ) -> None:
        self._source: Optional[int | str] = source
        self._capture: Optional[cv2.VideoCapture] = None
        self._frame_size = frame_size

    async def start(self) -> None:
        if self._capture is not None:
            return
        loop = asyncio.get_running_loop()
        success = await loop.run_in_executor(None, self._open)
        if not success:
            raise RuntimeError(f"카메라 소스를 열 수 없습니다: {self._source}")
        logger.info("카메라 스트림 시작 (source=%s)", self._source)

    async def stop(self) -> None:
        if self._capture is None:
            return
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._release)
        logger.info("카메라 스트림 종료")

    def read(self) -> tuple[bool, Optional[np.ndarray]]:
        if self._capture is None:
            return False, None
        ret, frame = self._capture.read()
        if not ret or frame is None:
            return False, None
        return True, frame

    def is_open(self) -> bool:
        return self._capture is not None and self._capture.isOpened()

    def _open(self) -> bool:
        capture = cv2.VideoCapture(self._source, cv2.CAP_ANY)
        if not capture or not capture.isOpened():
            if capture:
                capture.release()
            return False
        self._capture = capture
        try:
            self._apply_frame_size()
        except RuntimeError:
            logger.error("카메라 해상도를 적용하지 못해 스트림을 중단합니다.")
            self._release()
            return False
        return True

    def _release(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def _apply_frame_size(self) -> None:
        if self._capture is None or self._frame_size is None:
            return
        width, height = self._frame_size
        if width <= 0 or height <= 0:
            raise RuntimeError(
                f"잘못된 카메라 해상도 요청: width={width}, height={height}"
            )
        set_w = self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
        set_h = self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
        applied_width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        applied_height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info(
            "카메라 해상도 설정 요청 (requested=%dx%d) → 실제 장치 보고값 (actual=%dx%d)",
            width,
            height,
            applied_width,
            applied_height,
        )
        if not (set_w and set_h):
            raise RuntimeError("카메라 드라이버가 해상도 설정을 거부했습니다.")
        if applied_width != width or applied_height != height:
            raise RuntimeError(
                f"카메라 해상도 적용 실패: requested={width}x{height}, actual={applied_width}x{applied_height}"
            )
