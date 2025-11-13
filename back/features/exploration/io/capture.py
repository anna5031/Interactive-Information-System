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
        self._software_resize: Optional[tuple[int, int]] = None
        self._resize_logged = False

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
        if self._software_resize is not None:
            target_w, target_h = self._software_resize
            if target_w > 0 and target_h > 0:
                h, w = frame.shape[:2]
                interpolation = (
                    cv2.INTER_AREA if target_w < w or target_h < h else cv2.INTER_LINEAR
                )
                frame = cv2.resize(
                    frame, (target_w, target_h), interpolation=interpolation
                )
        return True, frame

    def is_open(self) -> bool:
        return self._capture is not None and self._capture.isOpened()

    def _open(self) -> bool:
        for backend in self._backend_candidates():
            capture = self._create_capture(backend)
            if not capture or not capture.isOpened():
                if capture:
                    capture.release()
                continue
            self._capture = capture
            logger.info("카메라 백엔드 선택: %s", self._backend_name(backend))
            self._apply_frame_size()
            return True
        return False

    def _release(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def _backend_candidates(self) -> list[Optional[int]]:
        candidates: list[Optional[int]] = []
        v4l2 = getattr(cv2, "CAP_V4L2", None)
        if isinstance(self._source, int) and v4l2 is not None:
            candidates.append(v4l2)
        candidates.append(cv2.CAP_ANY)
        return candidates

    def _create_capture(self, backend: Optional[int]) -> Optional[cv2.VideoCapture]:
        if backend is None:
            return cv2.VideoCapture(self._source)
        return cv2.VideoCapture(self._source, backend)

    def _backend_name(self, backend: Optional[int]) -> str:
        if backend is None:
            return "default"
        for attr in dir(cv2):
            if attr.startswith("CAP_") and getattr(cv2, attr) == backend:
                return attr
        return f"backend({backend})"

    def _apply_frame_size(self) -> None:
        if self._capture is None or self._frame_size is None:
            return
        width, height = self._frame_size
        if width <= 0 or height <= 0:
            logger.warning(
                "잘못된 카메라 해상도 요청: width=%s, height=%s (무시합니다)",
                width,
                height,
            )
            return
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
        if not (set_w and set_h) or applied_width != width or applied_height != height:
            self._software_resize = (width, height)
            if not self._resize_logged:
                logger.warning(
                    "카메라가 요청 해상도(%dx%d)를 지원하지 않습니다. 프레임을 소프트웨어로 리사이즈합니다.",
                    width,
                    height,
                )
                self._resize_logged = True
        else:
            self._software_resize = None
            self._resize_logged = False
