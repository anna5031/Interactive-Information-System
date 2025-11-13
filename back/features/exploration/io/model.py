from __future__ import annotations

"""YOLO pose 모델 래퍼."""

import logging
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
from ultralytics import YOLO  # type: ignore

from ..config import DetectionConfig

logger = logging.getLogger(__name__)


class PoseModel:
    def __init__(self, model_path: Path, device: str) -> None:
        self._path = model_path
        self._device = device
        self._model = self._load_model()

    def predict(
        self,
        frame: np.ndarray,
        config: DetectionConfig,
    ) -> Optional[Iterable]:
        try:
            return self._model.predict(
                frame,
                verbose=False,
                device=self._device,
                conf=config.confidence_threshold,
                iou=config.iou_threshold,
            )
        except Exception as exc:
            logger.exception("YOLO 포즈 추론 실패: %s", exc)
            return None

    def _load_model(self) -> YOLO:  # type: ignore[return-value]
        logger.info("YOLO 포즈 모델 로딩: %s", self._path)
        try:
            return YOLO(str(self._path))
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(f"YOLO 포즈 모델을 불러오지 못했습니다: {self._path}") from exc
