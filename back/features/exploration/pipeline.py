from __future__ import annotations

"""Async exploration pipeline using YOLO pose detection."""

import asyncio
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncIterator, Optional, Tuple

import cv2

import numpy as np

from app.events import VisionResultEvent
from .assistance import AssistanceClassifier
from .capture import CameraCapture, center_crop, resize_frame
from .config import ExplorationConfig, load_exploration_config
from .detector import PoseDetector
from .device import select_device
from .tracking import CentroidTracker, Track
from .constants import DIRECTION_LABELS_8

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class FPSEstimator:
    target_fps: Optional[float] = None
    alpha: float = 0.1
    _last_timestamp: Optional[float] = field(init=False, default=None, repr=False)
    _fps: float = field(init=False, default=30.0, repr=False)

    def __post_init__(self) -> None:
        self._last_timestamp = None
        self._fps = self.target_fps or self._fps

    def mark(self) -> None:
        now = time.monotonic()
        if self._last_timestamp is None:
            self._last_timestamp = now
            return
        delta = max(1e-3, now - self._last_timestamp)
        instant_fps = 1.0 / delta
        self._fps = (1 - self.alpha) * self._fps + self.alpha * instant_fps
        self._last_timestamp = now

    @property
    def current(self) -> float:
        return self._fps


class ExplorationPipeline:
    """Produce VisionResultEvent values based on live YOLO pose detection."""

    def __init__(self, config: Optional[ExplorationConfig] = None, log_details: bool = False):
        self.config = config or load_exploration_config()
        self._device = select_device(self.config.device)
        self._detector = PoseDetector(self.config.model, device=self._device)
        self._tracker = CentroidTracker(self.config.tracking)
        self._assistance = AssistanceClassifier(self.config.assistance)
        self._display_enabled = self.config.debug_display
        self._window_created = False
        self._log_details = log_details
        self._active_event = asyncio.Event()
        self._active_event.set()
        self._reference_image_path = self.config.camera.reference_image_path
        self._reference_resolution = self._load_reference_resolution(self._reference_image_path)
        self._resolution_logged = False

    @property
    def device(self) -> str:
        return self._device

    def suspend(self) -> None:
        if self._active_event.is_set():
            logger.info("Exploration pipeline suspended (QA active).")
        self._active_event.clear()

    def resume(self) -> None:
        if not self._active_event.is_set():
            logger.info("Exploration pipeline resumed.")
            self._active_event.set()

    async def stream(self) -> AsyncIterator[VisionResultEvent]:
        loop = asyncio.get_running_loop()
        capture = CameraCapture(self.config.camera)
        await loop.run_in_executor(None, capture.open)
        metadata = capture.metadata
        frame_width = metadata.width or (self.config.camera.frame_size[0] if self.config.camera.frame_size else 0)
        frame_height = metadata.height or (self.config.camera.frame_size[1] if self.config.camera.frame_size else 0)

        fps_estimator = FPSEstimator(target_fps=self.config.camera.target_fps)
        frame_idx = 0

        try:
            while True:
                await self._active_event.wait()
                frame = await loop.run_in_executor(None, capture.read)
                if frame is None:
                    logger.warning("Camera frame unavailable; stopping stream.")
                    break

                raw_height, raw_width = frame.shape[:2]
                if frame_width <= 0 or frame_height <= 0:
                    frame_height, frame_width = raw_height, raw_width

                self._ensure_resolution_logged(raw_width, raw_height)

                processed = resize_frame(frame, self.config.camera.frame_size)
                crop_offset = (0, 0)
                if self.config.crop_enabled:
                    processed, crop_offset = center_crop(processed, self.config.crop_ratio)

                try:
                    detections = await loop.run_in_executor(
                        None, self._detector.predict, processed
                    )
                except Exception as exc:
                    logger.exception("Pose detection failed: %s", exc)
                    await asyncio.sleep(0.1)
                    continue
                fps_estimator.mark()
                fps = max(1.0, fps_estimator.current)
                assignments = self._tracker.step(detections, frame_idx, fps)
                frame_idx += 1

                assignments_tuple = tuple(assignments)
                if self._display_enabled:
                    self._show_debug_frame(processed, assignments_tuple)

                decision = self._assistance.evaluate(assignments_tuple)
                event = self._build_event(
                    decision=decision,
                    frame_timestamp=time.time(),
                    frame_width=frame_width,
                    frame_height=frame_height,
                    crop_offset=crop_offset,
                )

                yield event
        except asyncio.CancelledError:
            logger.info("Exploration stream cancelled.")
            raise
        finally:
            await loop.run_in_executor(None, capture.close)
            if self._display_enabled and self._window_created:
                cv2.destroyWindow(self.config.debug_window_name)

    @staticmethod
    def _load_reference_resolution(path: Optional[Path]) -> Optional[Tuple[int, int]]:
        if not path:
            return None
        image = cv2.imread(str(path))
        if image is None:
            logger.warning("Reference image not found or unreadable: %s", path)
            return None
        height, width = image.shape[:2]
        return width, height

    def _ensure_resolution_logged(self, frame_width: int, frame_height: int) -> None:
        if self._resolution_logged:
            return

        if self._reference_resolution:
            ref_width, ref_height = self._reference_resolution
            matches = frame_width == ref_width and frame_height == ref_height
            logger.info(
                "Exploration camera frame: %dx%d. Reference image (%s): %dx%d. Resolution match: %s",
                frame_width,
                frame_height,
                self._reference_image_path if self._reference_image_path else "n/a",
                ref_width,
                ref_height,
                "same" if matches else "different",
            )
        else:
            if self._reference_image_path:
                logger.info(
                    "Exploration camera frame: %dx%d. Reference image %s unavailable; unable to compare.",
                    frame_width,
                    frame_height,
                    self._reference_image_path,
                )
            else:
                logger.info(
                    "Exploration camera frame: %dx%d. No reference image configured.",
                    frame_width,
                    frame_height,
                )

        self._resolution_logged = True

    def _build_event(
        self,
        decision,
        frame_timestamp: float,
        frame_width: int,
        frame_height: int,
        crop_offset: tuple[int, int],
    ) -> VisionResultEvent:
        has_target = decision.needs_assistance and decision.detection is not None
        target_position = None
        confidence = 0.0
        gaze_vector = None
        head_position = None
        foot_position = None
        target_pixel = None
        head_pixel = None
        foot_pixel = None
        direction_label = _direction_label(decision.track.angle_deg if decision.track else None)

        if decision.detection is not None:
            detection = decision.detection
            width = max(1.0, float(frame_width))
            height = max(1.0, float(frame_height))

            center = np.asarray(detection["bbox_center"], dtype=float)
            head_raw = detection.get("centroid")
            head_point = (
                np.asarray(head_raw, dtype=float) if head_raw is not None else center
            )
            foot_raw = detection.get("foot_point")
            foot_point = (
                np.asarray(foot_raw, dtype=float) if foot_raw is not None else center
            )

            target_position = _normalize_point(center, crop_offset, width, height)
            head_position = _normalize_point(head_point, crop_offset, width, height)
            foot_position = _normalize_point(foot_point, crop_offset, width, height)
            target_pixel = _frame_point(center, crop_offset)
            head_pixel = _frame_point(head_point, crop_offset)
            foot_pixel = _frame_point(foot_point, crop_offset)
            confidence = float(detection.get("conf", 0.0))

            if target_position is not None:
                gaze_vector = _compute_gaze_vector(*target_position)
        else:
            target_position = None
            head_position = None
            foot_position = None
            target_pixel = None
            head_pixel = None
            foot_pixel = None
            direction_label = None
            gaze_vector = None

        if self._log_details and decision.needs_assistance:
            logger.info(
                "Assistance target | track=%s stationary=%.2fs direction=%s",
                getattr(decision.track, "track_id", None),
                decision.stationary_duration,
                direction_label or "-",
            )

        return VisionResultEvent(
            has_target=has_target,
            target_position=target_position,
            gaze_vector=gaze_vector,
            confidence=confidence,
            timestamp=frame_timestamp,
            needs_assistance=decision.needs_assistance,
            head_position=head_position,
            foot_position=foot_position,
            frame_width=int(frame_width) if frame_width else None,
            frame_height=int(frame_height) if frame_height else None,
            target_pixel=target_pixel,
            head_pixel=head_pixel,
            foot_pixel=foot_pixel,
            direction_label=direction_label,
            stationary_duration=decision.stationary_duration,
        )

    def _show_debug_frame(
        self,
        frame: np.ndarray,
        assignments: Tuple[Tuple[Track, dict], ...],
    ) -> None:
        if not self._window_created:
            cv2.namedWindow(self.config.debug_window_name, cv2.WINDOW_NORMAL)
            self._window_created = True

        annotated = frame.copy()
        active_track_id = self._assistance.active_track_id
        for track, detection in assignments:
            bbox = detection.get("bbox")
            if bbox is None:
                continue
            x1, y1, x2, y2 = [int(v) for v in np.asarray(bbox).tolist()]
            if active_track_id is not None and track.track_id == active_track_id:
                color = (0, 255, 0)
            elif track.is_stationary:
                color = (0, 0, 255)
            else:
                color = (0, 165, 255)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = f"ID {track.track_id} {float(detection.get('conf', 0.0)):.2f}"
            cv2.putText(
                annotated,
                label,
                (x1, max(0, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            speed_text = f"{track.speed:.1f}px/s"
            cv2.putText(
                annotated,
                speed_text,
                (x1, y2 + 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
                1,
                cv2.LINE_AA,
            )

        cv2.imshow(self.config.debug_window_name, annotated)
        cv2.waitKey(1)


def _compute_gaze_vector(x_norm: float, y_norm: float) -> tuple[float, float]:
    dx = 0.5 - x_norm
    dy = 0.5 - y_norm
    length = math.hypot(dx, dy)
    if length == 0:
        return (0.0, 0.0)
    return (dx / length, dy / length)


def _normalize_point(
    point: Optional[np.ndarray],
    offset: Tuple[int, int],
    width: float,
    height: float,
) -> Optional[Tuple[float, float]]:
    if point is None:
        return None

    x = (float(point[0]) + offset[0]) / width
    y = (float(point[1]) + offset[1]) / height

    x_clamped = max(0.0, min(1.0, x))
    y_clamped = max(0.0, min(1.0, y))
    return (x_clamped, y_clamped)


def _frame_point(
    point: Optional[np.ndarray],
    offset: Tuple[int, int],
) -> Optional[Tuple[float, float]]:
    if point is None:
        return None
    return (
        float(point[0]) + offset[0],
        float(point[1]) + offset[1],
    )


def _direction_label(angle_deg: Optional[float]) -> Optional[str]:
    if angle_deg is None:
        return None
    normalized = angle_deg % 360.0
    sector = int((normalized + 22.5) // 45) % len(DIRECTION_LABELS_8)
    return DIRECTION_LABELS_8[sector]
