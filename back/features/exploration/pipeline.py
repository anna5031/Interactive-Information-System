from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Optional

import numpy as np

import cv2  # type: ignore
import torch

from ...devices.config import load_device_preferences

from .config import (
    AssistanceConfig,
    DetectionConfig,
    MappingConfig,
    TrackingConfig,
    SpeedMetric,
)
from .core import (
    AssistanceClassifier,
    FootPointEstimator,
    FootPointEstimatorConfig,
    FPSEstimator,
    CentroidTracker,
)
from .detection import build_detections
from .geometry import PixelToWorldMapper, load_pixel_to_world_mapper
from .io import CameraCapture, PoseModel
from .visual import OverlayRenderer

logger = logging.getLogger(__name__)


CameraSource = Optional[int | str]


class ExplorationPipeline:
    """Camera capture + YOLO pose + assistance detection."""

    def __init__(
        self,
        *,
        camera_source: CameraSource = None,
        camera_frame_size: Optional[tuple[int, int]] = None,
        show_overlay: bool = False,
        max_frames: Optional[int] = None,
        detection_config: Optional[DetectionConfig] = None,
        tracking_config: Optional[TrackingConfig] = None,
        foot_config: Optional[FootPointEstimatorConfig] = None,
        mapping_config: Optional[MappingConfig] = None,
        assistance_config: Optional[AssistanceConfig] = None,
    ) -> None:
        preferences = load_device_preferences()
        resolved_source = (
            camera_source if camera_source is not None else preferences.camera_source
        )
        self._camera_source: CameraSource = (
            resolved_source if resolved_source is not None else 0
        )
        resolved_frame = (
            camera_frame_size
            if camera_frame_size is not None
            else preferences.camera_frame_size
        )
        self._camera_frame_size: Optional[tuple[int, int]] = resolved_frame

        self._model_path = Path("yolo11n-pose.pt")
        self._show_overlay = show_overlay
        self._window_name = "Exploration - YOLO Pose"
        self._window_created = False
        self._max_frames = max_frames

        self._device = self._select_device()
        self._pose_model = PoseModel(self._model_path, self._device)

        self._tracking_config = tracking_config or TrackingConfig()
        self._tracking = CentroidTracker(self._tracking_config)
        self._assistance = AssistanceClassifier(assistance_config or AssistanceConfig())
        self._detection_config = detection_config or DetectionConfig()
        self._requires_world_speed = (
            self._tracking_config.speed_metric == SpeedMetric.WORLD
        )
        self._mapping_config = mapping_config or MappingConfig()
        if self._requires_world_speed and not self._mapping_config.enabled:
            raise RuntimeError(
                "월드 속도 기준을 사용하려면 픽셀→월드 매핑을 활성화해야 합니다."
            )
        self._pixel_mapper: Optional[PixelToWorldMapper] = load_pixel_to_world_mapper(
            self._mapping_config
        )
        if self._requires_world_speed and self._pixel_mapper is None:
            raise RuntimeError(
                "월드 속도 기준에 필요한 픽셀→월드 매퍼를 초기화하지 못했습니다."
            )
        self._foot_estimator = FootPointEstimator(
            foot_config,
            pixel_mapper=self._pixel_mapper,
            floor_z=self._mapping_config.floor_z_mm,
        )
        self._overlay_renderer = (
            OverlayRenderer(self._assistance.config) if self._show_overlay else None
        )
        self._camera = CameraCapture(
            self._camera_source, frame_size=self._camera_frame_size
        )

    async def run(self) -> None:
        logger.info("탐색 파이프라인 시작")
        await self._camera.start()
        try:
            await self._stream_camera_frames()
        finally:
            await self._camera.stop()
            self._destroy_overlay_window()
        logger.info("탐색 파이프라인 종료")
        if getattr(self, "_last_assistance_decision", None) and self._last_assistance_decision.transition_to_qa:
            logger.info("탐색 파이프라인에서 QA 전환 이벤트 발생")

    def _select_device(self) -> str:
        if torch.cuda.is_available():
            return "cuda:0"
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():  # type: ignore[attr-defined]
            return "mps"
        logger.warning("사용 가능한 GPU가 없어 CPU로 YOLO 추론을 수행합니다.")
        return "cpu"

    async def _stream_camera_frames(self) -> None:
        if self._show_overlay:
            self._detection_loop()
        else:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self._detection_loop)

    def _detection_loop(self) -> None:
        if not self._camera.is_open():
            logger.error("카메라 캡처가 초기화되지 않았습니다.")
            return

        frame_count = 0
        fps_estimator = FPSEstimator()

        while self._camera.is_open():
            ret, frame = self._camera.read()
            if not ret or frame is None:
                logger.warning("카메라 프레임을 읽지 못했습니다. 스트림을 종료합니다.")
                break

            results = self._pose_model.predict(frame, self._detection_config)
            if results is None:
                break

            result_list = list(results)
            if not result_list:
                continue
            result = result_list[0]

            detections = build_detections(result, self._detection_config)

            fps_estimator.mark()
            fps = max(1.0, fps_estimator.current)
            assignments = self._tracking.step(detections, frame_count, fps)
            frame_count += 1
            current_frame_idx = frame_count - 1

            for track, detection in assignments:
                foot_point = self._foot_estimator.update_track(track, detection)
                detection["foot_point"] = foot_point
                detection["walking"] = track.walking
                world_point = None
                if (
                    foot_point is not None
                    and self._pixel_mapper is not None
                ):
                    world_point = self._foot_estimator.map_to_world(foot_point)
                detection["foot_point_world"] = world_point
                track.foot_point_world = world_point
                if world_point is not None:
                    track.update_world_motion(world_point, current_frame_idx, fps)
                elif self._requires_world_speed:
                    raise RuntimeError(
                        f"월드 속도 기준이지만 track={track.track_id} 의 발 좌표를 월드 좌표로 변환하지 못했습니다."
                    )

            decision = self._assistance.evaluate(tuple(assignments))
            self._last_assistance_decision = decision
            active_id = (
                self._assistance.active_track_id if decision.needs_assistance else None
            )
            for track_obj in self._tracking.tracks.values():
                track_obj.assistance_active = (
                    active_id is not None and track_obj.track_id == active_id
                )
            if decision.needs_assistance and decision.track is not None:
                logger.info(
                    "도움이 필요한 사람 감지: track=%d stationary=%.2fs reason=%s",
                    decision.track.track_id,
                    decision.stationary_duration,
                    decision.reason,
                )
            if decision.needs_assistance and decision.track is not None:
                world_point = decision.track.foot_point_world
                if world_point is not None:
                    logger.info(
                        "타겟 월드 좌표: (%.1f, %.1f, %.1f)",
                        float(world_point[0]),
                        float(world_point[1]),
                        float(world_point[2]),
                    )
            if decision.transition_to_qa:
                logger.info("더미 넛지 패스 완료, QA 단계로 전환 요청")
                break

            annotated = result.plot(boxes=False) if self._show_overlay else None
            if self._show_overlay and annotated is not None:
                assert self._overlay_renderer is not None
                self._overlay_renderer.draw(annotated, assignments)
                try:
                    self._window_created = True
                    cv2.imshow(self._window_name, annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        logger.info("사용자 입력(q)으로 탐색 스트림을 종료합니다.")
                        break
                except cv2.error as exc:
                    logger.exception(
                        "디버그 오버레이 표시 중 OpenCV 오류 발생: %s", exc
                    )
                    break

            if self._max_frames is not None and frame_count >= self._max_frames:
                break

            # 로그에 현재 추정된 발 좌표를 남긴다.
            for track, detection in assignments:
                foot = detection.get("foot_point")
                if foot is None:
                    continue
                logger.info(
                    "track=%d foot_point=(%.1f, %.1f) walking=%s",
                    track.track_id,
                    float(foot[0]),
                    float(foot[1]),
                    "Y" if track.walking else "N",
                )

    def _destroy_overlay_window(self) -> None:
        if self._show_overlay and self._window_created:
            cv2.destroyWindow(self._window_name)
            self._window_created = False
