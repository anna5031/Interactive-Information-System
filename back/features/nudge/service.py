from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np

from features.exploration.config import MappingConfig
from features.exploration.geometry import PixelToWorldMapper, load_pixel_to_world_mapper
from features.homography import HomographyCalculator, HomographyConfig, ScreenConfig
from features.motor import (
    DummyMotorDriver,
    MotorAngles,
    MotorController,
    MotorDriver,
    MotorSettings,
    SerialMotorDriver,
    load_motor_settings,
)

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class NudgeResult:
    pixel: Tuple[float, float]
    foot_point_mm: Tuple[float, float, float]
    target_point_mm: Tuple[float, float, float]
    motor_angles: MotorAngles
    homography: np.ndarray


class NudgeService:
    """High-level helper that wires pixel→world, motor control, and homography."""

    def __init__(
        self,
        *,
        pixel_mapper: PixelToWorldMapper,
        motor_controller: MotorController,
        homography: HomographyCalculator,
        floor_z_mm: float = 0.0,
        target_plane_normal: Sequence[float] | None = None,
        target_plane_displacement_mm: float | None = None,
    ) -> None:
        self._pixel_mapper = pixel_mapper
        self._controller = motor_controller
        self._homography = homography
        self._floor_z_mm = floor_z_mm
        normal = (
            np.asarray(tuple(target_plane_normal), dtype=float)
            if target_plane_normal is not None
            else np.array([0.0, 0.0, 1.0], dtype=float)
        )
        norm = float(np.linalg.norm(normal))
        if norm < 1e-6:
            raise ValueError("target plane 법선 벡터가 유효하지 않습니다.")
        self._plane_normal = normal / norm
        self._plane_displacement_mm = (
            float(target_plane_displacement_mm)
            if target_plane_displacement_mm is not None
            else 0.0
        )

    @classmethod
    def from_configs(
        cls,
        *,
        mapping_config: Optional[MappingConfig] = None,
        motor_settings_path: Optional[Path] = None,
        motor_settings: Optional[MotorSettings] = None,
        use_dummy_motor: bool = False,
    ) -> NudgeService:
        mapping_cfg = mapping_config or MappingConfig()
        pixel_mapper = load_pixel_to_world_mapper(mapping_cfg)
        if pixel_mapper is None:
            raise RuntimeError("PixelToWorldMapper 초기화 실패 (캘리브레이션 파일 확인).")

        settings = motor_settings or load_motor_settings(motor_settings_path)
        driver: MotorDriver
        if use_dummy_motor:
            driver = DummyMotorDriver()
        else:
            driver = SerialMotorDriver(
                settings.serial.port,
                settings.serial.baudrate,
                settings.serial.timeout,
            )
        controller = MotorController(settings=settings, driver=driver)
        homography = build_homography_calculator(settings)
        plane_normal = settings.qa_geometry.normal or settings.qa_projection.ceiling_normal
        plane_displacement = (
            settings.space_geometry.ceiling_height_mm
            if settings.space_geometry.ceiling_height_mm
            else settings.qa_geometry.displacement_mm
            if settings.qa_geometry.displacement_mm
            else settings.qa_projection.displacement_mm
        )
        return cls(
            pixel_mapper=pixel_mapper,
            motor_controller=controller,
            homography=homography,
            floor_z_mm=mapping_cfg.floor_z_mm,
            target_plane_normal=plane_normal,
            target_plane_displacement_mm=plane_displacement,
        )

    def shutdown(self) -> None:
        self._controller.shutdown()

    def pixel_to_world(
        self,
        pixel: Sequence[float],
        plane_z_mm: Optional[float] = None,
    ) -> Tuple[float, float, float]:
        plane_z = self._floor_z_mm if plane_z_mm is None else float(plane_z_mm)
        world = self._pixel_mapper.pixel_to_world(
            (float(pixel[0]), float(pixel[1])),
            plane_z=plane_z,
        )
        if world is None:
            raise RuntimeError(
                f"픽셀→월드 좌표 변환 실패 (pixel={pixel}, plane_z={plane_z})"
            )
        return world

    def point_at_pixel(
        self,
        pixel: Sequence[float],
        *,
        plane_z_mm: Optional[float] = None,
        target_plane_override_mm: Optional[float] = None,
    ) -> NudgeResult:
        px = (float(pixel[0]), float(pixel[1]))
        foot = self.pixel_to_world(px, plane_z_mm)
        target = self._lift_to_target_plane(
            foot,
            target_plane_override_mm=target_plane_override_mm,
        )
        logger.info(
            "NudgeService: pixel=%s -> foot=(%.1f, %.1f, %.1f) target=(%.1f, %.1f, %.1f)",
            px,
            foot[0],
            foot[1],
            foot[2],
            target[0],
            target[1],
            target[2],
        )
        angles = self._controller.point_to(target)
        logger.info(
            "NudgeService: motor command tilt=%.2f°, pan=%.2f°",
            angles.tilt_deg,
            angles.pan_deg,
        )
        H = self._homography.calculate(
            pan_deg=angles.pan_deg,
            tilt_deg=angles.tilt_deg,
        )
        return NudgeResult(
            pixel=px,
            foot_point_mm=foot,
            target_point_mm=target,
            motor_angles=angles,
            homography=H,
        )

    def _lift_to_target_plane(
        self,
        foot_point_mm: Sequence[float],
        *,
        target_plane_override_mm: Optional[float] = None,
    ) -> Tuple[float, float, float]:
        target_displacement = (
            float(target_plane_override_mm)
            if target_plane_override_mm is not None
            else self._plane_displacement_mm
        )
        foot = np.asarray(tuple(foot_point_mm), dtype=float)
        denom = float(np.dot(self._plane_normal, self._plane_normal))
        if denom < 1e-9:
            raise RuntimeError("target plane 정의가 잘못되었습니다.")
        current = float(np.dot(self._plane_normal, foot))
        alpha = (target_displacement - current) / denom
        target = foot + alpha * self._plane_normal
        return (float(target[0]), float(target[1]), float(target[2]))


def build_homography_calculator(settings: MotorSettings) -> HomographyCalculator:
    qa = settings.qa_projection
    projector = settings.projector

    normal = np.array(qa.ceiling_normal, dtype=float)
    norm = np.linalg.norm(normal)
    if norm < 1e-6:
        raise ValueError("ceiling_normal 벡터가 유효하지 않습니다.")
    normal_tuple = tuple((normal / norm).tolist())

    screen = ScreenConfig(
        ceiling_normal=normal_tuple,
        displacement_m=qa.displacement_mm / 1000.0,
        screen_width_m=qa.screen_width_mm / 1000.0,
        screen_height_m=(
            qa.screen_width_mm * qa.screen_height_ratio / 1000.0
        ),
        roll_deg=qa.roll_deg,
    )
    config = HomographyConfig(
        projector_resolution=(projector.width_px, projector.height_px),
        horizontal_fov_deg=projector.horizontal_fov_deg,
        beam_offset_m=tuple(value / 1000.0 for value in projector.beam_offset_mm),
        origin_height_m=projector.pan_tilt_origin_height_mm / 1000.0,
        beam_wall_displacement_m=projector.beam_wall_displacement_m,
        input_resolution=(
            projector.input_image_width_px,
            projector.input_image_height_px,
        ),
    )
    return HomographyCalculator(screen, config)
