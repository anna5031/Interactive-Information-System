from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from features.exploration.config import MappingConfig
from features.exploration.geometry import load_pixel_to_world_mapper
from features.homography import HomographyCalculator, HomographyConfig, ScreenConfig
from features.motor import (
    MotorController,
    MotorSettings,
    SerialMotorDriver,
    DummyMotorDriver,
    load_motor_settings,
)


def _build_homography_calculator(settings: MotorSettings) -> HomographyCalculator:
    screen = ScreenConfig(
        ceiling_normal=tuple(v / np.linalg.norm(settings.qa_projection.ceiling_normal) for v in settings.qa_projection.ceiling_normal),  # type: ignore[arg-type]
        displacement_m=settings.qa_projection.displacement_mm / 1000.0,
        screen_width_m=settings.qa_projection.screen_width_mm / 1000.0,
        screen_height_m=(
            settings.qa_projection.screen_width_mm
            * settings.qa_projection.screen_height_ratio
            / 1000.0
        ),
        roll_deg=settings.qa_projection.roll_deg,
    )
    projector = settings.projector
    config = HomographyConfig(
        projector_resolution=(
            projector.width_px,
            projector.height_px,
        ),
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


def _parse_pixel(raw: str) -> Tuple[float, float]:
    parts = raw.strip().split()
    if len(parts) != 2:
        raise ValueError("좌표는 'x y' 형식으로 입력해주세요.")
    return float(parts[0]), float(parts[1])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="픽셀 좌표→월드→모터→호모그래피 데모 스크립트"
    )
    parser.add_argument(
        "--mapping-floor-z",
        type=float,
        default=MappingConfig().floor_z_mm,
        help="월드 좌표 계산 시 사용할 Z 평면(mm)",
    )
    parser.add_argument(
        "--motor-config",
        type=str,
        default="config/motor_settings.yaml",
        help="모터/프로젝터 설정 YAML 경로",
    )
    parser.add_argument(
        "--use-dummy-motor",
        action="store_true",
        help="실제 아두이노 대신 더미 드라이버 사용",
    )
    args = parser.parse_args()

    mapping_cfg = MappingConfig(floor_z_mm=args.mapping_floor_z)
    mapper = load_pixel_to_world_mapper(mapping_cfg)
    if mapper is None:
        raise SystemExit("PixelToWorldMapper 초기화 실패 (캘리브레이션 파일 확인).")

    motor_settings = load_motor_settings(Path(args.motor_config))
    driver = (
        DummyMotorDriver()
        if args.use_dummy_motor
        else SerialMotorDriver(
            motor_settings.serial.port,
            motor_settings.serial.baudrate,
            motor_settings.serial.timeout,
        )
    )

    controller = MotorController(settings=motor_settings, driver=driver)
    homography_calc = _build_homography_calculator(motor_settings)

    print("픽셀 좌표를 'x y' 형식으로 입력하세요. 종료하려면 q 입력.")
    try:
        while True:
            raw = input("pixel> ").strip()
            if raw.lower() in {"q", "quit", "exit"}:
                break
            try:
                px = _parse_pixel(raw)
            except ValueError as exc:
                print(f"[!] {exc}")
                continue

            world = mapper.pixel_to_world(px, plane_z=mapping_cfg.floor_z_mm)
            if world is None:
                print("[!] 월드 좌표 계산 실패")
                continue
            print(f"[PIXEL ] {px}")
            print(f"[WORLD ] x={world[0]:.1f}mm y={world[1]:.1f}mm z={world[2]:.1f}mm")

            target = (world[0], world[1], world[2])
            angles = controller.point_to(target)
            print(f"[MOTOR ] tilt={angles.tilt_deg:.2f}°, pan={angles.pan_deg:.2f}°")

            try:
                H = homography_calc.calculate(
                    pan_deg=angles.pan_deg,
                    tilt_deg=angles.tilt_deg,
                )
            except Exception as exc:
                print(f"[!] 호모그래피 계산 실패: {exc}")
                continue
            np.set_printoptions(precision=4, suppress=True)
            print("[HOMOGRAPHY]\n", H)

    finally:
        controller.shutdown()


if __name__ == "__main__":
    main()
