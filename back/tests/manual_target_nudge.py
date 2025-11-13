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
from features.nudge import NudgeService


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
    parser.add_argument(
        "--target-plane-mm",
        type=float,
        default=None,
        help="발 좌표를 투영할 천장/목표 평면의 displacement(mm). 기본값은 motor_settings.qa_projection.displacement_mm",
    )
    args = parser.parse_args()

    service = NudgeService.from_configs(
        mapping_config=MappingConfig(floor_z_mm=args.mapping_floor_z),
        motor_settings_path=Path(args.motor_config),
        use_dummy_motor=args.use_dummy_motor,
    )

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

            try:
                result = service.point_at_pixel(
                    px,
                    target_plane_override_mm=args.target_plane_mm,
                )
            except Exception as exc:
                print(f"[!] 요청 실패: {exc}")
                continue

            print(f"[PIXEL ] {px}")
            foot = result.foot_point_mm
            target = result.target_point_mm
            print(
                f"[FOOT  ] x={foot[0]:.1f}mm y={foot[1]:.1f}mm z={foot[2]:.1f}mm"
            )
            print(
                f"[TARGET] x={target[0]:.1f}mm y={target[1]:.1f}mm z={target[2]:.1f}mm"
            )
            angles = result.motor_angles
            print(f"[MOTOR ] tilt={angles.tilt_deg:.2f}°, pan={angles.pan_deg:.2f}°")

            np.set_printoptions(precision=4, suppress=True)
            print("[HOMOGRAPHY]\n", result.homography)

    finally:
        service.shutdown()


if __name__ == "__main__":
    main()
