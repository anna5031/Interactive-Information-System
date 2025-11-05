from __future__ import annotations

import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import load_config
from features.homography import load_calibration_bundle, PixelToWorldMapper


def test_pixel_to_world_outputs_millimetres() -> None:
    """
    Ensure pixel→world 변환 결과가 mm 스케일로 나오는지 확인한다.

    만약 캘리브레이션 데이터가 미터 단위라면, xy 성분이 10~20 수준으로
    머물 수 있는데 이때는 호모그래피가 심하게 틀어진다.
    """
    config = load_config()
    bundle = load_calibration_bundle(
        config.homography.files.camera_calibration_file,
        config.homography.files.camera_extrinsics_file,
    )
    mapper = PixelToWorldMapper(bundle)

    cx = float(bundle.intrinsics.matrix[0, 2])
    cy = float(bundle.intrinsics.matrix[1, 2])
    samples = [
        (cx, cy),
        (cx + 120.0, cy),
        (cx, cy + 120.0),
        (cx - 120.0, cy),
    ]

    floor_z = config.homography.floor_z_mm
    planar_magnitudes: list[float] = []
    for pixel in samples:
        world = mapper.pixel_to_world(pixel, plane_z=floor_z)
        if world is None:
            continue
        planar_magnitudes.append(math.hypot(world[0], world[1]))

    assert planar_magnitudes, "pixel_to_world 변환 실패 – 캘리브레이션 파일을 확인하세요."

    max_planar = max(planar_magnitudes)
    assert (
        max_planar >= 100.0
    ), (
        "pixel_to_world 결과 xy 성분이 100mm 미만입니다. "
        "캘리브레이션 결과가 미터 단위로 저장됐을 가능성이 높으므로 "
        "`camera_pose.py` 실행 시 단위를 mm로 맞춰 다시 추정하거나, "
        "mapper 출력에 1000배 스케일을 적용하는 보정을 추가해야 합니다."
    )


def main() -> None:
    try:
        test_pixel_to_world_outputs_millimetres()
    except AssertionError as exc:
        print(f"[FAIL] {exc}")
        raise SystemExit(1) from exc
    print("[PASS] pixel_to_world 결과가 mm 스케일로 확인되었습니다.")


if __name__ == "__main__":
    main()
