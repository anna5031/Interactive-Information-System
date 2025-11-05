from __future__ import annotations

"""
Manual test script that points the motor at a specific camera pixel.

Usage
-----
python -m tests.manual_motor_point_target --x 500 --y 500

The script converts the given pixel (camera image coordinates) to a world-space
point using the calibrated homography bundle and then issues a single update to
the real motor controller. The resulting pan/tilt command is sent over the
configured serial link so you can verify physical motion.
"""

import argparse
import asyncio
import sys
import time
from pathlib import Path
from typing import Tuple

# Ensure `back/` package root is importable when running as a module script.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import AppConfig, load_config
from app.events import VisionResultEvent
from features.homography import PixelToWorldMapper, load_calibration_bundle
from features.motor import RealMotorController


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Point the motor toward a pixel in camera space."
    )
    parser.add_argument("--x", type=float, default=500.0, help="Pixel X coordinate")
    parser.add_argument("--y", type=float, default=500.0, help="Pixel Y coordinate")
    parser.add_argument(
        "--plane-z",
        type=float,
        default=None,
        help="World Z plane to intersect (defaults to floor_z_mm from config)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=1.0,
        help="Confidence value for the synthetic vision event",
    )
    return parser.parse_args()


def _ensure_calibration_files(config: AppConfig) -> None:
    cam_calib = config.homography.files.camera_calibration_file
    cam_extr = config.homography.files.camera_extrinsics_file
    missing: list[Path] = [path for path in (cam_calib, cam_extr) if not path.exists()]
    if missing:
        pretty = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(
            f"Calibration assets missing: {pretty}. "
            "Generate them or update BACKEND_CAMERA_* env variables."
        )


async def _run_test(
    pixel: Tuple[float, float],
    *,
    confidence: float,
    plane_z_override: float | None,
) -> None:
    config = load_config()
    _ensure_calibration_files(config)

    if config.motor.backend == "stub":
        print(
            "[WARN] Motor backend is configured as 'stub'. "
            "Set BACKEND_MOTOR_BACKEND=serial (or appropriate) to drive hardware.",
            file=sys.stderr,
        )
        return

    bundle = load_calibration_bundle(
        config.homography.files.camera_calibration_file,
        config.homography.files.camera_extrinsics_file,
    )
    mapper = PixelToWorldMapper(bundle)

    plane_z = plane_z_override if plane_z_override is not None else config.homography.floor_z_mm
    world_point = mapper.pixel_to_world(pixel, plane_z=plane_z)
    if world_point is None:
        print(
            f"[ERROR] Unable to map pixel {pixel} to world plane Z={plane_z:.2f}. "
            "Check calibration inputs.",
            file=sys.stderr,
        )
        return

    print(f"[INFO] Pixel {pixel} maps to world {world_point}")
    controller = RealMotorController(
        motor_config=config.motor,
        homography_config=config.homography,
        mapper=mapper,
    )

    vision_event = VisionResultEvent(
        has_target=True,
        target_position=None,
        gaze_vector=None,
        confidence=confidence,
        timestamp=time.time(),
        frame_width=None,
        frame_height=None,
        target_pixel=pixel,
        head_pixel=None,
        foot_pixel=pixel,
        head_position=None,
        foot_position=None,
        direction_label=None,
    )

    try:
        state = await controller.update(vision_event)
    except Exception as exc:
        print(f"[ERROR] Motor update failed: {exc}", file=sys.stderr)
        raise
    else:
        print(
            "[INFO] Motor command issued – "
            f"pan={state.pan:.2f}°, tilt={state.tilt:.2f}°, has_target={state.has_target}"
        )
        if state.world_target:
            print(f"[INFO] Controller aim world point: {state.world_target}")
    finally:
        await controller.shutdown()


if __name__ == "__main__":
    args = _parse_args()
    try:
        asyncio.run(
            _run_test(
                (args.x, args.y),
                confidence=args.confidence,
                plane_z_override=args.plane_z,
            )
        )
    except KeyboardInterrupt:
        print("\n[INFO] Test aborted by user.")
