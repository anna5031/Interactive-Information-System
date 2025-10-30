from __future__ import annotations

"""Default parameters for homography calculation."""

from pathlib import Path

# Calibration files
CALIBRATION_DIR = Path(__file__).resolve().parent / "calibration"
CAMERA_CALIBRATION_FILE = CALIBRATION_DIR / "camera_calib.npz"
CAMERA_EXTRINSICS_FILE = CALIBRATION_DIR / "camera_extrinsics.npz"

# Projection surface ('floor', 'wall', etc.)
TARGET_PLANE = "floor"
FLOOR_Z_MM = 0.0

# Projector pose (world coordinates, millimetres)
PROJECTOR_POSITION_MM = (0.0, 0.0, 1200.0)

# Projected footprint (millimetres)
FOOTPRINT_WIDTH_MM = 1000.0
FOOTPRINT_HEIGHT_MM = 600.0

# Forward offset from foot position (millimetres)
FOOT_OFFSET_MM = 350.0

# Homography smoothing
SMOOTHING_ALPHA = 0.25
