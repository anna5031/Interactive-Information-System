"""Central configuration for the Streamlit inference app."""
from __future__ import annotations

from pathlib import Path

# Base paths -----------------------------------------------------------------
BASE_DIR: Path = Path(__file__).resolve().parent
MODEL_ROOT: Path = BASE_DIR / "model"
YOLO_MODEL_DIR: Path = MODEL_ROOT / "yolo"
RFDETR_MODEL_DIR: Path = MODEL_ROOT / "rf-detr"
TEST_IMAGE_DIR: Path = BASE_DIR / "test_image"

# File patterns --------------------------------------------------------------
IMAGE_EXTENSIONS: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
YOLO_EXTENSIONS: tuple[str, ...] = (".pt",)
RFDETR_EXTENSIONS: tuple[str, ...] = (".pth",)

# UI defaults ----------------------------------------------------------------
DEFAULT_CONFIDENCE: float = 0.25
CONFIDENCE_MIN: float = 0.0
CONFIDENCE_MAX: float = 1.0
CONFIDENCE_STEP: float = 0.01

DEFAULT_IOU: float = 0.7
IOU_MIN: float = 0.1
IOU_MAX: float = 0.95
IOU_STEP: float = 0.05

DEVICE_OPTIONS: tuple[str, ...] = ("auto", "cuda", "mps", "cpu")

# Caching keys ---------------------------------------------------------------
MODEL_CACHE_VERSION: str = "2024-10-12"
PREDICTION_CACHE_VERSION: str = "2024-10-12"


def format_confidence(value: float) -> str:
    """Format confidence scores uniformly with two decimal places."""
    return f"{value:.2f}"
