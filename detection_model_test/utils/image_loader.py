"""Utility helpers for loading test images with caching."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import streamlit as st
from PIL import Image

from config import IMAGE_EXTENSIONS, TEST_IMAGE_DIR


@st.cache_data(show_spinner=False, ttl=None)
def list_test_images(source: Path | None = None) -> list[Path]:
    """Return a sorted list of image paths from the given directory."""
    directory = source or TEST_IMAGE_DIR
    if not directory.exists():
        return []

    def _iter_files() -> Iterable[Path]:
        for ext in IMAGE_EXTENSIONS:
            yield from directory.glob(f"*{ext}")

    images = sorted(
        {path.resolve() for path in _iter_files() if path.is_file()},
        key=lambda path: path.name.lower(),
    )
    return list(images)


@st.cache_data(show_spinner=False, ttl=None)
def load_image(image_path: Path) -> Image.Image:
    """Load and return an image in RGB format."""
    with Image.open(image_path) as img:
        if img.mode != "RGB":
            return img.convert("RGB")
        return img.copy()

