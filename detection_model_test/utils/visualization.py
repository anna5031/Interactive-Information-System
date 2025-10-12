"""Utility functions for annotating images from detection results."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont

from inference.base import Detection


def _color_from_label(label: str) -> tuple[int, int, int]:
    """Generate a deterministic color for the given label."""
    palette = (
        (255, 99, 71),
        (65, 105, 225),
        (60, 179, 113),
        (255, 165, 0),
        (138, 43, 226),
        (220, 20, 60),
        (0, 128, 128),
        (250, 128, 114),
    )
    if not label:
        return palette[0]
    return palette[hash(label) % len(palette)]


def annotate_image(image: Image.Image, detections: Iterable[Detection]) -> Image.Image:
    """Return a copy of `image` with bounding boxes and labels drawn."""
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)

    try:
        font = ImageFont.load_default()
    except OSError:
        font = None

    for detection in detections:
        x1, y1, x2, y2 = detection.box
        color = _color_from_label(detection.label)
        draw.rectangle(((x1, y1), (x2, y2)), outline=color, width=2)

        label = f"{detection.label} {detection.confidence_display}"
        if font:
            text_size = draw.textbbox((0, 0), label, font=font)
            padding = 2
            box_width = text_size[2] - text_size[0] + padding * 2
            box_height = text_size[3] - text_size[1] + padding * 2
        else:
            box_width = len(label) * 6
            box_height = 12
            padding = 2

        text_origin = (x1, max(y1 - box_height, 0))
        background_shape = [
            (text_origin[0], text_origin[1]),
            (text_origin[0] + box_width, text_origin[1] + box_height),
        ]
        draw.rectangle(background_shape, fill=color)
        draw.text(
            (text_origin[0] + padding, text_origin[1] + padding),
            label,
            fill=(0, 0, 0),
            font=font,
        )

    return annotated

