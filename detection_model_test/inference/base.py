"""Shared dataclasses and helpers for inference backends."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import torch
from PIL import Image

from config import DEVICE_OPTIONS, format_confidence


@dataclass(frozen=True)
class Detection:
    """A normalized detection output."""

    label: str
    confidence: float
    box: tuple[float, float, float, float]

    @property
    def confidence_display(self) -> str:
        return format_confidence(self.confidence)


@dataclass(frozen=True)
class InferenceOutput:
    """Model predictions for an image."""

    detections: list[Detection]


class InferenceBackend(Protocol):
    """Simple protocol for inference backends."""

    def predict_batch(self, images: Sequence[Image.Image], confidence: float) -> list[InferenceOutput]:
        """Run inference on a batch of images."""


def resolve_device(choice: str) -> str:
    """Resolve a preferred device string based on runtime availability."""
    choice = choice.lower()
    if choice not in DEVICE_OPTIONS:
        return "cpu"

    if choice == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():  # type: ignore[attr-defined]
            return "mps"
        return "cpu"
    if choice == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if choice == "mps":
        return "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"
    return "cpu"


def describe_runtime(device: str) -> str:
    """Return a human-friendly description for the active device."""
    normalized = device.lower()
    if normalized.startswith("cuda") and torch.cuda.is_available():
        index = 0
        if ":" in normalized:
            try:
                index = int(normalized.split(":")[1])
            except ValueError:
                index = 0
        name = torch.cuda.get_device_name(index)
        return f"CUDA · {name}"
    if normalized == "mps":
        return "MPS · Apple Silicon"
    return "CPU"
