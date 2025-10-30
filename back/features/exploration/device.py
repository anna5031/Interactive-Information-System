from __future__ import annotations

"""Inference device selection helpers."""

import logging
from typing import Iterable, Optional

import torch

from .config import DeviceConfig

logger = logging.getLogger(__name__)


def _mps_available() -> bool:
    return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()


def _available_devices() -> list[str]:
    devices: list[str] = []
    if torch.cuda.is_available():
        devices.append("cuda:0")
    if _mps_available():
        devices.append("mps")
    devices.append("cpu")
    return devices


def _find_preferred_device(
    preferences: Iterable[str], available: Iterable[str]
) -> Optional[str]:
    available_set = list(available)
    for preferred in preferences:
        normalized = preferred.strip()
        if not normalized:
            continue
        if normalized in available_set:
            return normalized
        if normalized.startswith("cuda"):
            for device in available_set:
                if device.startswith("cuda"):
                    return device
    return None


def select_device(config: DeviceConfig) -> str:
    """Select the best available inference device."""
    if config.force_device:
        logger.info("Using forced device: %s", config.force_device)
        return config.force_device

    available = _available_devices()
    chosen = _find_preferred_device(config.device_preference, available)
    if chosen:
        logger.info("Selected preferred device: %s", chosen)
        return chosen

    fallback = available[0]
    logger.info("Falling back to device: %s", fallback)
    return fallback
