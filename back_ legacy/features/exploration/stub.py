from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import AsyncIterator, Tuple

from app.events import VisionResultEvent


@dataclass(slots=True)
class ExplorationStubConfig:
    interval: float = 0.5
    detection_delay: float = 5.0
    tracking_duration: float = 12.0
    stable_duration: float = 3.0
    cooldown_duration: float = 6.0


class ExplorationStub:
    """탐색 단계에서 사용할 더미 Vision 결과 생성기."""

    def __init__(self, config: ExplorationStubConfig):
        self._config = config
        self._state = _ExplorationState.SCANNING
        self._state_started_at = time.monotonic()
        self._cycle_index = 0
        self._start_position: Tuple[float, float] = (0.25, 0.68)
        self._goal_position: Tuple[float, float] = (0.52, 0.45)

    async def stream(self) -> AsyncIterator[VisionResultEvent]:
        while True:
            event = self._next_event()
            yield event
            await asyncio.sleep(self._config.interval)

    def suspend(self) -> None:  # no-op for stub
        return

    def resume(self) -> None:  # no-op for stub
        return

    def _next_event(self) -> VisionResultEvent:
        now_monotonic = time.monotonic()
        self._update_state(now_monotonic)
        timestamp = time.time()

        if self._state in (_ExplorationState.SCANNING, _ExplorationState.COOLDOWN):
            return VisionResultEvent(
                has_target=False,
                target_position=None,
                gaze_vector=None,
                confidence=0.0,
                timestamp=timestamp,
            )

        elapsed = now_monotonic - self._state_started_at
        tracking_total = self._config.tracking_duration
        progress = min(elapsed / tracking_total, 1.0)

        x = _lerp(self._start_position[0], self._goal_position[0], progress)
        y = _lerp(self._start_position[1], self._goal_position[1], progress)

        jitter_phase = self._cycle_index + elapsed
        x += 0.015 * math.sin(jitter_phase * 1.1) * (1 if self._cycle_index % 2 else -1)
        y += 0.012 * math.cos(jitter_phase * 0.9)

        x, y = _clamp_tuple((x, y))

        gaze_dx = 0.5 - x
        gaze_dy = 0.45 - y
        gaze_vector = _normalize_tuple((gaze_dx, gaze_dy))

        stable_ratio = min(
            1.0,
            max(
                0.0,
                (elapsed - tracking_total)
                / max(self._config.stable_duration, 1e-6),
            ),
        )
        confidence = 0.65 + 0.3 * max(progress, stable_ratio)

        return VisionResultEvent(
            has_target=True,
            target_position=_clamp_tuple((x, y)),
            gaze_vector=gaze_vector,
            confidence=min(1.0, confidence),
            timestamp=timestamp,
        )

    def _update_state(self, now_monotonic: float) -> None:
        elapsed = now_monotonic - self._state_started_at

        if self._state == _ExplorationState.SCANNING:
            if elapsed >= self._config.detection_delay:
                self._enter_tracking(now_monotonic)
        elif self._state == _ExplorationState.TRACKING:
            total = self._config.tracking_duration + self._config.stable_duration
            if elapsed >= total:
                self._transition_to(_ExplorationState.COOLDOWN, now_monotonic)
        elif self._state == _ExplorationState.COOLDOWN:
            if elapsed >= self._config.cooldown_duration:
                self._transition_to(_ExplorationState.SCANNING, now_monotonic)

    def _enter_tracking(self, now_monotonic: float) -> None:
        self._cycle_index += 1
        if self._cycle_index % 2 == 0:
            self._start_position = (0.22, 0.66)
        else:
            self._start_position = (0.78, 0.64)
        self._goal_position = (0.52, 0.45)
        self._transition_to(_ExplorationState.TRACKING, now_monotonic)

    def _transition_to(self, state: "_ExplorationState", now_monotonic: float) -> None:
        self._state = state
        self._state_started_at = now_monotonic


def _clamp_tuple(
    values: Tuple[float, float], min_value: float = 0.0, max_value: float = 1.0
) -> Tuple[float, float]:
    return (
        max(min_value, min(max_value, values[0])),
        max(min_value, min(max_value, values[1])),
    )


def _normalize_tuple(values: Tuple[float, float]) -> Tuple[float, float]:
    length = math.hypot(values[0], values[1])
    if length == 0:
        return (0.0, 0.0)
    return (values[0] / length, values[1] / length)


def _lerp(start: float, end: float, progress: float) -> float:
    return start + (end - start) * progress


class _ExplorationState(Enum):
    SCANNING = auto()
    TRACKING = auto()
    COOLDOWN = auto()
