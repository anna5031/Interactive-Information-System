from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from features.exploration.config import SpeedMetric, TrackingConfig
from features.exploration.core.tracking import Track


def _make_track(metric: SpeedMetric) -> Track:
    config = TrackingConfig(
        speed_metric=metric,
        stationary_speed_threshold=5.0,
        stationary_duration_seconds=0.5,
        angle_speed_threshold=5.0,
    )
    return Track(track_id=1, position=np.zeros(2), last_frame=0, config=config)


def test_pixel_metric_stationary_detection() -> None:
    track = _make_track(SpeedMetric.PIXEL)
    fps = 30.0
    track.update_pixel_motion([0.0, 0.0], 0, fps)

    for frame in range(1, 20):
        track.update_pixel_motion([0.0, 0.0], frame, fps)

    assert track.is_stationary
    assert track.stationary_time >= 0.5

    track.update_pixel_motion([50.0, 0.0], 21, fps)
    assert track.stationary_time == pytest.approx(0.0)
    assert not track.is_stationary


def test_world_metric_overrides_pixel_motion() -> None:
    track = _make_track(SpeedMetric.WORLD)
    fps = 30.0

    track.update_pixel_motion([0.0, 0.0], 0, fps)
    track.update_world_motion(np.array([0.0, 0.0, 0.0]), 0, fps)

    for frame in range(1, 20):
        track.update_world_motion(np.array([0.0, 0.0, 0.0]), frame, fps)

    assert track.is_stationary

    track.update_world_motion(np.array([500.0, 0.0, 0.0]), 40, fps)
    assert track.speed > track.config.stationary_speed_threshold
    assert not track.is_stationary

    previous_speed = track.speed
    track.update_pixel_motion([1000.0, 0.0], 41, fps)
    assert track.speed == pytest.approx(previous_speed)
