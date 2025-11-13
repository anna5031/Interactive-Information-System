from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from features.exploration.config import AssistanceConfig, TrackingConfig
from features.exploration.core.assistance import AssistanceClassifier
from features.exploration.core.tracking import Track


def _build_track() -> Track:
    tracking_cfg = TrackingConfig()
    track = Track(track_id=1, position=np.zeros(2), last_frame=0, config=tracking_cfg)
    track.is_stationary = True
    track.stationary_time = 2.0
    track.foot_point_world = np.array([2000.0, 2000.0], dtype=float)
    return track


def test_dummy_pass_prevents_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    config = AssistanceConfig(
        stationary_seconds_required=0.0,
        cooldown_seconds=0.0,
        approach_timeout_seconds=20.0,
        dummy_nudge_pass_enabled=True,
        dummy_nudge_pass_seconds=5.0,
    )
    classifier = AssistanceClassifier(config)
    track = _build_track()
    detection = {"foot_point": np.array([0.0, 0.0])}

    times = iter([0.0, 0.0, 6.0, 6.0])
    monkeypatch.setattr(
        "features.exploration.core.assistance.time.monotonic",
        lambda: next(times),
    )

    decision_first = classifier.evaluate(((track, detection),))
    assert decision_first.needs_assistance and decision_first.track is track

    decision_second = classifier.evaluate(((track, detection),))
    assert not decision_second.needs_assistance
    assert classifier.active_track_id is None
    assert track.assistance_dismissed
