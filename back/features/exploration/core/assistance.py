from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from ..config import AssistanceConfig
from .tracking import Track

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class AssistanceDecision:
    needs_assistance: bool
    track: Optional[Track] = None
    detection: Optional[Dict[str, np.ndarray]] = None
    reason: str = ""
    stationary_duration: float = 0.0
    transition_to_qa: bool = False


class AssistanceClassifier:
    def __init__(self, config: AssistanceConfig):
        self.config = config
        self._active_track_id: Optional[int] = None
        self._last_positive_at: float = 0.0
        self._last_stationary_duration: float = 0.0
        self._dismissed_tracks: set[int] = set()

    def evaluate(
        self, assignments: Tuple[Tuple[Track, Dict[str, np.ndarray]], ...]
    ) -> AssistanceDecision:
        now = time.monotonic()
        best_candidate: Optional[Tuple[Track, Dict[str, np.ndarray]]] = None
        active_track: Optional[Tuple[Track, Dict[str, np.ndarray]]] = None

        for track, detection in assignments:
            if track.track_id in self._dismissed_tracks:
                continue
            if (
                self._active_track_id is not None
                and track.track_id == self._active_track_id
            ):
                active_track = (track, detection)
            if not track.is_stationary:
                continue
            if track.stationary_time < self.config.stationary_seconds_required:
                continue
            if (
                best_candidate is None
                or track.stationary_time > best_candidate[0].stationary_time
            ):
                best_candidate = (track, detection)

        if self._active_track_id is not None:
            if active_track is not None:
                track, detection = active_track
                dismiss_reason = self._should_dismiss(track)
                if dismiss_reason:
                    self._dismiss_track(track, reason=dismiss_reason)
                    if dismiss_reason == "dummy-pass":
                        return AssistanceDecision(
                            needs_assistance=False,
                            track=None,
                            detection=None,
                            reason="dummy-pass",
                            transition_to_qa=True,
                        )
                else:
                    self._last_positive_at = now
                    self._last_stationary_duration = track.stationary_time
                    return AssistanceDecision(
                        needs_assistance=True,
                        track=track,
                        detection=detection,
                        reason="tracking",
                        stationary_duration=track.stationary_time,
                    )
            if now - self._last_positive_at < self.config.cooldown_seconds:
                return AssistanceDecision(
                    needs_assistance=True,
                    track=None,
                    detection=None,
                    reason="cooldown-hold",
                    stationary_duration=self._last_stationary_duration,
                )
            self._active_track_id = None
            self._last_stationary_duration = 0.0

        if best_candidate is not None:
            track, detection = best_candidate
            if track.track_id in self._dismissed_tracks:
                return AssistanceDecision(needs_assistance=False)
            self._assign_track(track)
            return AssistanceDecision(
                needs_assistance=True,
                track=track,
                detection=detection,
                reason="stationary",
                stationary_duration=track.stationary_time,
            )

        return AssistanceDecision(needs_assistance=False)

    def _assign_track(self, track: Track) -> None:
        self._active_track_id = track.track_id
        self._last_positive_at = time.monotonic()
        self._last_stationary_duration = track.stationary_time
        track.assistance_assigned_at = self._last_positive_at
        track.assistance_initial_distance = self._distance_to_origin(track)
        track.assistance_dismissed = False

    def _dismiss_track(self, track: Track, reason: str = "") -> None:
        self._dismissed_tracks.add(track.track_id)
        self._active_track_id = None
        self._last_stationary_duration = 0.0
        self._last_positive_at = 0.0
        track.assistance_assigned_at = None
        track.assistance_initial_distance = None
        track.assistance_dismissed = True
        if reason:
            logger.info(
                "타겟 해제: track=%d reason=%s", track.track_id, reason
            )

    def _should_dismiss(self, track: Track) -> Optional[str]:
        assigned_at = track.assistance_assigned_at
        if assigned_at is None:
            return None
        initial_distance = track.assistance_initial_distance
        now = time.monotonic()
        elapsed = now - assigned_at
        distance = self._distance_to_origin(track)

        if (
            self.config.dummy_nudge_pass_enabled
            and elapsed >= self.config.dummy_nudge_pass_seconds
        ):
            logger.info(
                "더미 넛지 패스 충족: track=%d elapsed=%.1fs",
                track.track_id,
                elapsed,
            )
            return "dummy-pass"

        if distance is None or initial_distance is None:
            return (
                "timeout"
                if elapsed >= self.config.approach_timeout_seconds
                else None
            )

        if elapsed >= self.config.approach_timeout_seconds:
            return "timeout"

        if initial_distance - distance >= self.config.approach_tolerance_mm:
            track.assistance_assigned_at = now
            track.assistance_initial_distance = distance
            return None

        if distance - initial_distance >= self.config.retreat_threshold_mm:
            return "retreat"

        return None

    def _distance_to_origin(self, track: Track) -> Optional[float]:
        if track.foot_point_world is None:
            return None
        x = float(track.foot_point_world[0]) - self.config.target_origin_x_mm
        y = float(track.foot_point_world[1]) - self.config.target_origin_y_mm
        return float(np.hypot(x, y))

    @property
    def active_track_id(self) -> Optional[int]:
        return self._active_track_id
