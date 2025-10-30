from __future__ import annotations

"""Assistance decision heuristics."""

import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from .config import AssistanceConfig
from .tracking import Track


@dataclass(slots=True)
class AssistanceDecision:
    needs_assistance: bool
    track: Optional[Track] = None
    detection: Optional[Dict[str, np.ndarray]] = None
    reason: str = ""
    stationary_duration: float = 0.0


class AssistanceClassifier:
    """Determine whether any tracked person needs assistance."""

    def __init__(self, config: AssistanceConfig):
        self.config = config
        self._active_track_id: Optional[int] = None
        self._last_positive_at: float = 0.0
        self._last_stationary_duration: float = 0.0

    def evaluate(
        self, assignments: Tuple[Tuple[Track, Dict[str, np.ndarray]], ...]
    ) -> AssistanceDecision:
        now = time.monotonic()
        best_candidate: Optional[Tuple[Track, Dict[str, np.ndarray]]] = None

        for track, detection in assignments:
            if not track.is_stationary:
                continue
            if track.stationary_time < self.config.stationary_seconds_required:
                continue
            if (
                best_candidate is None
                or track.stationary_time > best_candidate[0].stationary_time
            ):
                best_candidate = (track, detection)

        if best_candidate is not None:
            track, detection = best_candidate
            self._active_track_id = track.track_id
            self._last_positive_at = now
            self._last_stationary_duration = track.stationary_time
            return AssistanceDecision(
                needs_assistance=True,
                track=track,
                detection=detection,
                reason="stationary",
                stationary_duration=track.stationary_time,
            )

        if (
            self._active_track_id is not None
            and now - self._last_positive_at < self.config.cooldown_seconds
        ):
            # Maintain positive state briefly to avoid rapid toggling.
            for track, detection in assignments:
                if track.track_id == self._active_track_id:
                    return AssistanceDecision(
                        needs_assistance=True,
                        track=track,
                        detection=detection,
                        reason="cooldown-hold",
                        stationary_duration=track.stationary_time,
                    )
            # If the tracked person temporarily disappeared, keep signaling using last data.
            return AssistanceDecision(
                needs_assistance=True,
                track=None,
                detection=None,
                reason="cooldown-hold",
                stationary_duration=self._last_stationary_duration,
            )

        # Reset if cooldown elapsed or no matches.
        if self._active_track_id is not None and now - self._last_positive_at >= self.config.cooldown_seconds:
            self._active_track_id = None
            self._last_stationary_duration = 0.0

        return AssistanceDecision(needs_assistance=False)
