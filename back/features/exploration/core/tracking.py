from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..config import TrackingConfig


@dataclass(slots=True)
class Track:
    track_id: int
    position: np.ndarray
    last_frame: int
    config: TrackingConfig
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=float))
    smoothed_velocity: np.ndarray = field(
        default_factory=lambda: np.zeros(2, dtype=float)
    )
    speed: float = 0.0
    angle_deg: float | None = None
    missed: int = 0
    stationary_time: float = 0.0
    is_stationary: bool = False
    foot_point: Optional[np.ndarray] = None
    foot_history: Deque[np.ndarray] = field(default_factory=lambda: deque(maxlen=6))
    foot_point_world: Optional[np.ndarray] = None
    walking: bool = False
    assistance_active: bool = False
    assistance_assigned_at: Optional[float] = None
    assistance_initial_distance: Optional[float] = None
    assistance_dismissed: bool = False

    def update(self, position: Sequence[float], frame_idx: int, fps: float) -> None:
        new_position = np.asarray(position, dtype=float)
        frame_gap = max(1, frame_idx - self.last_frame)
        displacement = new_position - self.position
        fps = fps if fps > 0 else 30.0
        instant_velocity = displacement * fps / frame_gap
        alpha = max(0.0, min(1.0, self.config.velocity_smoothing))
        blended_alpha = alpha**frame_gap if frame_gap > 1 else alpha
        self.smoothed_velocity = (
            blended_alpha * self.smoothed_velocity
            + (1.0 - blended_alpha) * instant_velocity
        )
        self.velocity = instant_velocity
        self.speed = float(np.linalg.norm(self.smoothed_velocity))

        if self.speed <= self.config.stationary_speed_threshold:
            self.stationary_time += frame_gap / fps
        else:
            self.stationary_time = 0.0

        self.is_stationary = (
            self.stationary_time >= self.config.stationary_duration_seconds
        )

        effective_velocity = (
            np.zeros_like(self.smoothed_velocity)
            if self.is_stationary
            else self.smoothed_velocity
        )

        effective_speed = float(np.linalg.norm(effective_velocity))
        if effective_speed >= self.config.angle_speed_threshold:
            self.angle_deg = (
                math.degrees(math.atan2(-effective_velocity[1], effective_velocity[0]))
                + 360.0
            ) % 360.0
        else:
            self.angle_deg = None

        self.position = new_position
        self.last_frame = frame_idx
        self.missed = 0


class CentroidTracker:
    def __init__(self, config: TrackingConfig):
        self.config = config
        self.tracks: Dict[int, Track] = {}
        self.next_id = 1

    def step(
        self,
        detections: List[Dict[str, np.ndarray]],
        frame_idx: int,
        fps: float,
    ) -> List[Tuple[Track, Dict[str, np.ndarray]]]:
        if not detections:
            self._age_tracks()
            return []

        centroids = np.array([det["centroid"] for det in detections], dtype=float)

        if not self.tracks:
            assignments = []
            for det_idx, centroid in enumerate(centroids):
                track = self._make_track(centroid, frame_idx)
                assignments.append((track, detections[det_idx]))
            return assignments

        track_ids = list(self.tracks.keys())
        track_positions = np.array(
            [self.tracks[tid].position for tid in track_ids], dtype=float
        )
        dist_matrix = np.linalg.norm(
            track_positions[:, None, :] - centroids[None, :, :], axis=2
        )

        assigned_tracks: set[int] = set()
        assigned_dets: set[int] = set()
        assignments: List[Tuple[Track, Dict[str, np.ndarray]]] = []

        while dist_matrix.size > 0:
            min_idx = np.unravel_index(
                np.argmin(dist_matrix, axis=None), dist_matrix.shape
            )
            min_dist = dist_matrix[min_idx]
            if not np.isfinite(min_dist) or min_dist > self.config.distance_threshold:
                break
            track_idx, det_idx = min_idx
            track_id = track_ids[track_idx]
            track = self.tracks[track_id]
            track.update(centroids[det_idx], frame_idx, fps)
            assignments.append((track, detections[det_idx]))
            assigned_tracks.add(track_id)
            assigned_dets.add(det_idx)
            dist_matrix[track_idx, :] = np.inf
            dist_matrix[:, det_idx] = np.inf

        for tid in track_ids:
            if tid not in assigned_tracks:
                track = self.tracks[tid]
                track.missed += 1
                if track.missed > self.config.max_age:
                    del self.tracks[tid]

        for det_idx, centroid in enumerate(centroids):
            if det_idx in assigned_dets:
                continue
            track = self._make_track(centroid, frame_idx)
            assignments.append((track, detections[det_idx]))

        return assignments

    def _make_track(self, centroid: np.ndarray, frame_idx: int) -> Track:
        track = Track(
            track_id=self.next_id,
            position=np.asarray(centroid, dtype=float),
            last_frame=frame_idx,
            config=self.config,
        )
        self.tracks[self.next_id] = track
        self.next_id += 1
        return track

    def _age_tracks(self) -> None:
        to_delete = []
        for track_id, track in self.tracks.items():
            track.missed += 1
            if track.missed > self.config.max_age:
                to_delete.append(track_id)
        for track_id in to_delete:
            del self.tracks[track_id]
