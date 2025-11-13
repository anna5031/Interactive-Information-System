from __future__ import annotations

"""Overlay rendering utilities for the exploration pipeline."""

from dataclasses import dataclass
from typing import Iterable, Tuple, Dict

import cv2  # type: ignore
import numpy as np

from ..config import AssistanceConfig
from ..core import Track


Assignments = Iterable[Tuple[Track, Dict[str, np.ndarray]]]


@dataclass(slots=True)
class OverlayRenderer:
    assistance_config: AssistanceConfig

    def draw(self, frame: np.ndarray, assignments: Assignments) -> None:
        stationary_req = self.assistance_config.stationary_seconds_required

        for track, detection in assignments:
            centroid = detection.get("centroid")
            if centroid is None:
                continue

            cx, cy = int(centroid[0]), int(centroid[1])
            cv2.circle(frame, (cx, cy), 15, (0, 0, 255), -1)
            cv2.putText(
                frame,
                f"ID {track.track_id}",
                (cx + 6, cy - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            if track.is_stationary:
                text = f"{track.stationary_time:.1f}s"
                cv2.putText(
                    frame,
                    text,
                    (cx + 6, cy + 12),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 200, 0),
                    1,
                    cv2.LINE_AA,
                )

            foot = detection.get("foot_point")
            if foot is not None:
                fx, fy = int(foot[0]), int(foot[1])
                cv2.circle(frame, (fx, fy), 18, (0, 0, 255), -1)

            bbox = detection.get("bbox") if detection is not None else None
            if bbox is None:
                continue

            x1, y1, x2, y2 = map(int, bbox)
            if track.assistance_active:
                color = (0, 0, 255)
                label = "TARGET"
            elif track.stationary_time >= stationary_req:
                color = (0, 255, 0)
                label = "STILL"
            else:
                color = (255, 0, 0)
                label = None

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            if label:
                cv2.putText(
                    frame,
                    label,
                    (x1, max(y1 - 8, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                    cv2.LINE_AA,
                )
