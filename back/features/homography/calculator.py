from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import cv2  # type: ignore
import numpy as np


@dataclass(slots=True)
class ScreenConfig:
    ceiling_normal: Tuple[float, float, float]
    displacement_m: float
    screen_width_m: float
    screen_height_m: float
    roll_deg: float = 0.0


@dataclass(slots=True)
class HomographyConfig:
    projector_resolution: Tuple[int, int]
    horizontal_fov_deg: float
    beam_offset_m: Tuple[float, float, float]
    origin_height_m: float
    beam_wall_displacement_m: float
    input_resolution: Tuple[int, int]


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm < 1e-9:
        return vec
    return vec / norm


class HomographyCalculator:
    def __init__(self, screen: ScreenConfig, config: HomographyConfig) -> None:
        self.screen = screen
        self.config = config

    def calculate(self, pan_deg: float, tilt_deg: float) -> np.ndarray:
        pan = math.radians(pan_deg)
        tilt = math.radians(tilt_deg)

        nudge_vec = np.array(
            [
                math.cos(pan) * math.cos(tilt),
                math.sin(pan),
                math.cos(pan) * math.sin(tilt),
            ],
            dtype=np.float64,
        )

        origin = np.array([0.0, 0.0, self.config.origin_height_m], dtype=np.float64)
        beam_offset = np.array(self.config.beam_offset_m, dtype=np.float64)
        beam_position = origin + self._rotation_matrix(pan, tilt) @ beam_offset

        K = self._intrinsics()
        R_wc, axes = self._camera_axes(nudge_vec)
        wall_center = self._wall_intersection(axes[2])

        u, v, _ = self._plane_basis()
        half_w = self.screen.screen_width_m / 2.0
        half_h = self.screen.screen_height_m / 2.0

        corners = np.stack(
            [
                wall_center + (-half_w) * u + half_h * v,
                wall_center + half_w * u + half_h * v,
                wall_center + half_w * u - half_h * v,
                wall_center + (-half_w) * u - half_h * v,
            ],
            axis=0,
        )

        dst_pts = self._project_points(K, R_wc, corners)
        src_width, src_height = self.config.input_resolution
        src_pts = np.array(
            [[0, 0], [src_width - 1, 0], [src_width - 1, src_height - 1], [0, src_height - 1]],
            dtype=np.float64,
        )
        H, _ = cv2.findHomography(src_pts, dst_pts, method=0)
        if H is None:
            raise RuntimeError("Homography computation failed.")
        return H

    def _rotation_matrix(self, pan: float, tilt: float) -> np.ndarray:
        Rz = np.array(
            [
                [math.cos(pan), -math.sin(pan), 0.0],
                [math.sin(pan), math.cos(pan), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        Ry = np.array(
            [
                [math.cos(-tilt), 0.0, math.sin(-tilt)],
                [0.0, 1.0, 0.0],
                [-math.sin(-tilt), 0.0, math.cos(-tilt)],
            ],
            dtype=np.float64,
        )
        return Rz @ Ry

    def _intrinsics(self) -> np.ndarray:
        width, height = self.config.projector_resolution
        h_fov = math.radians(self.config.horizontal_fov_deg)
        fx = (width / 2.0) / math.tan(h_fov / 2.0)
        fy = fx
        cx, cy = width / 2.0, height / 2.0
        return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)

    def _camera_axes(self, forward: np.ndarray):
        z = _normalize(forward)
        down = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        if abs(np.dot(z, down)) > 0.99:
            down = np.array([0.0, -1.0, 0.0], dtype=np.float64)
        x = _normalize(np.cross(down, z))
        y = _normalize(np.cross(z, x))
        R_wc = np.vstack([x, y, z])
        return R_wc, (x, y, z)

    def _wall_intersection(self, forward: np.ndarray) -> np.ndarray:
        n = _normalize(np.array(self.screen.ceiling_normal, dtype=np.float64))
        point = n * self.screen.displacement_m
        direction = _normalize(forward)
        denom = np.dot(n, direction)
        if abs(denom) < 1e-9:
            raise RuntimeError("Beam vector nearly parallel to plane.")
        t = np.dot(n, point) / denom
        if t <= 0:
            raise RuntimeError("Beam vector does not intersect the plane.")
        return direction * t

    def _plane_basis(self):
        n = _normalize(np.array(self.screen.ceiling_normal, dtype=np.float64))
        world_y = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        u = world_y - np.dot(world_y, n) * n
        if np.linalg.norm(u) < 1e-9:
            helper = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            u = helper - np.dot(helper, n) * n
        u = _normalize(u)
        if self.screen.roll_deg:
            angle = math.radians(self.screen.roll_deg)
            u = self._rodrigues(u, n, angle)
        v = _normalize(np.cross(n, u))
        return u, v, n

    @staticmethod
    def _rodrigues(vec: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
        axis = _normalize(axis)
        c, s = math.cos(angle), math.sin(angle)
        return vec * c + np.cross(axis, vec) * s + axis * np.dot(axis, vec) * (1 - c)

    @staticmethod
    def _project_points(K: np.ndarray, R_wc: np.ndarray, pts: np.ndarray) -> np.ndarray:
        Xc = (R_wc @ pts.T).T
        projection = (K @ Xc.T).T
        return projection[:, :2] / projection[:, 2:3]
