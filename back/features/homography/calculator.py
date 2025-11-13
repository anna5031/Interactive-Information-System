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
    """Port of nudge_test/motor/H_calculator.py."""

    def __init__(self, screen: ScreenConfig, config: HomographyConfig) -> None:
        self.screen = screen
        self.config = config
        self._intrinsics = self._build_intrinsics()

    def calculate(self, pan_deg: float, tilt_deg: float) -> np.ndarray:
        z_rad = math.radians(pan_deg)
        y_rad = -math.radians(tilt_deg)

        R = self._beam_rotation_matrix(z_rad, y_rad)
        beam_origin = np.array(
            [0.0, 0.0, self.config.origin_height_m], dtype=np.float64
        )
        beam_offset = np.array(self.config.beam_offset_m, dtype=np.float64)
        beam_position = beam_origin + R @ beam_offset

        nudge_vector = R @ np.array([1.0, 0.0, 0.0], dtype=np.float64)

        screen_normal = np.array(self.screen.ceiling_normal, dtype=np.float64)
        nudge_center = self._compute_plane_intersection(
            screen_normal,
            self.screen.displacement_m,
            nudge_vector,
            beam_position,
        )

        u, v = self._plane_basis(screen_normal, self.screen.roll_deg)
        half_w = self.screen.screen_width_m * 0.5
        half_h = self.screen.screen_height_m * 0.5

        corners = np.stack(
            [
                nudge_center + (-half_w) * u + (+half_h) * v,
                nudge_center + (+half_w) * u + (+half_h) * v,
                nudge_center + (+half_w) * u + (-half_h) * v,
                nudge_center + (-half_w) * u + (-half_h) * v,
            ],
            axis=0,
        )

        R_wc, _ = self._camera_axes_from_rotation(R)
        dst_pts = self._project_points_world(self._intrinsics, R_wc, corners, beam_position)

        src_w, src_h = self.config.input_resolution
        src_pts = np.array(
            [
                [0.0, 0.0],
                [src_w - 1.0, 0.0],
                [src_w - 1.0, src_h - 1.0],
                [0.0, src_h - 1.0],
            ],
            dtype=np.float64,
        )
        H, _ = cv2.findHomography(src_pts, dst_pts, method=0)
        if H is None:
            raise RuntimeError("Homography computation failed.")
        return H

    def _build_intrinsics(self) -> np.ndarray:
        width, height = self.config.projector_resolution
        h = math.radians(self.config.horizontal_fov_deg)
        fx = (width / 2.0) / math.tan(h / 2.0)
        fy = fx
        cx, cy = width / 2.0, height / 2.0
        return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)

    @staticmethod
    def _beam_rotation_matrix(z_rad: float, y_rad: float) -> np.ndarray:
        Rz = np.array(
            [
                [math.cos(z_rad), -math.sin(z_rad), 0.0],
                [math.sin(z_rad), math.cos(z_rad), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        Ry = np.array(
            [
                [math.cos(y_rad), 0.0, math.sin(y_rad)],
                [0.0, 1.0, 0.0],
                [-math.sin(y_rad), 0.0, math.cos(y_rad)],
            ],
            dtype=np.float64,
        )
        return Rz @ Ry

    @staticmethod
    def _camera_axes_from_rotation(R: np.ndarray):
        forward = _normalize(R[:, 0])
        right = _normalize(-R[:, 1])
        down = _normalize(-R[:, 2])
        R_wc = np.vstack([right, down, forward])
        return R_wc, (right, down, forward)

    @staticmethod
    def _compute_plane_intersection(
        normal: np.ndarray,
        displacement_m: float,
        ray_direction: np.ndarray,
        ray_origin: np.ndarray,
    ) -> np.ndarray:
        n = _normalize(normal)
        P0 = n * displacement_m
        d = _normalize(ray_direction)
        o = np.asarray(ray_origin, dtype=np.float64)
        denom = float(np.dot(n, d))
        if abs(denom) < 1e-9:
            raise RuntimeError("Beam vector nearly parallel to plane.")
        t = float(np.dot(n, P0 - o) / denom)
        if t <= 0:
            raise RuntimeError("Beam does not intersect the plane.")
        return o + d * t

    @staticmethod
    def _plane_basis(normal: np.ndarray, roll_deg: float) -> Tuple[np.ndarray, np.ndarray]:
        n = _normalize(normal)
        world_y = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        u = world_y - np.dot(world_y, n) * n
        if np.linalg.norm(u) < 1e-9:
            helper = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            u = helper - np.dot(helper, n) * n
        u = _normalize(u)
        if np.dot(u, world_y) < 0:
            u = -u
        if abs(roll_deg) > 1e-9:
            u = HomographyCalculator._rodrigues_rotate(u, n, math.radians(roll_deg))
        v = _normalize(np.cross(n, u))
        return u, v

    @staticmethod
    def _rodrigues_rotate(vec: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
        axis_n = _normalize(axis)
        c, s = math.cos(angle), math.sin(angle)
        return (
            vec * c
            + np.cross(axis_n, vec) * s
            + axis_n * np.dot(axis_n, vec) * (1 - c)
        )

    @staticmethod
    def _project_points_world(
        K: np.ndarray,
        R_wc: np.ndarray,
        pts_world: np.ndarray,
        cam_origin: np.ndarray,
    ) -> np.ndarray:
        pts = np.asarray(pts_world, dtype=np.float64)
        origin = np.asarray(cam_origin, dtype=np.float64)
        Xc = (R_wc @ (pts - origin).T).T
        uvw = (K @ Xc.T).T
        return uvw[:, :2] / uvw[:, 2:3]
