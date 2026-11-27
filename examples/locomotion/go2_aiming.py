"""Estimate marker depth from RGB and geometric LiDAR and solve two-axis laser aiming."""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class Detection:
    center: np.ndarray
    corners: np.ndarray
    size: np.ndarray


@dataclass(frozen=True)
class Fusion:
    status: str
    point_camera: np.ndarray | None = None
    depth: float = 0.0
    count: int = 0
    residual: float = 0.0


def detect_icon(rgb: np.ndarray, detector: cv2.aruco.ArucoDetector) -> Detection | None:
    """Detect ID 7 and intersect its image diagonals to locate the marker center."""
    corners, ids, _ = detector.detectMarkers(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
    if ids is None:
        return None
    matches = np.nonzero(ids == 7)[0]
    if len(matches) != 1:
        return None
    quad = corners[matches[0]].reshape(4, 2)
    homogeneous = np.column_stack((quad, np.ones(4)))
    center = np.cross(np.cross(homogeneous[0], homogeneous[2]), np.cross(homogeneous[1], homogeneous[3]))
    if abs(center[2]) < 1e-8:
        return None
    _, _, width, height = cv2.boundingRect(quad)
    return Detection(center[:2] / center[2], quad, np.array([width, height]))


def fuse_depth(detection: Detection | None, camera_points: np.ndarray, intrinsics: np.ndarray) -> Fusion:
    """Intersect the detected pixel ray with a robust plane fitted to nearby LiDAR returns."""
    if detection is None:
        return Fusion("icon unavailable")
    points = camera_points[np.isfinite(camera_points).all(axis=1) & (camera_points[:, 2] > 0.05)]
    if len(points) < 6:
        return Fusion("insufficient LiDAR returns")
    pixels = points[:, :2] / points[:, 2, None] * intrinsics.diagonal()[:2] + intrinsics[:2, 2]
    selected = (np.abs(pixels - detection.center) < np.maximum(12, detection.size * 0.85)).all(axis=1)
    points, pixels = points[selected], pixels[selected]
    if len(points) < 6:
        return Fusion("no matching LiDAR patch")
    order = np.argsort(points[:, 2])
    points, pixels = points[order], pixels[order]
    groups = np.split(np.arange(len(points)), np.flatnonzero(np.diff(points[:, 2]) > 0.08) + 1)
    ray = np.linalg.solve(intrinsics, [*detection.center, 1.0])
    fits = []
    rng = np.random.default_rng(0)
    for group in groups:
        patch, uv = points[group], pixels[group]
        if len(patch) >= 6:
            triples = np.array([rng.choice(len(patch), 3, replace=False) for _ in range(64)])
            a, b, c = (patch[triples[:, i]] for i in range(3))
            normals = np.cross(b - a, c - a)
            lengths = np.linalg.norm(normals, axis=1)
            valid = lengths > 1e-8
            if valid.any():
                normals, a = normals[valid] / lengths[valid, None], a[valid]
                errors = np.abs(patch @ normals.T - np.sum(a * normals, axis=1))
                mask = errors[:, np.argmax((errors < 0.006).sum(axis=0))] < 0.006
                if mask.sum() >= 6 and mask.mean() >= 0.7:
                    inliers = patch[mask]
                    center = inliers.mean(axis=0)
                    _, singular, vh = np.linalg.svd(inliers - center, full_matrices=False)
                    if singular[1] >= 0.012 and singular[2] / singular[1] <= 0.15:
                        normal = vh[-1]
                        residual = np.sqrt(np.mean(((inliers - center) @ normal) ** 2))
                        denominator = normal @ ray
                        hull = cv2.convexHull(uv[mask].astype(np.float32))
                        enclosed = cv2.pointPolygonTest(hull, tuple(map(float, detection.center)), False) >= 0
                        if residual <= 0.006 and abs(denominator) >= 0.15 and enclosed:
                            depth = (normal @ center) / denominator
                            if depth > 0:
                                fits.append(Fusion("valid", ray * depth, depth, int(mask.sum()), residual))
    return fits[0] if len(fits) == 1 else Fusion("ambiguous, sparse or nonplanar surface")


class AimController:
    """Track fused targets with bounded command speed and acceleration."""

    def __init__(self, limits: np.ndarray):
        self.limits = limits.copy()
        self.command = np.zeros(2)
        self.goal = np.zeros(2)
        self.velocity = np.zeros(2)
        self.last_measurement = -np.inf
        self.status = "waiting for fusion"

    def observe(self, angles: np.ndarray | None, time: float):
        """Update the goal from solved yaw and pitch angles or hold the command."""
        if angles is None:
            self.goal[:] = self.command
            self.velocity[:] = 0
            self.status = "target unavailable"
        else:
            self.goal[:] = angles
            self.last_measurement = time
            self.status = "tracking"

    def step(self, time: float, dt: float) -> np.ndarray:
        """Advance the command and hold measurements older than 0.25 seconds."""
        if self.status == "tracking" and time - self.last_measurement > 0.25:
            self.goal[:] = self.command
            self.velocity[:] = 0
            self.status = "measurement stale"
        error = self.goal - self.command
        acceleration = np.deg2rad(180)
        desired = np.sign(error) * np.minimum(np.deg2rad(60), np.sqrt(2 * acceleration * np.abs(error)))
        self.velocity += np.clip(desired - self.velocity, -acceleration * dt, acceleration * dt)
        movement = self.velocity * dt
        arrived = (np.abs(movement) >= np.abs(error)) & (movement * error >= 0)
        self.command += np.where(arrived, error, movement)
        self.velocity[arrived] = 0
        self.command[:] = np.clip(self.command, self.limits[:, 0], self.limits[:, 1])
        return self.command.copy()


@dataclass(frozen=True)
class RangeGeom:
    kind: str
    size: np.ndarray


def cast_ranges(
    origin: np.ndarray,
    directions: np.ndarray,
    positions: np.ndarray,
    rotations: np.ndarray,
    geoms: tuple[RangeGeom, ...],
    max_range: float,
) -> np.ndarray:
    """Cast a batch of geometric rays against moving primitives and the ground plane."""
    distances = np.full(len(directions), max_range)
    downward = directions[:, 2] < -1e-8
    ground = np.full(len(directions), np.inf)
    ground[downward] = -origin[2] / directions[downward, 2]
    distances = np.minimum(distances, np.where(ground >= 0, ground, np.inf))
    for geom, position, rotation in zip(geoms, positions, rotations):
        start = (origin - position) @ rotation
        rays = directions @ rotation
        if geom.kind == "box":
            parallel = np.abs(rays) < 1e-10
            safe = np.where(parallel, 1.0, rays)
            lo, hi = (-geom.size - start) / safe, (geom.size - start) / safe
            near = np.max(np.where(parallel, -np.inf, np.minimum(lo, hi)), axis=1)
            far = np.min(np.where(parallel, np.inf, np.maximum(lo, hi)), axis=1)
            valid = (far >= np.maximum(near, 0)) & ~np.any(parallel & (np.abs(start) > geom.size), axis=1)
            hit = np.where(valid, np.where(near >= 0, near, far), np.inf)
        else:
            n = 3 if geom.kind == "sphere" else 2
            a = np.sum(rays[:, :n] ** 2, axis=1)
            b = rays[:, :n] @ start[:n]
            c = np.dot(start[:n], start[:n]) - geom.size[0] ** 2
            discriminant = b * b - a * c
            root = np.sqrt(np.maximum(discriminant, 0))
            safe = np.maximum(a, 1e-12)
            near, far = (-b - root) / safe, (-b + root) / safe
            hit = np.where(near >= 0, near, far)
            valid = (discriminant >= 0) & (a > 1e-12) & (hit >= 0)
            if geom.kind == "cylinder":
                valid &= np.abs(start[2] + rays[:, 2] * hit) <= geom.size[1]
                side = np.where(valid, hit, np.inf)
                caps = np.full((len(rays), 2), np.inf)
                moving = np.abs(rays[:, 2]) > 1e-10
                caps[moving] = (np.array([-geom.size[1], geom.size[1]]) - start[2]) / rays[moving, 2, None]
                xy = start[:2] + rays[:, None, :2] * np.where(np.isfinite(caps), caps, 0)[..., None]
                cap_valid = (caps >= 0) & (np.sum(xy**2, axis=-1) <= geom.size[0] ** 2)
                hit = np.minimum(side, np.min(np.where(cap_valid, caps, np.inf), axis=1))
            else:
                hit = np.where(valid, hit, np.inf)
        distances = np.minimum(distances, hit)
    return distances
