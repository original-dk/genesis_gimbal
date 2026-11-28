"""Convert calibrated camera pixels and optical depth into gimbal angles."""

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from .config import DEFAULT_CONFIG, PixelGimbalConfig


@dataclass(frozen=True)
class GimbalJoint:
    transform: np.ndarray
    axis: np.ndarray
    limits: np.ndarray


@dataclass(frozen=True)
class MassPoint:
    mass: float
    com: np.ndarray


class GimbalSolver:
    """Solve laser direction using calibrated geometry in the carrier frame."""

    def __init__(self, config: PixelGimbalConfig = DEFAULT_CONFIG):
        for pose in (config.gimbal_to_base, config.yaw.origin, config.pitch.origin, config.camera_to_base):
            if (
                np.shape(pose.pos) != (3,)
                or np.shape(pose.rpy_deg) != (3,)
                or not np.isfinite([pose.pos, pose.rpy_deg]).all()
            ):
                raise ValueError("Installation poses require finite xyz and roll/pitch/yaw triples.")
        self.mount = np.eye(4)
        self.mount[:3, 3] = config.gimbal_to_base.pos
        self.mount[:3, :3] = Rotation.from_euler("xyz", config.gimbal_to_base.rpy_deg, degrees=True).as_matrix()
        self.joints = []
        for joint in (config.yaw, config.pitch):
            transform = np.eye(4)
            transform[:3, 3] = joint.origin.pos
            transform[:3, :3] = Rotation.from_euler("xyz", joint.origin.rpy_deg, degrees=True).as_matrix()
            axis = np.array(joint.axis)
            if not np.isfinite(axis).all() or not np.isclose(np.linalg.norm(axis), 1):
                raise ValueError("Joint axes must be finite unit vectors.")
            limits = np.deg2rad(joint.limits_deg)
            if not np.isfinite(limits).all() or limits[0] >= limits[1]:
                raise ValueError("Joint limits must be finite and ordered.")
            self.joints.append(GimbalJoint(transform, axis, limits))
        self.limits = np.stack([joint.limits for joint in self.joints])
        self.aperture = np.array(config.laser_origin)
        self.direction = np.array(config.laser_direction)
        if not np.isfinite(self.aperture).all() or not np.isclose(np.linalg.norm(self.direction), 1):
            raise ValueError("Laser origin must be finite and direction must be a unit vector.")
        self.masses = [
            MassPoint(entry.mass, np.array(entry.com))
            for entry in (config.yaw_mass, config.pitch_mass, config.adapter_mass)
        ]

    def forward(self, angles: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return the laser origin, unit direction and mass-weighted link positions."""
        transform = self.mount.copy()
        moment = np.zeros(3)
        for i, (joint, angle) in enumerate(zip(self.joints, angles)):
            motion = np.eye(4)
            motion[:3, :3] = Rotation.from_rotvec(joint.axis * angle).as_matrix()
            transform = transform @ joint.transform @ motion
            entry = self.masses[i]
            moment += entry.mass * (transform[:3, :3] @ entry.com + transform[:3, 3])
        entry = self.masses[2]
        moment += entry.mass * (transform[:3, :3] @ entry.com + transform[:3, 3])
        return transform[:3, :3] @ self.aperture + transform[:3, 3], transform[:3, :3] @ self.direction, moment

    def residual(self, angles: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Return the direction residual for a carrier-frame target."""
        origin, direction, _ = self.forward(angles)
        vector = target - origin
        distance = np.linalg.norm(vector)
        return vector / max(distance, 1e-6) - direction

    def solve(self, target: np.ndarray, current: np.ndarray) -> np.ndarray | None:
        """Return feasible yaw and pitch angles for the estimated target."""
        if target.shape != (3,) or not np.isfinite(target).all():
            return None
        angles = np.clip(current, self.limits[:, 0], self.limits[:, 1])
        for _ in range(40):
            error = self.residual(angles, target)
            if np.linalg.norm(error) < 1e-7:
                return angles
            jacobian = np.column_stack(
                [(self.residual(angles + offset, target) - error) / 1e-5 for offset in np.eye(2) * 1e-5]
            )
            change = np.linalg.lstsq(jacobian, -error, rcond=None)[0]
            angles = np.clip(angles + np.clip(change, -0.2, 0.2), self.limits[:, 0], self.limits[:, 1])
        return angles if np.linalg.norm(self.residual(angles, target)) < 1e-4 else None


class PixelGimbalSolver:
    """Convert undistorted image pixels and optical-axis depth to yaw and pitch."""

    def __init__(self, config: PixelGimbalConfig = DEFAULT_CONFIG):
        self.intrinsics = np.array(
            [[config.camera_fx, 0, config.camera_cx], [0, config.camera_fy, config.camera_cy], [0, 0, 1]]
        )
        self.camera_to_carrier = np.eye(4)
        optical = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
        self.camera_to_carrier[:3, :3] = (
            Rotation.from_euler("xyz", config.camera_to_base.rpy_deg, degrees=True).as_matrix() @ optical
        )
        self.camera_to_carrier[:3, 3] = config.camera_to_base.pos
        if self.intrinsics.shape != (3, 3) or not np.isfinite(self.intrinsics).all():
            raise ValueError("Camera intrinsics must be a finite 3x3 matrix.")
        if self.intrinsics[0, 0] <= 0 or self.intrinsics[1, 1] <= 0 or not np.allclose(self.intrinsics[2], [0, 0, 1]):
            raise ValueError("Camera intrinsics must have positive focal lengths and a pinhole bottom row.")
        transform = self.camera_to_carrier
        if transform.shape != (4, 4) or not np.isfinite(transform).all():
            raise ValueError("Camera extrinsics must be a finite 4x4 camera-to-carrier transform.")
        rotation = transform[:3, :3]
        if (
            not np.allclose(transform[3], [0, 0, 0, 1])
            or not np.allclose(rotation.T @ rotation, np.eye(3))
            or not np.isclose(np.linalg.det(rotation), 1)
        ):
            raise ValueError("Camera-to-carrier transform must be rigid.")
        self.intrinsics = self.intrinsics.copy()
        self.camera_to_carrier = transform.copy()
        self.gimbal = GimbalSolver(config)

    def solve(self, x_px: float, y_px: float, depth_m: float) -> np.ndarray | None:
        """Return yaw and pitch in radians for a pixel and positive optical-axis depth in meters.

        Pixel coordinates use the calibrated image's top-left origin, with X right and Y down.
        Depth measures camera optical Z. Pixels must be undistorted and use the calibration resolution.
        Unreachable targets return None. Invalid measurements raise ValueError.
        """
        if not np.isfinite([x_px, y_px, depth_m]).all() or depth_m <= 0:
            raise ValueError("Pixel coordinates must be finite and optical depth must be finite and positive.")
        ray = np.linalg.solve(self.intrinsics, [x_px, y_px, 1.0])
        point_camera = ray * (depth_m / ray[2])
        target = self.camera_to_carrier[:3, :3] @ point_camera + self.camera_to_carrier[:3, 3]
        return self.gimbal.solve(target, np.zeros(2))
