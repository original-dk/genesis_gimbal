"""Numerical checks for the Go2 RGB, range and laser targeting pipeline."""

from pathlib import Path

import cv2
import numpy as np

from examples.locomotion.go2_aiming import (
    AimController,
    Detection,
    GimbalSolver,
    RangeGeom,
    cast_ranges,
    detect_icon,
    fuse_depth,
)


def test_marker_detection():
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    marker = cv2.aruco.generateImageMarker(dictionary, 7, 160)
    image = np.full((240, 320, 3), 255, dtype=np.uint8)
    image[40:200, 80:240] = marker[..., None]
    detection = detect_icon(image, cv2.aruco.ArucoDetector(dictionary))
    assert detection is not None
    assert np.allclose(detection.center, [159.5, 119.5], atol=1)


def test_tilted_plane_fusion_and_sparse_rejection():
    intrinsics = np.array([[400.0, 0, 320], [0, 400, 240], [0, 0, 1]])
    x, y = np.meshgrid(np.linspace(-0.1, 0.1, 11), np.linspace(-0.1, 0.1, 11))
    points = np.column_stack((x.reshape(121), y.reshape(121), (1.5 + 0.2 * x + 0.3 * y).reshape(121)))
    detection = Detection(np.array([320.0, 240.0]), np.zeros((4, 2)), np.array([80.0, 80.0]))
    fused = fuse_depth(detection, points, intrinsics)
    assert fused.point_camera is not None
    assert np.allclose(fused.point_camera, [0, 0, 1.5], atol=1e-6)
    assert fuse_depth(detection, points[:11], intrinsics).point_camera is None
    assert fuse_depth(None, points, intrinsics).point_camera is None


def test_range_occlusion():
    geoms = (RangeGeom("box", np.array([0.1, 0.2, 0.2])), RangeGeom("sphere", np.array([0.2])))
    ranges = cast_ranges(
        np.array([0.0, 0, 0.5]),
        np.array([[1.0, 0, 0], [0, 0, -1.0]]),
        np.array([[1.0, 0, 0.5], [2.0, 0, 0.5]]),
        np.stack([np.eye(3)] * 2),
        geoms,
        3,
    )
    assert np.allclose(ranges, [0.9, 0.5])


def test_aiming_geometry_and_stale_hold():
    solver = GimbalSolver(Path(__file__).parents[1] / "genesis/assets/urdf/go2/urdf/go2_gimbal.urdf")
    for reference in (np.array([0.2, -0.3]), np.array([1.2, 0.4]), np.array([-0.5, 0.2])):
        origin, direction, _ = solver.forward(reference)
        target = origin + 2 * direction
        angles = solver.solve(target, np.zeros(2))
        assert angles is not None
        assert np.linalg.norm(solver.residual(angles, target)) < 1e-4
    controller = AimController(solver)
    origin, direction, _ = solver.forward(np.array([0.5, 0.1]))
    controller.observe(origin + 2 * direction, 0)
    controller.step(0, 0.02)
    command = controller.command.copy()
    controller.step(0.3, 0.02)
    assert controller.status == "measurement stale"
    assert np.allclose(controller.command, command)
