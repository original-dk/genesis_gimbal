"""Build the moving-target scene and orchestrate RGB, LiDAR and laser aiming in Genesis."""

from pathlib import Path

import cv2
import numpy as np
import torch
from go2_aiming import AimController, Fusion, GimbalSolver, RangeGeom, cast_ranges, detect_icon, fuse_depth

import genesis as gs
from genesis.utils.geom import quat_to_R
from genesis.utils.misc import tensor_to_array


class TargetScene:
    """Own the play scene sensors and command the gimbal from fused measurements."""

    def __init__(self, show_rgb: bool = True):
        self.show_rgb = show_rgb
        parameters = cv2.aruco.DetectorParameters()
        parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), parameters)
        self.solver = GimbalSolver(Path(gs.__file__).parent / "assets/urdf/go2/urdf/go2_gimbal.urdf")
        self.controller = AimController(self.solver)
        self.camera = None
        self.scene = None
        self.robot = None
        self.links_idx = []
        self.gimbal_dofs = []
        self.geoms: tuple[RangeGeom, ...] = ()
        self.fusion = Fusion("waiting for fusion")
        self.rgb = None
        self.detection = None
        self.next_measurement = 0.0
        self.next_report = 0.0
        self.beam = None
        self.spot = None
        self.cloud = None
        self.target_carrier = None
        self.hit_distance = 10.0
        self.measured_points = np.empty((0, 3))
        self.camera_pos = np.array([0.275, 0.0, 0.10])
        self.lidar_pos = np.array([0.268, 0.0, 0.155])
        self.optical_rotation = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
        azimuth, elevation = np.meshgrid(np.deg2rad(np.linspace(-60, 60, 121)), np.deg2rad(np.linspace(-30, 30, 41)))
        self.ray_directions = np.stack(
            (np.cos(elevation) * np.cos(azimuth), np.cos(elevation) * np.sin(azimuth), np.sin(elevation)), axis=-1
        ).reshape(4961, 3)

    def build(self, scene: gs.Scene, robot):
        """Add the marker box, six obstacles and the calibrated front camera before scene build."""
        self.scene = scene
        self.robot = robot
        self.gimbal_dofs = [
            robot.get_joint(name).dofs_idx_local[0] for name in ("gimbal_yaw_joint", "gimbal_pitch_joint")
        ]
        self.links_idx = [robot.get_link("base").idx, robot.get_link("gimbal_pitch").idx]
        target = scene.add_entity(gs.morphs.URDF(file="urdf/go2/target_scene/target.urdf", pos=(1.5, -0.35, 0.25)))
        self.links_idx.append(target.links[0].idx)
        geoms = [RangeGeom("box", np.array([0.08, 0.10, 0.25]))]
        # The lateral shift keeps the central tag inside the CAD mount's yaw travel.
        for kind, size, pos, yaw, color in (
            ("box", (0.12, 0.12, 0.12), (1.1, 0.13, 0.12), 0.25, (0.18, 0.45, 0.85)),
            ("sphere", (0.12,), (0.95, -0.65, 0.12), 0.0, (0.95, 0.68, 0.08)),
            ("cylinder", (0.14, 0.25), (1.8, -1.07, 0.25), 0.0, (0.15, 0.7, 0.42)),
            ("box", (0.22, 0.18, 0.28), (2.65, 0.30, 0.28), -0.3, (0.62, 0.3, 0.78)),
            ("sphere", (0.20,), (2.9, -0.65, 0.20), 0.0, (0.1, 0.65, 0.78)),
            ("cylinder", (0.14, 0.38), (3.2, 1.0, 0.38), 0.0, (0.85, 0.5, 0.2)),
        ):
            if kind == "box":
                volume = 8 * np.prod(size)
                morph = gs.morphs.Box(size=tuple(2 * value for value in size), pos=pos, euler=(0, 0, np.rad2deg(yaw)))
            elif kind == "sphere":
                volume = 4 * np.pi * size[0] ** 3 / 3
                morph = gs.morphs.Sphere(radius=size[0], pos=pos)
            else:
                volume = np.pi * size[0] ** 2 * 2 * size[1]
                morph = gs.morphs.Cylinder(radius=size[0], height=2 * size[1], pos=pos)
            entity = scene.add_entity(
                morph, material=gs.materials.Rigid(rho=1 / volume), surface=gs.surfaces.Default(color=color)
            )
            self.links_idx.append(entity.links[0].idx)
            geoms.append(RangeGeom(kind, np.array(size)))
        self.geoms = tuple(geoms)
        self.camera = scene.add_camera(res=(640, 480), fov=60, near=0.01, far=10, env_idx=0, debug=False)

    def reset(self, time: float):
        """Restart aiming from zero after the robot's episode reset."""
        self.controller = AimController(self.solver)
        self.next_measurement = time

    def update(self, time: float, dt: float, targets: torch.Tensor):
        """Sense at 15 Hz, update smooth gimbal commands and draw the physical laser ray."""
        assert self.robot is not None and self.scene is not None and self.camera is not None
        positions = tensor_to_array(self.robot.solver.get_links_pos(self.links_idx, envs_idx=0))[0]
        rotations = quat_to_R(tensor_to_array(self.robot.solver.get_links_quat(self.links_idx, envs_idx=0))[0])
        base_pos, base_rotation = positions[0], rotations[0]
        camera_pos = base_pos + base_rotation @ self.camera_pos
        optical = base_rotation @ self.optical_rotation
        if time + 1e-8 >= self.next_measurement:
            self.next_measurement += 1 / 15
            self.camera.set_pose(pos=camera_pos, lookat=camera_pos + optical[:, 2], up=-optical[:, 1])
            self.rgb = tensor_to_array(self.camera.render(rgb=True)[0])
            self.detection = detect_icon(self.rgb, self.detector)
            origin = base_pos + base_rotation @ self.lidar_pos
            directions = self.ray_directions @ base_rotation.T
            distances = cast_ranges(origin, directions, positions[2:], rotations[2:], self.geoms, 3.0)
            hits = distances < 3.0 - 1e-8
            self.measured_points = origin + directions[hits] * distances[hits, None]
            if self.cloud is not None:
                self.scene.clear_debug_object(self.cloud)
            stride = max(1, (len(self.measured_points) + 499) // 500)
            self.cloud = None
            if len(self.measured_points) > 0:
                self.cloud = self.scene.draw_debug_spheres(
                    self.measured_points[::stride], radius=0.003, color=(0, 1, 0.6, 1)
                )
            camera_points = (self.measured_points - camera_pos) @ optical
            self.fusion = fuse_depth(self.detection, camera_points, self.camera.intrinsics)
            self.target_carrier = None
            if self.fusion.point_camera is not None:
                self.target_carrier = self.optical_rotation @ self.fusion.point_camera + self.camera_pos
            self.controller.observe(self.target_carrier, time)
        command = self.controller.step(time, dt)
        # Differentiate gravitational potential at the measured angles for feedforward.
        angles = tensor_to_array(self.robot.get_dofs_position(self.gimbal_dofs, envs_idx=0))[0]
        gravity_local = base_rotation.T @ np.array([0.0, 0.0, -9.81])
        compensation = np.zeros(2)
        for i, delta in enumerate(np.eye(2) * 1e-5):
            _, _, moment_before = self.solver.forward(angles - delta)
            _, _, moment_after = self.solver.forward(angles + delta)
            compensation[i] = -np.dot(gravity_local, moment_after - moment_before) / 2e-5
        targets[:] = torch.tensor(
            np.clip(command + compensation / 5.0, self.solver.limits[:, 0], self.solver.limits[:, 1]),
            device=gs.device,
        )
        aperture = positions[1] + rotations[1] @ self.solver.aperture
        direction = rotations[1] @ self.solver.direction
        self.hit_distance = cast_ranges(aperture, direction[None], positions[2:], rotations[2:], self.geoms, 10.0)[0]
        endpoint = aperture + direction * self.hit_distance
        if self.beam is not None:
            self.scene.clear_debug_object(self.beam)
        if self.spot is not None:
            self.scene.clear_debug_object(self.spot)
            self.spot = None
        self.beam = self.scene.draw_debug_line(aperture, endpoint, radius=0.001, color=(1, 0, 0, 1))
        if self.hit_distance < 10.0 - 1e-8:
            self.spot = self.scene.draw_debug_sphere(endpoint - 0.001 * direction, radius=0.004, color=(1, 0, 0, 1))
        if self.show_rgb and self.rgb is not None:
            image = cv2.cvtColor(self.rgb, cv2.COLOR_RGB2BGR)
            endpoint_camera = (endpoint - camera_pos) @ optical
            if self.hit_distance < 10.0 - 1e-8 and endpoint_camera[2] > 0:
                pixel = self.camera.intrinsics @ endpoint_camera
                cv2.circle(image, tuple(map(int, pixel[:2] / pixel[2])), 3, (0, 0, 255), -1)
            if self.detection is not None:
                xy = tuple(map(int, self.detection.center))
                cv2.drawMarker(image, xy, (0, 255, 0), cv2.MARKER_CROSS, 16, 1)
            text = f"{self.controller.status} | Z={self.fusion.depth:.3f}m N={self.fusion.count}"
            cv2.putText(image, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.imshow("Go2 RGB + LiDAR targeting", image)
            if cv2.waitKey(1) == 27:
                cv2.destroyWindow("Go2 RGB + LiDAR targeting")
                self.show_rgb = False
        if time >= self.next_report:
            self.next_report = time + 1.0
            print(
                f"Aim: {self.controller.status}, fusion: {self.fusion.status}, depth={self.fusion.depth:.3f}m, laser={self.hit_distance:.3f}m"
            )
