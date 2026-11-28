"""Default PixelGimbalSolver calibration. Replace camera calibration for each installation."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PoseConfig:
    pos: tuple[float, float, float]
    rpy_deg: tuple[float, float, float]


@dataclass(frozen=True)
class JointConfig:
    origin: PoseConfig
    axis: tuple[float, float, float]
    limits_deg: tuple[float, float]


@dataclass(frozen=True)
class MassConfig:
    mass: float
    com: tuple[float, float, float]


@dataclass(frozen=True)
class PixelGimbalConfig:
    # REQUIRED CALIBRATION: supply fx/fy/cx/cy from your camera SDK or calibration.
    # Defaults describe the simulated 640x480 image with 60-degree vertical FOV.
    camera_fx: float = 415.69219381653056
    camera_fy: float = 415.69219381653056
    camera_cx: float = 320.0
    camera_cy: float = 240.0
    # REQUIRED INSTALLATION: camera housing pose in the base frame (meters, degrees).
    # Zero RPY faces base +X with image up along base +Z. Optical-axis conversion is internal.
    camera_to_base: PoseConfig = PoseConfig((0.275, 0.0, 0.10), (0.0, 0.0, 0.0))
    # Base axes: X forward, Y left, Z up. All rotations use extrinsic XYZ (roll, pitch, yaw).
    gimbal_to_base: PoseConfig = PoseConfig(
        (-0.022126798366208, 0.0016289479923417, 0.08918682987185276), (0.0, 0.0, 90.0)
    )
    yaw: JointConfig = JointConfig(PoseConfig((0.0, 0.0, 0.0317), (0.0, 0.0, 0.0)), (0.0, 0.0, 1.0), (-180.0, 180.0))
    pitch: JointConfig = JointConfig(
        PoseConfig((0.0, -0.01057479729582, 0.0366), (0.0, 0.0, 0.0)), (0.0, -1.0, 0.0), (-90.0, 90.0)
    )
    # Full aperture position and unit direction in the moving pitch-link frame.
    laser_origin: tuple[float, float, float] = (0.00575, 0.0385747972958, -0.0121803398875)
    laser_direction: tuple[float, float, float] = (-1.0, 0.0, 0.0)
    # CAD mass properties used by simulation gravity compensation (kg, meters).
    yaw_mass: MassConfig = MassConfig(
        0.09807013899698717, (-2.9142785719978524e-05, -0.025472843933735407, 0.02969295394642258)
    )
    pitch_mass: MassConfig = MassConfig(
        0.023674555110400886, (0.003255085629150929, 0.018255788562034938, -0.008541412508569051)
    )
    adapter_mass: MassConfig = MassConfig(
        0.00868385161508989, (0.0047987152752195245, 0.04417999316253926, -0.018209153498764587)
    )


DEFAULT_CONFIG = PixelGimbalConfig()
