# Go2 velocity policy and automatic laser aiming

Run the trained policy with the new gimbal asset and target scene:

```bash
PYTHONPATH=. python examples/locomotion/go2_eval.py \
  -e go2-velocity-standing --ckpt 1000
```

Choose the checkpoint number that exists in `logs/go2-velocity-standing/`.
The experiment must contain `cfgs.pkl` and `model_<iteration>.pt`.
The policy and its saved observation scales remain the locomotion controller.
The target scene is built by evaluation. Training retains its flat-ground scene.

Focus the Genesis viewer and hold Up/Down to move forward/backward, Left/Right
to turn and A/D to move sideways. Releasing keys sends a zero velocity command.
Gimbal aiming runs automatically. The RGB window shows the detected marker center,
estimated depth, fit count and aim status. The red dot is the projected physical
laser hit. Escape closes the RGB window while sensing continues. The 3D viewer
shows the beam, hit spot and a subsampled point cloud.

For bounded runs with window creation disabled:

```bash
PYTHONPATH=. python examples/locomotion/go2_eval.py \
  -e go2-velocity-standing --ckpt 1000 --headless --steps 500
```

Offscreen RGB sensing still requires a working OpenGL rendering context.
The environment needs the existing Genesis dependencies plus the runtime
dependencies of the repository's RSL-RL source. OpenCV must provide `cv2.aruco`.

## Measurement and control

### Independent pixel/depth interface

`go2_pixel_gimbal_config.py` contains the default `PixelGimbalConfig`.
Replace `camera_fx`, `camera_fy`, `camera_cx`, `camera_cy` and `camera_to_base`
with the calibration of your camera and installation. Default values describe
this simulation. Installation poses use meters and roll/pitch/yaw in degrees.
A camera with zero RPY faces base +X, with image up along base +Z.

```python
from dataclasses import replace
from examples.locomotion.go2_gimbal_solver import PixelGimbalSolver
from examples.locomotion.go2_pixel_gimbal_config import DEFAULT_CONFIG, PoseConfig

config = replace(
    DEFAULT_CONFIG,
    camera_fx=415.69219381653056, camera_fy=415.69219381653056,
    camera_cx=320.0, camera_cy=240.0,
    camera_to_base=PoseConfig(pos=(0.275, 0.0, 0.10), rpy_deg=(0.0, 0.0, 0.0)),
)
solver = PixelGimbalSolver(config)
angles = solver.solve(x_px=440, y_px=300, depth_m=1.2)
if angles is not None:
    yaw, pitch = angles
```

`PixelGimbalSolver()` uses the default configuration. The solver reads explicit
geometry from that configuration and requires no URDF. It includes the gimbal
mount, both joint origins and axes, limits and the full three-dimensional laser
aperture position. CAD mass properties in the configuration serve the simulation's
gravity compensation. Update simulation URDF geometry as well when physically
changing the simulated mount.

Depth is camera optical Z in meters. Pixels must be undistorted at the calibrated
resolution. Output angles use radians. Unreachable targets return None and
invalid measurements raise ValueError.

The scene ports the seven movable targets from `simple_pan_tilt_gimbal`.
All targets have mass 1 kg. Positions are shifted 0.35 m toward the robot's
right so the initial marker falls inside the current CAD gimbal mount's yaw
travel. The canonical 130 mm ArUco marker is DICT_4X4_50, ID 7. It is attached
to the central red box as a visual mesh. Its physical support surface is 2 mm
behind the marker plane.

RGB resolution is 640 by 480 with a 60 degree vertical FOV. The optical camera
origin is `(0.275, 0, 0.10)` m in the robot base frame. The geometric 3D LiDAR
origin is `(0.268, 0, 0.155)` m. Its 4961 rays cover 120 by 60 degrees with a
3 m maximum range. These are provisional sensor installation calibrations.

RGB detection uses marker corners and their diagonal intersection. LiDAR
returns are projected into the camera, gated around the detection, split at
depth discontinuities and fitted with RANSAC and SVD. The detected pixel ray
intersects the fitted plane to provide optical depth and a carrier-frame point.
Sparse, collinear, ambiguous and nonplanar patches produce an unavailable result.
The camera depth buffer, marker size and target pose provide no depth estimate.

The simulated range sensor uses vectorized geometric intersections with the
seven Genesis primitive geoms at their current link poses and the ground plane.
The robot and marker visual are excluded from ranging, matching the source
scene's device/marker filters. Adding a new obstacle requires adding its range
geometry to `TargetScene`. The laser uses the same range scene with a 10 m limit.
It stops at the nearest surface, including intervening obstacles.

The independent numerical inverse solver uses fixed mounting transforms, joint
axes and limits from `go2_pixel_gimbal_config.py`. The laser aperture is
`(0.00575, 0.0385747972958, -0.0121803398875)` m in the CAD pitch-link frame,
and emits along local -X. It includes both pivot and aperture offsets.
The motor command has a 60 degree/s speed limit and 180 degree/s^2 acceleration
limit with gravity feedforward. Invalid or unreachable measurements hold the
command, and tracking stops after 0.25 s without a fresh measurement.
Sensing runs at 15 Hz simulation time. Control runs at the environment's 50 Hz.

Perception passes the detected pixel and estimated optical depth to the independent
pixel solver. The solver performs backprojection and the fixed camera-to-carrier conversion internally.
Target link poses are used by the simulated range sensor to generate geometric
returns, as in the source MuJoCo range sensor. They are excluded from the aiming
solver and fusion inputs. Simulation measurements are ideal and uncalibrated.

## Validation

Numerical checks in `tests/test_go2_targeting.py` cover marker detection, tilted
plane depth, rejection of a single scan line, occlusion, inverse kinematics and
stale-command hold. Local Genesis CPU validation builds the actual scene,
renders RGB, detects the marker, obtains fused depth and runs the gimbal loop
for 10 simulation seconds with standing leg commands. The trained locomotion
checkpoint resides on the user's server and requires server-side play validation.
