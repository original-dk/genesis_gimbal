# Go2 with two-axis gimbal

Load `urdf/go2/urdf/go2_gimbal.urdf` through the Genesis asset loader.
Keep `dae/`, `gimbal_meshes/` and `urdf/` together when moving this asset.

The original `urdf/go2.urdf` and its meshes remain the reference Go2 model.
The extended URDF preserves every original link and joint definition and appends:

```text
base
  mount_base                 (fixed)
    gimbal_carrier           (fixed)
      gimbal_base            (fixed)
        gimbal_yaw           (gimbal_yaw_joint)
          gimbal_pitch       (gimbal_pitch_joint)
            laser_adapter    (fixed)
```

The robot has 12 leg joints and two gimbal joints. The locomotion training and
evaluation environment loads this asset directly. Its policy controls the 12 leg
joints by name and observes their states. Separate position controllers hold both
gimbal axes at zero with provisional gains of kp=5 and kd=0.1 and torque limits
of +/-0.2 N m. Episode resets restore the full robot state including both axes.
The policy action dimension remains 12 and its observation dimension remains 45.
Training with this asset includes the added accessory mass and inertia.

## Geometry and registration

Accessory meshes are exported from a STEP AP214 copy of the open SolidWorks
`GO2_EDU_Integrated` assembly on 2026-11-27. The assembly includes
`FlushAssemblyV3`, the Go2 base plate and the gimbal carrier. The original
SolidWorks document remains open with its existing unsaved edits.

The base plate registration to Go2 reuses the CAD-to-Go2 registration documented
in `D:/pycharm_project/GO2_EDU_Integrated_URDF/README_仿真说明.md`:
translation `(-0.022126798366208, 0.0016289479923417, 0.059186829871672)` m
and yaw `pi/2`. That registration has an approximately 0.9 mm hip-center
alignment difference between the CAD and the original Go2 model.

The current assembly places the carrier 8 mm above the plate frame and the
gimbal base 22 mm above the carrier frame. Gimbal frames use the CAD axis
orientation. Yaw rotates about local +Z. Pitch rotates about local -Y.
The yaw joint is 31.7 mm above the gimbal base. The pitch joint is offset
`(0, -10.57479729582, 36.6)` mm from yaw.

Motor stator and rotor groups follow the existing FlushAssemblyV3 simulation
model. CAD motor shells are split at the visible cap seams to assign mass and
geometry to the two moving links. This is an approximation of motor internals.
The locating adapter is a separate fixed link on the pitch link. Laser hardware
and screws remain with their corresponding moving link.

## Provisional physics parameters

Both axes have provisional limits of +/-90 degrees, effort 0.2 N m, speed
2 rad/s and damping 0.005 N m s/rad. Replace these values after measuring the
hardware and checking mechanical travel and cable clearance.

Masses and inertias are estimates from CAD solid geometry and uniform material
densities: PETG 1270 kg/m^3 for the base plate and carrier, aluminum 2700 kg/m^3
for the motor shells, brackets and adapter, steel 7850 kg/m^3 for screws, brass
8500 kg/m^3 for the laser housing and 1850 kg/m^3 for the PCB envelope.
The added mass is approximately 0.60625 kg. Replace estimates with measured
link masses and inertias before identifying control gains.

Collision meshes use accessory CAD geometry. Contact behavior depends on the
importer's mesh handling. Full-range self-collision and dynamic stability need
simulation and hardware validation.

## Validation and provenance

Every original link and joint compares identically as canonical XML. Mesh paths
resolve, the model is a connected single-root tree, and MuJoCo loads 14 movable
joints after temporary conversion of the original DAE visuals for that importer.
Positive and negative gimbal rotations preserve every original Go2 link frame.
The source URDF SHA-256 is:

`4f306754e9b3d73930ac8362aa456eb8912f2e886665618e7eced9627c1704a4`

The current SolidWorks STEP export SHA-256 is:

`afd25f919ea0cdcd1926ff0b5d7d4b74ed4334acb0b295cae1e3fc2de6660b3e`
