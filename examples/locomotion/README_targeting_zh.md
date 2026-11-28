# Go2 速度策略与激光自动瞄准

[English](README_targeting.md)

## 运行 play

在项目根目录运行，加载已有速度策略、云台模型和目标场景：

```bash
python examples/locomotion/go2_eval.py -e go2-velocity --ckpt 10000
```

`-e` 是实验目录名，`--ckpt` 是实际保存的 checkpoint 编号。
例如上述命令读取 `logs/go2-velocity/cfgs.pkl` 和 `logs/go2-velocity/model_10000.pt`。
行走使用已训练策略及其保存的观测缩放，云台使用独立的感知与瞄准控制器。
评估时创建目标场景，训练仍使用平地场景。

点击 Genesis 主窗口使其获得焦点：

- 按住 ↑/↓：前进、后退。
- 按住 ←/→：左转、右转。
- 按住 A/D：向左、向右侧移。
- 松开按键：速度指令归零。

云台自动瞄准。独立 RGB 窗口显示标记中心、估计深度、平面拟合点数和瞄准状态。
红点表示实际激光落点投影。RGB 窗口中按 Esc 可关闭显示，后台感知继续运行。
主窗口显示激光束、落点和下采样后的点云。

关闭窗口显示并运行指定步数：

```bash
python examples/locomotion/go2_eval.py -e go2-velocity --ckpt 10000 --headless --steps 500
```

离屏 RGB 渲染仍需要可用的 OpenGL 上下文。
环境需要 Genesis 依赖、项目内 RSL-RL 源码的运行依赖，以及提供 `cv2.aruco` 的 OpenCV。

## 独立的像素与深度求解接口

默认配置位于 [go2_targeting/config.py](go2_targeting/config.py)，类型为 `PixelGimbalConfig`。
求解器位于 [go2_targeting/solver.py](go2_targeting/solver.py)，只需要 NumPy、SciPy 和配置。

每台相机和每次安装需要填写以下标定项：

| 配置项 | 含义与单位 |
| --- | --- |
| `camera_fx`、`camera_fy` | 相机焦距，单位像素 |
| `camera_cx`、`camera_cy` | 相机主点在图像中的像素坐标 |
| `camera_to_base.pos` | 相机安装位置，机身坐标系，单位米 |
| `camera_to_base.rpy_deg` | 相机安装偏角，roll、pitch、yaw，单位度 |

默认值描述当前仿真相机。真机应使用 SDK 或相机标定提供的内参。
机身坐标轴为 X 向前、Y 向左、Z 向上。
相机安装 RPY 为零时朝机身正前方，图像上方朝机身 +Z。
类内部完成相机光学坐标轴转换，配置时可以直接微调位置和角度。

以下示例可放在 `examples/locomotion/` 下的脚本中：

```python
from dataclasses import replace
from go2_targeting.solver import PixelGimbalSolver
from go2_targeting.config import DEFAULT_CONFIG, PoseConfig

config = replace(
    DEFAULT_CONFIG,
    camera_fx=415.69219381653056,
    camera_fy=415.69219381653056,
    camera_cx=320.0,
    camera_cy=240.0,
    camera_to_base=PoseConfig(
        pos=(0.275, 0.0, 0.10),
        rpy_deg=(0.0, 0.0, 0.0),
    ),
)
solver = PixelGimbalSolver(config)
angles = solver.solve(x_px=440, y_px=300, depth_m=1.2)
if angles is not None:
    yaw, pitch = angles  # 弧度
```

在项目根目录的其他 Python 程序中，可以通过
`examples.locomotion.go2_targeting.solver` 和
`examples.locomotion.go2_targeting.config` 导入。

`PixelGimbalSolver()` 直接使用默认配置。配置还包括云台安装位姿、两级关节原点、
转轴、角度范围，以及完整三维激光出射点和方向。求角度时无需读取 URDF。
配置中的 CAD 质量和重心用于仿真重力补偿。
修改仿真中的实际安装结构时，还需要同步修改对应 URDF。

### 像素与深度的约定

- `x_px`、`y_px` 是拍摄图像的像素坐标。左上角为原点，X 向右、Y 向下。
- 图像坐标应已去畸变，并与内参对应的分辨率、裁剪方式保持一致。
- 可视化窗口缩放、黑边和工具栏会改变鼠标坐标，需要先换算回原始图像像素。
- `depth_m` 是相机光学轴 Z 方向的深度，单位米。
- 返回顺序是 yaw、pitch，单位弧度。不可达时返回 `None`，非法输入抛出 `ValueError`。

例如相机向上微调 2 mm，就把 `camera_to_base.pos` 的 Z 增加 `0.002`。
相机向左偏转 1°，就把 `camera_to_base.rpy_deg` 的 yaw 增加 `1.0`。

## 场景与传感器

场景移植自 `simple_pan_tilt_gimbal`，包含七个质量为 1 kg 的可动物理目标。
目标位置整体向狗的右侧偏移 0.35 m，使初始标记位于 CAD 云台安装方向的可达区域。
中央红色箱体上固定 130 mm 的 ArUco 标记，字典为 `DICT_4X4_50`，ID 为 7。
标记是外观网格，其后方 2 mm 的箱体表面作为测距支撑面。

| 项目 | 默认配置 |
| --- | --- |
| RGB 图像 | 640×480，垂直视场角 60° |
| 相机安装位置 | 机身坐标系 `(0.275, 0, 0.10)` m |
| LiDAR 安装位置 | 机身坐标系 `(0.268, 0, 0.155)` m |
| LiDAR 扫描 | 4961 条射线，水平 120°、垂直 60°，最远 3 m |
| 激光射程 | 最远 10 m |

这些安装参数是仿真默认标定，真机需要按实际安装测量。

## 感知和控制流程

1. 从 RGB 检测 ID 7 的四个角点，用对角线交点计算标记中心像素。
2. 将 LiDAR 返回点投影到相机中，在检测区域附近选择点云。
3. 按深度断层分组，再使用 RANSAC 和 SVD 拟合局部平面。
4. 将标记中心的像素射线与拟合平面相交，得到光学轴深度。
5. 将像素 XY 和深度交给 `PixelGimbalSolver.solve()`，内部反投影、转换到机身坐标并求解关节角度。
6. 控制器平滑更新关节指令，仿真射线显示光束及最近落点。

稀疏、共线、多解或非平面的点云区域会产生不可用结果。
深度估计来自点云局部平面，相机深度缓冲、标记尺寸和目标真实位姿均不参与深度求解。

模拟 LiDAR 用向量化几何求交处理七个 Genesis 基本几何体的实时 link 位姿和地面。
机器人与标记外观被排除在测距范围外，与原场景的过滤逻辑一致。
增加障碍物时，需要在 `TargetScene` 中注册对应测距几何。
激光使用同一套测距场景，停止在最近表面，前方其他障碍物也可以截断光束。

目标 link 位姿仅用于模拟测距传感器生成返回点，融合与瞄准求解器接收测量结果。
这套传感器模型生成理想几何测量，真机误差仍需标定和验证。

## 云台与激光几何

逆解使用默认配置中的安装位姿、关节轴和限位，包括转轴偏移及出光孔偏移。
激光出射点相对于随俯仰轴转动的 pitch link：

```python
laser_origin = (0.00575, 0.0385747972958, -0.0121803398875)  # 米
laser_direction = (-1.0, 0.0, 0.0)  # pitch link 局部 -X
```

瞄准指令速度限制为 60°/s，加速度限制为 180°/s²，并使用重力前馈补偿。
测量失效或目标不可达时保持当前指令，超过 0.25 s 没有新鲜有效测量时停止追踪。
感知按仿真时间 15 Hz 更新，控制使用环境的 50 Hz 周期。

## 验证范围

`tests/test_go2_targeting.py` 覆盖标记检测、倾斜平面深度、单扫描线拒绝、遮挡、
像素深度到角度的转换、逆解和过期指令保持。
本机 Genesis CPU 验证包含实际场景构建、RGB 渲染、标记检测、融合深度，
以及站姿腿部指令下 10 秒的云台闭环运行。
已训练的行走 checkpoint 位于用户服务器，行走状态下的 play 效果需要在服务器验证。
