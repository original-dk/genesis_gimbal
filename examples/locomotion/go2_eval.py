import argparse
import math
import os
import pickle

import cv2
import torch
from go2_env import Go2Env
from go2_target_scene import TargetScene

import genesis as gs
from genesis.vis.keybindings import Key, KeyAction, Keybind
from rsl_rl.runners import OnPolicyRunner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e", "--exp-name", type=str, default="go2-walking", help="Experiment name; also the log directory under logs/"
    )
    parser.add_argument("--ckpt", type=int, default=100, help="Checkpoint iteration to load")
    parser.add_argument("--speed", type=float, default=0.5, help="Linear command speed in m/s")
    parser.add_argument("--yaw-rate", type=float, default=0.5, help="Turning command speed in rad/s")
    parser.add_argument("--headless", action="store_true", help="Run sensing and aiming without windows")
    parser.add_argument("--steps", type=int, default=0, help="Stop after this many steps (zero runs continuously)")
    args = parser.parse_args()
    if any(not math.isfinite(value) or value < 0 for value in (args.speed, args.yaw_rate)):
        parser.error("Speed and yaw rate must be finite and nonnegative.")
    if args.steps < 0:
        parser.error("Steps must be nonnegative.")

    gs.init(backend=gs.cpu)

    log_dir = f"logs/{args.exp_name}"
    with open(f"logs/{args.exp_name}/cfgs.pkl", "rb") as f:
        env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(f)
    reward_cfg["reward_scales"] = {}

    targeting = TargetScene(show_rgb=not args.headless)
    env = Go2Env(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=not args.headless,
        scene_setup=targeting.build,
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    runner.load(os.path.join(log_dir, f"model_{args.ckpt}.pt"))
    policy = runner.get_inference_policy(device=gs.device)

    pressed = [False] * 6

    # Viewer callbacks record keys. The simulation thread owns tensor writes.
    def set_pressed(index: int, value: bool):
        pressed[index] = value

    for index, key in enumerate(() if args.headless else (Key.UP, Key.DOWN, Key.LEFT, Key.RIGHT, Key.A, Key.D)):
        env.scene.viewer.register_keybinds(
            Keybind(f"velocity_press_{index}", key, KeyAction.PRESS, callback=set_pressed, args=(index, True)),
            Keybind(f"velocity_release_{index}", key, KeyAction.RELEASE, callback=set_pressed, args=(index, False)),
            overwrite=True,
        )
    print("Focus the viewer. Hold Up/Down: forward/backward, Left/Right: turn, A/D: strafe. Release to stop.")
    print("Command tracking depends on the velocity ranges used to train this checkpoint.")
    commands = torch.zeros(3, dtype=gs.tc_float, device=gs.device)
    obs_dict = env.reset()
    step = 0
    with torch.no_grad():
        while (args.headless or env.scene.viewer.is_alive()) and (args.steps == 0 or step < args.steps):
            forward, backward, left, right, strafe_left, strafe_right = pressed
            commands[0] = args.speed * (forward - backward)
            commands[1] = args.speed * (strafe_left - strafe_right)
            commands[2] = args.yaw_rate * (left - right)
            env.set_commands(commands)
            targeting.update(step * env.dt, env.dt, env.gimbal_target_pos)
            obs_dict = env.get_observations()
            actions = policy(obs_dict)
            obs_dict, _, dones, _ = env.step(actions)
            step += 1
            if dones[0]:
                targeting.reset(step * env.dt)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

"""
# evaluation
python examples/locomotion/go2_eval.py -e go2-walking --ckpt 100
"""
