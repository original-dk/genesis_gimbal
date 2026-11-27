import argparse
import math
import os
import pickle

import torch

from rsl_rl.runners import OnPolicyRunner

import genesis as gs
from genesis.vis.keybindings import Key, KeyAction, Keybind

from go2_env import Go2Env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e", "--exp-name", type=str, default="go2-walking", help="Experiment name; also the log directory under logs/"
    )
    parser.add_argument("--ckpt", type=int, default=100, help="Checkpoint iteration to load")
    parser.add_argument("--speed", type=float, default=0.5, help="Linear command speed in m/s")
    parser.add_argument("--yaw-rate", type=float, default=0.5, help="Turning command speed in rad/s")
    args = parser.parse_args()
    if any(not math.isfinite(value) or value < 0 for value in (args.speed, args.yaw_rate)):
        parser.error("Speed and yaw rate must be finite and nonnegative.")

    gs.init(backend=gs.cpu)

    log_dir = f"logs/{args.exp_name}"
    with open(f"logs/{args.exp_name}/cfgs.pkl", "rb") as f:
        env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(f)
    reward_cfg["reward_scales"] = {}

    env = Go2Env(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=True,
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    runner.load(os.path.join(log_dir, f"model_{args.ckpt}.pt"))
    policy = runner.get_inference_policy(device=gs.device)

    pressed = [False] * 6

    # Viewer callbacks record keys. The simulation thread owns tensor writes.
    def set_pressed(index: int, value: bool):
        pressed[index] = value

    for index, key in enumerate((Key.UP, Key.DOWN, Key.LEFT, Key.RIGHT, Key.A, Key.D)):
        env.scene.viewer.register_keybinds(
            Keybind(f"velocity_press_{index}", key, KeyAction.PRESS, callback=set_pressed, args=(index, True)),
            Keybind(f"velocity_release_{index}", key, KeyAction.RELEASE, callback=set_pressed, args=(index, False)),
            overwrite=True,
        )
    print("Focus the viewer. Hold Up/Down: forward/backward, Left/Right: turn, A/D: strafe. Release to stop.")
    print("Command tracking depends on the velocity ranges used to train this checkpoint.")
    commands = torch.zeros(3, dtype=gs.tc_float, device=gs.device)
    obs_dict = env.reset()
    with torch.no_grad():
        while env.scene.viewer.is_alive():
            forward, backward, left, right, strafe_left, strafe_right = pressed
            commands[0] = args.speed * (forward - backward)
            commands[1] = args.speed * (strafe_left - strafe_right)
            commands[2] = args.yaw_rate * (left - right)
            env.set_commands(commands)
            obs_dict = env.get_observations()
            actions = policy(obs_dict)
            obs_dict, rews, dones, infos = env.step(actions)


if __name__ == "__main__":
    main()

"""
# evaluation
python examples/locomotion/go2_eval.py -e go2-walking --ckpt 100
"""
