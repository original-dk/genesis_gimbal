import argparse
import math
import os
import pickle
from pathlib import Path

from rsl_rl.runners import OnPolicyRunner

import genesis as gs

from go2_env import Go2Env


def get_train_cfg(exp_name):
    train_cfg_dict = {
        "algorithm": {
            "class_name": "PPO",
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.01,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 0.001,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        "actor": {
            "class_name": "MLPModel",
            "hidden_dims": [512, 256, 128],
            "activation": "elu",
            "distribution_cfg": {
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        },
        "critic": {
            "class_name": "MLPModel",
            "hidden_dims": [512, 256, 128],
            "activation": "elu",
        },
        "obs_groups": {
            "actor": ["policy"],
            "critic": ["policy"],
        },
        "num_steps_per_env": 24,
        "save_interval": 100,
        "run_name": exp_name,
        "logger": "tensorboard",
    }

    return train_cfg_dict


def get_cfgs():
    env_cfg = {
        "num_actions": 12,
        # joint/link names
        "default_joint_angles": {  # [rad]
            "FL_hip_joint": 0.0,
            "FR_hip_joint": 0.0,
            "RL_hip_joint": 0.0,
            "RR_hip_joint": 0.0,
            "FL_thigh_joint": 0.8,
            "FR_thigh_joint": 0.8,
            "RL_thigh_joint": 1.0,
            "RR_thigh_joint": 1.0,
            "FL_calf_joint": -1.5,
            "FR_calf_joint": -1.5,
            "RL_calf_joint": -1.5,
            "RR_calf_joint": -1.5,
        },
        "joint_names": [
            "FR_hip_joint",
            "FR_thigh_joint",
            "FR_calf_joint",
            "FL_hip_joint",
            "FL_thigh_joint",
            "FL_calf_joint",
            "RR_hip_joint",
            "RR_thigh_joint",
            "RR_calf_joint",
            "RL_hip_joint",
            "RL_thigh_joint",
            "RL_calf_joint",
        ],
        # PD
        "kp": 20.0,
        "kd": 0.5,
        # termination
        "termination_if_roll_greater_than": 10,  # degree
        "termination_if_pitch_greater_than": 10,
        # base pose
        "base_init_pos": [0.0, 0.0, 0.42],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "episode_length_s": 20.0,
        "resampling_time_s": 4.0,
        "action_scale": 0.25,
        "simulate_action_latency": True,
        "clip_actions": 100.0,
    }
    obs_cfg = {
        "obs_scales": {
            "lin_vel": 2.0,
            "ang_vel": 0.25,
            "dof_pos": 1.0,
            "dof_vel": 0.05,
        },
    }
    reward_cfg = {
        "tracking_sigma": 0.25,
        "base_height_target": 0.3,
        "feet_height_target": 0.075,
        "reward_scales": {
            "tracking_lin_vel": 1.0,
            "tracking_ang_vel": 0.2,
            "lin_vel_z": -1.0,
            "base_height": -50.0,
            "action_rate": -0.01,
            "similar_to_default": -0.1,
            "standing_dof_vel": -0.02,
            "standing_pose": -0.5,
        },
    }
    command_cfg = {
        "num_commands": 3,
        "standing_probability": 0.2,
        "lin_vel_x_range": [-1.5, 1.5],
        "lin_vel_y_range": [-1.0, 1.0],
        "ang_vel_range": [-1.5, 1.5],
    }

    return env_cfg, obs_cfg, reward_cfg, command_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e", "--exp-name", type=str, default="go2-walking", help="Experiment name; also the log directory under logs/"
    )
    parser.add_argument("-b", "--num-envs", type=int, default=4096, help="Number of parallel environments")
    parser.add_argument("--max-iterations", type=int, default=1001, help="Number of additional learning iterations")
    parser.add_argument("--resume", action="store_true", help="Continue training with the saved experiment configuration")
    parser.add_argument("--ckpt", type=int, help="Checkpoint iteration to resume from (default: latest)")
    parser.add_argument("--max-forward-speed", type=float, default=1.5, help="Forward/backward speed limit in m/s")
    parser.add_argument("--max-lateral-speed", type=float, default=1.0, help="Lateral speed limit in m/s")
    parser.add_argument("--max-yaw-rate", type=float, default=1.5, help="Turning speed limit in rad/s")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    args = parser.parse_args()
    if args.ckpt is not None and (not args.resume or args.ckpt < 0):
        parser.error("--ckpt requires --resume and a nonnegative iteration.")
    if args.max_iterations <= 0 or args.num_envs <= 0:
        parser.error("Iterations and number of environments must be positive.")
    if any(
        not math.isfinite(value) or value <= 0
        for value in (args.max_forward_speed, args.max_lateral_speed, args.max_yaw_rate)
    ):
        parser.error("Velocity limits must be finite and positive.")

    log_dir = f"logs/{args.exp_name}"
    checkpoint_path = None
    if args.resume:
        if args.ckpt is None:
            checkpoints = [
                (int(path.stem[6:]), path)
                for path in Path(log_dir).glob("model_*.pt")
                if path.stem[6:].isdigit()
            ]
            if not checkpoints:
                parser.error(f"No checkpoints found in {log_dir}.")
            _, checkpoint_path = max(checkpoints)
        else:
            checkpoint_path = Path(log_dir) / f"model_{args.ckpt}.pt"
        if not checkpoint_path.is_file() or not Path(log_dir, "cfgs.pkl").is_file():
            parser.error(f"Resume requires a checkpoint and cfgs.pkl in {log_dir}.")
        with open(f"{log_dir}/cfgs.pkl", "rb") as f:
            env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(f)
        print(f"Resuming {checkpoint_path} with saved configuration (including velocity ranges).")
    else:
        if os.path.exists(log_dir):
            parser.error(f"Experiment {log_dir} already exists. Use --resume or a new experiment name.")
        env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
        for name, limit in (
            ("lin_vel_x_range", args.max_forward_speed),
            ("lin_vel_y_range", args.max_lateral_speed),
            ("ang_vel_range", args.max_yaw_rate),
        ):
            command_cfg[name] = [-limit, limit]
        train_cfg = get_train_cfg(args.exp_name)
        os.makedirs(log_dir)
        with open(f"{log_dir}/cfgs.pkl", "wb") as f:
            pickle.dump([env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg], f)

    # Training throughput comes from the 4096 parallel environments, which need a GPU.
    gs.init(backend=gs.gpu, precision="32", logging_level="warning", seed=args.seed, performance_mode=True)

    env = Go2Env(
        num_envs=args.num_envs, env_cfg=env_cfg, obs_cfg=obs_cfg, reward_cfg=reward_cfg, command_cfg=command_cfg
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    if checkpoint_path is not None:
        runner.load(str(checkpoint_path))

    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    main()

"""
# training
python examples/locomotion/go2_train.py
"""
