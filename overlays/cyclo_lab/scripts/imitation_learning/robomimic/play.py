# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play and evaluate a trained policy from robomimic.

This script loads a robomimic policy and plays it in an Isaac Lab environment.

Args:
    task: Name of the environment.
    checkpoint: Path to the robomimic policy checkpoint.
    horizon: If provided, override the step horizon of each rollout.
    num_rollouts: If provided, override the number of rollouts.
    seed: If provided, overeride the default random seed.
    norm_factor_min: If provided, minimum value of the action space normalization factor.
    norm_factor_max: If provided, maximum value of the action space normalization factor.
"""

"""Launch Isaac Sim Simulator first."""


import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Evaluate robomimic policy for Isaac Lab environment.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--checkpoint", type=str, default=None, help="Pytorch model checkpoint to load.")
parser.add_argument("--horizon", type=int, default=800, help="Step horizon of each rollout.")
parser.add_argument("--num_rollouts", type=int, default=1, help="Number of rollouts.")
parser.add_argument("--seed", type=int, default=101, help="Random seed.")
parser.add_argument(
    "--norm_factor_min", type=float, default=None, help="Optional: minimum value of the normalization factor."
)
parser.add_argument(
    "--norm_factor_max", type=float, default=None, help="Optional: maximum value of the normalization factor."
)
parser.add_argument(
    "--action_mode",
    type=str,
    default="inference",
    choices=["record", "inference", "mimic_ik"],
    help="Action space mode for envs that define init_action_cfg (joint vs IK).",
)
parser.add_argument(
    "--remap_ffw_sg2_actions",
    action="store_true",
    default=False,
    help="Swap head/lift action indices for FFW-SG2 joint HDF5 datasets (19-dim).",
)
parser.add_argument(
    "--no_remap_ffw_sg2_actions",
    action="store_true",
    default=False,
    help="Disable automatic FFW-SG2 head/lift action remapping.",
)
parser.add_argument(
    "--scripted_l_motion",
    action="store_true",
    default=False,
    help="Enable scripted base L-motion during play (policy controls manipulation).",
)
parser.add_argument("--enable_pinocchio", default=False, action="store_true", help="Enable Pinocchio.")


# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

if args_cli.enable_pinocchio:
    # Import pinocchio before AppLauncher to force the use of the version installed by IsaacLab and not the one installed by Isaac Sim
    # pinocchio is required by the Pink IK controllers and the GR1T2 retargeter
    import pinocchio  # noqa: F401

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import copy
import gymnasium as gym
import torch

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils

if args_cli.enable_pinocchio:
    import isaaclab_tasks.manager_based.manipulation.pick_place  # noqa: F401

from isaaclab_tasks.utils import parse_env_cfg

import cyclo_lab  # noqa: F401
from cyclo_lab.real_world_tasks.manager_based.FFW_SG2.pick_place_l_table.ltable_kinematic_l_motion import (
    LTableKinematicLMotion,
)

_FFW_SG2_JOINT_ACTION_DIM = 19


def remap_ffw_sg2_joint_actions_to_env(actions):
    """Swap head/lift slots from observation layout to ActionManager layout.

    Observation / legacy joint-HDF5 order:
      [..., head_joint1(16), lift_joint(17), head_joint2(18)]
    ActionManager / fixed joint-convert order:
      [..., lift_joint(16), head_joint1(17), head_joint2(18)]
    """
    if actions.shape[-1] != _FFW_SG2_JOINT_ACTION_DIM:
        return actions
    mapped = actions.copy()
    mapped[..., 16] = actions[..., 17]
    mapped[..., 17] = actions[..., 16]
    return mapped


def actions_look_like_obs_head_lift_layout(actions) -> bool:
    """Heuristic: obs layout has head1 @16 (higher mean) and lift @17 (lower mean)."""
    import numpy as np

    arr = np.asarray(actions)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.shape[-1] != _FFW_SG2_JOINT_ACTION_DIM:
        return False
    return float(arr[..., 16].mean()) > float(arr[..., 17].mean())


def rollout(policy, env, success_term, horizon, device):
    """Perform a single rollout of the policy in the environment.

    Args:
        policy: The robomimicpolicy to play.
        env: The environment to play in.
        horizon: The step horizon of each rollout.
        device: The device to run the policy on.

    Returns:
        terminated: Whether the rollout terminated.
        traj: The trajectory of the rollout.
    """
    policy.start_episode()
    obs_dict, _ = env.reset()
    traj = dict(actions=[], obs=[], next_obs=[])
    l_motion_ctrl = LTableKinematicLMotion(env) if args_cli.scripted_l_motion else None
    l_motion_prev_active = False

    for i in range(horizon):
        # Prepare observations
        obs = copy.deepcopy(obs_dict["policy"])
        for ob in obs:
            obs[ob] = torch.squeeze(obs[ob])

        # Identify image keys from env.cfg
        image_keys = [
            name
            for name, term_cfg in vars(env.cfg.observations.policy).items()
            if hasattr(term_cfg, "func") and term_cfg.func.__name__ == "image" and term_cfg.params.get("data_type") == "rgb"
        ]

        # Process image observations
        for key in image_keys:
            if key in obs_dict["policy"].keys():
                # Convert from chw uint8 to hwc normalized float
                image = torch.squeeze(obs_dict["policy"][key])
                image = image.permute(2, 0, 1).clone().float()
                image = image / 255.0
                image = image.clip(0.0, 1.0)
                obs[key] = image

        traj["obs"].append(obs)

        # Compute actions
        actions = policy(obs)

        if args_cli.remap_ffw_sg2_actions:
            actions = remap_ffw_sg2_joint_actions_to_env(actions)

        # Unnormalize actions
        if args_cli.norm_factor_min is not None and args_cli.norm_factor_max is not None:
            actions = (
                (actions + 1) * (args_cli.norm_factor_max - args_cli.norm_factor_min)
            ) / 2 + args_cli.norm_factor_min

        actions = torch.from_numpy(actions).to(device=device).view(1, env.action_space.shape[1])

        # Apply actions
        obs_dict, _, terminated, truncated, _ = env.step(actions)
        obs = obs_dict["policy"]

        if l_motion_ctrl is not None:
            env_ids = torch.arange(env.num_envs, device=env.device)
            l_motion_ctrl.step_interval(env_ids)
            if l_motion_ctrl.is_active() and not l_motion_prev_active:
                print(f"[INFO] Scripted L-motion started at step {i}")
            if l_motion_ctrl.is_done() and l_motion_prev_active:
                print(f"[INFO] Scripted L-motion completed at step {i}")
            l_motion_prev_active = l_motion_ctrl.is_active()

        # Record trajectory
        traj["actions"].append(actions.tolist())
        traj["next_obs"].append(obs)

        # Check if rollout was successful
        if bool(success_term.func(env, **success_term.params)[0]):
            return True, traj
        elif terminated or truncated:
            return False, traj

    return False, traj


def main():
    """Run a trained policy from robomimic with Isaac Lab environment."""
    if args_cli.no_remap_ffw_sg2_actions:
        args_cli.remap_ffw_sg2_actions = False
    elif args_cli.remap_ffw_sg2_actions:
        pass
    elif args_cli.task and "FFW-SG2" in args_cli.task:
        # Legacy joint HDF5s / policies used observation order [head1, lift, head2].
        # New action_data_converter joint exports use ActionManager order [lift, head1, head2]
        # — pass --no_remap_ffw_sg2_actions for those.
        args_cli.remap_ffw_sg2_actions = True
        print(
            "[INFO] FFW-SG2: enabling head/lift action remap (obs→ActionManager). "
            "Use --no_remap_ffw_sg2_actions if the policy was trained on a fixed "
            "joint-convert export (lift@16, head@17/18)."
        )

    # parse configuration
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1, use_fabric=not args_cli.disable_fabric)

    if hasattr(env_cfg, "init_action_cfg"):
        env_cfg.init_action_cfg(args_cli.action_mode)

    # Set observations to dictionary mode for Robomimic
    env_cfg.observations.policy.concatenate_terms = False

    # Set termination conditions
    env_cfg.terminations.time_out = None

    # Disable recorder
    env_cfg.recorders = None

    # Extract success checking function
    success_term = env_cfg.terminations.success
    env_cfg.terminations.success = None

    # Create environment
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    # Set seed
    torch.manual_seed(args_cli.seed)
    env.seed(args_cli.seed)

    # Acquire device
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)

    # Load policy
    policy, _ = FileUtils.policy_from_checkpoint(ckpt_path=args_cli.checkpoint, device=device, verbose=True)

    # Run policy
    results = []
    for trial in range(args_cli.num_rollouts):
        print(f"[INFO] Starting trial {trial}")
        terminated, traj = rollout(policy, env, success_term, args_cli.horizon, device)
        results.append(terminated)
        print(f"[INFO] Trial {trial}: {terminated}\n")

    print(f"\nSuccessful trials: {results.count(True)}, out of {len(results)} trials")
    print(f"Success rate: {results.count(True) / len(results)}")
    print(f"Trial Results: {results}\n")

    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
