# Copyright 2025 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
# Play / evaluate a trained *LeRobot* policy (ACT / VQ-BeT) in an Isaac Lab
# environment. This mirrors scripts/imitation_learning/robomimic/play.py but
# swaps the robomimic policy for a LeRobot policy:
#   robomimic:  FileUtils.policy_from_checkpoint(...) ; actions = policy(obs)
#   lerobot  :  ACTPolicy.from_pretrained(...)        ; action  = policy.select_action(obs)
#
# Run inside the container with the Isaac Sim python (`python`), after
# installing lerobot into that interpreter:
#     $ISAACLAB_PATH/_isaac_sim/python.sh -m pip install --no-deps lerobot==0.3.3
#
#     python scripts/sim2real/imitation_learning/inference/eval_policy_sim.py \
#         --task Cyclo-Real-Pick-Place-LTable-FFW-SG2-v0 \
#         --robot_type FFW_SG2 \
#         --policy JSHNSL/act_ffw_ltable --policy_type act \
#         --num_rollouts 10 --enable_cameras

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate a LeRobot policy in an Isaac Lab environment.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O.")
parser.add_argument("--task", type=str, required=True, help="Name of the task.")
parser.add_argument("--robot_type", type=str, default="FFW_SG2", choices=["OMY", "FFW_SG2", "FFW_SH5"])
parser.add_argument("--policy", type=str, required=True, help="HF repo_id or local LeRobot checkpoint dir.")
parser.add_argument("--policy_type", type=str, default="act", choices=["act", "vqbet", "diffusion"])
parser.add_argument("--horizon", type=int, default=800, help="Step horizon of each rollout.")
parser.add_argument("--num_rollouts", type=int, default=10, help="Number of rollouts.")
parser.add_argument("--seed", type=int, default=101, help="Random seed.")
parser.add_argument(
    "--action_mode", type=str, default="inference", choices=["record", "inference", "mimic_ik"],
    help="Action space mode for envs that define init_action_cfg.",
)
parser.add_argument("--remap_ffw_sg2_actions", action="store_true", default=False,
                    help="Force head/lift action remap for FFW-SG2 (19-dim).")
parser.add_argument("--no_remap_ffw_sg2_actions", action="store_true", default=False,
                    help="Disable automatic FFW-SG2 head/lift action remap.")
parser.add_argument("--scripted_l_motion", action="store_true", default=False,
                    help="Enable scripted base L-motion during play (policy controls manipulation).")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

from isaaclab_tasks.utils import parse_env_cfg

import cyclo_lab  # noqa: F401  (registers Cyclo-* tasks)
from cyclo_lab.real_world_tasks.manager_based.FFW_SG2.pick_place_l_table.ltable_kinematic_l_motion import (
    LTableKinematicLMotion,
)

_FFW_SG2_JOINT_ACTION_DIM = 19

# Camera key(s) per robot (matches ROBOT_CONFIGS in isaaclab2lerobot.py)
CAMERA_KEYS = {
    "OMY": ["cam_wrist", "cam_top"],
    "FFW_SG2": ["cam_head"],
    "FFW_SH5": ["cam_head"],
}


def remap_ffw_sg2_joint_actions_to_env(actions):
    """Swap head/lift slots from joint_pos_target layout to ActionManager layout.

    Training HDF5 (joint convert) stores actions in observation order:
        [..., head_joint1, lift_joint, head_joint2]
    while ActionManager applies:
        [..., lift_joint, head_joint1, head_joint2]
    """
    if actions.shape[-1] != _FFW_SG2_JOINT_ACTION_DIM:
        return actions
    mapped = actions.clone()
    mapped[..., 16] = actions[..., 17]
    mapped[..., 17] = actions[..., 16]
    return mapped


def load_policy(policy_type: str, path: str, device: str):
    if policy_type == "act":
        from lerobot.policies.act.modeling_act import ACTPolicy
        policy = ACTPolicy.from_pretrained(path)
    elif policy_type == "vqbet":
        from lerobot.policies.vqbet.modeling_vqbet import VQBeTPolicy
        policy = VQBeTPolicy.from_pretrained(path)
    elif policy_type == "diffusion":
        from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
        policy = DiffusionPolicy.from_pretrained(path)
    else:
        raise ValueError(f"Unknown policy_type: {policy_type}")
    policy.eval()
    policy.to(device)
    return policy


def build_lerobot_obs(policy_obs: dict, robot_type: str, device: str):
    """Assemble the LeRobot policy input from the env's 'policy' obs group.

    Uses the same terms the dataset was built from: 19-dim `joint_pos` and the
    `cam_head` RGB image (NOT the raw full-articulation joint state).
    """
    state = policy_obs["joint_pos"].float()
    if state.ndim == 1:
        state = state.unsqueeze(0)  # (19,) -> (1, 19)
    obs = {"observation.state": state.to(device)}

    for cam_key in CAMERA_KEYS[robot_type]:
        rgb = policy_obs[cam_key].float()          # (H, W, 3) or (1, H, W, 3), 0..255
        if rgb.ndim == 3:
            rgb = rgb.unsqueeze(0)
        img = (rgb.permute(0, 3, 1, 2) / 255.0).clip(0.0, 1.0)  # (1, 3, H, W)
        obs[f"observation.images.{cam_key}"] = img.to(device)

    return obs


def rollout(policy, env, success_term, horizon, device, remap, robot_type, trial):
    """Run a single rollout; return True if the success term fires."""
    if hasattr(policy, "reset"):
        policy.reset()  # clear ACT action-chunk queue between episodes
    obs_dict, _ = env.reset()
    l_motion_ctrl = LTableKinematicLMotion(env) if args_cli.scripted_l_motion else None
    lm_prev = None  # last (dual_grasped, active, done) for change-logging

    for i in range(horizon):
        if not simulation_app.is_running():
            break
        with torch.inference_mode():
            obs = build_lerobot_obs(obs_dict["policy"], robot_type, device)
            action = policy.select_action(obs)  # (1, 19) torch tensor

        if trial == 0 and i < 5:
            jp = obs["observation.state"][0]
            a = action[0]
            print(f"[dbg s{i}] state[min/max]={jp.min():.3f}/{jp.max():.3f} | "
                  f"action[min/max]={a.min():.3f}/{a.max():.3f} | "
                  f"|action-state|max={(a - jp).abs().max():.4f}", flush=True)

        if remap:
            action = remap_ffw_sg2_joint_actions_to_env(action)
        action = action.to(device).view(1, env.action_space.shape[1])

        obs_dict, _, terminated, truncated, _ = env.step(action)

        if l_motion_ctrl is not None:
            env_ids = torch.arange(env.num_envs, device=env.device)
            l_motion_ctrl.step_interval(env_ids)
            # --- debug: does the scripted L-motion actually trigger? ---
            try:
                grasped = bool(l_motion_ctrl._dual_grasped_mask(env_ids).any())
            except Exception:
                grasped = None
            active = bool(l_motion_ctrl.is_active())
            done = bool(l_motion_ctrl.is_done())
            lm_state = (grasped, active, done)
            if lm_state != lm_prev:
                print(f"[Lmotion t{trial} s{i}] dual_grasped={grasped} "
                      f"active={active} done={done}", flush=True)
                lm_prev = lm_state
            # grasp-condition components (trial 0, every 50 steps) -> see WHY latch fails
            if trial == 0 and i % 50 == 0:
                try:
                    rb = env.scene["robot"]
                    li = rb.joint_names.index("gripper_l_joint1")
                    ri = rb.joint_names.index("gripper_r_joint1")
                    gl = rb.data.joint_pos[0, li].item()
                    gr = rb.data.joint_pos[0, ri].item()
                    bp = env.scene["cardboard_box"].data.root_pos_w[0]
                    lp = env.scene["left_eef"].data.target_pos_w[0, 0, :]
                    rp = env.scene["right_eef"].data.target_pos_w[0, 0, :]
                    dl = (bp - lp).norm().item()
                    dr = (bp - rp).norm().item()
                    dm = (bp - (lp + rp) * 0.5).norm().item()
                    print(f"[grasp s{i}] grip L/R={gl:.2f}/{gr:.2f} (need>=0.2) | "
                          f"box->L/R/mid={dl:.2f}/{dr:.2f}/{dm:.2f} (need<0.18)", flush=True)
                except Exception as e:
                    print(f"[grasp s{i}] debug err: {e}", flush=True)

        if bool(success_term.func(env, **success_term.params)[0]):
            return True
        elif bool(terminated[0]) or bool(truncated[0]):
            return False

    return False


def main():
    # Auto-enable FFW-SG2 head/lift remap (same logic as robomimic play.py).
    remap = False
    if args_cli.no_remap_ffw_sg2_actions:
        remap = False
    elif args_cli.remap_ffw_sg2_actions:
        remap = True
    elif args_cli.task and "FFW-SG2" in args_cli.task:
        remap = True

    device = args_cli.device

    # --- environment (mirror robomimic play.py setup) ---
    env_cfg = parse_env_cfg(args_cli.task, device=device, num_envs=1, use_fabric=not args_cli.disable_fabric)
    if hasattr(env_cfg, "init_action_cfg"):
        env_cfg.init_action_cfg(args_cli.action_mode)
    env_cfg.observations.policy.concatenate_terms = False  # obs as a dict of terms
    env_cfg.terminations.time_out = None                   # run the full horizon
    env_cfg.recorders = None                               # no HDF5 recorder
    success_term = env_cfg.terminations.success            # keep the success checker
    env_cfg.terminations.success = None                    # but don't auto-reset on it

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    torch.manual_seed(args_cli.seed)
    env.seed(args_cli.seed)

    # --- policy ---
    print(f"[INFO] Loading {args_cli.policy_type} policy from {args_cli.policy} ...", flush=True)
    policy = load_policy(args_cli.policy_type, args_cli.policy, device)
    print(f"[INFO] Policy loaded. remap_ffw_sg2={remap}. Starting rollouts on {device}", flush=True)

    results = []
    for trial in range(args_cli.num_rollouts):
        print(f"[INFO] Starting trial {trial}", flush=True)
        ok = rollout(policy, env, success_term, args_cli.horizon, device, remap, args_cli.robot_type, trial)
        results.append(ok)
        print(f"[INFO] Trial {trial}: success={ok}\n", flush=True)

    n = len(results)
    print(f"\n[RESULT] {args_cli.policy_type} @ {args_cli.policy}", flush=True)
    print(f"Successful trials: {results.count(True)} / {n}", flush=True)
    print(f"Success rate: {results.count(True) / max(1, n):.2%}", flush=True)

    env.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user, shutting down...")
    finally:
        simulation_app.close()
