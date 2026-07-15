#!/usr/bin/env python3
# Copyright 2026 EYKOREA
"""In-sim smoke: FFW-SG2 L-table episode R reset (no headset).

Exercises the real Isaac env + FFWSG2Sdk reset path:
  1) Displace arms / grippers / lift / base from home
  2) Pollute DDS teleop caches (as leftover VR would)
  3) Run the R sequence: _request_pose_reset -> env.reset -> teleop.reset
     -> publish_observations (root restore)
  4) Assert joints/base near home and caches no longer override live state

Run inside cyclo_lab:
  cd /workspace/cyclo_lab
  ./third_party/IsaacLab/isaaclab.sh -p \\
    scripts/sim2real/imitation_learning/dds_sdk/smoke_episode_reset.py \\
    --task Cyclo-Real-Pick-Place-LTable-FFW-SG2-v0 --device cuda --headless
"""

from __future__ import annotations

import argparse
import math
import multiprocessing
import sys
import time

if multiprocessing.get_start_method() != "spawn":
    multiprocessing.set_start_method("spawn", force=True)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="In-sim smoke for episode R reset.")
parser.add_argument(
    "--task",
    type=str,
    default="Cyclo-Real-Pick-Place-LTable-FFW-SG2-v0",
    help="Gym task id",
)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument(
    "--hold-s",
    type=float,
    default=3.0,
    help="Seconds to hold/step after each reset so the GUI is visible.",
)
parser.add_argument(
    "--cycles",
    type=int,
    default=3,
    help="How many displace→R-reset cycles to run for visual confirmation.",
)
parser.add_argument(
    "--dirty-s",
    type=float,
    default=2.5,
    help="Seconds to show the displaced/dirty pose before each R reset.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(vars(args_cli))
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
import types  # noqa: E402
from unittest.mock import MagicMock  # noqa: E402

# Headless smoke: avoid pynput needing a real X display for the keyboard listener.
_pynput = types.ModuleType("pynput")
_pynput_keyboard = types.ModuleType("pynput.keyboard")
_pynput_keyboard.Listener = MagicMock(return_value=MagicMock(start=MagicMock(), stop=MagicMock()))
_pynput.keyboard = _pynput_keyboard
sys.modules["pynput"] = _pynput
sys.modules["pynput.keyboard"] = _pynput_keyboard

import cyclo_lab  # noqa: F401, E402
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

sys.path.append("/workspace/cyclo_lab/scripts/sim2real/imitation_learning")
from dds_sdk.ffw_sg2_sdk import FFWSG2Sdk  # noqa: E402

# EventCfg authored home (lift exact; arms may still see tiny residual after writes).
HOME = {
    "arm_l_joint1": 0.75,
    "arm_l_joint4": -2.30,
    "arm_r_joint1": 0.75,
    "arm_r_joint4": -2.30,
    "head_joint1": 0.549,
    "lift_joint": 0.0365,
    "gripper_l_joint1": 0.0,
    "gripper_r_joint1": 0.0,
}

PASSED = 0
FAILED = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASSED, FAILED
    if cond:
        PASSED += 1
        print(f"  PASS  {name}")
    else:
        FAILED += 1
        print(f"  FAIL  {name}" + (f" — {detail}" if detail else ""))


def _disable_reset_noise(env_cfg) -> None:
    """Prefer deterministic event terms when present; never delete required ones."""
    events = getattr(env_cfg, "events", None)
    if events is None:
        return
    # Keep set_robot_joint_pose + reset_scene_to_default. Soften arm noise for asserts.
    term = getattr(events, "randomize_ffw_sg2_joint_state", None)
    if term is not None and hasattr(term, "params"):
        term.params["std"] = 0.0
        term.params["mean"] = 0.0


def _joint_map(robot) -> dict[str, float]:
    names = list(robot.data.joint_names)
    pos = robot.data.joint_pos[0].detach().cpu().tolist()
    return {n: float(v) for n, v in zip(names, pos)}


def _set_joint_targets(robot, values: dict[str, float], device) -> None:
    names = list(robot.data.joint_names)
    pos = robot.data.joint_pos.clone()
    vel = torch.zeros_like(pos)
    for name, val in values.items():
        if name not in names:
            continue
        idx = names.index(name)
        pos[0, idx] = float(val)
    env_ids = torch.tensor([0], device=device)
    robot.write_joint_state_to_sim(pos, vel, env_ids=env_ids)
    robot.set_joint_position_target(pos, env_ids=env_ids)


def _displace_root(robot, device, dx=0.25, dy=0.15, dyaw=0.7) -> None:
    root = robot.data.root_state_w[0:1, 0:7].clone()
    # Isaac Lab root pose: (x, y, z, qw, qx, qy, qz)
    x, y, z, qw0, qx0, qy0, qz0 = [float(v) for v in root[0].tolist()]
    qw = math.cos(dyaw / 2.0)
    qz = math.sin(dyaw / 2.0)
    # q_new = q_delta(z) * q_old
    qw1 = qw * qw0 - qz * qz0
    qx1 = qw * qx0 + qz * qy0
    qy1 = qw * qy0 - qz * qx0
    qz1 = qw * qz0 + qz * qw0
    root[0, 0] = x + dx
    root[0, 1] = y + dy
    root[0, 3] = qw1
    root[0, 4] = qx1
    root[0, 5] = qy1
    root[0, 6] = qz1
    env_ids = torch.tensor([0], device=device)
    robot.write_root_pose_to_sim(root, env_ids=env_ids)
    robot.write_root_velocity_to_sim(torch.zeros(1, 6, device=device), env_ids=env_ids)


def _step_hold(env, teleop, joint_map: dict[str, float], seconds: float) -> None:
    """Step physics while holding a joint pose so the viewport stays live."""
    if seconds <= 0 or not simulation_app.is_running():
        return
    hold_action = torch.tensor(
        [[joint_map.get(n, 0.0) for n in teleop.joint_names]],
        device=env.device,
        dtype=torch.float32,
    )
    t0 = time.time()
    while time.time() - t0 < seconds and simulation_app.is_running():
        try:
            env.step(hold_action)
        except Exception:
            env.sim.render()


def _pollute_teleop_caches(teleop) -> None:
    with teleop.lock:
        teleop._accept_teleop_cmds = True
        teleop.left_arm_trajectory_cmd = {
            "arm_l_joint1": 0.05,
            "arm_l_joint4": -0.50,
        }
        teleop.right_arm_trajectory_cmd = {
            "arm_r_joint1": 0.08,
            "arm_r_joint4": -0.55,
        }
        teleop._left_gripper_cmd = 0.95
        teleop._right_gripper_cmd = 0.90
        teleop._left_gripper_smooth = 0.90
        teleop.head_joint_trajectory_cmd = {"head_joint1": 0.05, "head_joint2": 0.0}
        teleop.lift_joint_trajectory_cmd = {"lift_joint": -0.20}
        teleop._started = True
        teleop._reset_state = False


def _run_r_reset(teleop, env, home_root, device) -> None:
    teleop._started = False
    teleop._reset_state = True
    teleop._request_pose_reset()
    env.reset()
    teleop.reset()
    teleop._pending_reset_pose = True
    if teleop._home_root_pose is None:
        teleop._home_root_pose = home_root.to(device)
    teleop.publish_observations()


def main() -> int:
    print("=== In-sim episode R reset smoke ===", flush=True)
    print(f"task={args_cli.task} device={args_cli.device}", flush=True)

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    if hasattr(env_cfg, "init_action_cfg"):
        env_cfg.init_action_cfg("record")
    env_cfg.seed = args_cli.seed
    if hasattr(env_cfg.terminations, "time_out"):
        env_cfg.terminations.time_out = None
    if hasattr(env_cfg.terminations, "success"):
        env_cfg.terminations.success = None
    _disable_reset_noise(env_cfg)
    print("[smoke] creating env...", flush=True)

    env: ManagerBasedRLEnv = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    print("[smoke] env ready; creating FFWSG2Sdk...", flush=True)
    teleop = FFWSG2Sdk(env, mode="record")
    print("[smoke] teleop ready", flush=True)

    env.reset()
    teleop.reset()
    teleop._accept_teleop_cmds = True
    teleop.publish_observations()
    robot = env.scene["robot"]
    device = env.device

    home_before = _joint_map(robot)
    home_root = robot.data.root_state_w[0:1, 0:7].clone()
    print(
        f"[smoke] initial lift={home_before.get('lift_joint'):.4f} "
        f"arm_l1={home_before.get('arm_l_joint1'):.4f} "
        f"grip_l={home_before.get('gripper_l_joint1'):.4f}",
        flush=True,
    )
    home_ref = {
        k: home_before[k]
        for k in (
            "arm_l_joint1",
            "arm_l_joint4",
            "arm_r_joint1",
            "arm_r_joint4",
            "head_joint1",
            "lift_joint",
            "gripper_l_joint1",
            "gripper_r_joint1",
        )
        if k in home_before
    }

    dirty_joints = {
        "arm_l_joint1": 0.10,
        "arm_l_joint4": -1.00,
        "arm_r_joint1": 0.20,
        "arm_r_joint4": -1.10,
        "gripper_l_joint1": 0.80,
        "gripper_r_joint1": 0.75,
        "lift_joint": -0.15,
        "head_joint1": 0.10,
    }
    cycles = max(1, int(args_cli.cycles))
    dirty_s = float(args_cli.dirty_s)
    hold_s = float(args_cli.hold_s)
    home_xy = home_root[0, 0:3].detach().cpu()
    after = dict(home_ref)

    for cycle in range(1, cycles + 1):
        print(f"\n=== Cycle {cycle}/{cycles}: displace ===", flush=True)
        _set_joint_targets(robot, dirty_joints, device)
        _displace_root(robot, device, dx=0.20 + 0.03 * cycle, dy=0.12, dyaw=0.55)
        env.sim.forward()
        env.scene.update(dt=env.physics_dt)
        displaced = _joint_map(robot)
        displaced_root = robot.data.root_state_w[0, 0:3].detach().cpu()
        _pollute_teleop_caches(teleop)

        if cycle == 1:
            check(
                "pre-R lift moved off home",
                abs(displaced["lift_joint"] - home_ref["lift_joint"]) > 0.05,
                f"lift={displaced['lift_joint']:.4f}",
            )
            check(
                "pre-R base moved off home",
                float(torch.linalg.norm(displaced_root - home_xy)) > 0.05,
                f"root_xy=({displaced_root[0]:.3f},{displaced_root[1]:.3f})",
            )
            live_dirty = teleop._build_live_joint_state()
            check(
                "dirty caches override live lift before R",
                abs(live_dirty.get("lift_joint", 99) - (-0.20)) < 1e-6,
                f"got={live_dirty.get('lift_joint')}",
            )
            check(
                "dirty caches override live left grip before R",
                live_dirty.get("gripper_l_joint1", 0.0) > 0.5,
                f"got={live_dirty.get('gripper_l_joint1')}",
            )

        print(f"[smoke] showing dirty pose for {dirty_s:.1f}s...", flush=True)
        _step_hold(env, teleop, displaced, dirty_s)

        print(f"[smoke] cycle {cycle}: issuing R reset...", flush=True)
        _run_r_reset(teleop, env, home_root, device)
        after = _joint_map(robot)
        after_root = robot.data.root_state_w[0, 0:3].detach().cpu()

        for jn, target in home_ref.items():
            tol = 0.08 if jn.startswith("arm_") or jn.startswith("head_") else 0.03
            got = after.get(jn)
            check(
                f"cycle{cycle} post-R {jn} near home ({target:.4f})",
                got is not None and abs(got - target) <= tol,
                f"got={got}",
            )
        check(
            f"cycle{cycle} post-R base near home",
            float(torch.linalg.norm(after_root - home_xy)) < 0.05,
            f"delta={float(torch.linalg.norm(after_root - home_xy)):.4f}",
        )
        check(f"cycle{cycle} arm caches cleared", teleop.left_arm_trajectory_cmd is None)
        check(f"cycle{cycle} lift cache cleared", teleop.lift_joint_trajectory_cmd is None)
        check(
            f"cycle{cycle} gripper caches cleared",
            teleop._left_gripper_cmd is None and teleop._right_gripper_cmd is None,
        )
        check(f"cycle{cycle} teleop gated off", teleop._accept_teleop_cmds is False)

        print(f"[smoke] holding home pose for {hold_s:.1f}s...", flush=True)
        _step_hold(env, teleop, after, hold_s)

    # Extra gate / B re-arm checks once after the last cycle
    with teleop.lock:
        if teleop._accept_teleop_cmds:
            teleop.lift_joint_trajectory_cmd = teleop.lift_joint_trajectory_cmd or {}
            teleop.lift_joint_trajectory_cmd.update({"lift_joint": -0.3})
    check("gated late inject ignored", teleop.lift_joint_trajectory_cmd is None)

    teleop._accept_teleop_cmds = True
    teleop._started = True
    live_clean = teleop._build_live_joint_state()
    check(
        "post-B live lift follows sim home (not stale VR)",
        abs(live_clean.get("lift_joint", 99) - after["lift_joint"]) < 1e-5,
        f"live={live_clean.get('lift_joint')} sim={after['lift_joint']}",
    )
    check(
        "post-B live left arm follows sim (not stale VR)",
        abs(live_clean.get("arm_l_joint1", 99) - after["arm_l_joint1"]) < 1e-5,
        f"live={live_clean.get('arm_l_joint1')} sim={after['arm_l_joint1']}",
    )

    print(f"\nResult: {PASSED} passed, {FAILED} failed", flush=True)

    teleop.running = False
    try:
        env.close()
    except Exception as exc:
        print(f"[smoke] env.close warning: {exc}", flush=True)
    print("[smoke] done", flush=True)
    return 1 if FAILED else 0


if __name__ == "__main__":
    code = 1
    try:
        code = main()
    except BaseException as exc:
        import traceback

        print(f"[smoke] FATAL: {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()
        code = 1
    finally:
        try:
            simulation_app.close()
        except Exception as close_exc:
            print(f"[smoke] close warning: {close_exc}", flush=True)
    raise SystemExit(code)
