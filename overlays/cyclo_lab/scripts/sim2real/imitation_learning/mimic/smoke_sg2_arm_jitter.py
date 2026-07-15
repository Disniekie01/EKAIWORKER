#!/usr/bin/env python3
# Copyright 2026 EYKOREA
"""Validate SG2 arm-jitter fix (DiffIK vs joint-restore fight).

Offline: wiring + quat continuity.

In-sim (--sim): after one DiffIK step, displace arms +0.2 rad, then compare
``reset_to`` behaviors:

  OLD = restore recorded joint_position  → arms snap back (~0.20 rad)
  FIX = pin joints to live pose          → arms stay displaced (~0)

Pass when FIX Δ << OLD Δ.

  cd /workspace/cyclo_lab
  ./third_party/IsaacLab/_isaac_sim/python.sh \
    scripts/sim2real/imitation_learning/mimic/smoke_sg2_arm_jitter.py \
    --sim --device cuda --headless --enable_cameras
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[4]
DEFAULT_IK = ROOT / "datasets/ffw_sg2_l_table_ik.hdf5"
DEFAULT_GEN = ROOT / "datasets/ffw_sg2_l_table_generate.hdf5"

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


def _ensure_eef_quat_continuity(eef_pose: torch.Tensor) -> torch.Tensor:
    out = eef_pose.clone()
    quat = out[:, 3:7]
    for i in range(1, quat.shape[0]):
        if torch.dot(quat[i], quat[i - 1]) < 0:
            quat[i] *= -1
    return out


def _with_live_robot_joints(env, state: dict) -> dict:
    artic = state.get("articulation")
    if not isinstance(artic, dict) or "robot" not in artic:
        return state
    robot = artic["robot"]
    if not isinstance(robot, dict):
        return state
    robot_asset = env.scene["robot"]
    live_robot = {
        **robot,
        "joint_position": robot_asset.data.joint_pos.clone(),
        "joint_velocity": robot_asset.data.joint_vel.clone(),
    }
    return {**state, "articulation": {**artic, "robot": live_robot}}


def run_offline(ik_path: Path, gen_path: Path) -> None:
    print("=== Offline arm-jitter validation ===", flush=True)
    ann = (ROOT / "scripts/sim2real/imitation_learning/mimic/annotate_demos.py").read_text()
    rep = (ROOT / "scripts/imitation_learning/isaaclab_recorder/replay_demos.py").read_text()
    conv = (ROOT / "scripts/sim2real/imitation_learning/mimic/action_data_converter.py").read_text()
    mimic = (
        ROOT
        / "source/cyclo_lab/cyclo_lab/real_world_tasks/manager_based/FFW_SG2/pick_place/pick_place_mimic_env.py"
    ).read_text()

    check("annotate pins live joints", "_with_live_robot_joints" in ann and "skip_robot_joints" in ann)
    check("replay pins live joints for mimic_ik", "_with_live_robot_joints" in rep and 'skip_robot_joints=(args_cli.action_mode == "mimic_ik")' in rep)
    check("converter enforces quat continuity", "_ensure_eef_quat_continuity" in conv)
    check("converter IK trailing is lift then head", "lift_action,      # 16: lift joint" in conv)
    check("mimic enforces continuous eef quat", "_continuous_eef_quat" in mimic)

    poses = torch.zeros(4, 7)
    poses[:, 3] = 1.0
    poses[2, 3:7] = torch.tensor([-1.0, 0.0, 0.0, 0.0])
    fixed = _ensure_eef_quat_continuity(poses)
    dots = (fixed[1:, 3:7] * fixed[:-1, 3:7]).sum(dim=1)
    check("quat continuity removes double-cover flip", bool(torch.all(dots >= 0)))

    if gen_path.is_file() and gen_path.stat().st_size > 1000:
        import h5py

        with h5py.File(gen_path, "r") as f:
            a = np.array(f["data/demo_0/actions"])
        q = a[:, 11:15]
        flips = int(np.sum(np.sum(q[1:] * q[:-1], axis=1) < 0))
        q2 = q.copy()
        for i in range(1, len(q2)):
            if np.dot(q2[i], q2[i - 1]) < 0:
                q2[i] *= -1
        d = np.linalg.norm(np.diff(a[:, 8:15], axis=0), axis=1)
        a2 = a.copy()
        a2[:, 11:15] = q2
        d2 = np.linalg.norm(np.diff(a2[:, 8:15], axis=0), axis=1)
        check("generate demo has quat flips (legacy)", flips > 0, f"flips={flips}")
        check(
            "continuity cuts generate EE step max >2x",
            float(d2.max()) < float(d.max()) * 0.5,
            f"max {d.max():.3f} -> {d2.max():.3f}",
        )

    if ik_path.is_file() and ik_path.stat().st_size > 1000:
        import h5py

        with h5py.File(ik_path, "r") as f:
            a = np.array(f["data/demo_0/actions"])
        for arm, i0 in (("L", 0), ("R", 8)):
            q = a[:, i0 + 3 : i0 + 7]
            flips = int(np.sum(np.sum(q[1:] * q[:-1], axis=1) < 0))
            check(f"IK demo_0 {arm} quat flips==0", flips == 0, f"flips={flips}")


def _state_to_device(state: dict, device) -> dict:
    output: dict = {}
    for asset_type, assets in state.items():
        output[asset_type] = {}
        for asset_name, fields in assets.items():
            output[asset_type][asset_name] = {
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in fields.items()
            }
    return output


def _arm_joint_indices(robot) -> list[int]:
    names = list(robot.data.joint_names)
    return [i for i, n in enumerate(names) if n.startswith("arm_l_joint") or n.startswith("arm_r_joint")]


def _displace_then_restore(env, step_state, *, pin_live: bool, displace: float = 0.20) -> float:
    robot = env.scene["robot"]
    arm_idx = _arm_joint_indices(robot)
    pos = robot.data.joint_pos.clone()
    vel = robot.data.joint_vel.clone()
    pos[0, arm_idx] = pos[0, arm_idx] + displace
    env_ids = torch.tensor([0], device=env.device)
    robot.write_joint_state_to_sim(pos, vel, env_ids=env_ids)
    robot.set_joint_position_target(pos, env_ids=env_ids)
    env.sim.forward()
    j_before = robot.data.joint_pos[0, arm_idx].detach().clone()

    state = _state_to_device(step_state, env.device)
    if pin_live:
        state = _with_live_robot_joints(env, state)
    env.scene.reset_to(state, env_ids=None, is_relative=True)
    env.sim.forward()
    j_after = robot.data.joint_pos[0, arm_idx].detach().clone()
    return float((j_before - j_after).abs().mean().item())


def _load_ik_episode(path: Path, device: str):
    from isaaclab.utils.datasets import HDF5DatasetFileHandler

    handler = HDF5DatasetFileHandler()
    handler.open(str(path))
    names = list(handler.get_episode_names())
    if not names:
        raise RuntimeError(f"No demos in {path}")
    episode = handler.load_episode(names[0], device)
    handler.close()
    return episode


def run_sim(task: str, ik_path: Path, device: str) -> None:
    import gymnasium as gym
    import cyclo_lab  # noqa: F401
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab_tasks.utils import parse_env_cfg

    print("=== In-sim arm-jitter restore validation ===", flush=True)
    if not ik_path.is_file() or ik_path.stat().st_size < 1000:
        check("IK dataset present", False, f"missing {ik_path}")
        return

    episode = _load_ik_episode(ik_path, device)
    actions = episode.data["actions"]
    if not isinstance(actions, torch.Tensor):
        actions = torch.as_tensor(actions)
    st0 = episode.get_state(0)
    check("loaded IK episode state0", st0 is not None)
    if st0 is None:
        return

    env_cfg = parse_env_cfg(task, device=device, num_envs=1)
    if hasattr(env_cfg, "init_action_cfg"):
        env_cfg.init_action_cfg("mimic_ik")
    for term in ("time_out", "success", "object_dropped"):
        if hasattr(env_cfg.terminations, term):
            setattr(env_cfg.terminations, term, None)

    print(f"[sim] creating env task={task}...", flush=True)
    env: ManagerBasedRLEnv = gym.make(task, cfg=env_cfg).unwrapped
    env.sim.reset()
    env.reset()
    init = episode.data.get("initial_state")
    if init is not None:
        env.reset_to(init, None, is_relative=True)
    env.step(actions[0:1].to(env.device))
    print("[sim] seeded + 1 DiffIK step; measuring...", flush=True)

    # FIX first (no snap), then OLD (snaps back) — no second env.step needed.
    snap_fix = _displace_then_restore(env, st0, pin_live=True)
    print(f"[sim] FIX (pin live joints) mean |Δ| = {snap_fix:.5f} rad", flush=True)
    snap_old = _displace_then_restore(env, st0, pin_live=False)
    print(f"[sim] OLD (restore recorded) mean |Δ| = {snap_old:.5f} rad", flush=True)

    check("FIX leaves arms displaced (<0.02 rad)", snap_fix < 0.02, f"Δ={snap_fix:.5f}")
    check("OLD snaps arms back (≥0.10 rad)", snap_old > 0.10, f"Δ={snap_old:.5f}")
    check(
        "FIX << OLD (≥5x smaller)",
        snap_fix < snap_old / 5.0,
        f"old={snap_old:.5f} fix={snap_fix:.5f}",
    )
    env.close()


def main() -> int:
    if "--sim" in sys.argv:
        import multiprocessing

        if multiprocessing.get_start_method() != "spawn":
            multiprocessing.set_start_method("spawn", force=True)
        from isaaclab.app import AppLauncher

        launcher_parser = argparse.ArgumentParser(add_help=False)
        launcher_parser.add_argument("--sim", action="store_true")
        launcher_parser.add_argument(
            "--task",
            type=str,
            default="Cyclo-Real-Mimic-Pick-Place-LTable-FFW-SG2-v0",
        )
        launcher_parser.add_argument("--ik-file", type=str, default=str(DEFAULT_IK))
        launcher_parser.add_argument("--gen-file", type=str, default=str(DEFAULT_GEN))
        AppLauncher.add_app_launcher_args(launcher_parser)
        args_cli, _ = launcher_parser.parse_known_args()
        app_launcher = AppLauncher(vars(args_cli))
        simulation_app = app_launcher.app
        try:
            run_offline(Path(args_cli.ik_file), Path(args_cli.gen_file))
            run_sim(args_cli.task, Path(args_cli.ik_file), args_cli.device)
            print(f"\nResult: {PASSED} passed, {FAILED} failed", flush=True)
            code = 1 if FAILED else 0
        finally:
            simulation_app.close()
        raise SystemExit(code)

    parser = argparse.ArgumentParser(description="Validate SG2 arm-jitter fix.")
    parser.add_argument("--ik-file", type=str, default=str(DEFAULT_IK))
    parser.add_argument("--gen-file", type=str, default=str(DEFAULT_GEN))
    args, _ = parser.parse_known_args()
    run_offline(Path(args.ik_file), Path(args.gen_file))
    print(f"\nResult: {PASSED} passed, {FAILED} failed", flush=True)
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
