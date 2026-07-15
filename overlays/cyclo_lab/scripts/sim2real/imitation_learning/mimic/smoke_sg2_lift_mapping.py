#!/usr/bin/env python3
# Copyright 2026 EYKOREA
"""Smoke-test SG2 head/lift mapping used by Mimic Datagen.

Two tiers:

  1) Offline (default, ~1s, no Isaac app):
       - Prove IK layout detection + remapping
       - Fail loudly if the old bug (actions[:,16] as lift) returns
       - Prefer-obs episode helper path

  2) In-sim (--sim):
       - Create L-table Mimic env, inject remapped body cmds from an IK demo
       - Assert measured lift_joint tracks the *fixed* command, not head1

Offline:
  cd /workspace/cyclo_lab
  ./third_party/IsaacLab/_isaac_sim/python.sh \\
    scripts/sim2real/imitation_learning/mimic/smoke_sg2_lift_mapping.py

In-sim (headless):
  cd /workspace/cyclo_lab
  ./third_party/IsaacLab/isaaclab.sh -p \\
    scripts/sim2real/imitation_learning/mimic/smoke_sg2_lift_mapping.py \\
    --sim --task Cyclo-Real-Mimic-Pick-Place-LTable-FFW-SG2-v0 \\
    --device cuda --headless --enable_cameras
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[4]
DEFAULT_IK = ROOT / "datasets/ffw_sg2_l_table_ik.hdf5"
DEFAULT_RAW = ROOT / "datasets/ffw_sg2_l_table_raw.hdf5"

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


# --- helpers mirrored from cyclo_mimic_datagen (keep in sync with source) ---
_ACT_LIFT, _ACT_HEAD1, _ACT_HEAD2 = 16, 17, 18
_IK_HEAD1, _IK_HEAD2, _IK_LIFT = 16, 17, 18
_OBS_HEAD1, _OBS_LIFT, _OBS_HEAD2 = 16, 17, 18


def actions_are_ik_layout(actions: torch.Tensor) -> bool:
    if actions.ndim != 2 or actions.shape[1] < 19:
        return False
    return float(actions[:, _IK_HEAD1].mean().item()) > float(actions[:, _IK_LIFT].mean().item())


def body_joint_cmds_from_actions(actions: torch.Tensor) -> torch.Tensor:
    if actions_are_ik_layout(actions):
        return torch.cat(
            [actions[:, _IK_LIFT : _IK_LIFT + 1], actions[:, _IK_HEAD1 : _IK_HEAD2 + 1]],
            dim=1,
        )
    return actions[:, _ACT_LIFT : _ACT_HEAD2 + 1].clone()


def body_joint_cmds_from_joint_pos(joint_pos: torch.Tensor) -> torch.Tensor:
    lift = joint_pos[:, _OBS_LIFT : _OBS_LIFT + 1]
    head = joint_pos[:, [_OBS_HEAD1, _OBS_HEAD2]]
    return torch.cat([lift, head], dim=1)


def load_demo(path: Path):
    import h5py

    with h5py.File(path, "r") as f:
        actions = torch.tensor(f["data/demo_0/actions"][()], dtype=torch.float32)
        joints = torch.tensor(f["data/demo_0/obs/joint_pos_target"][()], dtype=torch.float32)
    return actions, joints


def run_offline(ik_path: Path, raw_path: Path) -> None:
    print("=== Offline SG2 lift/head smoke ===", flush=True)

    # Source still has the fix
    src = (ROOT / "scripts/sim2real/imitation_learning/mimic/cyclo_mimic_datagen.py").read_text()
    check("datagen still remaps IK layout", "actions_are_ik_layout" in src and "_IK_LIFT" in src)
    check("datagen prefers obs joints", "body_joint_cmds_from_episode" in src)

    raw_a, raw_j = load_demo(raw_path)
    ik_a, ik_j = load_demo(ik_path)

    check("IK detected as IK layout", actions_are_ik_layout(ik_a))
    check("RAW not IK layout", not actions_are_ik_layout(raw_a))

    obs_lift = ik_j[:, _OBS_LIFT]
    obs_head1 = ik_j[:, _OBS_HEAD1]
    old_bug = ik_a[:, 16]  # wrong: treat IK[16]=head1 as lift
    fixed = body_joint_cmds_from_actions(ik_a)[:, 0]
    from_obs = body_joint_cmds_from_joint_pos(ik_j)[:, 0]

    old_err = float((old_bug - obs_lift).abs().mean().item())
    fixed_err = float((fixed - obs_lift).abs().mean().item())
    head_match = float((old_bug - obs_head1).abs().mean().item())

    check("old bug injects head1 as lift", head_match < 0.05 and old_err > 0.3, f"headΔ={head_match:.4f} liftΔ={old_err:.4f}")
    check("fixed action remap tracks obs lift", fixed_err < 0.05, f"mean|Δ|={fixed_err:.4f}")
    check(
        "fixed lift is NOT near head1",
        float((fixed - obs_head1).abs().mean().item()) > 0.3,
        f"mean|Δ|={float((fixed - obs_head1).abs().mean().item()):.4f}",
    )
    check(
        "obs-derived lift matches remapped actions",
        float((from_obs - fixed).abs().mean().item()) < 0.05,
        f"mean|Δ|={float((from_obs - fixed).abs().mean().item()):.4f}",
    )

    # Spot-check: mid-demo lift should be depressed (grasp height), not ~0.4 (head)
    mid = ik_a.shape[0] // 2
    mid_lift = float(fixed[mid].item())
    mid_old = float(old_bug[mid].item())
    check("mid-demo fixed lift is grasp-height-ish (<0.1)", mid_lift < 0.1, f"lift={mid_lift:.4f}")
    check("mid-demo old bug looks like head (>0.2)", mid_old > 0.2, f"old={mid_old:.4f}")

    print(
        f"[offline] mid-demo: fixed_lift={mid_lift:.4f} old_bug={mid_old:.4f} "
        f"obs_lift={float(obs_lift[mid]):.4f} obs_head1={float(obs_head1[mid]):.4f}",
        flush=True,
    )


def run_sim(task: str, ik_path: Path, steps: int, device: str) -> None:
    """Inject remapped lift/head into Mimic env and check lift_joint tracks."""
    import multiprocessing

    if multiprocessing.get_start_method() != "spawn":
        multiprocessing.set_start_method("spawn", force=True)

    import gymnasium as gym
    import cyclo_lab  # noqa: F401
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab_tasks.utils import parse_env_cfg

    sys.path.insert(0, str(ROOT / "scripts/sim2real/imitation_learning/mimic"))
    from cyclo_mimic_datagen import (  # noqa: E402
        body_joint_cmds_from_actions,
        body_joint_cmds_from_joint_pos,
    )

    print("=== In-sim SG2 lift inject smoke ===", flush=True)
    ik_a, ik_j = load_demo(ik_path)
    body_fixed = body_joint_cmds_from_joint_pos(ik_j)  # preferred datagen path
    body_from_ik = body_joint_cmds_from_actions(ik_a)
    body_old = torch.stack([ik_a[:, 16], ik_a[:, 17], ik_a[:, 18]], dim=1)  # wrong for IK

    env_cfg = parse_env_cfg(task, device=device, num_envs=1)
    if hasattr(env_cfg, "init_action_cfg"):
        # Joint-space action vector so inject slots 16:19 are lift/head.
        env_cfg.init_action_cfg("record")
    if hasattr(env_cfg.terminations, "time_out"):
        env_cfg.terminations.time_out = None
    if hasattr(env_cfg.terminations, "success"):
        env_cfg.terminations.success = None
    if hasattr(env_cfg.terminations, "object_dropped"):
        env_cfg.terminations.object_dropped = None

    print(f"[sim] creating env task={task}...", flush=True)
    env: ManagerBasedRLEnv = gym.make(task, cfg=env_cfg).unwrapped
    env.reset()
    robot = env.scene["robot"]
    names = list(robot.data.joint_names)
    lift_idx = names.index("lift_joint")

    # Prefer a window where lift actually moves (early frames are often flat).
    lift_series = body_fixed[:, 0].numpy()
    best_start = 0
    best_std = -1.0
    window = min(steps, body_fixed.shape[0])
    for s in range(0, max(1, body_fixed.shape[0] - window + 1), max(1, window // 2)):
        std = float(np.std(lift_series[s : s + window]))
        if std > best_std:
            best_std, best_start = std, s
    # Also try end of demo (grasp → place often lowers torso).
    end_start = max(0, body_fixed.shape[0] - window)
    if float(np.std(lift_series[end_start:])) > best_std:
        best_start = end_start

    # Hold current joint targets; inject body cmds each step.
    n = window
    lift_targets = []
    lift_measured = []
    print(f"[sim] inject window start={best_start} steps={n} lift_std={float(np.std(lift_series[best_start:best_start+n])):.4f}", flush=True)
    for i in range(n):
        cmd = body_fixed[best_start + i]
        env._mimic_body_joint_cmd = cmd.to(env.device)
        # Build a flat action from current targets (dim must match ActionManager).
        act_dim = env.action_manager.total_action_dim
        action = torch.zeros(1, act_dim, device=env.device)
        # Fill with current joint targets mapped loosely: use zeros and rely on inject for 16:19.
        # Prefer last_action if available.
        if hasattr(env, "action_manager") and env.action_manager.action is not None:
            prev = env.action_manager.action
            if prev.shape[-1] == act_dim:
                action = prev.clone()
        action = env._inject_mimic_body_joints(action)
        # Assert inject wrote lift at slot 16
        if i == 0:
            check(
                "inject writes cmd lift to action[16]",
                float(abs(action[0, 16] - cmd[0]).item()) < 1e-5,
                f"a16={float(action[0,16]):.4f} cmd={float(cmd[0]):.4f}",
            )
            check(
                "inject does NOT write IK head1 as lift",
                float(abs(action[0, 16] - body_old[best_start + i, 0]).item()) > 0.05
                or float(abs(body_old[best_start + i, 0] - cmd[0]).item()) < 0.05,
                f"a16={float(action[0,16]):.4f} old={float(body_old[best_start + i,0]):.4f}",
            )
        env.step(action)
        lift_targets.append(float(cmd[0].item()))
        lift_measured.append(float(robot.data.joint_pos[0, lift_idx].item()))

    lift_targets_t = torch.tensor(lift_targets)
    lift_measured_t = torch.tensor(lift_measured)
    # Skip first few steps for actuator settle
    settle = min(10, n // 5)
    tgt = lift_targets_t[settle:]
    meas = lift_measured_t[settle:]
    err = float((meas - tgt).abs().mean().item())
    tgt_std = float(tgt.std().item())
    if tgt_std > 1e-3 and len(tgt) > 5:
        corr = float(np.corrcoef(tgt.numpy(), meas.numpy())[0, 1])
    else:
        corr = 1.0  # constant target: tracking error is the right metric
    old_mean = float(body_old[best_start : best_start + n, 0].mean().item())
    meas_mean = float(meas.mean().item())
    tgt_mean = float(tgt.mean().item())
    print(
        f"[sim] steps={n} mean|lift_err|={err:.4f} corr={corr:.3f} tgt_std={tgt_std:.4f} "
        f"meas_mean={meas_mean:.4f} target_mean={tgt_mean:.4f} "
        f"old_bug_mean={old_mean:.4f}",
        flush=True,
    )
    check("sim lift tracks remapped target (mean|err|<0.08)", err < 0.08, f"err={err:.4f}")
    check("sim lift correlates with target (corr>0.7)", corr > 0.7, f"corr={corr:.3f}")
    check(
        "sim lift mean closer to fixed than to old-bug head1",
        abs(meas_mean - tgt_mean) < abs(meas_mean - old_mean),
        f"meas={meas_mean:.4f} tgt={tgt_mean:.4f} old={old_mean:.4f}",
    )
    # Also confirm IK→body remap agrees with obs path for this demo
    check(
        "IK→body remap ≈ obs path (for this demo)",
        float(
            (body_from_ik[best_start : best_start + n, 0] - body_fixed[best_start : best_start + n, 0])
            .abs()
            .mean()
            .item()
        )
        < 0.05,
    )

    env.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-test SG2 lift/head datagen mapping.")
    parser.add_argument("--sim", action="store_true", help="Also run in-sim inject check.")
    parser.add_argument(
        "--task",
        type=str,
        default="Cyclo-Real-Mimic-Pick-Place-LTable-FFW-SG2-v0",
        help="Gym task for --sim",
    )
    parser.add_argument("--ik-file", type=str, default=str(DEFAULT_IK))
    parser.add_argument("--raw-file", type=str, default=str(DEFAULT_RAW))
    parser.add_argument("--steps", type=int, default=80, help="In-sim inject steps")
    parser.add_argument("--device", type=str, default="cuda")
    # When --sim, AppLauncher consumes remaining args before we get here if we
    # launch via isaaclab.sh -p. For dual-mode, parse known args only when not sim-first.
    args, _ = parser.parse_known_args()

    run_offline(Path(args.ik_file), Path(args.raw_file))

    if args.sim:
        # App must already be launched by caller when using isaaclab.sh -p with this
        # script structured below for sim entry.
        run_sim(args.task, Path(args.ik_file), args.steps, args.device)

    print(f"\nResult: {PASSED} passed, {FAILED} failed", flush=True)
    return 1 if FAILED else 0


if __name__ == "__main__":
    # Support: isaaclab.sh -p smoke... --sim  (AppLauncher first)
    if "--sim" in sys.argv:
        import multiprocessing

        if multiprocessing.get_start_method() != "spawn":
            multiprocessing.set_start_method("spawn", force=True)
        from isaaclab.app import AppLauncher

        launcher_parser = argparse.ArgumentParser(add_help=False)
        launcher_parser.add_argument("--sim", action="store_true")
        launcher_parser.add_argument("--task", type=str, default="Cyclo-Real-Mimic-Pick-Place-LTable-FFW-SG2-v0")
        launcher_parser.add_argument("--ik-file", type=str, default=str(DEFAULT_IK))
        launcher_parser.add_argument("--raw-file", type=str, default=str(DEFAULT_RAW))
        launcher_parser.add_argument("--steps", type=int, default=80)
        AppLauncher.add_app_launcher_args(launcher_parser)
        args_cli, _ = launcher_parser.parse_known_args()
        app_launcher = AppLauncher(vars(args_cli))
        simulation_app = app_launcher.app
        try:
            run_offline(Path(args_cli.ik_file), Path(args_cli.raw_file))
            run_sim(args_cli.task, Path(args_cli.ik_file), args_cli.steps, args_cli.device)
            print(f"\nResult: {PASSED} passed, {FAILED} failed", flush=True)
            code = 1 if FAILED else 0
        finally:
            simulation_app.close()
        raise SystemExit(code)

    raise SystemExit(main())
