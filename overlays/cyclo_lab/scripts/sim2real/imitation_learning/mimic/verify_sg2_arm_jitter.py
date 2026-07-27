#!/usr/bin/env python3
"""Offline checks for SG2 arm-jitter fixes (no Isaac app required)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[4]
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


def ensure_eef_quat_continuity(eef_pose: torch.Tensor) -> torch.Tensor:
    out = eef_pose.clone()
    quat = out[:, 3:7]
    for i in range(1, quat.shape[0]):
        if torch.dot(quat[i], quat[i - 1]) < 0:
            quat[i] *= -1
    return out


def without_robot_joint_state(state: dict) -> dict:
    artic = state.get("articulation")
    if not isinstance(artic, dict) or "robot" not in artic:
        return state
    robot = artic["robot"]
    filtered_robot = {k: v for k, v in robot.items() if "joint" not in str(k).lower()}
    return {**state, "articulation": {**artic, "robot": filtered_robot}}


def main() -> int:
    print("=== Arm jitter fix verification ===")
    ann = (ROOT / "scripts/sim2real/imitation_learning/mimic/annotate_demos.py").read_text()
    rep = (ROOT / "scripts/imitation_learning/isaaclab_recorder/replay_demos.py").read_text()
    conv = (ROOT / "scripts/sim2real/imitation_learning/mimic/action_data_converter.py").read_text()
    mimic = (
        ROOT
        / "source/cyclo_lab/cyclo_lab/real_world_tasks/manager_based/FFW_SG2/pick_place/pick_place_mimic_env.py"
    ).read_text()

    check("annotate pins live joints on restore", "skip_robot_joints" in ann and "_with_live_robot_joints" in ann)
    check("replay skips joints for mimic_ik", 'skip_robot_joints=(args_cli.action_mode == "mimic_ik")' in rep)
    check("converter enforces quat continuity", "_ensure_eef_quat_continuity" in conv)
    check("converter IK trailing is lift then head", "lift_action,      # 16: lift joint" in conv)
    check("mimic enforces continuous eef quat", "_continuous_eef_quat" in mimic)

    # Unit: quat flip removal
    poses = torch.zeros(4, 7)
    poses[:, 3] = 1.0
    poses[2, 3:7] = torch.tensor([-1.0, 0.0, 0.0, 0.0])  # same rot, flipped
    fixed = ensure_eef_quat_continuity(poses)
    dots = (fixed[1:, 3:7] * fixed[:-1, 3:7]).sum(dim=1)
    check("quat continuity removes double-cover flip", bool(torch.all(dots >= 0)))

    # Unit: joint filter
    state = {
        "articulation": {
            "robot": {
                "joint_position": torch.zeros(1, 31),
                "joint_velocity": torch.zeros(1, 31),
                "root_pose": torch.zeros(1, 7),
            }
        },
        "rigid_object": {"cardboard_box": {"root_pose": torch.zeros(1, 7)}},
    }
    filtered = without_robot_joint_state(state)
    robot = filtered["articulation"]["robot"]
    check("pins live joint_position", "joint_position" not in robot)
    check("filter drops joint_velocity", "joint_velocity" not in robot)
    check("filter keeps root_pose", "root_pose" in robot)
    check("filter keeps objects", "cardboard_box" in filtered["rigid_object"])

    # Generate HDF5: show flip reduction if present
    gen = ROOT / "datasets/ffw_sg2_l_table_generate.hdf5"
    if gen.is_file() and gen.stat().st_size > 1000:
        import h5py

        with h5py.File(gen, "r") as f:
            a = np.array(f["data/demo_0/actions"])
        q = a[:, 11:15]
        flips = int(np.sum(np.sum(q[1:] * q[:-1], axis=1) < 0))
        q2 = q.copy()
        for i in range(1, len(q2)):
            if np.dot(q2[i], q2[i - 1]) < 0:
                q2[i] *= -1
        flips2 = int(np.sum(np.sum(q2[1:] * q2[:-1], axis=1) < 0))
        check("generate demo_0 R has flips (pre-existing)", flips > 0, f"flips={flips}")
        check("continuity can zero those flips", flips2 == 0, f"flips2={flips2}")
        d = np.linalg.norm(np.diff(a[:, 8:15], axis=0), axis=1)
        a2 = a.copy()
        a2[:, 11:15] = q2
        d2 = np.linalg.norm(np.diff(a2[:, 8:15], axis=0), axis=1)
        check(
            "continuity cuts generate EE step max",
            float(d2.max()) < float(d.max()) * 0.5,
            f"max {d.max():.3f} -> {d2.max():.3f}",
        )

    print(f"\nResult: {PASSED} passed, {FAILED} failed")
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
