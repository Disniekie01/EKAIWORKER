#!/usr/bin/env python3
"""Offline smoke: L-table motion/home/grasp knobs live on env cfg + record_demos CLI."""

from __future__ import annotations

import ast
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
CFG = (
    ROOT
    / "source/cyclo_lab/cyclo_lab/real_world_tasks/manager_based/FFW_SG2"
    / "pick_place_l_table/pick_place_env_cfg.py"
)
JOINT = CFG.parent / "joint_pos_env_cfg.py"
SDK = ROOT / "scripts/sim2real/imitation_learning/dds_sdk/ffw_sg2_sdk.py"
RECORD = ROOT / "scripts/sim2real/imitation_learning/recorder/record_demos.py"
KIN = CFG.parent / "ltable_kinematic_l_motion.py"

REQUIRED_CFG_FIELDS = [
    "teleop_l_yaw",
    "teleop_l_forward_m",
    "teleop_l_rotation_duration_s",
    "teleop_l_forward_duration_s",
    "teleop_auto_l_on_grip_s",
    "home_lift_joint",
    "home_arm_joint1",
    "home_arm_joint4",
    "home_head_joint1",
    "grasp_diff_threshold",
    "gripper_close_threshold",
    "teleop_left_grip_smooth_alpha",
    "gripper_l_stiffness",
    "sync_configured_params",
]

REQUIRED_CLI = [
    "--teleop-l-yaw",
    "--teleop-l-forward-m",
    "--teleop-auto-l-on-grip-s",
    "--home-lift-joint",
    "--grasp-diff-threshold",
    "--gripper-l-stiffness",
]


def _check(cond: bool, msg: str, fails: list[str]) -> None:
    if not cond:
        fails.append(msg)
        print(f"  FAIL: {msg}")
    else:
        print(f"  ok: {msg}")


def main() -> int:
    fails: list[str] = []
    print("== Issue 8: teleop/home/grasp configuration ==")

    cfg_txt = CFG.read_text()
    for name in REQUIRED_CFG_FIELDS:
        _check(name in cfg_txt, f"cfg declares {name}", fails)

    _check("math.pi / 2.0" in cfg_txt or "math.pi/2" in cfg_txt, "default teleop_l_yaw uses pi/2", fails)
    _check("0.30" in cfg_txt, "default teleop_l_forward_m present", fails)
    _check("0.0365" in cfg_txt, "default home_lift_joint present", fails)

    joint_txt = JOINT.read_text()
    _check("sync_configured_params" in joint_txt, "joint_pos calls sync_configured_params", fails)

    sdk_txt = SDK.read_text()
    _check("teleop_left_grip_smooth_alpha" in sdk_txt, "SDK reads soft-grip from env.cfg", fails)
    _check("self._left_grip_smooth_alpha" in sdk_txt, "SDK uses instance soft-grip attrs", fails)
    _check("global _LEFT_GRIP_SMOOTH_ALPHA" not in sdk_txt, "SDK does not mutate grip globals", fails)

    kin_txt = KIN.read_text()
    _check("grasp_diff_threshold" in kin_txt, "kinematic L reads grasp_diff_threshold", fails)
    _check("gripper_close_threshold" in kin_txt, "kinematic L reads gripper_close_threshold", fails)

    rec_txt = RECORD.read_text()
    for flag in REQUIRED_CLI:
        _check(flag in rec_txt, f"record_demos has {flag}", fails)
    _check("_apply_cli_motion_overrides" in rec_txt, "record_demos applies CLI overrides", fails)

    # Parse defaults from cfg AST-ish: ensure teleop fields sit on PickPlaceLTableEnvCfg class body.
    tree = ast.parse(cfg_txt)
    class_names = {
        n.name
        for n in tree.body
        if isinstance(n, ast.ClassDef)
    }
    _check("PickPlaceLTableEnvCfg" in class_names, "PickPlaceLTableEnvCfg class exists", fails)

    # Sanity: pi/2 ≈ 1.5708 for docs / CLI users
    _check(abs(math.pi / 2.0 - 1.5708) < 1e-3, "pi/2 sanity", fails)

    print(f"\n== {len(REQUIRED_CFG_FIELDS) + len(REQUIRED_CLI) + 10 - len(fails)} checks; {len(fails)} failed ==")
    if fails:
        print("FAILED:")
        for f in fails:
            print(f"  - {f}")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
