#!/usr/bin/env python3
"""Verify SG2 head/lift index mapping across RAW / IK / datagen helpers / joint convert.

Standalone (no Isaac app) — reimplements the small mapping helpers so we can
check them against on-disk HDF5 demos.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[4]

PASSED = 0
FAILED = 0

_ACT_LIFT, _ACT_HEAD1, _ACT_HEAD2 = 16, 17, 18
_IK_HEAD1, _IK_HEAD2, _IK_LIFT = 16, 17, 18
_OBS_HEAD1, _OBS_LIFT, _OBS_HEAD2 = 16, 17, 18


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASSED, FAILED
    if cond:
        PASSED += 1
        print(f"  PASS  {name}")
    else:
        FAILED += 1
        print(f"  FAIL  {name}" + (f" — {detail}" if detail else ""))


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


def reorder_obs_to_am(joint_targets: torch.Tensor) -> torch.Tensor:
    reordered = joint_targets.clone()
    reordered[..., 16] = joint_targets[..., 17]
    reordered[..., 17] = joint_targets[..., 16]
    return reordered


def load_demo(path: Path):
    import h5py

    with h5py.File(path, "r") as f:
        actions = torch.tensor(f["data/demo_0/actions"][()], dtype=torch.float32)
        joints = torch.tensor(f["data/demo_0/obs/joint_pos_target"][()], dtype=torch.float32)
    return actions, joints


def main() -> int:
    print("=== SG2 head/lift mapping verification ===")
    # Confirm helpers match source file constants
    src = (ROOT / "scripts/sim2real/imitation_learning/mimic/cyclo_mimic_datagen.py").read_text()
    check("datagen documents IK vs AM layouts", "IK-converted actions" in src and "ActionManager / RAW" in src)
    check("datagen prefers episode helper", "body_joint_cmds_from_episode" in src)
    check("datagen detects IK layout", "actions_are_ik_layout" in src)
    conv = (ROOT / "scripts/sim2real/imitation_learning/mimic/action_data_converter.py").read_text()
    check("converter uses torch-safe clone reorder", "joint_targets.clone()" in conv)
    check(
        "converter joint→IK uses lift@16 head@17:19 for AM/raw",
        "slice(17,19),slice(16,17)" in conv.replace(" ", ""),
    )

    raw_a, raw_j = load_demo(ROOT / "datasets/ffw_sg2_l_table_raw.hdf5")
    ik_a, ik_j = load_demo(ROOT / "datasets/ffw_sg2_l_table_ik.hdf5")

    corr_raw = float(np.corrcoef(raw_a[:, 16].numpy(), raw_j[:, 17].numpy())[0, 1])
    check("RAW action[16] is lift (corr vs obs lift@17)", corr_raw > 0.95, f"corr={corr_raw:.3f}")
    check("RAW is not IK layout", not actions_are_ik_layout(raw_a))

    corr_ik_h = float(np.corrcoef(ik_a[:, 16].numpy(), ik_j[:, 16].numpy())[0, 1])
    corr_ik_l = float(np.corrcoef(ik_a[:, 18].numpy(), ik_j[:, 17].numpy())[0, 1])
    check("IK action[16] is head1", corr_ik_h > 0.95, f"corr={corr_ik_h:.3f}")
    check("IK action[18] is lift", corr_ik_l > 0.95, f"corr={corr_ik_l:.3f}")
    check("IK detected as IK layout", actions_are_ik_layout(ik_a))

    raw_body = body_joint_cmds_from_actions(raw_a)
    obs_body = body_joint_cmds_from_joint_pos(raw_j)
    # Actions vs obs targets can differ by tracking lag; require high corr per channel.
    for i, name in enumerate(("lift", "head1", "head2")):
        c = float(np.corrcoef(raw_body[:, i].numpy(), obs_body[:, i].numpy())[0, 1])
        check(f"RAW action→body {name} corr vs obs", c > 0.9, f"corr={c:.3f}")

    ik_body = body_joint_cmds_from_actions(ik_a)
    ik_obs_body = body_joint_cmds_from_joint_pos(ik_j)
    for i, name in enumerate(("lift", "head1", "head2")):
        c = float(np.corrcoef(ik_body[:, i].numpy(), ik_obs_body[:, i].numpy())[0, 1])
        check(f"IK action→body {name} corr vs obs", c > 0.9, f"corr={c:.3f}")
    wrong = float((ik_body[:, 0] - ik_a[:, 16]).abs().mean().item())
    right = float((ik_body[:, 0] - ik_a[:, 18]).abs().mean().item())
    check("IK lift channel is NOT head1 (old bug)", wrong > 0.05, f"mean|d|={wrong:.4f}")
    check("IK lift channel matches IK[:,18]", right < 0.05, f"mean|d|={right:.4f}")

    # joint→IK extract from AM/raw
    head = raw_a[:, 17:19]
    lift = raw_a[:, 16:17]
    check("raw→IK head slice matches AM head", torch.allclose(head, raw_a[:, 17:19]))
    check("raw→IK lift slice matches AM lift", torch.allclose(lift, raw_a[:, 16:17]))

    am = reorder_obs_to_am(raw_j)
    check("torch reorder: [16]=obs lift", torch.allclose(am[:, 16], raw_j[:, 17]))
    check("torch reorder: [17]=obs head1", torch.allclose(am[:, 17], raw_j[:, 16]))
    check("torch reorder: [18]=obs head2", torch.allclose(am[:, 18], raw_j[:, 18]))

    print(f"\nResult: {PASSED} passed, {FAILED} failed")
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
