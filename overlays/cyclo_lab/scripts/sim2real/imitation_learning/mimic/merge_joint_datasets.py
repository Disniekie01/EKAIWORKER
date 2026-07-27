#!/usr/bin/env python3
# Copyright 2026 EYKOREA
"""Merge raw + generate-joint demos into one joint-schema training HDF5 (Experimental).

Standalone tool — does not modify the Mimic pipeline. Writes ``*_mixed.hdf5`` only.

Schema (both sources normalized):
  actions              ActionManager joint order
  obs/joint_pos        same AM trailing order (SG2)
  obs/joint_pos_target same AM trailing order (SG2)
  obs/joint_vel        finite-diff of joint_pos
  optional eef poses + camera keys (intersection across selected demos)

Episode attrs: success, seed, source ("raw"|"generate"), source_demo, num_samples

Example:
  ./third_party/IsaacLab/_isaac_sim/python.sh \\
    scripts/sim2real/imitation_learning/mimic/merge_joint_datasets.py \\
    --robot_type FFW_SG2 \\
    --raw_file datasets/ffw_sg2_l_table_raw.hdf5 \\
    --joint_file datasets/ffw_sg2_l_table_joint.hdf5 \\
    --output_file datasets/ffw_sg2_l_table_mixed.hdf5 \\
    --raw_ratio 1.0 --generate_ratio 0.5 --seed 0
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import h5py
import numpy as np

ROBOT_ACTION_DIM = {
    "OMY": 7,
    "FFW_SG2": 19,
    "FFW_SH5": 57,
}

_KNOWN_CAM_PREFIXES = ("cam_",)
_REQUIRED_OBS = ("joint_pos", "joint_pos_target")
_OPTIONAL_EEF = ("left_eef_pose", "right_eef_pose", "eef_pose")


def _reorder_sg2_obs_to_am(arr: np.ndarray) -> np.ndarray:
    """Observation trailing [head1, lift, head2] → AM [lift, head1, head2]."""
    if arr.ndim != 2 or arr.shape[1] != 19:
        return arr
    out = arr.copy()
    out[:, 16] = arr[:, 17]
    out[:, 17] = arr[:, 16]
    # index 18 (head2) unchanged
    return out


def _finite_diff_vel(joint_pos: np.ndarray) -> np.ndarray:
    vel = np.zeros_like(joint_pos, dtype=np.float32)
    if joint_pos.shape[0] > 1:
        vel[1:] = joint_pos[1:] - joint_pos[:-1]
    return vel


def _is_success(demo: h5py.Group) -> bool:
    if "success" not in demo.attrs:
        return True
    return bool(demo.attrs["success"])


def _list_successful_demos(path: Path) -> list[str]:
    with h5py.File(path, "r") as f:
        if "data" not in f:
            raise ValueError(f"No /data group in {path}")
        names = []
        for name in sorted(f["data"].keys(), key=lambda n: (len(n), n)):
            if _is_success(f["data"][name]):
                names.append(name)
        return names


def _sample_names(names: list[str], ratio: float, rng: random.Random) -> list[str]:
    if not names:
        return []
    ratio = float(np.clip(ratio, 0.0, 1.0))
    k = int(math.floor(len(names) * ratio + 1e-9))
    if ratio > 0 and k == 0:
        k = 1
    if k >= len(names):
        return list(names)
    picked = list(names)
    rng.shuffle(picked)
    return sorted(picked[:k], key=lambda n: (len(n), n))


def _demo_camera_keys(demo: h5py.Group) -> set[str]:
    obs = demo.get("obs")
    if obs is None:
        return set()
    keys = set()
    for k in obs.keys():
        if k.startswith(_KNOWN_CAM_PREFIXES) or k.startswith("cam"):
            # Keep configured camera-like arrays (N,H,W,3)
            arr = obs[k]
            if getattr(arr, "ndim", 0) == 4 and arr.shape[-1] == 3:
                keys.add(k)
    return keys


def _load_normalized_episode(
    demo: h5py.Group,
    *,
    robot_type: str,
    source: str,
    source_demo: str,
    camera_keys: list[str],
) -> dict:
    expected = ROBOT_ACTION_DIM[robot_type]
    if "actions" not in demo:
        raise KeyError("actions")
    actions = np.asarray(demo["actions"], dtype=np.float32)
    if actions.ndim == 1:
        actions = actions.reshape(-1, expected)
    if actions.shape[1] != expected:
        raise ValueError(f"action dim {actions.shape[1]} != {expected}")

    obs_in = demo["obs"]
    for req in _REQUIRED_OBS:
        if req not in obs_in:
            raise KeyError(req)

    joint_pos = np.asarray(obs_in["joint_pos"], dtype=np.float32)
    joint_tgt = np.asarray(obs_in["joint_pos_target"], dtype=np.float32)
    if joint_pos.ndim == 1:
        joint_pos = joint_pos.reshape(-1, expected)
    if joint_tgt.ndim == 1:
        joint_tgt = joint_tgt.reshape(-1, expected)
    if joint_pos.shape[1] != expected or joint_tgt.shape[1] != expected:
        raise ValueError("joint_pos / joint_pos_target dim mismatch")

    # Raw actions are already AM; obs joint_* are often observation order on SG2.
    if robot_type == "FFW_SG2":
        joint_pos = _reorder_sg2_obs_to_am(joint_pos)
        joint_tgt = _reorder_sg2_obs_to_am(joint_tgt)

    n = min(actions.shape[0], joint_pos.shape[0], joint_tgt.shape[0])
    actions = actions[:n]
    joint_pos = joint_pos[:n]
    joint_tgt = joint_tgt[:n]
    joint_vel = _finite_diff_vel(joint_pos)

    obs_out: dict[str, np.ndarray] = {
        "joint_pos": joint_pos,
        "joint_pos_target": joint_tgt,
        "joint_vel": joint_vel,
    }
    for key in _OPTIONAL_EEF:
        if key in obs_in:
            obs_out[key] = np.asarray(obs_in[key], dtype=np.float32)[:n]

    for cam in camera_keys:
        if cam not in obs_in:
            raise KeyError(f"missing camera {cam}")
        obs_out[cam] = np.asarray(obs_in[cam], dtype=np.uint8)[:n]

    seed = int(demo.attrs["seed"]) if "seed" in demo.attrs else None
    return {
        "actions": actions,
        "obs": obs_out,
        "source": source,
        "source_demo": source_demo,
        "success": True,
        "seed": seed,
        "num_samples": int(n),
    }


def _write_episode(group: h5py.Group, ep: dict) -> None:
    group.create_dataset("actions", data=ep["actions"], compression="gzip")
    obs_g = group.create_group("obs")
    for k, v in ep["obs"].items():
        if v.dtype == np.uint8:
            obs_g.create_dataset(k, data=v, compression="gzip")
        else:
            obs_g.create_dataset(k, data=v, compression="gzip")
    group.attrs["success"] = bool(ep["success"])
    group.attrs["source"] = str(ep["source"])
    group.attrs["source_demo"] = str(ep["source_demo"])
    group.attrs["num_samples"] = int(ep["num_samples"])
    if ep["seed"] is not None:
        group.attrs["seed"] = int(ep["seed"])


def merge_joint_datasets(
    *,
    raw_file: Path,
    joint_file: Path,
    output_file: Path,
    robot_type: str,
    raw_ratio: float,
    generate_ratio: float,
    seed: int,
) -> dict:
    if robot_type not in ROBOT_ACTION_DIM:
        raise ValueError(f"Unsupported robot_type {robot_type}")
    if not raw_file.is_file():
        raise FileNotFoundError(raw_file)
    if not joint_file.is_file():
        raise FileNotFoundError(joint_file)

    rng = random.Random(seed)
    raw_names = _sample_names(_list_successful_demos(raw_file), raw_ratio, rng)
    gen_names = _sample_names(_list_successful_demos(joint_file), generate_ratio, rng)
    if not raw_names and not gen_names:
        raise RuntimeError("No demos selected (check ratios / success flags)")

    # Camera intersection across selected demos
    cam_sets: list[set[str]] = []
    with h5py.File(raw_file, "r") as fr, h5py.File(joint_file, "r") as fj:
        for name in raw_names:
            cam_sets.append(_demo_camera_keys(fr["data"][name]))
        for name in gen_names:
            cam_sets.append(_demo_camera_keys(fj["data"][name]))
    if not cam_sets:
        camera_keys: list[str] = []
    else:
        inter = set.intersection(*cam_sets) if cam_sets else set()
        # Prefer stable camera order
        preferred = ["cam_head", "cam_wrist_left", "cam_wrist_right", "cam_wrist", "cam_top"]
        camera_keys = [c for c in preferred if c in inter]
        camera_keys += sorted(k for k in inter if k not in camera_keys)

    print(
        f"[merge] selected raw={len(raw_names)}/{len(_list_successful_demos(raw_file))} "
        f"generate={len(gen_names)}/{len(_list_successful_demos(joint_file))} "
        f"cameras={camera_keys}",
        flush=True,
    )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    if output_file.exists():
        output_file.unlink()

    written = {"raw": 0, "generate": 0}
    demo_idx = 0
    env_args = None
    with h5py.File(raw_file, "r") as fr, h5py.File(joint_file, "r") as fj, h5py.File(
        output_file, "w"
    ) as fo:
        if "env_args" in fr["data"].attrs:
            env_args = fr["data"].attrs["env_args"]
        elif "env_args" in fj["data"].attrs:
            env_args = fj["data"].attrs["env_args"]
        data_out = fo.create_group("data")
        if env_args is not None:
            data_out.attrs["env_args"] = env_args

        for name in raw_names:
            ep = _load_normalized_episode(
                fr["data"][name],
                robot_type=robot_type,
                source="raw",
                source_demo=name,
                camera_keys=camera_keys,
            )
            g = data_out.create_group(f"demo_{demo_idx}")
            _write_episode(g, ep)
            written["raw"] += 1
            demo_idx += 1

        for name in gen_names:
            ep = _load_normalized_episode(
                fj["data"][name],
                robot_type=robot_type,
                source="generate",
                source_demo=name,
                camera_keys=camera_keys,
            )
            g = data_out.create_group(f"demo_{demo_idx}")
            _write_episode(g, ep)
            written["generate"] += 1
            demo_idx += 1

        data_out.attrs["total"] = sum(
            int(data_out[d].attrs["num_samples"]) for d in data_out.keys()
        )
        data_out.attrs["merge_meta"] = json.dumps(
            {
                "raw_file": str(raw_file),
                "joint_file": str(joint_file),
                "raw_ratio": raw_ratio,
                "generate_ratio": generate_ratio,
                "seed": seed,
                "robot_type": robot_type,
                "cameras": camera_keys,
                "counts": written,
            }
        )

    print(
        f"[merge] wrote {demo_idx} demos → {output_file} "
        f"(raw={written['raw']}, generate={written['generate']})",
        flush=True,
    )
    return {"output": str(output_file), "counts": written, "cameras": camera_keys, "n_demos": demo_idx}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge raw + joint demos into mixed training HDF5 (Experimental).")
    p.add_argument("--raw_file", type=str, required=True)
    p.add_argument("--joint_file", type=str, required=True, help="Generate-branch *_joint.hdf5")
    p.add_argument("--output_file", type=str, required=True)
    p.add_argument("--robot_type", type=str, default="FFW_SG2", choices=list(ROBOT_ACTION_DIM))
    p.add_argument("--raw_ratio", type=float, default=1.0, help="Keep-fraction of successful raw demos [0,1]")
    p.add_argument(
        "--generate_ratio",
        type=float,
        default=1.0,
        help="Keep-fraction of successful generate-joint demos [0,1]",
    )
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        merge_joint_datasets(
            raw_file=Path(args.raw_file),
            joint_file=Path(args.joint_file),
            output_file=Path(args.output_file),
            robot_type=args.robot_type,
            raw_ratio=args.raw_ratio,
            generate_ratio=args.generate_ratio,
            seed=args.seed,
        )
    except Exception as exc:
        print(f"[merge] ERROR: {exc}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
