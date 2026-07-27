#!/usr/bin/env python3
# Copyright 2026 EYKOREA
"""Validate experimental raw+joint merge (isolation + optional live merge).

  python3 scripts/sim2real/imitation_learning/mimic/smoke_sg2_merge_datasets.py
  python3 .../smoke_sg2_merge_datasets.py --live
"""

from __future__ import annotations

import argparse
import json
import math
import tempfile
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
MERGE = ROOT / "scripts/sim2real/imitation_learning/mimic/merge_joint_datasets.py"
DASH = ROOT / "sg2_ltable_dashboard.py"
DEFAULT_RAW = ROOT / "datasets/ffw_sg2_l_table_raw.hdf5"
DEFAULT_JOINT = ROOT / "datasets/ffw_sg2_l_table_joint.hdf5"

PASSED = 0
FAILED = 0
SKIPPED = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASSED, FAILED
    if cond:
        PASSED += 1
        print(f"  PASS  {name}")
    else:
        FAILED += 1
        print(f"  FAIL  {name}" + (f" — {detail}" if detail else ""))


def skip(name: str, reason: str) -> None:
    global SKIPPED
    SKIPPED += 1
    print(f"  SKIP  {name} — {reason}")


def run_offline() -> None:
    print("=== Offline merge validation ===", flush=True)
    dash = DASH.read_text()
    merge = MERGE.read_text()

    check("merge_joint_datasets.py exists", MERGE.is_file())
    check("merge CLI has source attrs", 'source' in merge and "source_demo" in merge)
    check("merge derives joint_vel", "joint_vel" in merge and "_finite_diff_vel" in merge)
    check(
        "PIPELINE_STEPS unchanged",
        'PIPELINE_STEPS = ("ik", "annotate", "generate", "joint", "lerobot")' in dash,
    )
    check("merge not in PIPELINE_STEPS tuple", "merge" not in dash[dash.find("PIPELINE_STEPS") : dash.find("PIPELINE_STEPS") + 120])
    check("default lerobot still uses joint", "dataset_file {paths['joint']}" in dash or 'paths["joint"]' in dash)
    check("Experimental Merge card", "Merge datasets" in dash and "Experimental" in dash)
    check("dashboard /api/merge", "/api/merge" in dash and "def launch_merge" in dash)
    check("LeRobot from mixed experimental", "launch_lerobot_from_mixed" in dash)

    # Unit: SG2 reorder + FD vel
    import importlib.util

    spec = importlib.util.spec_from_file_location("merge_joint_datasets", MERGE)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    x = np.zeros((3, 19), dtype=np.float32)
    x[:, 16] = 1.0  # head1 in obs order
    x[:, 17] = 2.0  # lift
    y = mod._reorder_sg2_obs_to_am(x)
    check("SG2 obs→AM swap", float(y[0, 16]) == 2.0 and float(y[0, 17]) == 1.0)
    jp = np.arange(10, dtype=np.float32).reshape(10, 1)
    vel = mod._finite_diff_vel(np.tile(jp, (1, 19)))
    check("joint_vel FD", float(vel[0, 0]) == 0.0 and float(vel[1, 0]) == 1.0)


def _count_success(path: Path) -> int:
    with h5py.File(path, "r") as f:
        n = 0
        for name in f["data"].keys():
            demo = f["data"][name]
            if "success" not in demo.attrs or bool(demo.attrs["success"]):
                n += 1
        return n


def _make_fixture_pair(joint: Path, fixture_dir: Path, robot: str) -> tuple[Path, Path]:
    """Build tiny raw+joint HDF5s from an existing joint file for live smoke."""
    import shutil

    raw_out = fixture_dir / "raw_fix.hdf5"
    joint_out = fixture_dir / "joint_fix.hdf5"
    with h5py.File(joint, "r") as src:
        names = [n for n in sorted(src["data"].keys(), key=lambda x: (len(x), x)) if _is_success_demo(src["data"][n])]
        if len(names) < 2:
            raise RuntimeError("need >=2 successful demos in joint file for fixtures")
        # First half-ish → raw fixture, all → joint fixture (ratios still apply)
        raw_names = names[: max(1, len(names) // 3)]
        for out_path, keep in ((raw_out, raw_names), (joint_out, names)):
            with h5py.File(out_path, "w") as dst:
                data = dst.create_group("data")
                if "env_args" in src["data"].attrs:
                    data.attrs["env_args"] = src["data"].attrs["env_args"]
                for i, name in enumerate(keep):
                    src["data"].copy(name, data, name=f"demo_{i}")
                    data[f"demo_{i}"].attrs["success"] = True
    return raw_out, joint_out


def _is_success_demo(demo: h5py.Group) -> bool:
    if "success" not in demo.attrs:
        return True
    return bool(demo.attrs["success"])


def run_live(raw: Path, joint: Path, robot: str) -> None:
    print("=== Live merge validation ===", flush=True)
    use_fixtures = False
    if not raw.is_file() or raw.stat().st_size < 1000:
        use_fixtures = True
    else:
        try:
            with h5py.File(raw, "r") as f:
                if "data" not in f or len(f["data"]) == 0:
                    use_fixtures = True
        except OSError:
            use_fixtures = True
    if not joint.is_file() or joint.stat().st_size < 1000:
        skip("live merge", f"joint missing: {joint}")
        return

    import importlib.util

    spec = importlib.util.spec_from_file_location("merge_joint_datasets", MERGE)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)

    with tempfile.TemporaryDirectory(prefix="sg2_merge_smoke_") as td:
        td_path = Path(td)
        if use_fixtures:
            print("[live] raw unusable; building fixtures from joint demos", flush=True)
            raw, joint = _make_fixture_pair(joint, td_path, robot)

        n_raw = _count_success(raw)
        n_gen = _count_success(joint)
        raw_ratio = 1.0
        gen_ratio = 0.5 if n_gen >= 2 else 1.0

        def _expected(n: int, ratio: float) -> int:
            ratio = float(np.clip(ratio, 0.0, 1.0))
            if n == 0 or ratio <= 0:
                return 0
            k = int(math.floor(n * ratio + 1e-9))
            if ratio > 0 and k == 0:
                k = 1
            return n if k >= n else k

        expected_raw = _expected(n_raw, raw_ratio)
        expected_gen = _expected(n_gen, gen_ratio)

        out = td_path / "mixed.hdf5"
        summary = mod.merge_joint_datasets(
            raw_file=raw,
            joint_file=joint,
            output_file=out,
            robot_type=robot,
            raw_ratio=raw_ratio,
            generate_ratio=gen_ratio,
            seed=0,
        )
        check("wrote output", out.is_file())
        with h5py.File(out, "r") as f:
            demos = list(f["data"].keys())
            sources = [f["data"][d].attrs.get("source") for d in demos]
            check("source attrs only raw|generate", all(s in ("raw", "generate") for s in sources))
            check(
                "ratio counts roughly match",
                summary["counts"]["raw"] == expected_raw and summary["counts"]["generate"] == expected_gen,
                f"summary={summary['counts']} expected raw={expected_raw} gen={expected_gen}",
            )
            cams_ref = None
            dim = {"FFW_SG2": 19, "FFW_SH5": 57, "OMY": 7}[robot]
            for d in demos:
                demo = f["data"][d]
                check(f"{d} has actions", "actions" in demo)
                check(f"{d} has joint_vel", "joint_vel" in demo["obs"])
                jp = np.asarray(demo["obs"]["joint_pos"])
                jv = np.asarray(demo["obs"]["joint_vel"])
                check(f"{d} joint_vel shape", jp.shape == jv.shape, f"{jp.shape} vs {jv.shape}")
                check(f"{d} action dim", np.asarray(demo["actions"]).shape[1] == dim)
                cams = sorted(k for k in demo["obs"].keys() if k.startswith("cam"))
                if cams_ref is None:
                    cams_ref = cams
                else:
                    check(f"{d} camera keys match", cams == cams_ref, f"{cams} vs {cams_ref}")
                check(f"{d} source_demo attr", "source_demo" in demo.attrs)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--raw-file", default=str(DEFAULT_RAW))
    parser.add_argument("--joint-file", default=str(DEFAULT_JOINT))
    parser.add_argument("--robot_type", default="FFW_SG2")
    args = parser.parse_args()

    run_offline()
    if args.live:
        run_live(Path(args.raw_file), Path(args.joint_file), args.robot_type)
    else:
        # Auto-run live if datasets exist
        if DEFAULT_RAW.is_file() and DEFAULT_JOINT.is_file():
            run_live(Path(args.raw_file), Path(args.joint_file), args.robot_type)
        else:
            skip("live merge", "pass --live or provide datasets")

    print(f"\nResult: {PASSED} passed, {FAILED} failed, {SKIPPED} skipped", flush=True)
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
