#!/usr/bin/env python3
# Copyright 2026 EYKOREA
"""Validate experimental LeRobot format v3.0 export wiring (isolation + optional live convert).

Does not change default Mimic pipeline: PIPELINE_STEPS still ends with v2.1 LeRobot export.

  cd /workspace/cyclo_lab
  ./third_party/IsaacLab/_isaac_sim/python.sh \\
    scripts/sim2real/imitation_learning/mimic/smoke_sg2_lerobot_v3.py

  # Optional live convert (needs setup_lerobot_v3_env.sh + joint hdf5):
  lerobot-python scripts/sim2real/imitation_learning/mimic/smoke_sg2_lerobot_v3.py --live
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
CONVERTER = ROOT / "scripts/sim2real/imitation_learning/data_converter/isaaclab2lerobot.py"
SETUP = ROOT / "scripts/sim2real/imitation_learning/data_converter/setup_lerobot_v3_env.sh"
DASH = ROOT / "sg2_ltable_dashboard.py"
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


def resolve_v3_python() -> str | None:
    candidates = [
        os.environ.get("LEROBOT_V3_PYTHON", "").strip(),
        shutil.which("lerobot-python-v3") or "",
        os.path.expanduser("~/lerobot_env_v3/bin/python3"),
        "/root/lerobot_env_v3/bin/python3",
    ]
    for path in candidates:
        if path and os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    return None


def run_offline() -> None:
    print("=== Offline LeRobot v3 (Experimental) validation ===", flush=True)
    conv = CONVERTER.read_text()
    dash = DASH.read_text()

    check("converter has --dataset_format", "--dataset_format" in conv and 'choices=["v2", "v3"]' in conv)
    check("converter default remains v2", 'default="v2"' in conv)
    check("converter has convert_v21_dataset_to_v30", "convert_v21_dataset_to_v30" in conv)
    check("converter refuses default venv for v3", "intentionally not used for conversion" in conv)
    check("setup_lerobot_v3_env.sh exists", SETUP.is_file())
    check(
        "PIPELINE_STEPS unchanged (no v3 insert)",
        'PIPELINE_STEPS = ("ik", "annotate", "generate", "joint", "lerobot")' in dash,
    )
    check("default lerobot cmd has no --dataset_format v3", "--dataset_format v3" not in _default_lerobot_cmd(dash))
    check("dashboard Experimental v3 card", "LeRobot v3.0 export" in dash and "Experimental" in dash)
    check("dashboard /api/lerobot_v3", '"/api/lerobot_v3"' in dash or "/api/lerobot_v3" in dash)
    check("dashboard launch_lerobot_v3", "def launch_lerobot_v3" in dash)
    check("dashboard does not add merge to PIPELINE_STEPS", "merge" not in _pipeline_steps_tuple(dash))


def _default_lerobot_cmd(dash: str) -> str:
    """Extract the else:# lerobot command block approximately."""
    marker = 'else:  # lerobot'
    i = dash.find(marker)
    if i < 0:
        return ""
    return dash[i : i + 400]


def _pipeline_steps_tuple(dash: str) -> str:
    i = dash.find("PIPELINE_STEPS = ")
    if i < 0:
        return ""
    return dash[i : i + 120]


def run_live(joint: Path, task: str, robot: str) -> None:
    print("=== Live LeRobot v3 convert (optional) ===", flush=True)
    py_v3 = resolve_v3_python()
    if py_v3 is None:
        skip("live convert", "lerobot-python-v3 / LEROBOT_V3_PYTHON not found (run setup_lerobot_v3_env.sh)")
        return
    if not joint.is_file() or joint.stat().st_size < 1000:
        skip("live convert", f"joint hdf5 missing/empty: {joint}")
        return

    lerobot_py = shutil.which("lerobot-python")
    if not lerobot_py:
        # Fall back to common container path
        candidate = os.path.expanduser("~/lerobot_env/bin/python3")
        lerobot_py = candidate if os.path.isfile(candidate) else sys.executable

    out_root = ROOT / "datasets" / "lerobot_v3" / "_smoke_v3"
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        lerobot_py,
        str(CONVERTER),
        f"--task={task}",
        "--robot_type",
        robot,
        "--dataset_file",
        str(joint),
        "--dataset_format",
        "v3",
        "--repo_id",
        str(out_root),
        "--root",
        str(out_root),
        "--frame_skip",
        "10",
    ]
    print(f"[live] {' '.join(cmd)}", flush=True)
    env = os.environ.copy()
    env["LEROBOT_V3_PYTHON"] = py_v3
    proc = subprocess.run(cmd, cwd=str(ROOT), env=env)
    check("live export exit 0", proc.returncode == 0, f"exit={proc.returncode}")

    info_path = out_root / "meta" / "info.json"
    if not info_path.is_file():
        check("v3 meta/info.json exists", False, str(info_path))
        return
    info = json.loads(info_path.read_text())
    check("codebase_version is v3.0", info.get("codebase_version") == "v3.0", repr(info.get("codebase_version")))

    # Chunked file layout (v3): data/chunk-*/file_*.parquet or file-*.parquet
    data_files = list((out_root / "data").rglob("file*.parquet")) if (out_root / "data").is_dir() else []
    ep_files = list((out_root / "data").rglob("episode_*.parquet")) if (out_root / "data").is_dir() else []
    check(
        "v3 uses chunked file-*.parquet (not only episode_*)",
        len(data_files) > 0,
        f"file*= {len(data_files)}, episode_*= {len(ep_files)}",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-test experimental LeRobot v3.0 export")
    parser.add_argument("--live", action="store_true", help="Also run a short live v3 export if venv+hdf5 exist")
    parser.add_argument("--joint-file", type=str, default=str(DEFAULT_JOINT))
    parser.add_argument("--task", type=str, default="Cyclo-Real-Pick-Place-LTable-FFW-SG2-v0")
    parser.add_argument("--robot_type", type=str, default="FFW_SG2")
    args = parser.parse_args()

    run_offline()
    if args.live:
        run_live(Path(args.joint_file), args.task, args.robot_type)
    else:
        skip("live convert", "pass --live to attempt export")

    print(f"\nResult: {PASSED} passed, {FAILED} failed, {SKIPPED} skipped", flush=True)
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
