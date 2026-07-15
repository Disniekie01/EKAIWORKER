#!/usr/bin/env python3
# Copyright 2026 EYKOREA
"""Smoke-test dual wrist + head camera collection for SG2 L-table.

Offline (no Isaac app): verifies source wiring + LeRobot camera discovery
against existing HDF5 (head-only legacy still discovers cam_head).

In-sim (--sim): creates Mimic L-table env, steps once, asserts
obs contains cam_head / cam_wrist_left / cam_wrist_right with RGB shapes.

Offline:
  cd /workspace/cyclo_lab
  ./third_party/IsaacLab/_isaac_sim/python.sh \\
    scripts/sim2real/imitation_learning/recorder/smoke_sg2_wrist_cams.py

In-sim:
  cd /workspace/cyclo_lab
  ./third_party/IsaacLab/_isaac_sim/python.sh \\
    scripts/sim2real/imitation_learning/recorder/smoke_sg2_wrist_cams.py \\
    --sim --device cuda --headless --enable_cameras
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
EXPECTED_CAMS = ("cam_head", "cam_wrist_left", "cam_wrist_right")

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


def run_offline(raw_hdf5: Path) -> None:
    print("=== Offline dual-wrist camera wiring ===", flush=True)

    scene_cfg = (
        ROOT
        / "source/cyclo_lab/cyclo_lab/real_world_tasks/manager_based/FFW_SG2/pick_place_l_table/pick_place_env_cfg.py"
    ).read_text()
    joint_cfg = (
        ROOT
        / "source/cyclo_lab/cyclo_lab/real_world_tasks/manager_based/FFW_SG2/pick_place_l_table/joint_pos_env_cfg.py"
    ).read_text()
    lerobot = (
        ROOT / "scripts/sim2real/imitation_learning/data_converter/isaaclab2lerobot.py"
    ).read_text()
    sdk = (ROOT / "scripts/sim2real/imitation_learning/dds_sdk/ffw_sg2_sdk.py").read_text()
    bc = (
        ROOT
        / "source/cyclo_lab/cyclo_lab/real_world_tasks/manager_based/FFW_SG2/pick_place_l_table/agents/robomimic/bc_rnn_image.json"
    ).read_text()

    for cam in EXPECTED_CAMS:
        check(f"scene declares {cam}", f"{cam}: CameraCfg" in scene_cfg or f"{cam}:" in scene_cfg)
        check(f"obs term {cam}", f"{cam} = ObsTerm" in scene_cfg)
        check(f"CameraCfg spawn {cam}", f"self.scene.{cam} = CameraCfg" in joint_cfg)
        check(f"robomimic rgb lists {cam}", f'"{cam}"' in bc)

    check("LeRobot FFW_SG2 lists wrist cams", "cam_wrist_left" in lerobot and "cam_wrist_right" in lerobot)
    check("LeRobot discovers cams from HDF5", "discover_cameras_in_hdf5" in lerobot)
    check("SDK publishes wrist cams", 'self._publish_camera("cam_wrist_left")' in sdk)
    check("SDK publishes wrist cams (right)", 'self._publish_camera("cam_wrist_right")' in sdk)
    check("SDK skips missing scene cams", "if cam_name not in self.env.scene.keys()" in sdk)

    # Avoid importing isaaclab2lerobot (pulls lerobot); mirror discovery with h5py.
    cfg_cams = list(EXPECTED_CAMS)
    check("config camera order head+left+right", '"cam_wrist_left"' in lerobot and '"cam_wrist_right"' in lerobot)

    if raw_hdf5.is_file():
        import h5py

        discovered = []
        with h5py.File(raw_hdf5, "r") as f:
            for demo_name in f["data"].keys():
                obs = f["data"][demo_name].get("obs")
                if obs is None:
                    continue
                for cam_name in cfg_cams:
                    if cam_name in obs and cam_name not in discovered:
                        discovered.append(cam_name)
        check(
            "legacy raw HDF5 still discovers cam_head",
            "cam_head" in discovered,
            f"keys={discovered}",
        )
        check(
            "legacy raw does not invent missing wrist keys",
            "cam_wrist_left" not in discovered and "cam_wrist_right" not in discovered,
            f"keys={discovered}",
        )
        print(f"[offline] discovered in {raw_hdf5.name}: {discovered}", flush=True)
    else:
        print(f"[offline] skip HDF5 discovery (missing {raw_hdf5})", flush=True)


def run_sim(task: str, device: str, hold_s: float = 0.0) -> None:
    import time

    import gymnasium as gym
    import torch
    import cyclo_lab  # noqa: F401
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab_tasks.utils import parse_env_cfg

    print("=== In-sim dual-wrist camera smoke ===", flush=True)
    env_cfg = parse_env_cfg(task, device=device, num_envs=1)
    if hasattr(env_cfg, "init_action_cfg"):
        env_cfg.init_action_cfg("record")
    for term in ("time_out", "success", "object_dropped"):
        if hasattr(env_cfg.terminations, term):
            setattr(env_cfg.terminations, term, None)

    print(f"[sim] creating env task={task}...", flush=True)
    env: ManagerBasedRLEnv = gym.make(task, cfg=env_cfg).unwrapped
    env.reset()

    for cam in EXPECTED_CAMS:
        check(f"scene entity {cam}", cam in env.scene.keys())

    # Step so camera sensors produce RGB.
    act_dim = env.action_manager.total_action_dim
    action = torch.zeros(1, act_dim, device=env.device)
    if env.action_manager.action is not None and env.action_manager.action.shape[-1] == act_dim:
        action = env.action_manager.action.clone()
    obs, _, _, _, _ = env.step(action)
    policy = obs["policy"] if isinstance(obs, dict) and "policy" in obs else obs

    for cam in EXPECTED_CAMS:
        present = cam in policy
        check(f"policy obs has {cam}", present)
        if present:
            img = policy[cam]
            shape = tuple(img.shape)
            # Expected (N, H, W, C) or (N, C, H, W)
            ok = len(shape) == 4 and (shape[-1] == 3 or shape[1] == 3)
            check(f"{cam} is RGB tensor", ok, f"shape={shape}")
            if ok and shape[-1] == 3:
                check(f"{cam} HxW is 376x672", shape[1] == 376 and shape[2] == 672, f"shape={shape}")
            elif ok and shape[1] == 3:
                check(f"{cam} HxW is 376x672", shape[2] == 376 and shape[3] == 672, f"shape={shape}")
            # Non-trivial image (not all zeros) — rendering may be dark but variance > 0 typically
            var = float(img.float().var().item())
            check(f"{cam} has signal (var>0)", var > 0.0, f"var={var:.6f}")

    if hold_s > 0:
        print(f"[sim] holding viewport for {hold_s:.0f}s — inspect cameras in Isaac GUI", flush=True)
        t0 = time.time()
        while time.time() - t0 < hold_s:
            try:
                env.step(action)
            except Exception:
                env.sim.render()

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
        launcher_parser.add_argument(
            "--raw-file",
            type=str,
            default=str(ROOT / "datasets/ffw_sg2_l_table_raw.hdf5"),
        )
        launcher_parser.add_argument(
            "--hold-s",
            type=float,
            default=60.0,
            help="Seconds to keep the Isaac GUI open after checks (GUI mode).",
        )
        AppLauncher.add_app_launcher_args(launcher_parser)
        args_cli, _ = launcher_parser.parse_known_args()
        app_launcher = AppLauncher(vars(args_cli))
        simulation_app = app_launcher.app
        try:
            run_offline(Path(args_cli.raw_file))
            hold = float(getattr(args_cli, "hold_s", 60.0) or 0.0)
            if bool(getattr(args_cli, "headless", False)):
                hold = 0.0
            run_sim(args_cli.task, args_cli.device, hold_s=hold)
            print(f"\nResult: {PASSED} passed, {FAILED} failed", flush=True)
            code = 1 if FAILED else 0
        finally:
            simulation_app.close()
        raise SystemExit(code)

    parser = argparse.ArgumentParser(description="Smoke dual wrist camera wiring.")
    parser.add_argument(
        "--raw-file",
        type=str,
        default=str(ROOT / "datasets/ffw_sg2_l_table_raw.hdf5"),
    )
    args, _ = parser.parse_known_args()
    run_offline(Path(args.raw_file))
    print(f"\nResult: {PASSED} passed, {FAILED} failed", flush=True)
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
