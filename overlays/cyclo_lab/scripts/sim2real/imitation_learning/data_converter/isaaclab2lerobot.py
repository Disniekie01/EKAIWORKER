# Copyright 2025 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Taehyeong Kim

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset

_FFW_SH5_JOINT_NAMES = (
    [f"arm_l_joint{i}" for i in range(1, 8)]
    + [f"finger_l_joint{i}" for i in range(1, 21)]
    + [f"arm_r_joint{i}" for i in range(1, 8)]
    + [f"finger_r_joint{i}" for i in range(1, 21)]
    + ["head_joint1", "head_joint2", "lift_joint"]
)

ROBOT_CONFIGS = {
    "OMY": {
        "expected_dim": 7,
        "joint_names": [
            "joint1", "joint2", "joint3", "joint4",
            "joint5", "joint6", "rh_r1_joint",
        ],
        "cameras": {
            "cam_wrist": {"height": 480, "width": 848},
            "cam_top": {"height": 480, "width": 848},
        }
    },
    "FFW_SG2": {
        "expected_dim": 19,
        "joint_names": [
            "arm_l_joint1", "arm_l_joint2", "arm_l_joint3", "arm_l_joint4",
            "arm_l_joint5", "arm_l_joint6", "arm_l_joint7", "gripper_l_joint1",
            "arm_r_joint1", "arm_r_joint2", "arm_r_joint3", "arm_r_joint4",
            "arm_r_joint5", "arm_r_joint6", "arm_r_joint7", "gripper_r_joint1",
            "head_joint1", "head_joint2", "lift_joint",
        ],
        "cameras": {
            "cam_head": {"height": 376, "width": 672},
            "cam_wrist_left": {"height": 376, "width": 672},
            "cam_wrist_right": {"height": 376, "width": 672},
        }
    },
    "FFW_SH5": {
        "expected_dim": 57,
        "joint_names": list(_FFW_SH5_JOINT_NAMES),
        "cameras": {
            "cam_head": {"height": 376, "width": 672},
        }
    },
}


def discover_cameras_in_hdf5(dataset_file: str, robot_type: str) -> dict:
    """Return configured cameras that actually exist in the HDF5 (union across demos).

    New SG2 recordings include head + both wrists; older head-only files still convert.
    Image HxW is taken from the first demo that contains each camera when available.
    """
    if robot_type not in ROBOT_CONFIGS:
        raise ValueError(f"Unsupported robot type: {robot_type}")
    configured = ROBOT_CONFIGS[robot_type]["cameras"]
    discovered: dict[str, dict] = {}
    with h5py.File(dataset_file, "r") as f:
        for demo_name in f["data"].keys():
            obs = f["data"][demo_name].get("obs")
            if obs is None:
                continue
            for cam_name, cam_cfg in configured.items():
                if cam_name in discovered or cam_name not in obs:
                    continue
                arr = obs[cam_name]
                if arr.ndim == 4 and arr.shape[-1] == 3:
                    discovered[cam_name] = {
                        "height": int(arr.shape[1]),
                        "width": int(arr.shape[2]),
                    }
                else:
                    discovered[cam_name] = dict(cam_cfg)
    if not discovered:
        # Fall back to configured list (demos will skip if keys are missing).
        discovered = {k: dict(v) for k, v in configured.items()}
    # Preserve configured key order.
    return {k: discovered[k] for k in configured if k in discovered}


def get_env_features(fps: int, robot_type: str, cameras: dict | None = None):
    if robot_type not in ROBOT_CONFIGS:
        raise ValueError(f"Unsupported robot type: {robot_type}")
    
    config = ROBOT_CONFIGS[robot_type]
    cameras = cameras if cameras is not None else config["cameras"]
    
    # Build action and observation.state features
    features = {
        "action": {
            "dtype": "float32",
            "shape": (config["expected_dim"],),
            "names": config["joint_names"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (config["expected_dim"],),
            "names": config["joint_names"],
        }
    }
    
    # Add camera features
    for cam_name, cam_cfg in cameras.items():
        features[f"observation.images.{cam_name}"] = {
            "dtype": "video",
            "shape": [cam_cfg["height"], cam_cfg["width"], 3],
            "names": ["height", "width", "channels"],
            "video_info": {
                "video.height": cam_cfg["height"],
                "video.width": cam_cfg["width"],
                "video.codec": "libx264",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "video.fps": fps,
                "video.channels": 3,
                "has_audio": False,
            },
        }
    
    return features

def process_data(
    dataset: LeRobotDataset,
    task: str,
    demo_group: h5py.Group,
    demo_name: str,
    frame_skip: int,
    robot_type: str,
    camera_keys: list[str] | None = None,
) -> bool:
    """
    Process a single demonstration group from the HDF5 dataset
    and add it into the LeRobot dataset.
    """
    if robot_type not in ROBOT_CONFIGS:
        raise ValueError(f"Unsupported robot type: {robot_type}")
    
    config = ROBOT_CONFIGS[robot_type]
    camera_keys = list(camera_keys) if camera_keys is not None else list(config["cameras"].keys())
    
    try:
        # Load action and joint position data
        actions = np.array(demo_group['actions'], dtype=np.float32)
        joint_pos = np.array(demo_group['obs/joint_pos'], dtype=np.float32)
        
        # Load camera images based on robot type
        camera_data = {}
        for cam_key in camera_keys:
            camera_data[cam_key] = np.array(demo_group[f'obs/{cam_key}'], dtype=np.uint8)
            
    except KeyError as e:
        print(f"Demo {demo_name} is not valid (missing key: {e}), skipping...")
        return False

    if actions.shape[0] < 10:
        print(f"Demo {demo_name} has insufficient frames ({actions.shape[0]}), skipping...")
        return False

    # Ensure actions and joint positions are 2D arrays
    if actions.ndim == 1:
        actions = actions.reshape(-1, config["expected_dim"])
    if joint_pos.ndim == 1:
        joint_pos = joint_pos.reshape(-1, config["expected_dim"])
    
    if actions.shape[1] != config["expected_dim"]:
        print(
            f"Demo {demo_name} action dim {actions.shape[1]} != "
            f"expected {config['expected_dim']} for {robot_type}, skipping..."
        )
        return False
    if joint_pos.shape[1] != config["expected_dim"]:
        print(
            f"Demo {demo_name} joint_pos dim {joint_pos.shape[1]} != "
            f"expected {config['expected_dim']} for {robot_type}, skipping..."
        )
        return False

    total_state_frames = actions.shape[0]

    # Process each frame
    for frame_index in tqdm(range(total_state_frames), desc=f"Processing demo {demo_name}"):
        if frame_index < frame_skip:
            continue
        
        # Build frame dictionary
        frame = {
            "action": actions[frame_index],
            "observation.state": joint_pos[frame_index],
        }
        
        # Add camera images
        for cam_key in camera_keys:
            frame[f"observation.images.{cam_key}"] = camera_data[cam_key][frame_index]
        
        dataset.add_frame(frame=frame, task=task)

    return True

def resolve_lerobot_v3_python() -> str:
    """Return the experimental lerobot>=0.4 interpreter used for v2.1→v3.0 conversion.

    Never falls back to the default ``lerobot-python`` (0.3.3) venv.
    """
    candidates = [
        os.environ.get("LEROBOT_V3_PYTHON", "").strip(),
        shutil.which("lerobot-python-v3") or "",
        os.path.expanduser("~/lerobot_env_v3/bin/python3"),
        "/root/lerobot_env_v3/bin/python3",
    ]
    for path in candidates:
        if path and os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    raise RuntimeError(
        "LeRobot format v3.0 export requires an experimental venv with lerobot>=0.4.\n"
        "Run: scripts/sim2real/imitation_learning/data_converter/setup_lerobot_v3_env.sh\n"
        "Or set LEROBOT_V3_PYTHON / install alias lerobot-python-v3 pointing at that interpreter.\n"
        "Default lerobot-python (0.3.3 → v2.1) is intentionally not used for conversion."
    )


def convert_v21_dataset_to_v30(
    *,
    root: str | Path,
    repo_id: str,
    push_to_hub: bool = False,
) -> None:
    """Convert a local LeRobot v2.1 dataset folder to v3.0 in place via upstream converter."""
    root = Path(root).resolve()
    if not (root / "meta" / "info.json").is_file():
        raise FileNotFoundError(f"Not a LeRobot dataset root (missing meta/info.json): {root}")

    py = resolve_lerobot_v3_python()
    cmd = [
        py,
        "-m",
        "lerobot.scripts.convert_dataset_v21_to_v30",
        f"--repo-id={repo_id}",
        f"--root={root}",
        f"--push-to-hub={'true' if push_to_hub else 'false'}",
    ]
    print(f"[v3] converting v2.1 → v3.0 with: {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"v2.1→v3.0 conversion failed (exit {proc.returncode}). "
            "Ensure lerobot-python-v3 has lerobot>=0.4."
        )

    info_path = root / "meta" / "info.json"
    try:
        import json

        info = json.loads(info_path.read_text())
        version = info.get("codebase_version")
    except Exception as exc:
        raise RuntimeError(f"Could not read {info_path} after conversion: {exc}") from exc
    if version != "v3.0":
        raise RuntimeError(f"Expected codebase_version v3.0 after convert, got {version!r}")
    print(f"[v3] conversion ok: {root} (codebase_version={version})", flush=True)


def convert_isaaclab_to_lerobot(
    task: str,
    repo_id: str,
    robot_type: str,
    dataset_file: str,
    fps: int,
    push_to_hub: bool = False,
    frame_skip: int = 3,
    root: str = "./datasets/lerobot/sim2real_data",
    dataset_format: str = "v2",
):
    """Convert an IsaacLab HDF5 dataset into LeRobot format (v2.1 default, optional v3.0).

    When ``dataset_format='v3'`` (experimental): write v2.1 first with the installed
    lerobot 0.3.3 writer, then convert to v3.0 using the separate ``lerobot-python-v3``
    interpreter. Default v2 path is unchanged.
    """
    if dataset_format not in ("v2", "v3"):
        raise ValueError(f"dataset_format must be 'v2' or 'v3', got {dataset_format!r}")

    hdf5_files = [dataset_file]
    now_episode_index = 0

    cameras = discover_cameras_in_hdf5(dataset_file, robot_type)
    camera_keys = list(cameras.keys())
    print(f"LeRobot cameras for {robot_type}: {camera_keys}")
    print(f"LeRobot dataset_format={dataset_format} root={root}", flush=True)

    # Create a new LeRobot dataset (always v2.1 writer from pinned lerobot 0.3.3)
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        robot_type=robot_type,
        features=get_env_features(fps, robot_type, cameras=cameras),
        root=root,
    )

    # Process each HDF5 dataset file
    for hdf5_id, hdf5_file in enumerate(hdf5_files):
        print(f"[{hdf5_id+1}/{len(hdf5_files)}] Processing HDF5 file: {hdf5_file}")
        with h5py.File(hdf5_file, "r") as f:
            demo_names = list(f["data"].keys())
            print(f"Found {len(demo_names)} demos: {demo_names}")

            for demo_name in tqdm(demo_names, desc="Processing each demo"):
                demo_group = f["data"][demo_name]

                # Skip unsuccessful demonstrations
                if "success" in demo_group.attrs and not demo_group.attrs["success"]:
                    print(f"Demo {demo_name} not successful, skipping...")
                    continue

                valid = process_data(
                    dataset, task, demo_group, demo_name, frame_skip, robot_type, camera_keys=camera_keys
                )

                if valid:
                    now_episode_index += 1
                    dataset.save_episode()
                    print(f"Saved episode {now_episode_index} successfully")

    if now_episode_index == 0:
        raise RuntimeError(f"No episodes exported from {dataset_file}; aborting")

    if dataset_format == "v3":
        # Hub push for v3 happens after convert (converter may also push when requested).
        convert_v21_dataset_to_v30(root=root, repo_id=repo_id, push_to_hub=push_to_hub)
    elif push_to_hub:
        dataset.push_to_hub()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert IsaacLab dataset to LeRobot format")
    parser.add_argument("--task", type=str, required=True, help="Task name (e.g., OMY_Pickup)")
    parser.add_argument(
        "--robot_type",
        type=str,
        default="OMY",
        choices=["OMY", "FFW_SG2", "FFW_SH5"],
        help="Robot type (OMY, FFW_SG2, or FFW_SH5)",
    )
    parser.add_argument("--dataset_file", type=str, default="./datasets/dataset.hdf5", help="Path to dataset HDF5 file")
    parser.add_argument("--fps", type=int, default=10, help="Frames per second for dataset (default: 10)")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether to push dataset to HuggingFace Hub")
    parser.add_argument("--frame_skip", type=int, default=2, help="Frame skip rate (default: 2)")
    parser.add_argument(
        "--dataset_format",
        type=str,
        default="v2",
        choices=["v2", "v3"],
        help="LeRobot dataset format: v2 (default, v2.1) or v3 (experimental v3.0 via separate venv)",
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    if "--dataset_format" in sys.argv:
        # Peek early so default repo_id lands under lerobot_v3/ when exporting v3.
        peek = argparse.ArgumentParser(add_help=False)
        peek.add_argument("--dataset_format", default="v2")
        peek_args, _ = peek.parse_known_args()
        fmt = peek_args.dataset_format
    else:
        fmt = "v2"
    default_root = (
        f"./datasets/lerobot_v3/{timestamp}" if fmt == "v3" else f"./datasets/lerobot/{timestamp}"
    )
    parser.add_argument("--repo_id", type=str, default=default_root, help=f"Repo ID (default: {default_root})")
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Local dataset root directory (defaults to --repo_id when it is a local path)",
    )

    args = parser.parse_args()
    root = args.root if args.root is not None else args.repo_id

    convert_isaaclab_to_lerobot(
        task=args.task,
        repo_id=args.repo_id,
        robot_type=args.robot_type,
        dataset_file=args.dataset_file,
        fps=args.fps,
        push_to_hub=args.push_to_hub,
        frame_skip=args.frame_skip,
        root=root,
        dataset_format=args.dataset_format,
    )
