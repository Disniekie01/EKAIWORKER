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

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _hand_curl_score(robot: Articulation, finger_joint_names: list[str]) -> torch.Tensor:
    """Heuristic curl score: higher means fingers are more closed."""
    indices = [robot.joint_names.index(name) for name in finger_joint_names if name in robot.joint_names]
    if not indices:
        return torch.zeros(robot.data.joint_pos.shape[0], device=robot.device)
    return robot.data.joint_pos[:, indices].mean(dim=1)


def object_dual_grasped(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    left_eef_cfg: SceneEntityCfg,
    right_eef_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    diff_threshold: float = 0.18,
    finger_close_threshold: float = 0.15,
) -> torch.Tensor:
    """Check if the object is grasped by both hands (proximity + finger curl)."""
    robot: Articulation = env.scene[robot_cfg.name]
    left_eef: FrameTransformer = env.scene[left_eef_cfg.name]
    right_eef: FrameTransformer = env.scene[right_eef_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]

    object_pos = obj.data.root_pos_w
    left_pos = left_eef.data.target_pos_w[:, 0, :]
    right_pos = right_eef.data.target_pos_w[:, 0, :]
    midpoint = (left_pos + right_pos) * 0.5

    left_fingers = [f"finger_l_joint{i}" for i in range(1, 21)]
    right_fingers = [f"finger_r_joint{i}" for i in range(1, 21)]
    left_curl = _hand_curl_score(robot, left_fingers)
    right_curl = _hand_curl_score(robot, right_fingers)

    close_threshold = torch.tensor(finger_close_threshold, device=env.device)
    both_closed = torch.logical_and(left_curl >= close_threshold, right_curl >= close_threshold)
    near_midpoint = torch.linalg.vector_norm(object_pos - midpoint, dim=1) < diff_threshold

    return torch.logical_and(both_closed, near_midpoint)
