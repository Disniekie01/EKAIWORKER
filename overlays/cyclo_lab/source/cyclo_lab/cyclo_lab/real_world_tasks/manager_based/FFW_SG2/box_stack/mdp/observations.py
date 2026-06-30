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

from .box_stack_events import get_left_table_drop_zone_world


def object_dual_grasped(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    left_eef_cfg: SceneEntityCfg,
    right_eef_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    diff_threshold: float = 0.18,
    gripper_close_threshold: float = 0.2,
) -> torch.Tensor:
    """Check if the object is grasped by both grippers simultaneously."""
    robot: Articulation = env.scene[robot_cfg.name]
    left_eef: FrameTransformer = env.scene[left_eef_cfg.name]
    right_eef: FrameTransformer = env.scene[right_eef_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]

    object_pos = obj.data.root_pos_w
    left_pos = left_eef.data.target_pos_w[:, 0, :]
    right_pos = right_eef.data.target_pos_w[:, 0, :]
    midpoint = (left_pos + right_pos) * 0.5

    left_gripper = robot.data.joint_pos[:, robot.joint_names.index("gripper_l_joint1")]
    right_gripper = robot.data.joint_pos[:, robot.joint_names.index("gripper_r_joint1")]

    close_threshold = torch.tensor(gripper_close_threshold, device=env.device)
    both_closed = torch.logical_and(left_gripper >= close_threshold, right_gripper >= close_threshold)
    near_midpoint = torch.linalg.vector_norm(object_pos - midpoint, dim=1) < diff_threshold

    return torch.logical_and(both_closed, near_midpoint)


def object_on_left_table(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    table_left_cfg: SceneEntityCfg,
    distance_threshold: float = 0.12,
) -> torch.Tensor:
    """Check if the object is placed on the left table drop zone."""
    obj: RigidObject = env.scene[object_cfg.name]
    drop_zone = get_left_table_drop_zone_world(env, table_left_cfg)
    distance = torch.linalg.vector_norm(obj.data.root_pos_w - drop_zone, dim=1)
    return distance < distance_threshold
