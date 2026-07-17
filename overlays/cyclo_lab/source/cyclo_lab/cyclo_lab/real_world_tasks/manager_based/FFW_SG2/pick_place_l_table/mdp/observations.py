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

from .ffw_sg2_l_table_events import (
    LEFT_TABLE_DROP_HEIGHT_TOLERANCE,
    LEFT_TABLE_EDGE_MARGIN,
    is_object_on_left_table_top,
)


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
    edge_margin: float = LEFT_TABLE_EDGE_MARGIN,
    height_tolerance: float = LEFT_TABLE_DROP_HEIGHT_TOLERANCE,
) -> torch.Tensor:
    """Check if the object is placed anywhere on the left table tabletop."""
    return is_object_on_left_table_top(
        env, object_cfg, table_left_cfg, edge_margin=edge_margin, height_tolerance=height_tolerance
    )


def base_planar_velocity(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Base planar velocity in the base frame: ``[linear_x, linear_y, angular_z]``.

    Matches the real ffw_sg2_rev1 base-velocity convention appended to the 19-joint state
    (state/action dims 19 -> 22). Only meaningful on the drivable base (FFW_SG2_MOBILE);
    on the welded stock base it stays ~0.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    lin_b = asset.data.root_lin_vel_b  # (num_envs, 3) linear velocity, base frame
    ang_b = asset.data.root_ang_vel_b  # (num_envs, 3) angular velocity, base frame
    return torch.cat([lin_b[:, :2], ang_b[:, 2:3]], dim=-1)
