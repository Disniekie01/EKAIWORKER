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

import math
import random
import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

# ---- Geometry (env-local frame, relative to the env origin) ----
# Tables are prim cuboids with legs; the body origin sits on the floor and the
# work surface is at z = TABLE_HEIGHT.
TABLE_HEIGHT = 0.72
BOX_HALF_HEIGHT = 0.0075

# A small pedestal sits on the front table under the box. The box overhangs it
# on every side so the grippers can slide underneath the overhanging edges to
# grasp the box. Box rests on top of the riser instead of flush on the table.
RISER_HEIGHT = 0.08
RISER_TOP_Z = TABLE_HEIGHT + RISER_HEIGHT
BOX_TOP_Z = RISER_TOP_Z + BOX_HALF_HEIGHT

# Front table: directly in front of the robot (origin on the floor).
# Spans X in [0.20, 0.80], Y in [-0.55, 0.55].
FRONT_TABLE_POS = (0.50, 0.00, 0.0)
# Left table rotated 90 deg about Z so its long axis (1.10) runs along X.
LEFT_TABLE_YAW = math.pi / 2.0
# Positioned so its far end lines up with the front table's far edge (X = 0.80)
# and it extends back toward the robot. Center X = 0.80 - 1.10/2 = 0.25.
# Near Y-edge at 0.85 - 0.60/2 = 0.55 meets the front table's far Y-edge.
LEFT_TABLE_POS = (0.25, 0.85, 0.0)

# Cardboard box starts centered on the front table, resting on top of the riser.
BOX_POS = (0.50, 0.00, BOX_TOP_Z)
# Riser center sits on the front table surface, directly under the box.
RISER_CENTER_Z = TABLE_HEIGHT + RISER_HEIGHT / 2.0
# When placed on the (bare, no-riser) left table, the box rests directly on the
# surface. The drop zone uses this height, not the raised front-table BOX_TOP_Z.
BOX_ON_TABLE_Z = TABLE_HEIGHT + BOX_HALF_HEIGHT
# Target drop zone: on the left table surface near the L-corner (reachable).
LEFT_TABLE_DROP_ZONE = (0.55, 0.70, BOX_ON_TABLE_Z)


def _write_pose(env: ManagerBasedEnv, asset, cur_env: int, pos_xyz, yaw: float = 0.0):
    position = torch.tensor([list(pos_xyz)], device=env.device) + env.scene.env_origins[cur_env, 0:3]
    orientation = math_utils.quat_from_euler_xyz(
        torch.tensor([0.0], device=env.device),
        torch.tensor([0.0], device=env.device),
        torch.tensor([yaw], device=env.device),
    )
    env_ids = torch.tensor([cur_env], device=env.device)
    asset.write_root_pose_to_sim(torch.cat([position, orientation], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(torch.zeros(1, 6, device=env.device), env_ids=env_ids)


def randomize_l_table_scene(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    table_front_cfg: SceneEntityCfg,
    table_left_cfg: SceneEntityCfg,
    box_cfg: SceneEntityCfg,
    box_riser_cfg: SceneEntityCfg | None = None,
    box_pose_range: dict[str, tuple[float, float]] | None = None,
):
    """Place the two prim tables in an L formation and spawn the box (on a riser)
    on the front table."""
    if env_ids is None:
        return

    if box_pose_range is None:
        box_pose_range = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}

    table_front = env.scene[table_front_cfg.name]
    table_left = env.scene[table_left_cfg.name]
    box = env.scene[box_cfg.name]
    box_riser = env.scene[box_riser_cfg.name] if box_riser_cfg is not None else None

    for cur_env in env_ids.tolist():
        _write_pose(env, table_front, cur_env, FRONT_TABLE_POS)
        _write_pose(env, table_left, cur_env, LEFT_TABLE_POS, yaw=LEFT_TABLE_YAW)

        dx = random.uniform(*box_pose_range.get("x", (0.0, 0.0)))
        dy = random.uniform(*box_pose_range.get("y", (0.0, 0.0)))
        dyaw = random.uniform(*box_pose_range.get("yaw", (0.0, 0.0)))

        # Riser sits on the table under the box, sharing its XY and yaw so the
        # box stays centered and overhangs the riser edges.
        if box_riser is not None:
            riser_pos = (BOX_POS[0] + dx, BOX_POS[1] + dy, RISER_CENTER_Z)
            _write_pose(env, box_riser, cur_env, riser_pos, yaw=dyaw)

        box_pos = (BOX_POS[0] + dx, BOX_POS[1] + dy, BOX_POS[2])
        _write_pose(env, box, cur_env, box_pos, yaw=dyaw)


def get_left_table_drop_zone_world(
    env: ManagerBasedEnv,
    table_left_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """World-frame drop zone position on the left table for each env."""
    num_envs = env.scene.env_origins.shape[0]
    rel = torch.tensor([list(LEFT_TABLE_DROP_ZONE)], device=env.device).expand(num_envs, -1)
    return env.scene.env_origins[:, 0:3] + rel
