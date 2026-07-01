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

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from .ffw_sg2_l_table_events import (
    LEFT_TABLE_DROP_HEIGHT_TOLERANCE,
    LEFT_TABLE_EDGE_MARGIN,
    is_object_on_left_table_top,
)


def task_done(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    table_left_cfg: SceneEntityCfg,
    edge_margin: float = LEFT_TABLE_EDGE_MARGIN,
    height_tolerance: float = LEFT_TABLE_DROP_HEIGHT_TOLERANCE,
) -> torch.Tensor:
    """Success when the cardboard box is placed on the left table tabletop."""
    return is_object_on_left_table_top(
        env, object_cfg, table_left_cfg, edge_margin=edge_margin, height_tolerance=height_tolerance
    )


def object_dropped(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    velocity_threshold: float = 2.0,
) -> torch.Tensor:
    """Failure when the object is falling too fast."""
    obj: RigidObject = env.scene[object_cfg.name]
    velocity = torch.linalg.vector_norm(obj.data.root_lin_vel_w, dim=1)
    return velocity > velocity_threshold
