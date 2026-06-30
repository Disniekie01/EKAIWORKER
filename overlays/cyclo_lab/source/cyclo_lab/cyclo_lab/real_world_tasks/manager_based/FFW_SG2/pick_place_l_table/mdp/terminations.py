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

from .ffw_sg2_l_table_events import get_left_table_drop_zone_world


def task_done(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    table_left_cfg: SceneEntityCfg,
    distance_threshold: float = 0.12,
) -> torch.Tensor:
    """Success when the cardboard box is placed on the left table."""
    obj: RigidObject = env.scene[object_cfg.name]
    drop_zone = get_left_table_drop_zone_world(env, table_left_cfg)
    distance = torch.linalg.vector_norm(obj.data.root_pos_w - drop_zone, dim=1)
    return distance < distance_threshold


def object_dropped(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    velocity_threshold: float = 2.0,
) -> torch.Tensor:
    """Failure when the object is falling too fast."""
    obj: RigidObject = env.scene[object_cfg.name]
    velocity = torch.linalg.vector_norm(obj.data.root_lin_vel_w, dim=1)
    return velocity > velocity_threshold
