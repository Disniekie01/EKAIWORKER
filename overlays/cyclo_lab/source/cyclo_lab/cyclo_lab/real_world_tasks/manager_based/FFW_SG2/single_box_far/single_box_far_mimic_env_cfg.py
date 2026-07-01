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

from isaaclab.envs.mimic_env_cfg import MimicEnvCfg
from isaaclab.utils import configclass

from cyclo_lab.real_world_tasks.manager_based.FFW_SG2.mimic_dual_arm import configure_dual_arm_box_mimic
from cyclo_lab.real_world_tasks.manager_based.FFW_SG2.single_box_far.joint_pos_env_cfg import (
    FFWSG2SingleBoxFarEnvCfg,
)


@configclass
class FFWSG2SingleBoxFarMimicEnvCfg(FFWSG2SingleBoxFarEnvCfg, MimicEnvCfg):
    """Mimic config for rear-table single box task."""

    def __post_init__(self):
        super().__post_init__()
        configure_dual_arm_box_mimic(
            self,
            datagen_name="single_box_far_rear_table",
            place_signal="box_on_rear_table",
            place_description="Place box on rear table",
        )
