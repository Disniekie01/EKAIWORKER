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

import gymnasium as gym

gym.register(
    id="Cyclo-Real-Single-Box-Far-FFW-SG2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:FFWSG2SingleBoxFarEnvCfg",
    },
    disable_env_checker=True,
)

_MIMIC_ENTRY = (
    "cyclo_lab.real_world_tasks.manager_based.FFW_SG2.pick_place.pick_place_mimic_env:"
    "FFWSG2PickPlaceMimicEnv"
)

gym.register(
    id="Cyclo-Real-Mimic-Single-Box-Far-FFW-SG2-v0",
    entry_point=_MIMIC_ENTRY,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.single_box_far_mimic_env_cfg:FFWSG2SingleBoxFarMimicEnvCfg",
    },
    disable_env_checker=True,
)
