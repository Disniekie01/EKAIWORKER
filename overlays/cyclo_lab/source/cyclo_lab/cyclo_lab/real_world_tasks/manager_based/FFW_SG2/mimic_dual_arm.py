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

"""Shared Mimic datagen wiring for dual-arm SG2 cardboard-box tasks."""

from __future__ import annotations

from isaaclab.envs.mimic_env_cfg import SubTaskConfig


def configure_dual_arm_box_mimic(
    cfg,
    *,
    datagen_name: str,
    place_signal: str,
    place_description: str,
    arm_side: str = "right",
) -> None:
    """Attach standard grasp → place subtasks for L-table style box demos."""
    cfg.datagen_config.name = datagen_name
    cfg.datagen_config.generation_guarantee = True
    cfg.datagen_config.generation_keep_failed = False
    cfg.datagen_config.generation_num_trials = 10
    cfg.datagen_config.generation_select_src_per_subtask = True
    cfg.datagen_config.generation_transform_first_robot_pose = False
    cfg.datagen_config.generation_interpolate_from_last_target_pose = True
    cfg.datagen_config.generation_relative = True
    cfg.datagen_config.max_num_failures = 25
    cfg.datagen_config.seed = 42

    subtask_configs = [
        SubTaskConfig(
            object_ref="cardboard_box",
            subtask_term_signal="dual_grasp_box",
            subtask_term_offset_range=(10, 20),
            selection_strategy="nearest_neighbor_object",
            selection_strategy_kwargs={"nn_k": 1},
            action_noise=0.003,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
            description="Dual grasp box",
            next_subtask_description=place_description,
        ),
        SubTaskConfig(
            object_ref="cardboard_box",
            subtask_term_signal=place_signal,
            subtask_term_offset_range=(5, 10),
            selection_strategy="nearest_neighbor_object",
            selection_strategy_kwargs={"nn_k": 1},
            action_noise=0.003,
            num_interpolation_steps=10,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
            description=place_description,
            next_subtask_description="Task complete",
        ),
        SubTaskConfig(
            object_ref=None,
            subtask_term_signal=None,
            subtask_term_offset_range=(0, 0),
            selection_strategy="random",
            selection_strategy_kwargs={},
            action_noise=0.0001,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        ),
    ]
    cfg.subtask_configs[f"{arm_side}_arm"] = subtask_configs
