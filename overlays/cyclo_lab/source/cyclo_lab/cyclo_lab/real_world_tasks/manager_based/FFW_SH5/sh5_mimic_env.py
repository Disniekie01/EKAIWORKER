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

"""Isaac Lab Mimic env for FFW-SH5 dual-arm + dexterous-hand tasks."""

from __future__ import annotations

import torch
from collections.abc import Sequence

import isaaclab.utils.math as PoseUtils
from isaaclab.envs import ManagerBasedRLMimicEnv, ManagerBasedRLEnvCfg

# Joint / IK action layout (57 dims):
# [left_eef(7), finger_l(20), right_eef(7), finger_r(20), head(2), lift(1)]
_IK_LEFT_EEF = slice(0, 7)
_IK_FINGER_L = slice(7, 27)
_IK_RIGHT_EEF = slice(27, 34)
_IK_FINGER_R = slice(34, 54)
_IK_HEAD = slice(54, 56)
_IK_LIFT = slice(56, 57)

_JOINT_FINGER_L = slice(7, 27)
_JOINT_FINGER_R = slice(34, 54)
_JOINT_HEAD = slice(54, 56)
_JOINT_LIFT = slice(56, 57)


class FFWSH5DualArmMimicEnv(ManagerBasedRLMimicEnv):
    """Mimic wrapper for SH5 box tasks (arms in IK, fingers from joint_pos_target)."""

    def __init__(self, cfg: ManagerBasedRLEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.robot_root_pos = self.scene["robot"].data.root_pos_w
        self.robot_root_quat = self.scene["robot"].data.root_quat_w

    def get_robot_eef_pose(self, eef_name: str, env_ids: Sequence[int] | None = None) -> torch.Tensor:
        if env_ids is None:
            env_ids = slice(None)

        if "right" in eef_name.lower():
            eef_pose = self.obs_buf["policy"]["right_eef_pose"][env_ids]
        elif "left" in eef_name.lower():
            eef_pose = self.obs_buf["policy"]["left_eef_pose"][env_ids]
        else:
            eef_pose = self.obs_buf["policy"]["right_eef_pose"][env_ids]

        eef_pos = eef_pose[:, :3]
        eef_quat = eef_pose[:, 3:7]
        return PoseUtils.make_pose(eef_pos, PoseUtils.matrix_from_quat(eef_quat))

    def _joint_pos_row(self, env_id: int) -> torch.Tensor:
        joint_pos_target = self.obs_buf["policy"]["joint_pos_target"]
        if joint_pos_target.dim() > 1:
            return joint_pos_target[env_id]
        return joint_pos_target

    def _eef_pose_row(self, key: str, env_id: int) -> torch.Tensor:
        eef_pose = self.obs_buf["policy"][key]
        if eef_pose.dim() > 1:
            return eef_pose[env_id, :7]
        return eef_pose[:7]

    def target_eef_pose_to_action(
        self,
        target_eef_pose_dict: dict,
        gripper_action_dict: dict,
        action_noise_dict: dict | None = None,
        env_id: int = 0,
    ) -> torch.Tensor:
        eef_name = list(self.cfg.subtask_configs.keys())[0]
        joint_row = self._joint_pos_row(env_id)
        finger_l = joint_row[_JOINT_FINGER_L]
        finger_r = joint_row[_JOINT_FINGER_R]
        head_action = joint_row[_JOINT_HEAD]
        lift_action = joint_row[_JOINT_LIFT]

        if "right" in eef_name.lower():
            target_pose = target_eef_pose_dict[eef_name]
            pos, rot = PoseUtils.unmake_pose(target_pose)
            right_pose_action = torch.cat([pos, PoseUtils.quat_from_matrix(rot)], dim=0)
            left_pose_action = self._eef_pose_row("left_eef_pose", env_id)
        elif "left" in eef_name.lower():
            target_pose = target_eef_pose_dict[eef_name]
            pos, rot = PoseUtils.unmake_pose(target_pose)
            left_pose_action = torch.cat([pos, PoseUtils.quat_from_matrix(rot)], dim=0)
            right_pose_action = self._eef_pose_row("right_eef_pose", env_id)
        else:
            right_pose_action = self._eef_pose_row("right_eef_pose", env_id)
            left_pose_action = self._eef_pose_row("left_eef_pose", env_id)

        action = torch.cat(
            [
                left_pose_action,
                finger_l,
                right_pose_action,
                finger_r,
                head_action,
                lift_action,
            ],
            dim=0,
        )
        return action.unsqueeze(0)

    def action_to_target_eef_pose(self, action: torch.Tensor) -> dict[str, torch.Tensor]:
        eef_name = list(self.cfg.subtask_configs.keys())[0]

        if "right" in eef_name.lower():
            target_eef_pos = action[:, _IK_RIGHT_EEF.start : _IK_RIGHT_EEF.start + 3]
            target_eef_quat = action[:, _IK_RIGHT_EEF.start + 3 : _IK_RIGHT_EEF.stop]
        elif "left" in eef_name.lower():
            target_eef_pos = action[:, _IK_LEFT_EEF.start : _IK_LEFT_EEF.start + 3]
            target_eef_quat = action[:, _IK_LEFT_EEF.start + 3 : _IK_LEFT_EEF.stop]
        else:
            target_eef_pos = action[:, _IK_RIGHT_EEF.start : _IK_RIGHT_EEF.start + 3]
            target_eef_quat = action[:, _IK_RIGHT_EEF.start + 3 : _IK_RIGHT_EEF.stop]

        target_eef_rot = PoseUtils.matrix_from_quat(target_eef_quat)
        target_eef_pose = PoseUtils.make_pose(target_eef_pos, target_eef_rot).clone()
        return {eef_name: target_eef_pose}

    def actions_to_gripper_actions(self, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        """Scalar finger-curl proxy for Mimic API (full hand pose comes from joint_pos_target)."""
        eef_name = list(self.cfg.subtask_configs.keys())[0]
        if "right" in eef_name.lower():
            curl = actions[:, _IK_FINGER_R].mean(dim=1, keepdim=True)
        elif "left" in eef_name.lower():
            curl = actions[:, _IK_FINGER_L].mean(dim=1, keepdim=True)
        else:
            curl = actions[:, _IK_FINGER_R].mean(dim=1, keepdim=True)
        return {eef_name: curl}

    def get_subtask_term_signals(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        if env_ids is None:
            env_ids = slice(None)

        signals = {}
        subtask_terms = self.obs_buf["subtask_terms"]
        for term_name, term_signal in subtask_terms.items():
            signals[term_name] = term_signal[env_ids]
        return signals
