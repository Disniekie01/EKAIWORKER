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

"""Shared constants and scene wiring for FFW-SH5 hand teleoperation tasks."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg, FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg

from isaaclab.markers.config import FRAME_MARKER_CFG
from cyclo_lab.assets.robots.FFW_SH5 import FFW_SH5_CFG

# When the FFW_SH5 USD is referenced onto {ENV_REGEX_NS}/Robot, its default
# prim (ffw_sh5_follower) is consumed into Robot, so at runtime every link
# lives under Robot/base_link/... (confirmed by the articulation root being
# "/base_link/base_link"). The camera parent (zed), arm_base_link and
# arm_*_link7 are therefore all nested below base_link.
SH5_FOLLOWER = "base_link"
SH5_URDF = (
    "/root/ros2_ws/install/ffw_description/share/ffw_description/urdf/"
    "ffw_sh5_rev1_follower/ffw_sh5_follower.urdf"
)

SH5_FINGER_L_JOINTS = [f"finger_l_joint{i}" for i in range(1, 21)]
SH5_FINGER_R_JOINTS = [f"finger_r_joint{i}" for i in range(1, 21)]
SH5_POLICY_JOINT_NAMES = (
    [f"arm_l_joint{i}" for i in range(1, 8)]
    + SH5_FINGER_L_JOINTS
    + [f"arm_r_joint{i}" for i in range(1, 8)]
    + SH5_FINGER_R_JOINTS
    + ["head_joint1", "head_joint2", "lift_joint"]
)


def attach_sh5_robot_and_sensors(scene) -> None:
    """Wire SH5 robot, head camera, and arm tip frame transformers onto a scene cfg."""
    scene.robot = FFW_SH5_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    scene.robot.spawn.semantic_tags = [("class", "robot")]
    scene.plane.semantic_tags = [("class", "ground")]

    scene.cam_head = CameraCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/{SH5_FOLLOWER}/head_link2/zed/cam_head",
        update_period=0.0,
        height=376,
        width=672,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=10.4,
            focus_distance=200.0,
            horizontal_aperture=20.955,
            clipping_range=(0.01, 100.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.0, 0.03, 0.0),
            rot=(0.5, 0.5, -0.5, -0.5),
            convention="isaac",
        ),
    )

    marker_cfg = FRAME_MARKER_CFG.copy()
    marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    marker_cfg.prim_path = "/Visuals/FrameTransformer"

    scene.right_eef = FrameTransformerCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/{SH5_FOLLOWER}/arm_base_link",
        debug_vis=False,
        visualizer_cfg=marker_cfg,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Robot/{SH5_FOLLOWER}/arm_r_link7",
                name="end_effector",
                offset=OffsetCfg(pos=[0.0, 0.0, -0.2]),
            ),
        ],
    )

    scene.left_eef = FrameTransformerCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/{SH5_FOLLOWER}/arm_base_link",
        debug_vis=False,
        visualizer_cfg=marker_cfg,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Robot/{SH5_FOLLOWER}/arm_l_link7",
                name="end_effector",
                offset=OffsetCfg(pos=[0.0, 0.0, -0.2]),
            ),
        ],
    )


def policy_joint_obs_params() -> dict:
    return {"joint_names": list(SH5_POLICY_JOINT_NAMES), "asset_name": "robot"}


def init_sh5_action_cfg(actions, mode: str, mdp, diff_ik_cfg_cls, diff_ik_action_cfg_cls, diff_ik_controller_cfg_cls):
    """Configure arm + finger actions for record/inference or mimic_ik modes."""
    if mode in ["record", "inference"]:
        actions.arm_l_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["arm_l_joint[1-7]"],
            scale=1.0,
            use_default_offset=False,
        )
        actions.finger_l_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["finger_l_joint[1-9]", "finger_l_joint1[0-9]", "finger_l_joint20"],
            scale=1.0,
            use_default_offset=False,
        )
        actions.arm_r_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["arm_r_joint[1-7]"],
            scale=1.0,
            use_default_offset=False,
        )
        actions.finger_r_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["finger_r_joint[1-9]", "finger_r_joint1[0-9]", "finger_r_joint20"],
            scale=1.0,
            use_default_offset=False,
        )
        actions.head_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["head_joint1", "head_joint2"],
            scale=1.0,
        )
        actions.lift_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["lift_joint"],
            scale=1.0,
        )
    elif mode in ["mimic_ik"]:
        actions.arm_l_action = diff_ik_action_cfg_cls(
            asset_name="robot",
            joint_names=["arm_l_joint[1-7]"],
            body_name="arm_l_link7",
            controller=diff_ik_controller_cfg_cls(
                command_type="pose",
                ik_params={"lambda_val": 0.05},
                ik_method="dls",
                use_relative_mode=False,
            ),
            body_offset=diff_ik_action_cfg_cls.OffsetCfg(pos=[0.0, 0.0, -0.2]),
        )
        actions.finger_l_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["finger_l_joint[1-9]", "finger_l_joint1[0-9]", "finger_l_joint20"],
            scale=1.0,
            use_default_offset=False,
        )
        actions.arm_r_action = diff_ik_action_cfg_cls(
            asset_name="robot",
            joint_names=["arm_r_joint[1-7]"],
            body_name="arm_r_link7",
            controller=diff_ik_controller_cfg_cls(
                command_type="pose",
                ik_params={"lambda_val": 0.05},
                ik_method="dls",
                use_relative_mode=False,
            ),
            body_offset=diff_ik_action_cfg_cls.OffsetCfg(pos=[0.0, 0.0, -0.2]),
        )
        actions.finger_r_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["finger_r_joint[1-9]", "finger_r_joint1[0-9]", "finger_r_joint20"],
            scale=1.0,
            use_default_offset=False,
        )
        actions.head_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["head_joint1", "head_joint2"],
            scale=1.0,
        )
        actions.lift_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["lift_joint"],
            scale=1.0,
        )
    else:
        raise ValueError(f"Unknown action mode: {mode}")
