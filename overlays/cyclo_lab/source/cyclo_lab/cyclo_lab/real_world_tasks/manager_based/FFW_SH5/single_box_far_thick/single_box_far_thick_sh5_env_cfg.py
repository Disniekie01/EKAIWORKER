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

import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg as RecordTerm
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg
from isaaclab.utils import configclass
from isaaclab.sensors import CameraCfg

from . import mdp
from cyclo_lab.real_world_tasks.manager_based.FFW_SH5._shared import (
    init_sh5_action_cfg,
    policy_joint_obs_params,
)
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from cyclo_lab.real_world_tasks.manager_based.FFW_SG2.single_box_far_thick.mdp.single_box_far_thick_events import (
    REAR_TABLE_DISTANCE_M,
)


@configclass
class SingleBoxFarThickSceneCfg(InteractiveSceneCfg):
    """Front pick table + rear place table 3 m behind the robot (thick box)."""

    robot: ArticulationCfg = MISSING
    left_eef: FrameTransformerCfg = MISSING
    right_eef: FrameTransformerCfg = MISSING

    table_front: AssetBaseCfg = MISSING
    table_rear: AssetBaseCfg = MISSING
    cardboard_box: AssetBaseCfg = MISSING
    box_riser: AssetBaseCfg = MISSING
    cam_head: CameraCfg = MISSING

    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, 0.0]),
        spawn=GroundPlaneCfg(),
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


@configclass
class ActionsCfg:
    arm_l_action: mdp.ActionTermCfg = MISSING
    finger_l_action: mdp.ActionTermCfg = MISSING
    arm_r_action: mdp.ActionTermCfg = MISSING
    finger_r_action: mdp.ActionTermCfg = MISSING
    lift_action: mdp.ActionTermCfg = MISSING
    head_action: mdp.ActionTermCfg = MISSING


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        actions = ObsTerm(func=mdp.last_action)
        joint_pos = ObsTerm(
            func=mdp.joint_pos_name,
            params={
                **policy_joint_obs_params(),
            },
        )
        joint_pos_target = ObsTerm(
            func=mdp.joint_pos_target_name,
            params={
                **policy_joint_obs_params(),
            },
        )
        left_eef_pose = ObsTerm(
            func=mdp.eef_pose,
            params={"eef_cfg": SceneEntityCfg("left_eef"), "robot_cfg": SceneEntityCfg("robot")},
        )
        right_eef_pose = ObsTerm(
            func=mdp.eef_pose,
            params={"eef_cfg": SceneEntityCfg("right_eef"), "robot_cfg": SceneEntityCfg("robot")},
        )
        cam_head = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("cam_head"), "data_type": "rgb", "normalize": False},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class SubtaskCfg(ObsGroup):
        dual_grasp_box = ObsTerm(
            func=mdp.object_dual_grasped,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "left_eef_cfg": SceneEntityCfg("left_eef"),
                "right_eef_cfg": SceneEntityCfg("right_eef"),
                "object_cfg": SceneEntityCfg("cardboard_box"),
            },
        )
        box_on_rear_table = ObsTerm(
            func=mdp.object_on_rear_table,
            params={
                "object_cfg": SceneEntityCfg("cardboard_box"),
                "table_rear_cfg": SceneEntityCfg("table_rear"),
                "distance_threshold": 0.12,
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()
    subtask_terms: SubtaskCfg = SubtaskCfg()


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=mdp.task_done,
        params={
            "object_cfg": SceneEntityCfg("cardboard_box"),
            "table_rear_cfg": SceneEntityCfg("table_rear"),
            "distance_threshold": 0.12,
        },
    )
    object_dropped = DoneTerm(
        func=mdp.object_dropped,
        params={
            "object_cfg": SceneEntityCfg("cardboard_box"),
            "velocity_threshold": 2.0,
        },
    )


@configclass
class SingleBoxFarThickSH5EnvCfg(ManagerBasedRLEnvCfg):
    """Thick box pick from front table and place on a rear table 3 m behind the robot."""

    scene: SingleBoxFarThickSceneCfg = SingleBoxFarThickSceneCfg(
        num_envs=4096,
        env_spacing=max(8.0, REAR_TABLE_DISTANCE_M + 5.0),
        replicate_physics=False,
    )
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    recorders: RecordTerm = RecordTerm()

    commands = None
    rewards = None
    events = None
    curriculum = None

    # VR teleop L-key motion (read by FFWSG2Sdk).
    teleop_l_yaw: float = math.pi
    teleop_l_forward_m: float = REAR_TABLE_DISTANCE_M - 0.7
    teleop_l_forward_duration_s: float = 8.0
    teleop_l_rotation_duration_s: float = 3.0
    teleop_l_target_label: str = "rear table"
    teleop_l_use_swerve: bool = True

    def __post_init__(self):
        self.decimation = 5
        self.episode_length_s = 60.0
        self.sim.dt = 0.01
        self.sim.render_interval = 2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625

    def init_action_cfg(self, mode: str):
        init_sh5_action_cfg(
            self.actions,
            mode,
            mdp,
            DifferentialInverseKinematicsActionCfg,
            DifferentialInverseKinematicsActionCfg,
            DifferentialIKControllerCfg,
        )
