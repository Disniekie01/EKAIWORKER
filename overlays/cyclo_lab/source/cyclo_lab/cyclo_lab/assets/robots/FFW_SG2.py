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
#
# Author: Taehyeong Kim

import re

from isaacsim.core.utils.stage import get_current_stage
from pxr import Usd

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.sim import (
    ArticulationRootPropertiesCfg,
    RigidBodyMaterialCfg,
    RigidBodyPropertiesCfg,
    UsdFileCfg,
)
from isaaclab.sim.spawners.from_files import from_files
from isaaclab.sim.utils import bind_physics_material, clone, make_uninstanceable

from cyclo_lab.assets.robots import CYCLO_LAB_ASSETS_DATA_DIR


_SG2_GRIPPER_TIP_MATERIAL = RigidBodyMaterialCfg(
    friction_combine_mode="max",
    restitution_combine_mode="min",
    static_friction=2.0,
    dynamic_friction=1.8,
    restitution=0.0,
)

# SG2 swerve base (same geometry as SH5; joint names differ in the USD).
SG2_SWERVE_STEERING_JOINTS = ("left_wheel_steer", "right_wheel_steer", "rear_wheel_steer")
SG2_SWERVE_WHEEL_JOINTS = ("left_wheel_drive", "right_wheel_drive", "rear_wheel_drive")
SG2_SWERVE_MODULE_X_OFFSETS = (0.1371, 0.1374, -0.289)
SG2_SWERVE_MODULE_Y_OFFSETS = (0.2554, -0.2554, 0.0)
SG2_SWERVE_MODULE_ANGLE_OFFSETS = (0.0, 0.0, 0.0)
SG2_SWERVE_WHEEL_RADIUS = 0.0865


def _is_sg2_gripper_contact_prim(prim_path: str) -> bool:
    """Return true for RH-P12 finger-link collision prims (both proximal and distal)."""
    path = prim_path.lower()
    if "/collisions/" not in path:
        return False
    if "_base" in path:
        return False
    return re.search(r"gripper_[lr]_rh_p12_rn_[lr][12](/|_|$)", path) is not None


def _collect_sg2_gripper_friction_prims(stage, prim_path: str) -> tuple[set[str], set[str]]:
    """Collect finger collision prims for left and right RH-P12 grippers."""
    left_paths: set[str] = set()
    right_paths: set[str] = set()
    for child_prim in _iter_robot_prims(stage, prim_path):
        child_path = str(child_prim.GetPath())
        if not _is_sg2_gripper_contact_prim(child_path):
            continue
        if "gripper_l_rh_p12_rn_" in child_path.lower():
            left_paths.add(child_path)
        elif "gripper_r_rh_p12_rn_" in child_path.lower():
            right_paths.add(child_path)
    return left_paths, right_paths


def _iter_robot_prims(stage, prim_path: str):
    robot_prim = stage.GetPrimAtPath(prim_path)
    if not robot_prim.IsValid():
        return ()
    return Usd.PrimRange(robot_prim)


@clone
def spawn_sg2_with_gripper_friction(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """Spawn SG2 and bind high-friction material to RH-P12 gripper finger pads."""
    prim = from_files.spawn_from_usd(prim_path, cfg, translation, orientation, **kwargs)

    material_path = f"{prim_path}/gripperTipPhysicsMaterial"
    _SG2_GRIPPER_TIP_MATERIAL.func(material_path, _SG2_GRIPPER_TIP_MATERIAL)

    stage = get_current_stage()
    make_uninstanceable(prim_path, stage)

    left_paths, right_paths = _collect_sg2_gripper_friction_prims(stage, prim_path)
    friction_prim_paths = left_paths | right_paths

    for friction_prim_path in friction_prim_paths:
        bind_physics_material(friction_prim_path, material_path)

    if friction_prim_paths:
        print(
            f"[SG2 gripper friction] bound high-friction material to "
            f"{len(left_paths)} left + {len(right_paths)} right finger collision prim(s).",
            flush=True,
        )
    else:
        print("[SG2 gripper friction] WARNING: no gripper collision prims matched.", flush=True)

    return prim


FFW_SG2_CFG = ArticulationCfg(
    spawn=UsdFileCfg(
        func=spawn_sg2_with_gripper_friction,
        usd_path=f"{CYCLO_LAB_ASSETS_DATA_DIR}/robots/FFW/FFW_SG2.usd",
        rigid_props=RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=32,
            solver_velocity_iteration_count=1,
        ),
        activate_contact_sensors=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            # Swerve base joints
            "left_wheel_drive": 0.0,
            "left_wheel_steer": 0.0,
            "right_wheel_drive": 0.0,
            "right_wheel_steer": 0.0,
            "rear_wheel_drive": 0.0,
            "rear_wheel_steer": 0.0,

            # Left arm joints
            **{f"arm_l_joint{i + 1}": 0.0 for i in range(7)},
            # Right arm joints
            **{f"arm_r_joint{i + 1}": 0.0 for i in range(7)},

            # Left and right gripper joints
            **{f"gripper_l_joint{i + 1}": 0.0 for i in range(4)},
            **{f"gripper_r_joint{i + 1}": 0.0 for i in range(4)},

            # Head joints
            "head_joint1": 0.0,
            "head_joint2": 0.0,

            # Lift joint
            "lift_joint": 0.0,
        },
    ),
    actuators={
        # Actuators for swerve base
        "base_steer": ImplicitActuatorCfg(
            joint_names_expr=list(SG2_SWERVE_STEERING_JOINTS),
            velocity_limit_sim=10.0,
            effort_limit_sim=100000.0,
            stiffness=10000.0,
            damping=100.0,
        ),
        "base_drive": ImplicitActuatorCfg(
            joint_names_expr=list(SG2_SWERVE_WHEEL_JOINTS),
            velocity_limit_sim=50.0,
            effort_limit_sim=100000.0,
            stiffness=0.0,
            damping=100.0,
        ),

        # Actuator for vertical lift joint
        "lift": ImplicitActuatorCfg(
            joint_names_expr=["lift_joint"],
            velocity_limit_sim=0.2,
            effort_limit_sim=1000000.0,
            stiffness=10000.0,
            damping=100.0,
        ),

        # Actuators for both arms
        "DY_80": ImplicitActuatorCfg(
            joint_names_expr=[
                "arm_l_joint[1-2]",
                "arm_r_joint[1-2]",
            ],
            velocity_limit_sim=15.0,
            effort_limit_sim=61.4,
            stiffness=600.0,
            damping=30.0,
        ),
        "DY_70": ImplicitActuatorCfg(
            joint_names_expr=[
                "arm_l_joint[3-6]",
                "arm_r_joint[3-6]",
            ],
            velocity_limit_sim=15.0,
            effort_limit_sim=31.7,
            stiffness=600.0,
            damping=20.0,
        ),
        "DP-42" : ImplicitActuatorCfg(
            joint_names_expr=[
                "arm_l_joint7",
                "arm_r_joint7",
            ],
            velocity_limit_sim=6.0,
            effort_limit_sim=5.1,
            stiffness=200.0,
            damping=3.0,
        ),

        # Left gripper: stiff enough to hold the box, not so stiff it snaps shut.
        "gripper_l": ImplicitActuatorCfg(
            joint_names_expr=["gripper_l_joint1"],
            velocity_limit_sim=2.2,
            effort_limit_sim=200.0,
            stiffness=3500.0,
            damping=10.0,
        ),
        "gripper_l_fingers": ImplicitActuatorCfg(
            joint_names_expr=["gripper_l_joint[2-4]"],
            velocity_limit_sim=6.5,
            effort_limit_sim=220.0,
            stiffness=5000.0,
            damping=14.0,
        ),
        # Right gripper unchanged (mimic-friendly slave joints).
        "gripper_r_master": ImplicitActuatorCfg(
            joint_names_expr=["gripper_r_joint1"],
            velocity_limit_sim=2.2,
            effort_limit_sim=200.0,
            stiffness=4000.0,
            damping=12.0,
        ),
        "gripper_r_slave": ImplicitActuatorCfg(
            joint_names_expr=["gripper_r_joint[2-4]"],
            effort_limit_sim=30.0,
            stiffness=2.0,
            damping=0.5,
        ),

        # Actuators for head joints
        "head": ImplicitActuatorCfg(
            joint_names_expr=["head_joint1", "head_joint2"],
            velocity_limit_sim=2.0,
            effort_limit_sim=30.0,
            stiffness=150.0,
            damping=3.0,
        ),
    }
)
