"""MDP for SH5 single_box_far_thick."""

from isaaclab.envs.mdp import *  # noqa: F401, F403

from cyclo_lab.real_world_tasks.manager_based.FFW_SG2.pick_place.mdp.observations import (  # noqa: F401
    eef_pose,
    joint_pos_name,
    joint_pos_target_name,
    last_action,
)
from cyclo_lab.real_world_tasks.manager_based.FFW_SH5.mdp.hand_observations import object_dual_grasped  # noqa: F401
from cyclo_lab.real_world_tasks.manager_based.FFW_SG2.single_box_far_thick.mdp.observations import object_on_rear_table  # noqa: F401
from cyclo_lab.real_world_tasks.manager_based.FFW_SG2.single_box_far_thick.mdp.terminations import *  # noqa: F401, F403
from cyclo_lab.real_world_tasks.manager_based.FFW_SG2.single_box_far_thick.mdp.single_box_far_thick_events import *  # noqa: F401, F403
