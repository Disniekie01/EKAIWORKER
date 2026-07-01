"""Visual-only marker overlay for the L-table left tabletop drop target."""

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg

from cyclo_lab.real_world_tasks.manager_based.FFW_SG2.pick_place_l_table.mdp.ffw_sg2_l_table_events import (
    LEFT_TABLE_EDGE_MARGIN,
    LEFT_TABLE_HALF_DEPTH,
    LEFT_TABLE_HALF_WIDTH,
    LEFT_TABLE_POS,
    LEFT_TABLE_QUAT_WXYZ,
    TABLE_HEIGHT,
)

# Flat slab matching TableLeft/geometry/tabletop (inset slightly from the rim).
_MARKER_DEPTH = 2.0 * (LEFT_TABLE_HALF_DEPTH - LEFT_TABLE_EDGE_MARGIN)
_MARKER_WIDTH = 2.0 * (LEFT_TABLE_HALF_WIDTH - LEFT_TABLE_EDGE_MARGIN)
_TABLETOP_Z = TABLE_HEIGHT + 0.003

L_TABLE_DROP_ZONE_MARKER_CFG = RigidObjectCfg(
    spawn=sim_utils.CuboidCfg(
        size=(_MARKER_DEPTH, _MARKER_WIDTH, 0.004),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            kinematic_enabled=True,
            disable_gravity=True,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.15, 0.85, 0.35),
            emissive_color=(0.08, 0.30, 0.12),
            roughness=0.35,
            metallic=0.0,
            opacity=0.45,
        ),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(LEFT_TABLE_POS[0], LEFT_TABLE_POS[1], _TABLETOP_Z),
        rot=LEFT_TABLE_QUAT_WXYZ,
    ),
)
