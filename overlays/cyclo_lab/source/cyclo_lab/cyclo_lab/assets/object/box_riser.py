import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg

# Small static pedestal placed under the cardboard box.
# Its footprint is smaller than the box (0.50 x 0.35), so the box edges
# overhang on every side and the grippers can reach underneath to grasp it.
# Kinematic + gravity-disabled so it stays fixed on the table like the tables.
RISER_SIZE = (0.20, 0.15, 0.08)

BOX_RISER_CFG = RigidObjectCfg(
    spawn=sim_utils.CuboidCfg(
        size=RISER_SIZE,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            kinematic_enabled=True,
            disable_gravity=True,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.25, 0.22, 0.2),
            roughness=0.8,
        ),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(0.50, 0.0, 0.72 + RISER_SIZE[2] / 2.0),
    ),
)
