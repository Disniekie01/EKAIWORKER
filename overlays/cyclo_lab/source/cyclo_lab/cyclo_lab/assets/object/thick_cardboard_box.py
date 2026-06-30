import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg

# Thicker cardboard box for the box-stack L-table task.
# Same footprint as CARDBOARD_BOX_CFG but 6 cm tall (vs 1.5 cm).
BOX_THICKNESS = 0.06
BOX_HALF_HEIGHT = BOX_THICKNESS / 2.0
TABLE_HEIGHT = 0.72
RISER_HEIGHT = 0.08

THICK_CARDBOARD_BOX_CFG = RigidObjectCfg(
    spawn=sim_utils.CuboidCfg(
        size=(0.50, 0.35, BOX_THICKNESS),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            linear_damping=0.05,
            angular_damping=0.05,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.06),
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=3.0,
            dynamic_friction=3.0,
            restitution=0.0,
            friction_combine_mode="max",
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.58, 0.44, 0.28),
            roughness=0.9,
        ),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(0.50, 0.0, TABLE_HEIGHT + RISER_HEIGHT + BOX_HALF_HEIGHT),
    ),
)
