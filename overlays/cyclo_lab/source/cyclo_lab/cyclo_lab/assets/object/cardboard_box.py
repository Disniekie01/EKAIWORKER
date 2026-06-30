import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg

# Flat cardboard box for dual-gripper pick-and-place tasks.
# Size: 50 cm x 35 cm x 1.5 cm (length x width x height) - large and very thin.
CARDBOARD_BOX_CFG = RigidObjectCfg(
    spawn=sim_utils.CuboidCfg(
        size=(0.50, 0.35, 0.015),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            # Low damping: high box damping fights the grippers when the box is
            # squeezed or carried (feels like the fingers are pushed open).
            linear_damping=0.05,
            angular_damping=0.05,
        ),
        # Very light empty cardboard — easy for both grippers to lift and hold.
        mass_props=sim_utils.MassPropertiesCfg(mass=0.02),
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=3.0,
            dynamic_friction=3.0,
            restitution=0.0,
            friction_combine_mode="max",
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.62, 0.48, 0.32),
            roughness=0.9,
        ),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        # Box rests on top of the 8 cm riser on the 0.72 m table.
        pos=(0.50, 0.0, 0.8075),
    ),
)
