from __future__ import annotations

from collections.abc import Callable

import isaacsim.core.utils.prims as prim_utils
from pxr import Usd, UsdGeom

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.sim import schemas
from isaaclab.sim.spawners import materials
from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
from isaaclab.sim.utils import bind_visual_material, clone
from isaaclab.utils import configclass


def _spawn_visual_cube(geom_path: str, size, pos, material_cfg):
    """Spawn a visual-only cube under geometry/."""
    mesh_path = f"{geom_path}/mesh"
    base = min(size)
    scale = [dim / base for dim in size]
    prim_utils.create_prim(
        mesh_path, "Cube", translation=pos, scale=scale, attributes={"size": base}
    )
    if material_cfg is not None:
        material_path = f"{geom_path}/material"
        material_cfg.func(material_path, material_cfg)
        bind_visual_material(mesh_path, material_path)


def _spawn_collision_cube(geom_path: str, size, pos, collision_props):
    """Spawn an invisible collision-only cube under geometry/."""
    mesh_path = f"{geom_path}/mesh"
    base = min(size)
    scale = [dim / base for dim in size]
    prim = prim_utils.create_prim(
        mesh_path, "Cube", translation=pos, scale=scale, attributes={"size": base}
    )
    if collision_props is not None:
        schemas.define_collision_properties(mesh_path, collision_props)
    # Hide the collision geometry so only the visual tabletop/legs are rendered.
    UsdGeom.Imageable(prim).MakeInvisible()


@clone
def spawn_table_with_legs(
    prim_path: str,
    cfg: "TableWithLegsCfg",
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Spawn a table (tabletop slab + 4 legs) as a single kinematic rigid body.

    Uses one tabletop collision mesh for physics stability and visual-only leg meshes.
    The body origin is at the floor center; the work surface sits at z = ``height``.
    """
    if not prim_utils.is_prim_path_valid(prim_path):
        prim_utils.create_prim(prim_path, "Xform", translation=translation, orientation=orientation)
    else:
        raise ValueError(f"A prim already exists at path: '{prim_path}'.")

    depth, width, height = cfg.depth, cfg.width, cfg.height
    tt = cfg.top_thickness
    leg = cfg.leg_thickness
    inset = cfg.leg_inset
    geom_root = f"{prim_path}/geometry"

    # Single collision slab on the tabletop (avoids multi-shape PhysX tensor issues).
    _spawn_collision_cube(
        f"{geom_root}/collision",
        (depth, width, tt),
        (0.0, 0.0, height - tt / 2.0),
        cfg.collision_props,
    )

    # Visual tabletop.
    _spawn_visual_cube(
        f"{geom_root}/tabletop",
        (depth, width, tt),
        (0.0, 0.0, height - tt / 2.0),
        cfg.visual_material,
    )

    # Visual legs only (no collision).
    leg_h = height - tt
    lz = leg_h / 2.0
    hx = depth / 2.0 - inset - leg / 2.0
    hy = width / 2.0 - inset - leg / 2.0
    hidden_legs = cfg.hidden_legs or ()
    for i, (sx, sy) in enumerate([(hx, hy), (hx, -hy), (-hx, hy), (-hx, -hy)]):
        if i in hidden_legs:
            continue
        _spawn_visual_cube(
            f"{geom_root}/leg_{i}",
            (leg, leg, leg_h),
            (sx, sy, lz),
            cfg.visual_material,
        )

    if cfg.mass_props is not None:
        schemas.define_mass_properties(prim_path, cfg.mass_props)
    if cfg.rigid_props is not None:
        schemas.define_rigid_body_properties(prim_path, cfg.rigid_props)

    return prim_utils.get_prim_at_path(prim_path)


@configclass
class TableWithLegsCfg(RigidObjectSpawnerCfg):
    """Configuration for a primitive table built from a tabletop slab and four legs."""

    func: Callable = spawn_table_with_legs

    depth: float = 0.60
    width: float = 0.80
    height: float = 0.72
    top_thickness: float = 0.04
    leg_thickness: float = 0.06
    leg_inset: float = 0.03
    hidden_legs: tuple[int, ...] = ()

    visual_material_path: str = "material"
    visual_material: materials.VisualMaterialCfg | None = None


_TABLE_COMMON = dict(
    height=0.72,
    rigid_props=sim_utils.RigidBodyPropertiesCfg(
        kinematic_enabled=True,
        disable_gravity=True,
    ),
    collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
    visual_material=sim_utils.PreviewSurfaceCfg(
        diffuse_color=(0.6, 0.62, 0.65),
        metallic=1.0,
        roughness=0.35,
    ),
)

TABLE_FRONT_CFG = RigidObjectCfg(
    spawn=TableWithLegsCfg(depth=0.60, width=1.10, hidden_legs=(2,), **_TABLE_COMMON),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.50, 0.0, 0.0)),
)

TABLE_LEFT_CFG = RigidObjectCfg(
    spawn=TableWithLegsCfg(depth=0.60, width=1.10, **_TABLE_COMMON),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.50, 0.85, 0.0)),
)
