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

"""Builds ``FFW_SG2_MOBILE.usd`` — the drivable variant of FFW_SG2.

    cd /workspace/cyclo_lab
    ./third_party/IsaacLab/isaaclab.sh -p scripts/tools/build_ffw_sg2_mobile_usd.py

The stock ``FFW_SG2.usd`` is authored for stationary manipulation and blocks driving in four
places. This script writes a thin layer that **references** the original and overrides only those
four things (~2 KB instead of a 41 MB copy, and it automatically tracks the original if it is
updated).

    1. A FixedJoint welds the chassis (the ``world`` link) to the simulation world.
    2. The wheel drive joints have +/-1080 deg stops (3 turns -> stops after 1.63 m).
    3. The left/right wheel collision meshes are disabled (only the rear wheel touches the ground).
    4. Self-collision is ON — enabling the wheel colliders makes them overlap the housing and the
       robot is launched.

The original USD is never modified. The stock task keeps working as-is.
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Build FFW_SG2_MOBILE.usd")
parser.add_argument("--force", action="store_true", help="Overwrite the existing file")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os

from pxr import Sdf, Usd, UsdPhysics, UsdShade

from cyclo_lab.assets.robots import CYCLO_LAB_ASSETS_DATA_DIR

FFW_DIR = f"{CYCLO_LAB_ASSETS_DATA_DIR}/robots/FFW"
SRC_NAME = "FFW_SG2.usd"
OUT = f"{FFW_DIR}/FFW_SG2_MOBILE.usd"

ROBOT = "/Root/ffw_sg2_follower"
MODULES = ("left", "rear", "right")

# Stand-in for continuous rotation. lower>upper (UsdPhysics' "unlimited" convention) is read
# literally by PhysX, which locks the wheel at ~0 rad, and RemoveProperty only clears the local
# opinion so the reference layer's +/-1080 survives. So we set an explicit large value (~27,000
# turns).
FREE_SPIN_DEG = 1.0e7

WHEEL_STATIC_FRICTION = 1.2
WHEEL_DYNAMIC_FRICTION = 1.0


def info(*args):
    # Kit swallows the buffer, so flushing is required.
    print(*args, flush=True)


def main() -> int:
    if os.path.exists(OUT) and not args_cli.force:
        info(f"[X] Already exists: {OUT}\n    Use --force to overwrite")
        return 1

    if os.path.exists(OUT):
        os.remove(OUT)

    info("=" * 88)
    info("Building FFW_SG2_MOBILE.usd")
    info("=" * 88)
    info(f"  reference source : {FFW_DIR}/{SRC_NAME}  (not modified)")
    info(f"  output           : {OUT}")

    stage = Usd.Stage.CreateNew(OUT)
    root = stage.DefinePrim("/Root")
    # Path relative to the same folder -> the reference does not break if the folder is moved.
    root.GetReferences().AddReference(f"./{SRC_NAME}")
    stage.SetDefaultPrim(root)

    changes = []

    # [1] Release the base fixing ---------------------------------------------
    fixed_joint = stage.GetPrimAtPath(f"{ROBOT}/FixedJoint")
    if not fixed_joint.IsValid():
        info(f"[X] FixedJoint not found: {ROBOT}/FixedJoint")
        return 1
    fixed_joint.SetActive(False)
    changes.append("FixedJoint disabled (floating base)")
    info("\n[1] FixedJoint disabled -> base fixing released")

    # [2] Release the wheel drive limit ---------------------------------------
    info(f"\n[2] Wheel drive limit: +/-1080 deg -> +/-{FREE_SPIN_DEG:.0e} deg")
    for module in MODULES:
        joint = stage.GetPrimAtPath(f"{ROBOT}/joints/{module}_wheel_drive")
        if not joint.IsValid():
            info(f"[X] Joint not found: {module}_wheel_drive")
            return 1
        for attr_name, value in (
            ("physics:lowerLimit", -FREE_SPIN_DEG),
            ("physics:upperLimit", FREE_SPIN_DEG),
        ):
            attr = joint.GetAttribute(attr_name) or joint.CreateAttribute(
                attr_name, Sdf.ValueTypeNames.Float
            )
            attr.Set(value)
        changes.append(f"{module}_wheel_drive unlimited")
        info(f"    {module}_wheel_drive")

    # [3] Enable the wheel colliders ------------------------------------------
    info("\n[3] Enable the wheel colliders (left/right were off)")
    wheel_colliders = []
    for module in MODULES:
        path = f"{ROBOT}/{module}_wheel_drive_link/collisions/{module}_wheel_drive_link/mesh"
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            info(f"[X] Collision prim not found: {path}")
            return 1
        attr = prim.GetAttribute("physics:collisionEnabled")
        before = attr.Get() if attr else None
        if not attr:
            attr = prim.CreateAttribute("physics:collisionEnabled", Sdf.ValueTypeNames.Bool)
        attr.Set(True)
        wheel_colliders.append(prim)
        changes.append(f"{module} wheel collision ON")
        info(f"    {module:5s} collisionEnabled: {before} -> True")

    # [4] Self-collision ON + wheel collision filter --------------------------
    # Enable self-collision so the arms do not pass through the torso, but the restored wheel
    # colliders overlap their own housing/chassis and launch the robot, so filter the six wheel
    # links against every body link (wheels collide only with the ground). Measured: ON without a
    # filter -> launched to z=700 m / after filtering -> root_z stable at 1.405.
    articulation = stage.GetPrimAtPath(ROBOT)
    attr = articulation.GetAttribute("physxArticulation:enabledSelfCollisions")
    if not attr:
        info("[X] enabledSelfCollisions attribute missing")
        return 1
    attr.Set(True)
    changes.append("self-collision ON")
    info("\n[4] enabledSelfCollisions -> True (+ wheel collision filter)")

    wheel_link_paths = [
        f"{ROBOT}/{module}_wheel_{kind}_link"
        for module in MODULES for kind in ("steer", "drive")
    ]
    body_link_paths = [
        str(p.GetPath())
        for p in Usd.PrimRange(articulation)
        if p.HasAPI(UsdPhysics.RigidBodyAPI)
    ]
    pair_count = 0
    for wheel_path in wheel_link_paths:
        wheel_prim = stage.GetPrimAtPath(wheel_path)
        if not wheel_prim.IsValid():
            info(f"[X] Wheel link not found: {wheel_path}")
            return 1
        rel = UsdPhysics.FilteredPairsAPI.Apply(wheel_prim).CreateFilteredPairsRel()
        for body_path in body_link_paths:
            if body_path != wheel_path:
                rel.AddTarget(body_path)
                pair_count += 1
    changes.append(f"wheel collision filter {pair_count} pairs")
    info(f"    {len(wheel_link_paths)} wheels x {len(body_link_paths)} bodies = {pair_count} filter pairs")

    # [5] Wheel traction material ---------------------------------------------
    info("\n[5] Wheel physics material + binding")
    material = UsdShade.Material.Define(stage, "/Root/wheelPhysicsMaterial")
    UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    material_api = UsdPhysics.MaterialAPI(material.GetPrim())
    material_api.CreateStaticFrictionAttr().Set(WHEEL_STATIC_FRICTION)
    material_api.CreateDynamicFrictionAttr().Set(WHEEL_DYNAMIC_FRICTION)
    material_api.CreateRestitutionAttr().Set(0.0)
    for prim in wheel_colliders:
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            material, UsdShade.Tokens.weakerThanDescendants, "physics"
        )
    changes.append(f"wheel material bound x{len(wheel_colliders)}")
    info(f"    static={WHEEL_STATIC_FRICTION} dynamic={WHEEL_DYNAMIC_FRICTION}"
         f" -> {len(wheel_colliders)} wheels")

    # [6] Per-body gravity ----------------------------------------------------
    # Gravity ON only for the base + wheels (wheel traction = needed for driving). Arms/lift/head/
    # grippers have gravity OFF -> they do not sag like the stock robot and hold as commanded by
    # the actuators (thumbstick lift, etc.). The stock robot had gravity fully OFF.
    info("\n[6] Per-body gravity (base+wheels ON, upper body OFF)")
    gravity_on_links = {"world"} | {
        f"{module}_wheel_{kind}_link" for module in MODULES for kind in ("steer", "drive")
    }
    on_count = off_count = 0
    for prim in Usd.PrimRange(articulation):
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        disable = prim.GetName() not in gravity_on_links
        attr = prim.GetAttribute("physxRigidBody:disableGravity") or prim.CreateAttribute(
            "physxRigidBody:disableGravity", Sdf.ValueTypeNames.Bool
        )
        attr.Set(disable)
        off_count += int(disable)
        on_count += int(not disable)
    changes.append(f"gravity ON {on_count} / OFF {off_count}")
    info(f"    gravity ON: {on_count} (base+wheels)  OFF: {off_count} (upper body)")

    stage.GetRootLayer().documentation = (
        "FFW_SG2 drivable variant. References FFW_SG2.usd and overrides only what the stock "
        "asset locks down for stationary manipulation: the world FixedJoint, the +/-1080 deg "
        "wheel stops, the disabled left/right wheel colliders. Self-collision is enabled (so "
        "the arms cannot pass through the torso) with the six wheel links filtered against all "
        "body links, so the wheels collide only with the ground. "
        "Regenerate with scripts/tools/build_ffw_sg2_mobile_usd.py"
    )
    stage.Save()

    info("\n" + "=" * 88)
    info(f"Done — {len(changes)} changes, {os.path.getsize(OUT) / 1024:.1f} KB (references the 41 MB original)")
    info("=" * 88)
    for change in changes:
        info(f"  - {change}")
    info("\n  Verify: ./third_party/IsaacLab/isaaclab.sh -p scripts/tools/check_ffw_sg2_mobile.py")
    return 0


if __name__ == "__main__":
    code = main()
    simulation_app.close()
