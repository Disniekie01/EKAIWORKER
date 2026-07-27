# Future fix: Swerve controller and wheel physics

**Status:** Deferred — listed as a future fix. Do **not** replace kinematic L-motion in the current SG2 L-table IL pipeline.

## Goal (when prioritized)

Integrate the swerve controller and wheel-physics implementation from the latest official robot package into the Isaac Lab environment, and evaluate whether kinematic base movement can be replaced with physics-based movement while keeping stable control, wheel motion, rotation, and proper data recording.

## Current behavior (keep)

L-table VR recording and Mimic datagen move the base with **kinematic root teleport** (`write_root_pose_to_sim`) plus rigid box carry:

- Grip stays stable during rotate (~90°) + forward drive
- L-motion is **outside** the action vector; annotate / datagen / replay depend on recorded `root_pose` / states
- Robot cfg uses `disable_gravity=True`; MDP actions are arms / grippers / head / lift only — **no wheel actions**
- Default flag: `teleop_l_use_swerve=False`

Even with `teleop_l_use_swerve=True`, the SDK still teleports the root and keeps wheels still (spinning wheels under teleport shakes the box out of the grippers).

## Upstream pieces already in-repo

| Piece | Path |
|-------|------|
| Sim swerve math | `scripts/sim2real/bringup/common/swerve_drive.py` |
| SG2 swerve joint constants / actuators | `source/cyclo_lab/.../assets/robots/FFW_SG2.py` (`SG2_SWERVE_*`) |
| Official ROS swerve controller | `ai_worker/ffw_swerve_drive_controller/` |
| Teleop / L-motion | `scripts/sim2real/imitation_learning/dds_sdk/ffw_sg2_sdk.py` |
| Kinematic Mimic L-motion | `.../pick_place_l_table/ltable_kinematic_l_motion.py` |
| Datagen state replay | `scripts/sim2real/imitation_learning/mimic/cyclo_mimic_datagen.py` |

## Future-fix scope (phased)

Not a drop-in swap — expect a multi-sprint project:

1. **Shared controller** — Drive base with `SwerveDriveController` + steer/drive joint targets (align with official package / bringup).
2. **Physics foundation** — Gravity on, wheel–ground contact/friction; prove on a drive-only scene before L-table pick/place.
3. **Recording contract** — Add base cmd / wheel targets to actions+obs, or keep state-based `root_pose` while physics drives motion; replace rigid box teleport with grip that survives base acceleration.
4. **Datagen / replay / policies** — Retime Mimic L segment for physics DT; update annotate/replay; retrain if action dim changes.

## Acceptance criteria (when picked up)

- Stable wheel motion and chassis rotation under physics
- Carried box remains graspable through L-motion
- Recording still produces train / replay-compatible demos
- Kinematic path remains available behind a flag until physics is proven

## Explicit non-goals for now

- Changing default `teleop_l_use_swerve` or the Full Mimic pipeline
- Removing kinematic L-motion from VR or `cyclo_mimic_datagen`
- Expanding BC action dim until the recording schema is designed

## Feasibility note

Full physics replacement is **not currently feasible as a drop-in**. Wheel DoFs and swerve math exist, but the IL stack assumes floating gravity-off + root teleport + rigid box carry. Safer near-term path: keep kinematic teleport for VR datagen stability; use true swerve physics on bringup / drive experiments only.
