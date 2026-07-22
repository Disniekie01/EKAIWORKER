# DATA COLLECTION GUIDE

> **Author:** Hun Kim · **Last updated:** 2026-07-21
> 
> FFW-SG2 USD drivable conversion — the **Data Collection** part.
> Scope: modify the AI Worker (FFW-SG2) USD to be **physically drivable** so mobile-base
> demonstrations can be teleoperated and recorded via VR.
> The base system (containers / VR / recording) comes from the repos below and is only linked here.
> This document covers **only what I added/modified**.
> The **Data Generation** part (IK → annotate → datagen → joint → LeRobot) is in a separate doc:
> [`DATA_GENERATION.md`](DATA_GENERATION.md).

---

## 0. Sources

| Source | Link | Provides |
|---|---|---|
| **EKAIWORKER** (base) | https://github.com/Disniekie01/EKAIWORKER | 3 containers, VR publisher/controller, recording pipeline, dashboard |
| **adb_vr_connect** | https://github.com/Disniekie01/EKAIWORKER/tree/main/adb_vr_connect | Quest USB tether connection |
| **ROBOTIS VR guide** | https://docs.robotis.com/docs/systems/aiworker/quick_start_guide/operation_guide/vr_teleoperation/ | VR operation procedure |

---

## 1. Overview

Collect L-table pick & place demonstrations in the **same format** as the real AI Worker (4 cameras + 22-dim state/action), producing `*_raw.hdf5`. Setup / VR / recording themselves are the base repo ([§0](#0-Sources)); below is **my change that makes mobile collection possible**.

```
── Data Collection ──────────────────────────────────────►  (→ Data Generation)
[setup] → [VR connect] → [record: drive base + manipulate] → *_raw.hdf5
```

| My contribution | Tag | Section |
|---|---|---|
| USD drivable conversion (`FFW_SG2_MOBILE`) — physical swerve driving instead of teleport | [Added] | [§2](#USD-drivable-conversion-FFW_SG2_MOBILE-Added) |
| VR base driving — A/B rotation, Y-button mode toggle, `/cmd_vel` RELIABLE | [Modified] | [§3](#3-vr-base-driving-controls-modified) |
| Session naming + mobile tasks in the dashboard | [Modified] | [§4](#4-changed-files-collection) |

### 1.1 Collection procedure (in order)

For per-stage details/commands, see the base repo:
[**EKAIWORKER (Niek)**](https://github.com/Disniekie01/EKAIWORKER). Only the order is shown here.

| # | Stage | Output | Notes |
|---|---|---|---|
| ① | Setup (`setup.sh`, 3 containers) | — | EKAIWORKER |
| ② | VR connection (USB tether) | — | adb_vr_connect |
| ③ | Record (dashboard → Launch Record → operate → N/R) | `*_raw.hdf5` | base controls ([§3]( [§3](#3-vr-base-driving-controls-modified))) |

The `*_raw.hdf5` is the input to the Data Generation pipeline
([`DATA_GENERATION.md`](DATA_GENERATION.md)).

---

## 2. USD drivable conversion `FFW_SG2_MOBILE` **[Added]**

The stock `FFW_SG2.usd` is authored for stationary manipulation, so the base moves by **kinematic
root teleport** → it fights physics (the carried box jitters) and the base trajectory did not carry
cleanly through datagen. Override layer referencing the stock USD lifts only the locks
(the stock asset is not modified):

| Stock lock | Fix |
|---|---|
| `FixedJoint` welds chassis to world | `fix_root_link=False` (free base) |
| Wheel drive limit ±1080° | removed (continuous rotation) |
| Left/right wheel colliders off | re-enabled |
| Gravity off | per-body: on for base + 6 wheels (traction), off for arms/lift/head/grippers (no sag) |
| — | self-collision on (arms ↔ torso); 6 wheel links filtered vs all body links (wheels touch only the ground) |
| — | a reset event restores the base to standing height |

The swerve controller converts `cmd_vel [linear_x, linear_y, angular_z]` → per-module steer angle +
wheel speed. Verified: `root_z ≈ 1.405` stable, 8.23 m in 10 s (96 % of commanded), holonomic
crab / spin-in-place confirmed.
Tools: `build_ffw_sg2_mobile_usd.py` (regenerate), `teleop_sg2_mobile.py` (keyboard driving).

## 3. VR base driving controls **[Modified]**

To drive the base during recording, the VR publisher was extended (base controls only; arm /
gripper / lift are unchanged from the base repo):

| Input | Action |
|---|---|
| Left thumbstick | Base translate (forward/back x, left/right y) |
| A button (hold) / B button (hold) | Rotate right / rotate left |
| Y button | Toggle `LIFT+HEAD` ↔ `LIFT+CMD_VEL` mode |

Key fix: `/cmd_vel` is published on **RELIABLE** QoS (was BEST_EFFORT). The SDK subscribes RELIABLE,
so the DDS mismatch had dropped every base command. Deploy/test: `DEPLOY_PLAN_B.md`.

---

## 4. Changed files

Path prefix: under `overlays/cyclo_lab/`.

### Added

| File | Purpose |
|---|---|
| `…/assets/robots/FFW_SG2_MOBILE.py` | Drivable-base articulation config |
| `…/data/robots/FFW/FFW_SG2_MOBILE.usd` | Override over the stock USD |
| `…/controllers/swerve.py` (+ `__init__.py`) | 3-module holonomic swerve controller |
| `scripts/tools/{build_ffw_sg2_mobile_usd, check_ffw_sg2_mobile, teleop_sg2_mobile}.py` | USD regenerate / regression check / keyboard driving |


### Modified

| File | Change |
|---|---|
| `scripts/…/dds_sdk/ffw_sg2_sdk.py` | subscribe `/cmd_vel`, physical swerve driving |
| `sg2_ltable_dashboard.py` | session naming; mobile tasks |
| `robotis_applications/…/vr_publisher_sg2.py` | Y toggle, A/B rotation, `/cmd_vel` RELIABLE QoS |

> The camera / base-velocity / mobile-cfg file changes are used during generation and are listed in
> [`DATA_GENERATION.md`](DATA_GENERATION.md).

## 5. Known limitations (Collection)

- **Camera extrinsics are placeholders** — calibrate against the physical rig.
- **Steering twitch at near-zero speed is sim-only** — the policy learns a `cmd_vel` (3 channels),
  not wheel commands, and the real robot's swerve controller converts it. The base trajectory and
  the `base_velocity` label are smooth.

