# DATA COLLECTION GUIDE

> **Author:** Hun Kim · **Last updated:** 2026-07-23
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
| **ROBOTIS VR guide** | https://docs.robotis.com/docs/systems/aiworker/quick_start_guide/operation_guide/vr_teleoperation/ | VR operation procedure |

---

## 1. Overview

Collect L-table pick & place demonstrations in the **same format** as the real AI Worker (4 cameras + 22-dim state/action), producing `*_raw.hdf5`. Setup / VR / recording themselves are the base repo ([§0](#0-Sources)); below is **my change that makes mobile collection possible**.

```
── Data Collection ──────────────────────────────────────►  (→ Data Generation)
[setup] → [VR connect] → [record: drive base + manipulate] → *_raw.hdf5
```

| Changes | Tag | Section |
|---|---|---|
| USD drivable conversion (`FFW_SG2_MOBILE`) — physical swerve driving instead of teleport | [Added] | [§2](#USD-drivable-conversion-FFW_SG2_MOBILE-Added) |
| VR base driving — A/B rotation, Y-button mode toggle, `/cmd_vel` RELIABLE | [Modified] | [§3](#3-vr-base-driving-controls-modified) |
| Session naming + mobile tasks in the dashboard | [Modified] | [§4](#4-changed-files-collection) |

### 1.1 Collection procedure

For per-stage details/commands, see the base repo:
[**EKAIWORKER (Niek)**](https://github.com/Disniekie01/EKAIWORKER). Only the order is shown here.

| # | Stage | Output | Notes |
|---|---|---|---|
| ① | Setup | — | Follow [0. 5-Minute Quickstart](https://github.com/Disniekie01/EKAIWORKER#5-minute-quickstart-fresh-machine) |
| ② | VR connection (USB or WIFI) | — | Follow [1. VR Device Setup](https://docs.robotis.com/docs/systems/aiworker/quick_start_guide/operation_guide/vr_teleoperation/#vr-device-setup), [2.1 USB](https://github.com/Disniekie01/EKAIWORKER#usb--adb-connection-recommended-for-recording) or [2.2 WIFI](https://github.com/Disniekie01/EKAIWORKER#wifi-connection-every-session) |
| ③ | Record | `*_raw.hdf5` | Follow [3. Start Record](https://github.com/Disniekie01/EKAIWORKER#1-record-demos-dashboard)|

The `*_raw.hdf5` is the input to the Data Generation pipeline ([`DATA_GENERATION.md`](DATA_GENERATION.md)).

---

## 2. USD drivable conversion `FFW_SG2_MOBILE` **[Added]**

`cyclo_lab/source/cyclo_lab/data/robots/FFW/FFW_SG2.usd` is the robot asset **provided by ROBOTIS**.
It is authored for **stationary manipulation only — the base cannot actually drive**: the chassis is
welded to the world, the wheel joints are limited to ±1080°, and the left/right wheel colliders are
off, so any "base motion" is a **kinematic root teleport**, not real wheel driving. That teleport
fights physics (the carried box jitters relative to the grippers) and did not carry cleanly through
datagen.

To make the base **physically drivable**, I added an override layer that references the stock ROBOTIS
USD and lifts only the locks below (the stock asset itself is never modified):

| Stock lock (ROBOTIS USD) | Fix |
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

---

## 3. VR teleoperation controls **[Modified]**

The goal was to mirror the AI Worker **leader (teleoperation kit)** controls on the Meta Quest 3
controllers. On the real leader, **clicking the thumbstick** switches the robot into base-driving
mode. On the Quest controllers that thumbstick-click was recognized unreliably,
so I remapped the mode switch to the **Y button** instead. Everything else follows the leader
mapping. The full control set:

**Always active (arms, grippers, lift — same as the base repo):**

| Input | Action |
|---|---|
| Hold both grips ~3 s | Activate arm teleoperation for the episode |
| Move both controllers | Arm end-effectors follow the hands (IK) |
| Trigger (L / R) | Close the left / right gripper (analog) |
| Right thumbstick (up/down) | Lift up / down |

**Base-driving mode (this fork's addition):**

| Input | Action |
|---|---|
| **Y button** | Toggle mode: `LIFT+HEAD` ↔ `LIFT+CMD_VEL` (drive) — replaces the leader's thumbstick-click, which mis-triggered on the Quest |
| Left thumbstick | Drive the base: forward/back = x, left/right = y (crab) |
| **A button (hold)** | Rotate right |
| **B button (hold)** | Rotate left |
| Right thumbstick (up/down) | Lift up / down (still active while driving) |

In `LIFT+HEAD` mode the left thumbstick pans/tilts the head; in `LIFT+CMD_VEL` mode it drives the
base. Deploy/test procedure: `DEPLOY_PLAN_B.md`.

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
| `robotis_applications/…/vr_publisher_sg2.py` | Y-button mode toggle (replaces the leader's thumbstick-click), A/B base rotation, `/cmd_vel` publishing |

> The camera / base-velocity / mobile-cfg file changes are used during generation and are listed in
> [`DATA_GENERATION.md`](DATA_GENERATION.md).

## 5. Known limitations (Collection)

- **Camera extrinsics are placeholders** — calibrate against the physical rig.
- **Steering twitch at near-zero speed is sim-only** — the policy learns a `cmd_vel` (3 channels),
  not wheel commands, and the real robot's swerve controller converts it. The base trajectory and
  the `base_velocity` label are smooth.

