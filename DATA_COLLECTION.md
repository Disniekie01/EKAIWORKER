# Data Collection Pipeline — FFW-SG2 VR Teleoperation

**Author:** Hun Kim (hun7728@hanyang.ac.kr) · **Last updated:** 2026-07-20

This document covers collecting bimanual manipulation **and mobile-base** demonstrations in
NVIDIA Isaac Sim, running them through the ROBOTIS IsaacLab-Mimic augmentation pipeline, and
exporting to the **LeRobot** format for imitation learning.

**Scope of this document.** The base system (containers, VR stack, recording, Mimic, LeRobot
converter, dashboard) comes from existing repositories — those are only summarized here with
links to their own documentation. The detail is reserved for **what this fork adds or modifies**:
the drivable mobile base, physics-driven 22-dim datagen, and real-robot camera/base-velocity
parity. Fork work is tagged **[ADDED]** / **[MODIFIED]** and itemized in
[Appendix A](#appendix-a--files-added--modified).

---

## 0. Attribution & Sources

**This work reuses existing repositories rather than reimplementing them.** Almost the entire
pipeline — containers, VR publisher/controller, Isaac Sim recording loop, IsaacLab-Mimic
augmentation, LeRobot converter, dashboard — comes from the base repository below. This document's
procedures are, for the most part, **derived from these existing sources**, not authored from
scratch.

| Source | Link | What it provides |
|---|---|---|
| **EKAIWORKER** (base repo) | https://github.com/Disniekie01/EKAIWORKER | The complete stack this fork builds on: 3-container setup, VR publisher/controller (`robotis_vuer`, `ai_worker`), Isaac Sim record pipeline, IsaacLab-Mimic datagen, `isaaclab2lerobot` converter, `sg2_ltable_dashboard.py`. Pins three upstream ROBOTIS repos (`cyclo_lab`, `ai_worker`, `robotis_applications`). |
| **adb_vr_connect** | https://github.com/Disniekie01/EKAIWORKER/tree/main/adb_vr_connect | One-time ADB/udev setup and the USB-tether connection procedure for the Meta Quest. |
| **ROBOTIS VR teleoperation guide** | https://docs.robotis.com/docs/systems/aiworker/quick_start_guide/operation_guide/vr_teleoperation/ | The official VR operation procedure (headset pairing, browser flow, grip-to-activate). |

**Reused as-is (documented, not authored here):** container bring-up (`setup.sh`, `docker/`), the
VR connection procedure (`adb_vr_connect`), the base VR publisher/controller/IK chain, the record
pipeline (`record_demos.py`, recorder manager), the IsaacLab-Mimic driver scripts, and the stock
L-table task/USD.

**This repo.** `AI_HUN` (https://github.com/hun7407-lgtm/AI_HUN) is an overlay on **EKAIWORKER**;
everything under `overlays/` is applied on top of the pinned upstreams by `setup.sh`. A team-repo
copy lives at `nimbuslab1/pai-korea_sprint2026` under `Initial_collecion/HunKim`.

---

## 1. What this fork adds

Everything below is the fork's own contribution; each links to its detailed section.

| Contribution | Tag | Section |
|---|---|---|
| **Drivable mobile base** (`FFW_SG2_MOBILE`) — swerve base driven by physics instead of root teleport | [ADDED] | [§4](#4-the-drivable-mobile-base) |
| **Physics-driven 22-dim datagen** — Mimic augmentation drives the base physically; 22 dims carried through every stage | [ADDED] | [§5](#5-mobile-22-dim-datagen) |
| **4-camera real-robot parity** — head stereo + two wrist cameras, CHW, matching `ffw_sg2_rev1` | [MODIFIED] | [§6.1](#61-cameras-modified) |
| **Base-velocity observation** — `[linear_x, linear_y, angular_z]`, state/action 19 → 22 | [ADDED] | [§6.2](#62-base-velocity-added) |
| **LeRobot converter** — auto-detect cameras + base velocity, emit the real schema | [MODIFIED] | [§6.3](#63-converter-modified) |
| **VR base driving** — A/B rotation, Y-button mode toggle, `/cmd_vel` on RELIABLE QoS | [MODIFIED] | [§7](#7-vr-base-driving-controls) |
| **Session-based dataset naming** + mobile tasks in the dashboard | [MODIFIED] | [Appendix A](#appendix-a--files-added--modified) |

---

## 2. Base system (summary — see linked docs)

The robot, task, containers, setup, VR connection, and the standard pipeline stages are all base-repo
functionality. Brief facts for context; follow the links for full procedures.

- **Robot** — ROBOTIS **FFW-SG2**, dual-arm mobile manipulator. Joint/action layout is **19-dim**
  (`arm_l 7, gripper_l 1, arm_r 7, gripper_r 1, head 2, lift 1`); the mobile task adds base
  `linear_x, linear_y, angular_z` → **22-dim**. Base = 3 swerve modules (left / right / rear).
- **Task** — L-table pick & place: grasp a box on the front table with both arms, reposition the
  base, place it on the left ("L") table.
- **Containers** — three Docker containers over ROS 2 (`ROS_DOMAIN_ID=30`, Fast DDS): `cyclo_lab`
  (Isaac Sim + recorder + Mimic + dashboard:8765), `robotis-applications` (Vuer VR publisher:8012),
  `ai_worker` (arm IK controller). Versions: Isaac Sim 5.1.0, Isaac Lab 2.3.0.
- **Setup** — `git clone https://github.com/hun7407-lgtm/AI_HUN.git AIWORKER && cd AIWORKER &&
  ./setup.sh ~/AIWORKER`. This clones the pinned upstreams and rsyncs `overlays/` on top.
  Prerequisites (not installed by `setup.sh`): Linux + X11, NVIDIA RTX GPU + driver, Docker +
  NVIDIA Container Toolkit, NGC login, Meta Quest 3. See EKAIWORKER README.
- **VR connection** — USB tether via `adb_vr_connect` (recommended) or WiFi. Full procedure:
  [adb_vr_connect](https://github.com/Disniekie01/EKAIWORKER/tree/main/adb_vr_connect),
  [EKAIWORKER README](https://github.com/Disniekie01/EKAIWORKER), and the
  [ROBOTIS VR guide](https://docs.robotis.com/docs/systems/aiworker/quick_start_guide/operation_guide/vr_teleoperation/).
- **Recording** — launch from `sg2_ltable_dashboard.py`; grip both controllers ~3 s to activate,
  operate, then `N` to save / `R` to discard. Output: `datasets/…_raw.hdf5`.
- **Mimic pipeline (base)** — five stages, run one at a time from the dashboard or manually:
  `IK convert → annotate → datagen → joint convert → LeRobot export`
  (`action_data_converter.py`, `annotate_demos.py`, `generate_dataset.py`, `isaaclab2lerobot.py`).
  The fork's changes to this pipeline for the mobile 22-dim path are in §5.

```
[Meta Quest] ─ pose + joystick ─► [robotis-applications] Vuer publisher
    ├─ arm/wrist poses ─► [ai_worker] IK ─► joint cmds ─┐
    └─ /cmd_vel (Twist) ───────────────────────────────┤
                                                        ▼
[cyclo_lab] Isaac Sim + record_demos.py + FFWSG2Sdk ─► raw.hdf5
    └─ IK convert ─► annotate ─► Mimic datagen ─► joint convert ─► LeRobot
                                                        ▼
                          LeRobot dataset (4 cameras + 22-dim state/action)
```

---

## 3. The stock task's mobile limitation (why this fork exists)

On the stock (base-repo) form, the base repositions by a **scripted L-motion that kinematically
teleports the robot root**. Two problems:
- The teleport fights physics — the carried box is still driven by physics while the root is
  teleported, so it **shakes / jitters** relative to the grippers.
- The kinematic-teleport base motion did **not carry cleanly through data generation**, so
  mobile-base trajectories could not be augmented reliably.

The rest of this document is the fork's fix: a physically drivable base (§4) used for both
recording and datagen (§5), plus the real-robot parity work (§6).

---

## 4. The Drivable Mobile Base **[ADDED]**

A **physically drivable** base USD (`FFW_SG2_MOBILE`) — the base moves by driving its swerve wheels
under physics, with no root teleport, so the base and the carried box stay physically consistent.

### 4.1 What changed in the USD

The stock `FFW_SG2.usd` is authored for stationary manipulation. `FFW_SG2_MOBILE` lifts its locks
in a `~2 KB` override layer that references the stock USD (the stock asset is never modified):

| Stock lock | Fix |
|---|---|
| `FixedJoint` welds chassis to world | `fix_root_link=False` (free base) |
| Wheel drive limit ±1080° | removed (continuous rotation) |
| Left/right wheel colliders off | re-enabled |
| Gravity off | **per-body**: on for base + 6 wheels (traction), off for arms/lift/head/grippers (no sag) |
| — | self-collision **on** (arms can't pass through torso); 6 wheel links filtered vs all body links (wheels touch only the ground) |
| — | a reset event lifts the base to standing height after `reset_scene_to_default` |

### 4.2 How it drives

The swerve controller converts a body-frame `cmd_vel` `[linear_x, linear_y, angular_z]` into
per-module steer angle + wheel speed. During recording the operator drives from `/cmd_vel`; the SDK
applies it as wheel targets (`_apply_swerve_cmd_vel(..., integrate_root=False)` → physical driving).
Verified: settles at `root_z ≈ 1.405`, drives 8.23 m in 10 s at 96 % of commanded speed; holonomic
crab / spin-in-place confirmed.

Tools: `scripts/tools/build_ffw_sg2_mobile_usd.py` (regenerate the USD),
`check_ffw_sg2_mobile.py` (6/6 regression), `teleop_sg2_mobile.py` (keyboard driving).

---

## 5. Mobile 22-dim datagen **[ADDED]**

The whole Mimic pipeline runs end-to-end on the **mobile 22-dim** data, driving the base
**physically** (not teleporting) during augmentation.

### 5.1 Two enablers

**(a) A mobile Mimic task.** `Cyclo-Real-Mimic-Pick-Place-LTable-Mobile-FFW-SG2-v0`
(`FFWSG2PickPlaceLTableMobileMimicEnvCfg`) — the same L-table datagen built on the drivable-base
env, so generated demos keep the `base_velocity` observation.

**(b) Physics-driven base replay (no teleport).** During the move subtask, instead of teleporting
the base to the recorded pose, the mimic env **replays the recorded base velocity as a swerve
`cmd_vel`** so the base physically drives — wheels turn, no teleport/physics jitter. Implemented in
`pick_place_l_table_mimic_env.py` (`_physics_drive_step`, batched swerve targets), gated on the
mobile cfg so the fixed-base task is unchanged.

### 5.2 22 dims through every stage

Base velocity is carried as the last 3 action channels from IK convert onward (mobile only;
fixed-base stays 19-dim):

| Stage | How 22-dim is produced |
|---|---|
| IK / joint convert | `action_data_converter.py` appends `obs/base_velocity` to the action when present |
| Annotate | the sim action manager is 19-dim, so the env is fed 19-dim and the exported action is re-packed to 22-dim after each episode (`annotate_demos.py`) |
| Datagen | `generate_dataset.py` post-processes the generated file to 22-dim after the sim app closes |
| LeRobot | `isaaclab2lerobot.py` accepts a pre-packed 22-dim action (no double append) and still builds the 22-dim state |

### 5.3 Running it

```bash
# Datagen — mobile task, physics-driven base. --generation_num_trials = successes to reach
# (generation_guarantee=True: failed attempts are retried, not counted).
python scripts/sim2real/imitation_learning/mimic/generate_dataset.py \
  --device cuda --num_envs 4 --task Cyclo-Real-Mimic-Pick-Place-LTable-Mobile-FFW-SG2-v0 \
  --generation_num_trials 20 --input_file ./datasets/<...>_annotate.hdf5 \
  --output_file ./datasets/<...>_generate.hdf5 --enable_cameras --headless
```
IK convert / annotate / joint convert / LeRobot use the same commands as the base pipeline (§2),
just with the mobile task ID; they produce 22-dim automatically on mobile data.

**Practical notes**
- **Env count is GPU-bound by camera rendering.** Each env renders 4 RGB-D cameras and the
  generated observations accumulate on the GPU until each demo is exported (sawtooth memory). On a
  24 GB GPU, `--num_envs 10` **OOMs**; `--num_envs 4` peaks ~20 GB and is safe.
- **Physics-drive success rate is lower than teleport.** Open-loop velocity replay drifts, so the
  box is occasionally misplaced and that attempt is tagged `success=False` (dropped downstream).
  Observed ~40 %; `generation_guarantee=True` retries until the requested number of *successes* is
  reached. Only successful demos survive the joint-convert filter.
- **Resuming to a target.** If interrupted, generate the remainder into a second file and merge:
  `python merge_hdf5_demos.py --inputs partA.hdf5 partB.hdf5 --output generate.hdf5`.
- **Steering twitch at ~0 speed is a sim-only artifact.** The policy learns a `cmd_vel` (3 base
  channels), not wheel commands; the real robot's own swerve controller (with its own low-speed
  handling) drives the wheels. The base trajectory and the `base_velocity` label are smooth.

---

## 6. Real-Robot Parity

The real `ffw_sg2_rev1` records **4** RGB cameras and a **22-dim** state; the stock sim recorded
1 camera and 19 dims. Two additions plus a converter change match the real schema.

### 6.1 Cameras **[MODIFIED]**

| LeRobot key | Sim camera | Mount | Resolution |
|---|---|---|---|
| `…rgb.cam_left_head` | `cam_head` (ZED left eye) | `head_link2/zed` | 376 × 672 |
| `…rgb.cam_right_head` | `cam_right_head` | `head_link2/zed`, mirrored | 376 × 672 |
| `…rgb.cam_left_wrist` | `cam_left_wrist` | `arm_l_link7` (D405 pose) | 424 × 240 |
| `…rgb.cam_right_wrist` | `cam_right_wrist` | `arm_r_link7` (D405 pose) | 424 × 240 |

- Head cameras are the two eyes of the head ZED, symmetric about the `zed` prim centre (±0.03 m in
  Y ⇒ ~0.06 m baseline). Wrist cameras use the RealSense **D405** pose from the USD
  (`arm_*_link7/visuals/d405`): local `pos (0.10683, 0, -0.07713)`, 180° about Y.
- Cameras are children of the robot links, so the rendered/recorded feed follows head/arm motion.
- **Extrinsics are calibrated placeholders** — verify against the physical rig before cross-domain
  training.

### 6.2 Base velocity **[ADDED]**

The mobile task records `obs/base_velocity = base_planar_velocity(env)` =
`[root_lin_vel_b.x, root_lin_vel_b.y, root_ang_vel_b.z]` (base frame). It is appended to state and
action → 22-dim, matching real.

### 6.3 Converter **[MODIFIED]**

`isaaclab2lerobot.py` auto-detects what a recording contains and emits the real-robot schema — no
flags needed:

```bash
lerobot-python scripts/sim2real/imitation_learning/data_converter/isaaclab2lerobot.py \
  --task Cyclo-Real-Pick-Place-LTable-Mobile-FFW-SG2-v0 --robot_type FFW_SG2 --fps 15 \
  --dataset_file ./datasets/<...>_joint.hdf5
```
- Detects camera streams and exports each as `observation.images.rgb.<name>`, **channels-first
  `[3, H, W]`** (`cam_head` → `cam_left_head`).
- Appends `obs/base_velocity` → **22 dims** (else 19); if the action is already 22-dim (packed by
  the mobile datagen, §5.2), uses it as-is (no double append).

### 6.4 Output schema (matches real `ffw_sg2_rev1`)

| Feature | Shape |
|---|---|
| `observation.state` | (22,) — 19 joints + `linear_x, linear_y, angular_z` |
| `action` | (22,) — same layout |
| `observation.images.rgb.cam_left_head` / `cam_right_head` | [3, 376, 672] |
| `observation.images.rgb.cam_left_wrist` / `cam_right_wrist` | [3, 424, 240] |

19 joints, in order: `arm_l_joint1..7, gripper_l_joint1, arm_r_joint1..7, gripper_r_joint1,
head_joint1, head_joint2, lift_joint`.

---

## 7. VR base-driving controls **[MODIFIED]**

To drive the base during recording (Plan B), the VR publisher was extended (base controls only;
the arm/gripper/lift controls are unchanged from the base repo):

| Input | Action |
|---|---|
| Left thumbstick | Base translate (forward/back = x, left/right = y) |
| **A button (hold)** | Rotate right |
| **B button (hold)** | Rotate left |
| **Y button** | Toggle `LIFT+HEAD` ↔ `LIFT+CMD_VEL` base-driving mode |

Critical fix: `/cmd_vel` is published on **RELIABLE** QoS (was BEST_EFFORT); the SDK subscribes
RELIABLE, so the DDS mismatch had silently dropped every base command. See
[`DEPLOY_PLAN_B.md`](DEPLOY_PLAN_B.md) for the deployment/test procedure.

---

## 8. Known Limitations

- **Mobile 22-dim datagen has a lower success rate than fixed-base teleport** (~40 %). The
  physics-driven base is open-loop, so drifting attempts misplace the box and are dropped;
  `generation_guarantee=True` retries until the requested successes are reached, so the run takes
  proportionally longer.
- **Datagen env count is limited by camera-render memory, not compute** — `--num_envs` must stay
  small on a 24 GB GPU (4 safe, 10 OOMs). See §5.3.
- **Camera extrinsics are placeholders** — calibrate against the physical rig.
- **Rendering four cameras is GPU-heavy.** On the validated workstation (**NVIDIA RTX PRO 5000
  Blackwell Laptop, 24 GB**) frame drops do not occur in normal use; on lower-spec machines four
  cameras may saturate the GPU and slow teleop. Frame drops affect only teleop smoothness, **not the
  recorded data** (each frame is still the correct image at its sim timestep). Run headless to drop
  the GUI render if needed.
- **The action's base velocity uses the measured twist** as a stand-in for the command (the swerve
  tracks `/cmd_vel` closely); near-zero speed carries mild noise the real swerve/inertia absorbs.

---

## Appendix A — Files Added / Modified

All paths are under `overlays/` (the version-controlled overlay), applied onto the live checkout by
`setup.sh` / `sync_overlay.sh`. Everything else in the repo is base-repo / upstream code, unchanged.

### Added

| File | Purpose |
|---|---|
| `cyclo_lab/…/assets/robots/FFW_SG2_MOBILE.py` | Drivable-base articulation config |
| `cyclo_lab/…/data/robots/FFW/FFW_SG2_MOBILE.usd` | ~2 KB override layer over the stock USD |
| `cyclo_lab/…/controllers/swerve.py` (+ `__init__.py`) | 3-module holonomic swerve IK controller |
| `cyclo_lab/scripts/tools/build_ffw_sg2_mobile_usd.py` | Regenerates `FFW_SG2_MOBILE.usd` |
| `cyclo_lab/scripts/tools/check_ffw_sg2_mobile.py` | Drive / holonomic regression (6 checks) |
| `cyclo_lab/scripts/tools/teleop_sg2_mobile.py` | Keyboard base driving (dev tool) |
| `DEPLOY_PLAN_B.md` | Deployment & test procedure for mobile recording |

### Modified

| File | Change |
|---|---|
| `cyclo_lab/…/pick_place_l_table/joint_pos_env_cfg.py` | 4 cameras; `FFWSG2PickPlaceLTableMobileEnvCfg`; base-velocity obs; reset event |
| `cyclo_lab/…/pick_place_l_table/pick_place_env_cfg.py` | Camera scene slots + observation terms; teleop flags |
| `cyclo_lab/…/pick_place_l_table/__init__.py` | Register the mobile task **and the mobile Mimic task** |
| `cyclo_lab/…/pick_place_l_table/pick_place_l_table_mimic_env_cfg.py` | Add `FFWSG2PickPlaceLTableMobileMimicEnvCfg` (mobile 22-dim datagen cfg) |
| `cyclo_lab/…/pick_place_l_table/mdp/observations.py` | `base_planar_velocity` observation |
| `cyclo_lab/…/pick_place_l_table/mdp/ffw_sg2_l_table_events.py` | `reset_mobile_base_standing` event |
| `cyclo_lab/scripts/…/mimic/cyclo_mimic_datagen.py` | Datagen: SG2 head/lift action layout, strip camera obs from source episodes, box-carry latch |
| `cyclo_lab/…/pick_place_l_table/ltable_kinematic_l_motion.py` | Robust dual-hand grasp detection (reject one-handed / partial grasps) |
| `cyclo_lab/…/pick_place_l_table/pick_place_l_table_mimic_env.py` | Parallel (multi-env) datagen stepping; **physics-driven base replay** (`_physics_drive_step`, batched swerve) replacing teleport on the mobile task |
| `cyclo_lab/scripts/…/mimic/action_data_converter.py` | IK / joint convert **append `obs/base_velocity` → 22-dim** on mobile (19-dim otherwise) |
| `cyclo_lab/scripts/…/mimic/annotate_demos.py` | Feed 19-dim to the env; **re-pack the exported action to 22-dim** from `obs/base_velocity` |
| `cyclo_lab/scripts/…/mimic/generate_dataset.py` | **Post-process the generated file to 22-dim** actions after the sim app closes |
| `cyclo_lab/scripts/…/data_converter/isaaclab2lerobot.py` | Auto-detect cameras (rgb, CHW) + base velocity → 22-dim; **accept pre-packed 22-dim actions** |
| `cyclo_lab/scripts/…/dds_sdk/ffw_sg2_sdk.py` | Subscribe `/cmd_vel`, physical swerve base driving |
| `cyclo_lab/sg2_ltable_dashboard.py` | Session-based dataset naming; mobile tasks in the task list |
| `robotis_applications/robotis_vuer/robotis_vuer/vr_publisher_sg2.py` | Y-button mode toggle; A/B base rotation; `/cmd_vel` → RELIABLE QoS |

---

*Base repository: [`EKAIWORKER`](https://github.com/Disniekie01/EKAIWORKER) · Fork:
[`AI_HUN`](https://github.com/hun7407-lgtm/AI_HUN). Upstream: ROBOTIS `cyclo_lab`, `ai_worker`,
`robotis_applications` (pinned in `setup.sh`).*
