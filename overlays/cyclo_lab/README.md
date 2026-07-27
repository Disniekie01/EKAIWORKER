# cyclo_lab

[![IsaacSim](https://img.shields.io/badge/IsaacSim-5.1.0-silver.svg)](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.3.0-silver)](https://isaac-sim.github.io/IsaacLab/main/index.html)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://docs.python.org/3/whatsnew/3.11.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/22.04/)
[![License](https://img.shields.io/badge/license-Apache2.0-yellow.svg)](https://opensource.org/license/apache-2-0)

https://github.com/user-attachments/assets/28347b4b-f90c-4a4f-8916-621f917d86cb

## Overview

**cyclo_lab** is a research-oriented repository based on [Isaac Lab](https://isaac-sim.github.io/IsaacLab), designed to enable reinforcement learning (RL) and imitation learning (IL) experiments using Robotis robots in simulation.
This project provides simulation environments, configuration tools, and task definitions tailored for Robotis hardware, leveraging NVIDIA Isaac Sim’s powerful GPU-accelerated physics engine and Isaac Lab’s modular RL pipeline.

> [!IMPORTANT]
> This repository currently depends on **IsaacLab v2.2.0** or higher.
>

## Installation (Docker)

Docker installation provides a consistent environment with all dependencies pre-installed.

**Prerequisites:**
- Docker and Docker Compose installed
- NVIDIA Container Toolkit installed
- NVIDIA GPU with appropriate drivers

**Steps:**

1. Clone cyclo_lab repository with submodules:

   ```bash
   git clone --recurse-submodules https://github.com/ROBOTIS-GIT/cyclo_lab.git
   cd cyclo_lab
   ```

   If you already cloned without submodules, initialize them:
   ```bash
   git submodule update --init --recursive
   ```

2. Build and start the Docker container:

   ```bash
   ./docker/container.sh start
   ```

3. Enter the container:

   ```bash
   ./docker/container.sh enter
   ```

**Docker Commands:**
- `./docker/container.sh start` - Build and start the container
- `./docker/container.sh enter` - Enter the running container
- `./docker/container.sh stop` - Stop the container
- `./docker/container.sh logs` - View container logs
- `./docker/container.sh clean` - Remove container and image

**What's included in the Docker image:**
- Isaac Sim 5.1.0
- Isaac Lab v2.3.0 (from third_party submodule)
- CycloneDDS 0.10.2 (from third_party submodule)
- robotis_dds_python (from third_party submodule)
- LeRobot 0.3.3 (in separate virtual environment at `~/lerobot_env`)
- All required dependencies and configurations

## Try examples

### Sim2Sim
<details>
<summary>Reinforcement learning</summary>

**OMY Reach Task**

```bash
# Train
python scripts/reinforcement_learning/rsl_rl/train.py --task Cyclo-Reach-OMY-v0 --num_envs=512 --headless

# Play
python scripts/reinforcement_learning/rsl_rl/play.py --task Cyclo-Reach-OMY-v0 --num_envs=16
```

**OMY Lift Task**

```bash
# Train
python scripts/reinforcement_learning/rsl_rl/train.py --task Cyclo-Lift-Cube-OMY-v0 --num_envs=512 --headless

# Play
python scripts/reinforcement_learning/rsl_rl/play.py --task Cyclo-Lift-Cube-OMY-v0 --num_envs=16
```

**OMY Open drawer Task**

```bash
# Train
python scripts/reinforcement_learning/rsl_rl/train.py --task Cyclo-Open-Drawer-OMY-v0 --num_envs=512 --headless

# Play
python scripts/reinforcement_learning/rsl_rl/play.py --task Cyclo-Open-Drawer-OMY-v0 --num_envs=16
```

**FFW-BG2 reach Task**

```bash
# Train
python scripts/reinforcement_learning/rsl_rl/train.py --task Cyclo-Reach-FFW-BG2-v0 --num_envs=512 --headless

# Play
python scripts/reinforcement_learning/rsl_rl/play.py --task Cyclo-Reach-FFW-BG2-v0 --num_envs=16
```

</details>

<details>
<summary>Imitation learning</summary>

>
> If you want to control a **SINGLE ROBOT** with the keyboard during playback, add `--keyboard` at the end of the play script.
>
> ```
> Key bindings:
> =========================== =========================
> Command                     Key
> =========================== =========================
> Toggle gripper (open/close) K      
> Move arm along x-axis       W / S   
> Move arm along y-axis       A / D
> Move arm along z-axis       Q / E
> Rotate arm along x-axis     Z / X
> Rotate arm along y-axis     T / G
> Rotate arm along z-axis     C / V
> =========================== =========================
> ```

**OMY Stack Task** (Stack the blocks in the following order: blue → red → green.)

```bash
# Teleop and record
python scripts/imitation_learning/isaaclab_recorder/record_demos.py --task Cyclo-Stack-Cube-OMY-IK-Rel-v0 --teleop_device keyboard --dataset_file ./datasets/dataset.hdf5 --num_demos 10

# Annotate
python scripts/imitation_learning/isaaclab_mimic/annotate_demos.py --device cuda --task Cyclo-Stack-Cube-OMY-IK-Rel-Mimic-v0 --auto --input_file ./datasets/dataset.hdf5 --output_file ./datasets/annotated_dataset.hdf5 --headless

# Mimic data
python scripts/imitation_learning/isaaclab_mimic/generate_dataset.py \
--device cuda --num_envs 100 --generation_num_trials 1000 \
--input_file ./datasets/annotated_dataset.hdf5 --output_file ./datasets/generated_dataset.hdf5 --headless

# Train
python scripts/imitation_learning/robomimic/train.py \
--task Cyclo-Stack-Cube-OMY-IK-Rel-v0 --algo bc \
--dataset ./datasets/generated_dataset.hdf5

# Play
python scripts/imitation_learning/robomimic/play.py \
--device cuda --task Cyclo-Stack-Cube-OMY-IK-Rel-v0 --num_rollouts 50 \
--checkpoint /PATH/TO/desired_model_checkpoint.pth
```

**FFW-BG2 Pick and Place Task** (Move the red stick into the basket.)

```bash
# Teleop and record
python scripts/imitation_learning/isaaclab_recorder/record_demos.py --task Cyclo-PickPlace-FFW-BG2-IK-Rel-v0 --teleop_device keyboard --dataset_file ./datasets/dataset.hdf5 --num_demos 10 --enable_cameras

# Annotate
python scripts/imitation_learning/isaaclab_mimic/annotate_demos.py --device cuda --task Cyclo-PickPlace-FFW-BG2-Mimic-v0 --input_file ./datasets/dataset.hdf5 --output_file ./datasets/annotated_dataset.hdf5 --enable_cameras

# Mimic data
python scripts/imitation_learning/isaaclab_mimic/generate_dataset.py \
--device cuda --num_envs 20 --generation_num_trials 300 \
--input_file ./datasets/annotated_dataset.hdf5 --output_file ./datasets/generated_dataset.hdf5 --enable_cameras --headless

# Train
python scripts/imitation_learning/robomimic/train.py \
--task Cyclo-PickPlace-FFW-BG2-IK-Rel-v0 --algo bc \
--dataset ./datasets/generated_dataset.hdf5

# Play
python scripts/imitation_learning/robomimic/play.py \
--device cuda --task Cyclo-PickPlace-FFW-BG2-IK-Rel-v0  --num_rollouts 50 \
--checkpoint /PATH/TO/desired_model_checkpoint.pth --enable_cameras
```
</details>

### Sim2Real
>
> **Important**
>
> OMY Hardware Setup:
> To run Sim2Real with the real OMY robot, you need to bring up the robot.
>
> This can be done using ROBOTIS’s [open_manipulator repository](https://github.com/ROBOTIS-GIT/open_manipulator.git).
> 
> AI WORKER Hardware Setup:
> To run Sim2Real with the real AI WORKER robot, you need to bring up the robot.
>
> This can be done using ROBOTIS’s [ai_worker repository](https://github.com/ROBOTIS-GIT/ai_worker.git).
>
> The training and inference of the collected dataset should be carried out using physical_ai_tools.
> This can be done using ROBOTIS’s [physical_ai_tools](https://github.com/ROBOTIS-GIT/physical_ai_tools)
>

<details>
<summary>Bringup</summary>

**ROBOTIS HAND VR Teleoperation in Isaac Sim**
[Introduction YouTube](https://www.youtube.com/watch?v=4D8if2lZY7o)

https://github.com/user-attachments/assets/67267195-db02-4dd9-83cf-00830b4bc13f

Launch the AI Worker SH5 model and DDS bridge

```bash
# Grid with SH5 AI Worker
python scripts/sim2real/bringup/sh5_dds_bringup.py --domain_id 30 --enable_gravity --enable_camera_views

# NVIDIA Simple Warehouse environment
python scripts/sim2real/bringup/sh5_dds_bringup.py --domain_id 30 --enable_gravity --enable_camera_views --enable_environment
```

</details>

<details>
<summary>Reinforcement learning</summary>

**OMY Reach Task**
[Introduction YouTube](https://www.youtube.com/watch?v=pSY0Gb5b5kI)

https://github.com/user-attachments/assets/6c27bdb1-3a6b-4686-a546-8f14f01e4abe

Run Sim2Real Reach Policy on OMY

```bash
# Train
python scripts/reinforcement_learning/rsl_rl/train.py --task Cyclo-Reach-OMY-v0 --num_envs=512 --headless

# Play (You must run rsl_rl play in order to generate the policy file.)
python scripts/reinforcement_learning/rsl_rl/play.py --task Cyclo-Reach-OMY-v0 --num_envs=16

# Sim2Real
python scripts/sim2real/reinforcement_learning/inference/OMY/reach/run_omy_reach.py --model_dir=<2025-07-10_08-47-09>
```

Replace <2025-07-10_08-47-09> with the actual timestamp folder name under:
```bash
logs/rsl_rl/reach_omy/
```
</details>

<details>
<summary>Imitation learning</summary>

**OMY Pick and Place Task**

**Sim2Sim**

https://github.com/user-attachments/assets/a6e75e80-203f-47d1-974b-d4c5435c15bc

**Sim2Real**

https://github.com/user-attachments/assets/8ec9d245-f8e0-4bcc-b683-0ea2864de495


* Teleop and record demos
```bash
python scripts/sim2real/imitation_learning/recorder/record_demos.py --task=Cyclo-Real-Pick-Place-Bottle-OMY-v0 --robot_type OMY --dataset_file ./datasets/omy_pick_place_task.hdf5 --num_demos 10 --enable_cameras

```

* Mimic generate dataset

```bash
# Data convert ee_pose action from joint action
python scripts/sim2real/imitation_learning/mimic/action_data_converter.py --robot_type OMY --input_file ./datasets/omy_pick_place_task.hdf5 --output_file ./datasets/processed_omy_pick_place_task.hdf5 --action_type ik

# Annotate dataset
python scripts/sim2real/imitation_learning/mimic/annotate_demos.py --task Cyclo-Real-Mimic-Pick-Place-Bottle-OMY-v0 --auto --input_file ./datasets/processed_omy_pick_place_task.hdf5 --output_file ./datasets/annotated_dataset.hdf5 --enable_cameras --headless

# Generate dataset
python scripts/sim2real/imitation_learning/mimic/generate_dataset.py --device cuda --num_envs 10 --task Cyclo-Real-Mimic-Pick-Place-Bottle-OMY-v0 --generation_num_trials 500 --input_file ./datasets/annotated_dataset.hdf5 --output_file ./datasets/generated_dataset.hdf5 --enable_cameras --headless

# Data convert joint action from ee_pose action
python scripts/sim2real/imitation_learning/mimic/action_data_converter.py --robot_type OMY --input_file ./datasets/generated_dataset.hdf5 --output_file ./datasets/processed_generated_dataset.hdf5 --action_type joint

```

* Data convert lerobot dataset from IsaacLab hdf dataset
```bash
lerobot-python scripts/sim2real/imitation_learning/data_converter/isaaclab2lerobot.py \
    --task=Cyclo-Real-Pick-Place-Bottle-OMY-v0 \
    --robot_type OMY \
    --dataset_file ./datasets/processed_generated_dataset.hdf5

```

* Inference in simulation
```bash
python scripts/sim2real/imitation_learning/inference/inference_demos.py --task Cyclo-Real-Pick-Place-Bottle-OMY-v0 --robot_type OMY --enable_cameras

```

**FFW SG2 Pick and Place Task**

https://github.com/user-attachments/assets/cdb3afca-f0db-4fb6-bf17-e361f3aa254b

* Teleop and record demos
```bash
python scripts/sim2real/imitation_learning/recorder/record_demos.py --task=Cyclo-Real-Pick-Place-FFW-SG2-v0 --robot_type FFW_SG2 --dataset_file ./datasets/ffw_sg2_raw.hdf5 --num_demos 4 --enable_cameras

```

* Mimic generate dataset
```bash

# Data convert ee_pose action from joint action
python scripts/sim2real/imitation_learning/mimic/action_data_converter.py --robot_type FFW_SG2 --input_file ./datasets/ffw_sg2_raw.hdf5 --output_file ./datasets/ffw_sg2_ik.hdf5 --action_type ik

# Annotate dataset
python scripts/sim2real/imitation_learning/mimic/annotate_demos.py --task Cyclo-Real-Mimic-Pick-Place-FFW-SG2-v0 --auto --input_file ./datasets/ffw_sg2_ik.hdf5 --output_file ./datasets/ffw_sg2_annotate.hdf5 --enable_cameras --headless

# Generate dataset
python scripts/sim2real/imitation_learning/mimic/generate_dataset.py --device cuda --num_envs 10 --task Cyclo-Real-Mimic-Pick-Place-FFW-SG2-v0 --generation_num_trials 500 --input_file ./datasets/ffw_sg2_annotate.hdf5 --output_file ./datasets/ffw_sg2_generate.hdf5 --enable_cameras --headless

# Data convert joint action from ee_pose action
python scripts/sim2real/imitation_learning/mimic/action_data_converter.py --robot_type FFW_SG2 --input_file ./datasets/ffw_sg2_generate.hdf5 --output_file ./datasets/ffw_sg2_final.hdf5 --action_type joint

```

* Data convert lerobot dataset from IsaacLab hdf dataset
```bash
lerobot-python scripts/sim2real/imitation_learning/data_converter/isaaclab2lerobot.py \
    --task=Cyclo-Real-Pick-Place-FFW-SG2-v0 \
    --robot_type FFW_SG2 \
    --dataset_file ./datasets/ffw_sg2_final.hdf5

```

* Inference in simulation
```bash
python scripts/sim2real/imitation_learning/inference/inference_demos.py --task Cyclo-Real-Pick-Place-FFW-SG2-v0  --robot_type FFW_SG2 --enable_cameras

```

### FFW-SG2 L-table Robomimic Play Notes

For `Cyclo-Real-Pick-Place-LTable-FFW-SG2-v0`, the play script supports a hybrid mode:
- policy controls arm/gripper/head/lift actions
- scripted base `L` movement runs after grasp latch

```bash
python scripts/imitation_learning/robomimic/play.py \
  --device cuda \
  --task Cyclo-Real-Pick-Place-LTable-FFW-SG2-v0 \
  --checkpoint /PATH/TO/model_epoch_20.pth \
  --num_rollouts 10 --horizon 2000 \
  --enable_cameras --action_mode inference \
  --scripted_l_motion
```

Key flags:
- `--scripted_l_motion`: enable scripted base rotate+forward `L` motion during play
- `--action_mode inference`: initializes action terms required by real-world SG2 envs
- `--remap_ffw_sg2_actions` / `--no_remap_ffw_sg2_actions`: control SG2 head/lift action index remapping

## Experimental / Recent fixes (SG2 L-table)

Work on this branch covers SG2 L-table imitation-learning fixes (issues 2–8), dashboard knobs, and optional experimental tools. Default Mimic pipeline steps are unchanged unless noted.

**Dashboard layout** (`sg2_ltable_dashboard.py`): **Config — L-motion / home / grasp** sits after Recording. **Merge datasets** and **LeRobot v3.0 export** (both Experimental) sit below the Mimic pipeline card.

**New training env:** see [docs/CREATE_TRAINING_ENV.md](docs/CREATE_TRAINING_ENV.md).

### Issue 2 — Head/lift action mapping

- **Problem:** Trailing action order for `lift` / `head1` / `head2` disagreed between AM, observations, and IK conversion, corrupting Mimic datagen and replay.
- **Fix:** Aligned trailing order to `lift`, `head1`, `head2` in datagen and converters; added a mapping verifier.
- **Where to edit:** `scripts/sim2real/imitation_learning/mimic/cyclo_mimic_datagen.py`, `action_data_converter.py`, `verify_sg2_action_mapping.py` (smoke: `smoke_sg2_lift_mapping.py`).

### Issue 3 — Dual wrist cameras

- **Problem:** L-table IL needed left/right wrist RGB in the scene, LeRobot export, DDS publish, and robomimic configs.
- **Fix:** Added `cam_wrist_left` / `cam_wrist_right` to scene/obs, discovery, DDS, and robomimic JSON; tuned wrist camera poses.
- **Where to edit:** `.../pick_place_l_table/joint_pos_env_cfg.py` (poses), obs terms in `pick_place_env_cfg.py`, `isaaclab2lerobot.py`, `ffw_sg2_sdk.py`, `agents/robomimic/bc_rnn_image.json` (smoke: `smoke_sg2_wrist_cams.py`).

### Issue 4 — Arm jitter (IK / annotate / datagen)

- **Problem:** `reset_to` fought DiffIK, and quaternion double-cover caused discontinuous arm targets / visible jitter.
- **Fix:** Prefer live robot joints via `_with_live_robot_joints`; enforce quat continuity across resets/replays.
- **Where to edit:** Mimic env / datagen / annotate paths under `.../mimic/` and `pick_place*_mimic_env.py`; validators `verify_sg2_arm_jitter.py`, `smoke_sg2_arm_jitter.py`.

### Issue 5 — Raw + generated merge (Experimental)

- **Problem:** Wanted a mixed dataset (human raw + Mimic joint) without altering the default Mimic chain.
- **Fix:** Standalone CLI `merge_joint_datasets.py` and dashboard **Merge datasets (Experimental)** → `*_mixed.hdf5`. Does **not** change default pipeline outputs.
- **Where to edit:** `scripts/sim2real/imitation_learning/mimic/merge_joint_datasets.py`, `sg2_ltable_dashboard.py` (smoke: `smoke_sg2_merge_datasets.py`).

### Issue 6 — LeRobot format 3.0 (Experimental)

- **Problem:** Optional export to LeRobot dataset format 3.0 alongside default v2.1.
- **Fix:** `--dataset_format {v2,v3}` on `isaaclab2lerobot.py`; `setup_lerobot_v3_env.sh`; dashboard **LeRobot v3.0 export (Experimental)**. Live smoke passed 14/14. Needs `lerobot_env_v3` with pinned `av>=12,<16`.
- **Where to edit:** `scripts/sim2real/imitation_learning/data_converter/isaaclab2lerobot.py`, `setup_lerobot_v3_env.sh`, dashboard experimental card (smoke: `smoke_sg2_lerobot_v3.py`).

### Issue 7 — Swerve / wheel physics (future)

- **Problem:** Physics-based swerve wheels vs kinematic root teleport for L-motion.
- **Fix:** Deferred. Kinematic root teleport remains the default (`teleop_l_use_swerve=False`).
- **Where to edit / read:** [docs/FUTURE_FIX_swerve_wheel_physics.md](docs/FUTURE_FIX_swerve_wheel_physics.md); L-motion in `ltable_kinematic_l_motion.py` and `ffw_sg2_sdk.py`.

### Issue 8 — Config of hardcoded motions/params

- **Problem:** L-yaw, forward distance, durations, home lift/pose, grasp thresholds, soft-grip / optional gripper PD lived as hardcodes.
- **Fix:** Fields on `PickPlaceLTableEnvCfg` + `sync_configured_params()`; `record_demos.py` CLI flags; dashboard Config card with shell-env fallback.
- **Where to edit:** `.../pick_place_l_table/pick_place_env_cfg.py`, `record_demos.py`, `sg2_ltable_dashboard.py` (smoke: `smoke_sg2_teleop_cfg.py`). Prefer env-cfg knobs for new tasks — see the [training env guide](docs/CREATE_TRAINING_ENV.md).

## License

This repository is licensed under the **Apache 2.0 License**. See [LICENSE](LICENSE) for details.

### Third-party components

- **Isaac Lab**: BSD-3-Clause License, see [LICENSE-IsaacLab](LICENSE-IsaacLab)
