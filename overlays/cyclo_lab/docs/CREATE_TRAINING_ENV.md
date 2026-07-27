# Creating a new training env (FFW SG2 / SH5)

Practical checklist for adding a manager-based real-world task under this repo. Prefer copying an existing task and renaming rather than scaffolding from scratch.

## Where tasks live

```
source/cyclo_lab/cyclo_lab/real_world_tasks/manager_based/
  FFW_SG2/          # dual-gripper SG2
  FFW_SH5/          # five-finger hands
  OMY/              # other robots
```

Good templates:

| Goal | Copy from |
|------|-----------|
| Front-table pick/place + base L-motion | `FFW_SG2/pick_place_l_table/` |
| Simpler scene, Mimic on shared mimic env | `FFW_SG2/single_box_far/` or `pick_place/` |

## Files to create (typical SG2 task)

Under `.../manager_based/FFW_SG2/<your_task>/`:

1. **`__init__.py`** — `gymnasium` registration for the play/record id and (if IL) Mimic id.
2. **Scene / pick-place env cfg** — e.g. `pick_place_env_cfg.py` or `*_env_cfg.py` (tables, objects, lights, MDP terms).
3. **`joint_pos_env_cfg.py`** — concrete env cfg used for recording / joint play (cameras, joint actions, initial state).
4. **Mimic (IL only)** — `*_mimic_env_cfg.py`; optionally a task-specific `*_mimic_env.py` if behavior differs from shared `pick_place/pick_place_mimic_env.py`.
5. **`mdp/`** — observations, terminations, events as needed.
6. **`agents/`** (optional) — e.g. `robomimic/bc_rnn_image.json` for BC training/play.

Parent package import: ensure `FFW_SG2/__init__.py` (or package discovery) loads your submodule the same way siblings do.

## Gym registration pattern

Mirror `pick_place_l_table/__init__.py`:

- Record/play: `id="Cyclo-Real-...-FFW-SG2-v0"`, `entry_point="isaaclab.envs:ManagerBasedRLEnv"`, `env_cfg_entry_point` → your `joint_pos_env_cfg:...EnvCfg`.
- Mimic: `id="Cyclo-Real-Mimic-...-FFW-SG2-v0"`, `entry_point` → mimic env class, `env_cfg_entry_point` → `*_mimic_env_cfg:...`.

Use distinct, stable ids; keep Mimic and non-Mimic pairs consistent with dashboard naming.

## Teleop / motion knobs (Issue 8 pattern)

Put L-yaw, forward distance, durations, home joints, grasp thresholds, soft-grip / gripper PD on the **env cfg** class (see `PickPlaceLTableEnvCfg` + `sync_configured_params()`), not scattered hardcodes.

Wire CLI overrides in `scripts/sim2real/imitation_learning/recorder/record_demos.py` and (if desired) the dashboard Config card / `TELEOP_*` env vars in `sg2_ltable_dashboard.py`.

## Dashboard registration

In `sg2_ltable_dashboard.py`, add an entry to `TASKS`:

```python
"My Task Label": {
    "id": "Cyclo-Real-...-FFW-SG2-v0",
    "mimic_id": "Cyclo-Real-Mimic-...-FFW-SG2-v0",
    "dataset": "ffw_sg2_my_task_raw.hdf5",
    "robot": "FFW_SG2",
},
```

`dataset` is the raw HDF5 basename the pipeline derives `*_ik`, `*_annotate`, `*_generate`, `*_joint`, LeRobot paths from.

## Dataset naming and pipeline

Default Mimic flow (dashboard or CLI):

1. **Record** — `record_demos.py --task <id> --robot_type FFW_SG2 --dataset_file ./datasets/<name>_raw.hdf5 --enable_cameras`
2. **IK convert** — `action_data_converter.py ... --action_type ik` → `*_ik.hdf5`
3. **Annotate** — `annotate_demos.py --task <mimic_id> ...`
4. **Generate** — `generate_dataset.py` / `cyclo_mimic_datagen.py`
5. **Joint convert** — `action_data_converter.py ... --action_type joint` → `*_joint.hdf5`
6. **Optional LeRobot** — `isaaclab2lerobot.py` (default v2.1; Experimental v3 via `--dataset_format v3`)

Experimental (not in Full pipeline): merge raw+joint (`merge_joint_datasets.py`), LeRobot v3.

## Cameras and action dims

- Add cameras on the robot/scene cfg used by `joint_pos_env_cfg` and list them in observation groups.
- Keep trailing head/lift order consistent with existing SG2 mapping (`lift`, `head1`, `head2`) across obs, converters, and datagen (Issue 2).
- Update robomimic / LeRobot feature lists if you add views (e.g. wrist cams — Issue 3).

## Quick verify

- Import/register: launch `record_demos.py` with your `--task` once.
- Smokes under `scripts/sim2real/imitation_learning/mimic/` and `recorder/` when touching mapping, jitter, merge, LeRobot, or teleop cfg.
- For base motion physics vs kinematic teleport, read [FUTURE_FIX_swerve_wheel_physics.md](FUTURE_FIX_swerve_wheel_physics.md) before changing defaults.

## Key paths (cheatsheet)

| Piece | Path |
|-------|------|
| SG2 tasks | `source/cyclo_lab/.../manager_based/FFW_SG2/` |
| L-table template | `.../pick_place_l_table/` |
| Record | `scripts/sim2real/imitation_learning/recorder/record_demos.py` |
| Mimic / convert | `scripts/sim2real/imitation_learning/mimic/` |
| LeRobot export | `scripts/sim2real/imitation_learning/data_converter/isaaclab2lerobot.py` |
| Dashboard | `sg2_ltable_dashboard.py` |
| Fixes overview | [README — Experimental / Recent fixes](../README.md#experimental--recent-fixes-sg2-l-table) |
