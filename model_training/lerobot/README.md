# Track A — LeRobot

Train ACT / Diffusion / VQ-BeT / Pi0 / Pi0.5 / SmolVLA policies on a prepared LeRobot
v3.0 dataset and evaluate them by rollout in Isaac Sim. ACT / Diffusion / VQ-BeT run
fully local; the VLA policies download pretrained weights from HF ([`03_train.md`](03_train.md)).

Producing the dataset (recording, HDF5 conversion) is upstream of this track.

## Steps

1. [`01_setup.md`](01_setup.md) — clone the AIWORKER stack, build the container, set up both Python environments.
2. [`02_dataset.md`](02_dataset.md) — point at the local dataset; unify camera resolution if needed.
3. [`03_train.md`](03_train.md) — train ACT / Diffusion / VQ-BeT / Pi0 / Pi0.5 / SmolVLA.
4. [`04_eval.md`](04_eval.md) — convert the policy and run a sim rollout.
5. [`05_hpo.md`](../../overlays/cyclo_lab/05_hpo.md) — (optional) Optuna hyperparameter
   search. Ships with the `cyclo_lab` overlay alongside the HPO scripts.

Errors: [`docs/TROUBLESHOOTING.md`](../docs/TROUBLESHOOTING.md).

## Where commands run

Each command block starts with a comment marking where to run it and the working
directory:

- `# [host] <path>` — on the host shell.
- `# [container: cyclo_lab] /workspace/cyclo_lab` — inside the container (after
  `./container.sh enter`).

Cloning, the container build/start, and file copies are on the host. Training and
evaluation run inside the `cyclo_lab` container, from `/workspace/cyclo_lab`. The host
`~/AIWORKER/cyclo_lab` is that same directory, mounted.

## Two Python environments

The container ships two interpreters with different roles. Do not mix them.

| Use | Interpreter | Python | lerobot | Data |
|-----|-------------|--------|---------|------|
| **Training** | `/root/lerobot_env` | 3.12 | 0.5.2 | v3.0 |
| **Sim rollout** | `$ISAACLAB_PATH/_isaac_sim/python.sh` | 3.11 | 0.3.3 | v2.1 |

Isaac Sim is on 3.11 and cannot install 0.5.2 (needs ≥3.12), so a 0.5.2-trained policy is
converted to 0.3.3 before rollout ([`04_eval.md`](04_eval.md)).

## Verified toolchain (training venv)

| Component | Version |
|-----------|---------|
| Python | 3.12.3 |
| lerobot | `github.com/ROBOTIS-GIT/lerobot-cyclo@2e9cd87` (reports 0.5.2) |
| torch / torchvision | 2.7.1+cu128 / 0.22.1+cu128 |
| numpy | 2.2.6 |
| diffusers | 0.38.0 (Diffusion policy) |

Full list: [`requirements.txt`](requirements.txt). Verified from a clean `python:3.12`
container; `lerobot_train --help` exposes the documented arguments.

## Scripts

All but one ship with the `cyclo_lab` overlay, so `setup.sh` puts them in the mounted
tree ([`01_setup.md`](01_setup.md)):

| Script | Source | Env | Purpose |
|--------|--------|-----|---------|
| `hpo_optuna.py`, `hpo_train_shim.py` | `overlays/cyclo_lab/` | training | Optuna HPO |
| `sim2real/.../inference/convert_policy_v05_to_v03.py` | `overlays/cyclo_lab/scripts/` | training | 0.5.2 policy → 0.3.3 for rollout |
| `sim2real/.../inference/eval_policy_sim.py` | `overlays/cyclo_lab/scripts/` | isaac | Isaac Sim rollout evaluation |
| `unify_resolution.py` (shared) | [`../scripts/`](../scripts/) — copy manually | python3 + ffmpeg | unify camera resolution |
