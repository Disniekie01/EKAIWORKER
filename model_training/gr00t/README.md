# Track B — GR00T N1.7

Fine-tune NVIDIA GR00T N1.7 (3B vision-language-action model) on a LeRobot v2.1 dataset,
using the ROBOTIS Isaac-GR00T fork so checkpoints stay compatible with `cyclo_intelligence`.

## Steps

1. [`01_setup.md`](01_setup.md) — clone the fork, build the Docker image, obtain the base model / gated backbone.
2. [`02_dataset.md`](02_dataset.md) — convert v3.0 to v2.1, place the modality config, set the task instruction.
3. [`03_train.md`](03_train.md) — run `launch_finetune.py`.
4. [`04_deploy.md`](04_deploy.md) — model files required for inference.

Errors: [`docs/TROUBLESHOOTING.md`](../docs/TROUBLESHOOTING.md).

## Where commands run

Each command block starts with a comment marking where to run it and the working
directory:

- `# [host] <path>` — on the host shell, at that directory.
- `# [container: gr00t] <path>` — inside the running `gr00t` container.

Build, dataset prep, and file management happen on the host. Training happens inside the
`gr00t` container, which is started with `docker run` (ephemeral, `--rm`).

## Default hardware

Written for an **RTX PRO 6000 Blackwell workstation (96 GB)**. At this VRAM the diffusion
action head is trained by default (recommended). Smaller cards (e.g. RTX PRO 5000
Blackwell, 24 GB) cannot train the head; reduced-VRAM fallback:
[`docs/TROUBLESHOOTING.md`](../docs/TROUBLESHOOTING.md#b3-train).

## Hugging Face requirement

One-time download: `nvidia/GR00T-N1.7-3B` (base) + gated `nvidia/Cosmos-Reason2-2B`
(accept once, token). Keep `HF_TOKEN` exported every run. Details: [`01_setup.md`](01_setup.md).

## Environment (verified)

| Component | Version |
|-----------|---------|
| Docker image | `gr00t:latest` (built from the fork; ~41 GB) |
| base OS | Ubuntu 22.04 |
| CUDA | 12.8 |
| Python | 3.10.12 |
| torch / torchvision | 2.7.1+cu128 / 0.22.1+cu128 |
| transformers | 4.57.3 |
| diffusers | 0.35.1 |
| numpy | 1.26.4 |
| flash_attn | 2.7.4.post1 |
| fork | `github.com/ROBOTIS-GIT/Isaac-GR00T-n1.7@e81d02b` |

## Configs

[`configs/`](configs/) holds the FFW_SG2 L-Table modality:

| File | Purpose |
|------|---------|
| `ffw_sg2_ltable_config.py` | modality config (19-dim, single head camera), registers `NEW_EMBODIMENT` |
| `modality.json` | dataset-side modality mapping (state/action/video/annotation) |
| `../scripts/unify_resolution.py` (shared) | unify camera resolution |
