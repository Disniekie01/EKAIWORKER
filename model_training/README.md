# aiworker-imitation-learning

Imitation-learning training pipelines for the ROBOTIS AIWORKER (FFW_SG2) humanoid,
box pick-and-place. Given a prepared dataset, each track goes from a bare machine through
setup, training, and evaluation on one computer. ACT / Diffusion / VQ-BeT run fully
local; the VLA policies (Pi0, Pi0.5, SmolVLA) and Track B need a one-time Hugging Face
download (Pi0/Pi0.5 tokenizer and the GR00T backbone are gated — token required).

- **`lerobot/`** — LeRobot policies (ACT, Diffusion, VQ-BeT, Pi0, Pi0.5, SmolVLA) + Optuna HPO.
- **`gr00t/`** — NVIDIA GR00T N1.7 (3B VLA) fine-tuning.

## Layout

```
docs/            Shared overview + troubleshooting (both tracks)
scripts/         Shared utilities (camera-resolution unification)
lerobot/         Track A — LeRobot training (numbered steps 01..05)
gr00t/           Track B — GR00T fine-tuning (numbered steps 01..04)
```

## Quick start

Read [`docs/00_overview.md`](docs/00_overview.md) first, then pick a track:

| Track | Start here | Runs in | Flow |
|-------|------------|---------|------|
| LeRobot | [`lerobot/README.md`](lerobot/README.md) | AIWORKER `cyclo_lab` container | setup → dataset → train → sim-rollout eval |
| GR00T | [`gr00t/README.md`](gr00t/README.md) | dedicated `gr00t` Docker image | setup → dataset → fine-tune → deploy |

## Embodiment contract

Both tracks target the same robot. Changing it means changing every step.

| Group | Index | DoF |
|-------|-------|-----|
| arm_left | 0:8 | 7 joints + 1 gripper |
| arm_right | 8:16 | 7 joints + 1 gripper |
| head | 16:18 | 2 |
| lift | 18:19 | 1 |
| (base) | 19:22 | linear_x, linear_y, angular_z — present in some datasets |

Camera and state/action dimensionality depend on the dataset; see each track's dataset step.

## Verified versions

Pinned toolchains, taken from the working environments (2026-07). See each track's
`01_setup` for exact install commands.

| | LeRobot track | GR00T track |
|---|---|---|
| Python | 3.12.3 | 3.10.12 |
| torch / torchvision | 2.7.1+cu128 / 0.22.1+cu128 | 2.7.1+cu128 / 0.22.1+cu128 |
| CUDA | 12.8 | 12.8 |
| numpy | 2.2.6 | 1.26.4 |
| framework | lerobot-cyclo @ 2e9cd87 | Isaac-GR00T-n1.7 @ e81d02b |

## License

Apache-2.0. Third-party components (LeRobot, Isaac-GR00T, GR00T N1.7 weights, Cosmos
backbone) retain their own licenses.
