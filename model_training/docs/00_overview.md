# Overview

## Goal

Train a box pick-and-place policy for the FFW_SG2 humanoid from demonstration data,
using two independent approaches.

## Data flow

```
prepared LeRobot dataset (local)
        |
        +--> Track A: LeRobot (ACT / Diffusion / VQ-BeT) --> policy --> sim rollout eval
        |
        +--> Track B: GR00T N1.7 fine-tune                --> model  --> deploy
                       (dataset converted v3.0 -> v2.1 first)
```

The Hugging Face Hub is optional, used only to share datasets/models between
people/machines — except the GR00T backbone download and the Track B dataset
conversion, which pull from the Hub.

## Track comparison

| | LeRobot | GR00T N1.7 |
|---|---|---|
| Type | imitation-learning policies | vision-language-action (3B) fine-tune |
| Policies | ACT, Diffusion, VQ-BeT | single diffusion action head on a frozen VLM backbone |
| Dataset format | LeRobot v3.0 | LeRobot v2.1 (converted from v3.0) |
| Language input | not used | task instruction per frame |
| Compute (single task) | one 24 GB GPU is enough | 40 GB+ for action-head tuning; 80 GB+ for full |
| Runtime | `cyclo_lab` container / venv | dedicated `gr00t` Docker image |

## When to use which

- **LeRobot** — smaller, faster to train, no gated dependencies, good default for a
  single well-defined task with enough demonstrations.
- **GR00T** — starts from a pretrained VLA; useful when leveraging language conditioning
  or transferring across tasks. Needs the gated Cosmos backbone and more VRAM.

## Embodiment contract

See the table in the root [`README`](../README.md). The same 19–22 dim state/action
layout and camera set must hold across dataset generation, both training tracks, and
deployment. A mismatch in one place breaks inference.

## Prerequisites

NVIDIA GPU + driver, Docker + NVIDIA Container Toolkit, `git`. Details: each track's
`01_setup.md`.
