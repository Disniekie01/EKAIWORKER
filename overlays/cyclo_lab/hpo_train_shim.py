#!/usr/bin/env python3
"""Run `lerobot_train` with EpisodeAwareSampler fixed for episode subsets.

Why this exists
---------------
lerobot 0.5.2 builds the sampler like this (scripts/lerobot_train.py):

    if hasattr(active_cfg, "drop_n_last_frames"):        # diffusion only
        sampler = EpisodeAwareSampler(
            dataset.meta.episodes["dataset_from_index"],  # GLOBAL frame offsets
            dataset.meta.episodes["dataset_to_index"],
            episode_indices_to_use=dataset.episodes,
            ...)

`meta.episodes` is NOT filtered when `--dataset.episodes` selects a subset: it
still holds the global frame offsets of every episode. The loaded dataset,
however, is renumbered 0..N-1. So for any non-prefix subset the sampler emits
indices past the end of the dataset:

    IndexError: Invalid key: 159794 is out of bounds for size 156964

Only **diffusion** has `drop_n_last_frames`, which is why ACT / VQ-BeT tune
fine on an episode subset and diffusion always dies.

hpo_optuna.py must pass `--dataset.episodes` (that is how the held-out val split
is made), so we launch training through this shim, which rebuilds the episode
boundaries in subset-local coordinates.

Usage: exactly like `python -m lerobot.scripts.lerobot_train <args>`.
"""
import os

import torch.utils.data

import lerobot.scripts.lerobot_train as lerobot_train
from lerobot.datasets import EpisodeAwareSampler as _RealSampler

# ---------------------------------------------------------------------------
# Fix 2: DataLoader workers segfault when decoding video.
#
#   RuntimeError: DataLoader worker (pid ...) is killed by signal: Segmentation fault.
#
# PyAV/FFmpeg use internal threads. PyTorch creates DataLoader workers with
# `fork`, and forking a threaded process leaves the child with corrupted locks
# -> random segfaults mid-training. This is NOT a resource problem; more RAM or
# a bigger GPU does not help.
#
# `spawn` starts each worker as a fresh interpreter that inherits nothing, so the
# threaded PyAV state never reaches it. (`forkserver` is NOT enough: its server
# process is itself forked from the parent, so if PyAV/FFmpeg is already loaded
# by then the server inherits the broken state and every worker forked from it
# does too.) Override with HPO_MP_CONTEXT=forkserver|fork.
# ---------------------------------------------------------------------------

_MP_CONTEXT = os.environ.get("HPO_MP_CONTEXT", "spawn")

_RealDataLoader = torch.utils.data.DataLoader


class _SafeDataLoader(_RealDataLoader):
    def __init__(self, *args, **kwargs):
        if kwargs.get("num_workers", 0) > 0 and _MP_CONTEXT != "fork":
            kwargs.setdefault("multiprocessing_context", _MP_CONTEXT)
            # Workers are expensive to start under forkserver/spawn, so keep them.
            kwargs.setdefault("persistent_workers", True)
        super().__init__(*args, **kwargs)


torch.utils.data.DataLoader = _SafeDataLoader


def _subset_local_sampler(
    dataset_from_indices,
    dataset_to_indices,
    episode_indices_to_use=None,
    **kwargs,
):
    """Rebuild episode boundaries relative to the loaded subset, then delegate."""
    if episode_indices_to_use is None:
        return _RealSampler(
            dataset_from_indices, dataset_to_indices, None, **kwargs
        )

    # The dataset concatenates the selected episodes in ascending order, so the
    # local start of each is just the running sum of the preceding lengths.
    selected = sorted(episode_indices_to_use)
    local_from, local_to, offset = [], [], 0
    for ep in selected:
        length = dataset_to_indices[ep] - dataset_from_indices[ep]
        local_from.append(offset)
        local_to.append(offset + length)
        offset += length

    # episode_indices_to_use is now None: local_from/local_to already contain
    # only the selected episodes.
    return _RealSampler(local_from, local_to, None, **kwargs)


lerobot_train.EpisodeAwareSampler = _subset_local_sampler

if __name__ == "__main__":
    lerobot_train.main()
