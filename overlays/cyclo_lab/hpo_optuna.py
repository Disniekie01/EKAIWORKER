#!/usr/bin/env python3
"""Optuna hyperparameter search for LeRobot policies (ACT / VQ-BeT / Diffusion).

The objective is **held-out validation loss**, not training loss. LeRobot's train
script has no validation split, and its built-in `eval_freq` only works for gym
envs (our Isaac Sim L-Table task is not one) -- so we split the dataset's
*episodes* into train/val ourselves:

  * each trial trains a SHORT run on the train episodes via `lerobot_train`
  * the resulting checkpoint is then scored on the held-out val episodes, using
    the exact same preprocessor pipeline lerobot trains with
  * Optuna minimises that val loss; bad trials are pruned early using the
    training loss stream

Val loss is a *proxy*. It ranks configs cheaply, but it does not measure task
success. Take the top few configs from here, train them full-length, and compare
real success rate with `eval_policy_sim.py` before picking a winner.

Run inside the cyclo_lab container:

    lerobot-pip install optuna                 # once

    # search + full training + upload, in one command
    lerobot-python hpo_optuna.py --policy act --n_trials 20 --steps 8000 \
        --then_train --full_steps 100000 --full_num_workers 8 --push_to_hub

    # search only (saves outputs/hpo/<policy>/best_params.json)
    lerobot-python hpo_optuna.py --policy act --n_trials 20 --steps 8000

    # train from the saved best params (no re-search)
    lerobot-python hpo_optuna.py --policy act --train_best --full_steps 100000

    # inspect afterwards
    lerobot-python hpo_optuna.py --policy act --report_only

The full training run is configurable like a normal lerobot_train invocation:
--full_steps / --full_batch_size / --full_num_workers / --save_freq /
--device / --video_backend / --train_output_dir, plus --train_extra to pass
arbitrary extra flags straight through.
"""
import argparse
import collections
import json
import random
import re
import shutil
import subprocess
import sys
from pathlib import Path

import optuna
import torch

# ---------------------------------------------------------------------------
# Search spaces. Each returns the `--policy.<field>=<value>` overrides for a
# trial, plus `batch_size` (a top-level train arg, not a policy field).
# Field names verified against lerobot 0.5.2 configuration_*.py.
# ---------------------------------------------------------------------------


def space_act(trial):
    return {
        "batch_size": trial.suggest_categorical("batch_size", [8, 16, 32]),
        "policy": {
            "optimizer_lr": trial.suggest_float("optimizer_lr", 1e-6, 1e-4, log=True),
            "optimizer_lr_backbone": trial.suggest_float(
                "optimizer_lr_backbone", 1e-6, 1e-4, log=True
            ),
            "optimizer_weight_decay": trial.suggest_float(
                "optimizer_weight_decay", 1e-6, 1e-2, log=True
            ),
            "chunk_size": trial.suggest_categorical("chunk_size", [50, 100]),
            "n_action_steps": trial.suggest_categorical("n_action_steps", [50, 100]),
            "dim_model": trial.suggest_categorical("dim_model", [256, 512]),
            "n_encoder_layers": trial.suggest_int("n_encoder_layers", 3, 6),
            "dropout": trial.suggest_float("dropout", 0.0, 0.3),
            "kl_weight": trial.suggest_float("kl_weight", 1.0, 50.0, log=True),
        },
    }


def space_vqbet(trial):
    return {
        "batch_size": trial.suggest_categorical("batch_size", [8, 16, 32]),
        "policy": {
            "optimizer_lr": trial.suggest_float("optimizer_lr", 1e-5, 1e-3, log=True),
            "n_obs_steps": trial.suggest_int("n_obs_steps", 3, 7),
            "action_chunk_size": trial.suggest_categorical("action_chunk_size", [3, 5, 8]),
            "vqvae_n_embed": trial.suggest_categorical("vqvae_n_embed", [16, 32, 64]),
            "gpt_n_layer": trial.suggest_int("gpt_n_layer", 4, 10),
            "gpt_hidden_dim": trial.suggest_categorical("gpt_hidden_dim", [256, 512]),
            "dropout": trial.suggest_float("dropout", 0.0, 0.3),
            "bet_softmax_temperature": trial.suggest_float(
                "bet_softmax_temperature", 0.01, 1.0, log=True
            ),
            "offset_loss_weight": trial.suggest_float(
                "offset_loss_weight", 1e3, 1e5, log=True
            ),
        },
    }


def space_diffusion(trial):
    return {
        "batch_size": trial.suggest_categorical("batch_size", [8, 16, 32]),
        "policy": {
            "optimizer_lr": trial.suggest_float("optimizer_lr", 1e-5, 1e-3, log=True),
            "n_obs_steps": trial.suggest_int("n_obs_steps", 2, 3),
            "horizon": trial.suggest_categorical("horizon", [32, 64]),
            "n_action_steps": trial.suggest_categorical("n_action_steps", [8, 16, 32]),
            "num_train_timesteps": trial.suggest_categorical(
                "num_train_timesteps", [50, 100]
            ),
            "beta_schedule": trial.suggest_categorical(
                "beta_schedule", ["squaredcos_cap_v2", "linear"]
            ),
            "diffusion_step_embed_dim": trial.suggest_categorical(
                "diffusion_step_embed_dim", [128, 256]
            ),
        },
    }


# ---------------------------------------------------------------------------
# CORE spaces (the default) -- only hyperparameters that val loss can rank FAIRLY.
#
# The objective is validation loss, so a hyperparameter may only be searched if
# it leaves the *loss definition* unchanged. Two kinds break that rule and are
# therefore FIXED here, not searched:
#
#   1. Prediction length: chunk_size / horizon / n_action_steps /
#      action_chunk_size. Predicting 8 steps ahead is inherently easier than 32,
#      so a shorter horizon always scores a lower loss -- Optuna would pick the
#      shortest one every time, even though long action chunks are usually what
#      makes these policies work.
#
#   2. Loss weights: kl_weight (ACT), offset_loss_weight (VQ-BeT, 1e3..1e5),
#      primary/secondary_code_loss_weight. These multiply terms of the loss, so
#      the search would simply drive them to their smallest value to shrink the
#      number, regardless of policy quality.
#
# What is left is fair game: learning rates, weight decay, dropout, batch size,
# and model capacity. Those change *how well* the same objective is optimised.
#
# Fewer dimensions also make the budget go further: 20 trials over 10 dims is
# close to random search; over 4-5 dims TPE can actually converge.
#
# Use --space full to search everything anyway (see the caveat above).
# ---------------------------------------------------------------------------


def core_act(trial):
    return {
        "batch_size": trial.suggest_categorical("batch_size", [8, 16, 32]),
        "policy": {
            "optimizer_lr": trial.suggest_float("optimizer_lr", 1e-6, 1e-4, log=True),
            "optimizer_lr_backbone": trial.suggest_float(
                "optimizer_lr_backbone", 1e-6, 1e-4, log=True
            ),
            "optimizer_weight_decay": trial.suggest_float(
                "optimizer_weight_decay", 1e-6, 1e-2, log=True
            ),
            "dropout": trial.suggest_float("dropout", 0.0, 0.3),
        },
    }


def core_vqbet(trial):
    return {
        "batch_size": trial.suggest_categorical("batch_size", [8, 16, 32]),
        "policy": {
            "optimizer_lr": trial.suggest_float("optimizer_lr", 1e-5, 1e-3, log=True),
            "optimizer_vqvae_lr": trial.suggest_float(
                "optimizer_vqvae_lr", 1e-4, 1e-2, log=True
            ),
            "optimizer_weight_decay": trial.suggest_float(
                "optimizer_weight_decay", 1e-7, 1e-3, log=True
            ),
            "dropout": trial.suggest_float("dropout", 0.0, 0.3),
        },
    }


def core_diffusion(trial):
    return {
        "batch_size": trial.suggest_categorical("batch_size", [8, 16, 32]),
        "policy": {
            "optimizer_lr": trial.suggest_float("optimizer_lr", 1e-5, 1e-3, log=True),
            "optimizer_weight_decay": trial.suggest_float(
                "optimizer_weight_decay", 1e-7, 1e-3, log=True
            ),
            "scheduler_warmup_steps": trial.suggest_categorical(
                "scheduler_warmup_steps", [100, 500, 1000]
            ),
        },
    }


SPACES = {
    "core": {"act": core_act, "vqbet": core_vqbet, "diffusion": core_diffusion},
    "full": {"act": space_act, "vqbet": space_vqbet, "diffusion": space_diffusion},
}
POLICIES = list(SPACES["core"])


class _ProbeTrial:
    """Records which hyperparameters a space function asks for (for logging)."""

    def __init__(self):
        self.params = []

    def suggest_float(self, name, low, high, **kw):
        self.params.append(name)
        return low

    def suggest_int(self, name, low, high, **kw):
        self.params.append(name)
        return low

    def suggest_categorical(self, name, choices):
        self.params.append(name)
        return choices[0]


def _space_params(space_fn):
    probe = _ProbeTrial()
    space_fn(probe)
    return probe.params

# diffusion requires horizon > n_action_steps + n_obs_steps; enforced in objective().


# ---------------------------------------------------------------------------
# Episode split
# ---------------------------------------------------------------------------


def split_episodes(repo_id, root, val_ratio, seed):
    """Deterministic train/val split over episode indices."""
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

    meta = LeRobotDatasetMetadata(repo_id, root=root)
    n = meta.total_episodes
    eps = list(range(n))
    random.Random(seed).shuffle(eps)
    n_val = max(1, int(round(n * val_ratio)))
    val = sorted(eps[:n_val])
    train = sorted(eps[n_val:])
    if not train:
        sys.exit(f"error: no training episodes left (total={n}, val_ratio={val_ratio})")
    return train, val


# ---------------------------------------------------------------------------
# Training (subprocess) with pruning on the streamed training loss
# ---------------------------------------------------------------------------

LOSS_RE = re.compile(r"loss:\s*([0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)")

# NOTE: we deliberately do NOT parse the step out of the log line. lerobot prints
# it through format_big_number(), which collapses 1000/1200/1400 all to "1K" --
# feeding those to trial.report() produces duplicate steps ("value is ignored
# because this step is already reported") and pruning silently stops working.
# lerobot logs exactly when `step % log_freq == 0`, so the Nth loss line is
# always step N * log_freq. We pass --log_freq explicitly and count lines.


# PyAV video decoding in forked DataLoader workers segfaults at random points
# (it is not fork-safe). It kills a trial mid-run after burning GPU time, so we
# detect it and retry the trial once with num_workers=0, which is slower but safe.
def _is_worker_crash(tail):
    text = "\n".join(tail)
    return "DataLoader worker" in text and (
        "Segmentation fault" in text or "exited unexpectedly" in text
    )


def run_training(trial, args, params, train_eps, out_dir):
    last_loss, tail, rc = _run_training_once(trial, args, params, train_eps, out_dir,
                                             num_workers=args.num_workers)
    if rc != 0 and args.num_workers > 0 and _is_worker_crash(tail):
        print(f"\n  DataLoader worker crashed (PyAV is not fork-safe). "
              f"Retrying this trial with --num_workers=0 ...\n", flush=True)
        shutil.rmtree(out_dir, ignore_errors=True)
        last_loss, tail, rc = _run_training_once(trial, args, params, train_eps, out_dir,
                                                 num_workers=0)
    if rc != 0:
        if not args.verbose:
            print(f"\n  --- lerobot_train exited {rc}; last {len(tail)} lines ---", flush=True)
            for ln in tail:
                print("  | " + ln, flush=True)
            print("  --- end of training output ---\n", flush=True)
        raise RuntimeError(f"lerobot_train exited {rc} (see output above)")
    return last_loss


def _run_training_once(trial, args, params, train_eps, out_dir, num_workers):
    """Run one training subprocess. Returns (last_loss, output_tail, returncode)."""
    # Trials train on an episode SUBSET, which trips a lerobot 0.5.2 bug for
    # diffusion (EpisodeAwareSampler keeps global frame offsets -> IndexError).
    # hpo_train_shim.py is lerobot_train with that sampler fixed.
    shim = Path(__file__).with_name("hpo_train_shim.py")
    cmd = [
        sys.executable, str(shim),
        f"--dataset.repo_id={args.repo_id}",
        f"--dataset.episodes={json.dumps(train_eps)}",
        f"--policy.type={args.policy}",
        f"--output_dir={out_dir}",
        f"--policy.device={args.device}",
        "--policy.push_to_hub=false",
        f"--num_workers={num_workers}",
        f"--dataset.video_backend={args.video_backend}",
        f"--steps={args.steps}",
        f"--save_freq={args.steps}",       # only the final checkpoint
        f"--log_freq={args.trial_log_freq}",
        f"--batch_size={params['batch_size']}",
        f"--seed={args.seed}",
    ]
    if args.root:
        cmd.append(f"--dataset.root={args.root}")
    for k, v in params["policy"].items():
        cmd.append(f"--policy.{k}={v}")
    # NOTE: --train_extra deliberately does NOT apply here. It is for the single
    # full run; applying it to every trial would e.g. spawn one wandb run per trial.

    print("  $ " + " ".join(cmd[2:]), flush=True)
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    )
    last_loss = None
    n_logs = 0
    # Keep the tail of the training output so a crash is not silent: without this
    # a failed trial only ever showed "lerobot_train exited 1" and the real cause
    # (OOM, missing dep, ...) was swallowed.
    tail = collections.deque(maxlen=40)
    for line in proc.stdout:
        tail.append(line.rstrip())
        m_loss = LOSS_RE.search(line)
        if m_loss:
            last_loss = float(m_loss.group(1))
            n_logs += 1
            step = n_logs * args.trial_log_freq      # exact: lerobot logs every log_freq
            # Intermediate report -> lets the pruner kill hopeless trials early.
            trial.report(last_loss, step)
            if trial.should_prune():
                proc.kill()
                proc.wait()
                raise optuna.TrialPruned(f"pruned at step {step} (train loss {last_loss})")
        elif args.verbose:
            print("    " + line.rstrip(), flush=True)
    proc.wait()
    # The caller decides what to do with a non-zero exit (it may retry).
    return last_loss, list(tail), proc.returncode


# ---------------------------------------------------------------------------
# Validation loss on held-out episodes (same pipeline lerobot trains with)
# ---------------------------------------------------------------------------


@torch.no_grad()
def val_loss(ckpt_dir, args, val_eps):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors
    from lerobot.utils.collate import lerobot_collate_fn

    pcfg = PreTrainedConfig.from_pretrained(ckpt_dir)
    pcfg.device = args.device
    policy = get_policy_class(pcfg.type).from_pretrained(ckpt_dir)
    policy.to(args.device)

    # policy.forward() is the *training* loss path. ACT only runs its VAE encoder
    # when self.training is True (otherwise mu/log_sigma are None and the KL term
    # blows up), so the module must stay in train mode. Turn off just the
    # stochastic / stat-updating layers so the val loss is deterministic and
    # comparable across trials.
    policy.train()
    for m in policy.modules():
        if isinstance(m, (torch.nn.Dropout, torch.nn.Dropout1d, torch.nn.Dropout2d)):
            m.eval()
        elif isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.eval()

    preprocessor, _ = make_pre_post_processors(policy_cfg=pcfg, pretrained_path=ckpt_dir)

    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

    meta = LeRobotDatasetMetadata(args.repo_id, root=args.root)
    ds = LeRobotDataset(
        args.repo_id, root=args.root, episodes=val_eps,
        delta_timestamps=resolve_delta_timestamps(pcfg, meta),
        video_backend=args.video_backend,
    )
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.val_batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=lerobot_collate_fn,
    )

    # Diffusion/VQ-BeT sample noise inside forward(); fix the seed so every trial
    # is scored on the same draws.
    torch.manual_seed(args.seed)

    total, count = 0.0, 0
    for i, batch in enumerate(loader):
        if args.val_batches and i >= args.val_batches:
            break
        batch = preprocessor(batch)
        loss, _ = policy.forward(batch)
        total += float(loss)
        count += 1
    if count == 0:
        raise RuntimeError("validation produced no batches")
    return total / count


# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description="Optuna HPO for LeRobot policies")
    ap.add_argument("--policy", required=True, choices=POLICIES)
    ap.add_argument("--space", choices=list(SPACES), default="core",
                    help="core (default): only hyperparameters that val loss can rank fairly "
                         "-- learning rates, weight decay, dropout, batch size. "
                         "full: also search chunk/horizon and loss weights, which BIAS the "
                         "objective (shorter horizons and smaller loss weights always score "
                         "lower loss regardless of quality). See the comment in this file.")
    ap.add_argument("--repo_id", default="JSHNSL/ffw_sg2_ltable_merge")
    ap.add_argument("--root", default=None, help="local dataset folder (instead of the Hub)")
    ap.add_argument("--n_trials", type=int, default=20)
    ap.add_argument("--n_jobs", type=int, default=1,
                    help="trials to run CONCURRENTLY. Each one is a full training run on the "
                         "same GPU, so >1 only pays off if a single trial leaves the GPU idle. "
                         "Too high = CUDA OOM. Start at 2 and watch nvidia-smi.")
    ap.add_argument("--steps", type=int, default=8000, help="steps per trial (keep SHORT)")
    ap.add_argument("--trial_log_freq", type=int, default=200,
                    help="how often a trial logs its loss = how often the pruner can act")
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--val_batches", type=int, default=40, help="0 = whole val set")
    ap.add_argument("--val_batch_size", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=8,
                    help="dataloader workers for trials + val (see --full_num_workers)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--video_backend", default="pyav",
                    help="dataset video backend (pyav avoids worker segfaults)")
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--out_root", default="./outputs/hpo")
    ap.add_argument("--study_name", default=None)
    ap.add_argument("--fresh", action="store_true",
                    help="do NOT resume: delete the existing study and start from zero")
    ap.add_argument("--keep_checkpoints", action="store_true",
                    help="keep each trial's checkpoint (default: delete, they add up fast)")
    ap.add_argument("--report_only", action="store_true", help="print results and exit")
    ap.add_argument("--verbose", action="store_true", help="stream full training output")
    # --- run the real training with the tuned hyperparameters ---
    ap.add_argument("--then_train", action="store_true",
                    help="after the search finishes, immediately start the FULL training run "
                         "with the winning hyperparameters (search + train in one command)")
    ap.add_argument("--train_best", action="store_true",
                    help="skip the search: load best_params.json and start the FULL training run")
    ap.add_argument("--full_steps", type=int, default=100_000,
                    help="steps for the full training run (--then_train / --train_best)")
    ap.add_argument("--full_batch_size", type=int, default=None,
                    help="override the TUNED batch_size for the full run (e.g. when it OOMs)")
    ap.add_argument("--full_num_workers", type=int, default=None,
                    help="dataloader workers for the full run (default: --num_workers)")
    ap.add_argument("--save_freq", type=int, default=None,
                    help="checkpoint every N steps in the full run (default: full_steps/5)")
    ap.add_argument("--train_output_dir", default=None,
                    help="output dir for the full run (default ./outputs/train/<policy>_tuned)")
    ap.add_argument("--train_extra", nargs=argparse.REMAINDER, default=[],
                    help="ALL remaining args are passed straight to lerobot_train. "
                         "Must come LAST, e.g.: --train_extra --policy.optimizer_lr=1e-5 --log_freq=100")
    # --- upload the trained policy right after --train_best finishes ---
    ap.add_argument("--push_to_hub", action="store_true",
                    help="after --train_best, upload the final checkpoint to the Hub")
    ap.add_argument("--hub_repo_id", default="JSHNSL/humanoid-imitation-learning",
                    help="model repo for --push_to_hub")
    ap.add_argument("--hub_name", default=None,
                    help="folder name under models/ (default <policy>_tuned)")
    args = ap.parse_args()

    study_name = args.study_name or f"hpo_{args.policy}"
    out_root = Path(args.out_root) / args.policy
    out_root.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{out_root / 'study.db'}"
    best_file = out_root / "best_params.json"

    # Train directly from a previous search's saved hyperparameters.
    if args.train_best:
        train_best(args, best_file)
        return

    # By default a re-run RESUMES the existing study (trials accumulate). --fresh
    # throws the old study away and starts from zero.
    if args.fresh and not args.report_only:
        try:
            optuna.delete_study(study_name=study_name, storage=storage)
            print(f"--fresh: deleted existing study '{study_name}'")
        except KeyError:
            pass                                  # nothing to delete
        best_file.unlink(missing_ok=True)

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="minimize",
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=args.steps // 4),
        sampler=optuna.samplers.TPESampler(seed=args.seed),
    )

    if args.report_only:
        report(study, args, best_file)
        return

    # Be explicit about resuming — otherwise "using an existing study" is a
    # surprise, and stale trials from an older run silently skew the search.
    if study.trials:
        done = sum(1 for t in study.trials if t.value is not None)
        failed = sum(1 for t in study.trials if t.state.name == "FAIL")
        print(f"RESUMING study '{study_name}': {len(study.trials)} existing trials "
              f"({done} completed, {failed} failed). {args.n_trials} more will be added.")
        print(f"  -> use --fresh to start over, or --study_name <name> for a separate study.\n")

    train_eps, val_eps = split_episodes(args.repo_id, args.root, args.val_ratio, args.seed)
    print(f"episodes: {len(train_eps)} train / {len(val_eps)} val  (val={val_eps})")

    # Show what is actually being searched -- with a warning when the space
    # contains hyperparameters that val loss cannot rank fairly.
    searched = _space_params(SPACES[args.space][args.policy])
    print(f"space={args.space}: searching {len(searched)} hyperparameters -> "
          f"{', '.join(searched)}")
    if args.space == "full":
        print("  WARNING: --space full searches chunk/horizon and loss weights. Val loss is\n"
              "  biased toward shorter horizons and smaller loss weights regardless of policy\n"
              "  quality -- the winner may be worse at the actual task. Prefer --space core.")
    print()

    def objective(trial):
        params = SPACES[args.space][args.policy](trial)
        p = params["policy"]

        # These constraints only apply when the shape params are actually searched
        # (--space full). In core they keep the lerobot defaults, which are valid.
        if args.policy == "diffusion" and "horizon" in p:
            if p["horizon"] <= p["n_action_steps"] + p["n_obs_steps"] - 1:
                raise optuna.TrialPruned("invalid horizon/n_action_steps combo")
            p["drop_n_last_frames"] = p["horizon"] - p["n_action_steps"] - p["n_obs_steps"] + 1
        if args.policy == "act" and "chunk_size" in p:
            p["n_action_steps"] = min(p["n_action_steps"], p["chunk_size"])

        out_dir = out_root / f"trial_{trial.number:03d}"
        if out_dir.exists():
            shutil.rmtree(out_dir)

        print(f"\n=== trial {trial.number} ===", flush=True)
        train_loss = run_training(trial, args, params, train_eps, out_dir)

        ckpt = out_dir / "checkpoints" / "last" / "pretrained_model"
        if not ckpt.is_dir():
            raise RuntimeError(f"no checkpoint at {ckpt}")

        vl = val_loss(ckpt, args, val_eps)
        print(f"  train_loss={train_loss:.5f}  VAL_LOSS={vl:.5f}", flush=True)
        trial.set_user_attr("train_loss", train_loss)

        if not args.keep_checkpoints:
            shutil.rmtree(out_dir, ignore_errors=True)
        return vl

    if args.n_jobs > 1:
        print(f"running {args.n_jobs} trials concurrently on one GPU "
              f"-- watch for CUDA OOM (nvidia-smi)\n")
    study.optimize(objective, n_trials=args.n_trials, n_jobs=args.n_jobs,
                   catch=(RuntimeError,))
    report(study, args, best_file)

    # Search -> full training in a single command.
    if args.then_train:
        print("\n" + "=" * 64)
        print("--then_train: starting the full training run with the winning config")
        train_best(args, best_file)


def _full_train_cmd(args, params):
    """Build the full-length training command from a params dict.

    Tuned hyperparameters come from `params`; everything else (steps, workers,
    save_freq, ...) comes from the CLI so the real run can be configured like a
    normal lerobot_train invocation. --full_batch_size overrides the tuned
    batch_size (useful when the tuned one OOMs at full model size).
    """
    out_dir = args.train_output_dir or f"./outputs/train/{args.policy}_tuned"
    workers = args.full_num_workers if args.full_num_workers is not None else args.num_workers
    batch_size = args.full_batch_size or params["batch_size"]
    save_freq = args.save_freq or max(1, args.full_steps // 5)

    cmd = [
        # Via the shim too: it is what makes DataLoader workers not segfault on
        # PyAV video decoding (the sampler fix is a no-op here -- the full run
        # uses every episode, so there is no subset).
        sys.executable, str(Path(__file__).with_name("hpo_train_shim.py")),
        f"--dataset.repo_id={args.repo_id}",
        f"--policy.type={args.policy}",
        f"--output_dir={out_dir}",
        f"--policy.device={args.device}",
        "--policy.push_to_hub=false",
        f"--num_workers={workers}",
        f"--dataset.video_backend={args.video_backend}",
        f"--steps={args.full_steps}",
        f"--save_freq={save_freq}",
        f"--batch_size={batch_size}",
        f"--seed={args.seed}",
    ]
    if args.root:
        cmd.append(f"--dataset.root={args.root}")
    for k, v in params.items():
        if k != "batch_size":
            cmd.append(f"--policy.{k}={v}")
    # Anything else the user wants to pass straight through to lerobot_train.
    cmd += args.train_extra
    return cmd, out_dir


def save_best(study, args, best_file):
    """Persist the winning hyperparameters so training can start from them."""
    best = study.best_trial
    payload = {
        "policy": args.policy,
        "repo_id": args.repo_id,
        "val_loss": best.value,
        "trial_number": best.number,
        "search_steps": args.steps,
        "space": args.space,
        "params": best.params,
    }
    best_file.write_text(json.dumps(payload, indent=2))
    print(f"\nsaved best hyperparameters -> {best_file}")
    return payload


def train_best(args, best_file):
    """Load saved hyperparameters and launch the full training run."""
    if not best_file.exists():
        sys.exit(f"error: {best_file} not found -- run the search first:\n"
                 f"  lerobot-python hpo_optuna.py --policy {args.policy} --n_trials 20")
    payload = json.loads(best_file.read_text())
    if payload.get("policy") != args.policy:
        sys.exit(f"error: {best_file} is for policy '{payload.get('policy')}', not '{args.policy}'")

    params = payload["params"]
    print(f"tuned hyperparameters (val_loss={payload['val_loss']:.5f}, "
          f"trial #{payload['trial_number']}):")
    for k, v in params.items():
        print(f"  {k} = {v}")

    cmd, out_dir = _full_train_cmd(args, params)
    print(f"\nstarting full training ({args.full_steps} steps) -> {out_dir}\n")
    print("  $ " + " ".join(cmd[2:]) + "\n", flush=True)
    rc = subprocess.call(cmd)
    if rc != 0:
        sys.exit(f"training failed (exit {rc})")

    ckpt = Path(out_dir) / "checkpoints" / "last" / "pretrained_model"
    print(f"\ndone -> {ckpt}")

    if args.push_to_hub:
        push_policy(args, ckpt, payload)


def push_policy(args, ckpt, payload):
    """Upload the trained checkpoint to models/<name>/ in the Hub model repo.

    Optional sharing step (training itself runs with --policy.push_to_hub=false so
    lerobot does not create its own repo).
    """
    from huggingface_hub import upload_file, upload_folder

    name = args.hub_name or f"{args.policy}_tuned"
    path_in_repo = f"models/{name}"

    print(f"\nuploading {ckpt} -> {args.hub_repo_id}:{path_in_repo} ...", flush=True)
    upload_folder(
        folder_path=str(ckpt),
        repo_id=args.hub_repo_id,
        repo_type="model",
        path_in_repo=path_in_repo,
    )

    # Ship the hyperparameters next to the weights so the run is reproducible.
    hp = Path(ckpt).parent / "hpo_best_params.json"
    hp.write_text(json.dumps(payload, indent=2))
    upload_file(
        path_or_fileobj=str(hp),
        path_in_repo=f"{path_in_repo}/hpo_best_params.json",
        repo_id=args.hub_repo_id,
        repo_type="model",
    )
    print(f"uploaded -> https://huggingface.co/{args.hub_repo_id}/tree/main/{path_in_repo}")


def report(study, args, best_file):
    done = [t for t in study.trials if t.value is not None]
    print("\n" + "=" * 64)
    print(f"trials: {len(study.trials)} total / {len(done)} completed / "
          f"{len([t for t in study.trials if t.state.name == 'PRUNED'])} pruned")
    if not done:
        print("no completed trials yet.")
        return

    print("\ntop 5 by val loss:")
    for t in sorted(done, key=lambda t: t.value)[:5]:
        print(f"  #{t.number:3d}  val_loss={t.value:.5f}  {t.params}")

    best = study.best_trial
    print(f"\nBEST: trial #{best.number}  val_loss={best.value:.5f}")
    save_best(study, args, best_file)

    if not args.then_train:
        print("\nStart the full training run with these hyperparameters:\n")
        print(f"  lerobot-python hpo_optuna.py --policy {args.policy} --train_best")
        print(f"  lerobot-python hpo_optuna.py --policy {args.policy} --train_best --push_to_hub"
              "   # + upload to models/")
    print("\n(then check REAL task success with eval_policy_sim.py -- val loss only ranks "
          "configs, it does not prove the policy succeeds at the task.)")


if __name__ == "__main__":
    main()
