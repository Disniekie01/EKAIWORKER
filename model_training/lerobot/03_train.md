# A3. Train

Run in the training venv from `/workspace/cyclo_lab`. Entry point:
`lerobot.scripts.lerobot_train`. Arguments use dotted paths (`--dataset.*`, `--policy.*`).

Common form:

```bash
# [container: cyclo_lab] /workspace/cyclo_lab
lerobot-python -m lerobot.scripts.lerobot_train \
  --dataset.repo_id=local/<name> \
  --dataset.root=./datasets/<name> \
  --policy.type=<act|diffusion|vqbet> \        # VLA policies use --policy.path instead
  --output_dir=./outputs/train/<run_name> \
  --batch_size=64 --steps=100000
```

If you unified resolution in 02, `<name>` is `<name>_uni256`. The trained policy lands in
`./outputs/train/<run_name>/checkpoints/last/pretrained_model` — local only, evaluated in
[`04_eval.md`](04_eval.md).

---

## ACT

Transformer encoder-decoder that predicts an action chunk. Robust default, fastest to
train, least data-hungry.

```bash
# [container: cyclo_lab] /workspace/cyclo_lab
lerobot-python -m lerobot.scripts.lerobot_train \
  --dataset.root=./datasets/<name> --dataset.repo_id=local/<name> \
  --policy.type=act \
  --policy.push_to_hub = false \
  --output_dir=./outputs/train/act_<name> \
  --batch_size=64 --steps=100000
```

- Required modules: none beyond the base install (uses `torchvision`, `einops`).
- Multiple cameras are fine as long as their resolutions match (see [`02_dataset.md`](02_dataset.md)).

## Diffusion

Diffusion Policy: denoises an action sequence conditioned on observations. Captures
multimodal behavior; slower inference; more data-hungry; sensitive to camera setup.

```bash
# [container: cyclo_lab] /workspace/cyclo_lab
lerobot-python -m lerobot.scripts.lerobot_train \
  --dataset.root=./datasets/<name> --dataset.repo_id=local/<name> \
  --policy.type=diffusion \
  --policy.push_to_hub = false \
  --output_dir=./outputs/train/diffusion_<name> \
  --batch_size=64 --steps=100000
```

- Required module: `diffusers` — installed in [`01_setup.md`](01_setup.md).
- **Camera shapes must match** — this policy triggers the shape check most often. Unify
  resolution first ([`02_dataset.md`](02_dataset.md)).
- Data-hungry: tens of episodes overfit and diverge in closed loop. Treat small-data runs
  as baselines; add episodes or enable augmentation
  (`--dataset.image_transforms.enable=true`).
- Multiple separate camera encoders raise memory; lower `--batch_size` if OOM.

## VQ-BeT

Vector-quantized behavior transformer: learns a codebook of action primitives, then a
transformer over codes. Handles multimodal behavior; needs enough data to populate the
codebook.

```bash
# [container: cyclo_lab] /workspace/cyclo_lab
lerobot-python -m lerobot.scripts.lerobot_train \
  --dataset.root=./datasets/<name> --dataset.repo_id=local/<name> \
  --policy.type=vqbet \
  --policy.push_to_hub = false \
  --output_dir=./outputs/train/vqbet_<name> \
  --batch_size=64 --steps=100000
```

- Required modules: none beyond the base install (VQ-VAE is vendored in lerobot; uses
  `einops`, `numpy`).
- With very little data the codebook underfits; prefer ACT as a baseline first.

## Pi0

π0: flow-matching VLA on a PaliGemma (3B) backbone. Finetune from the pretrained base
via `--policy.path` (not `--policy.type`):

```bash
# [container: cyclo_lab] /workspace/cyclo_lab
lerobot-python -m lerobot.scripts.lerobot_train \
  --dataset.root=./datasets/<name> --dataset.repo_id=local/<name> \
  --policy.path=lerobot/pi0 \
  --policy.push_to_hub = false \
  --output_dir=./outputs/train/pi0_<name> \
  --batch_size=32 --steps=100000
```

- Deps: `lerobot[pi]` extras — installed in [`01_setup.md`](01_setup.md).
- Pulls the **gated** PaliGemma tokenizer: accept it on HF once + `export HF_TOKEN`.
- Language-conditioned: the dataset's task strings are the instruction.

## Pi0.5

π0.5: π0 successor with better open-world generalization. Same invocation, different
base:

```bash
# [container: cyclo_lab] /workspace/cyclo_lab
lerobot-python -m lerobot.scripts.lerobot_train \
  --dataset.root=./datasets/<name> --dataset.repo_id=local/<name> \
  --policy.path=lerobot/pi05_base \
  --policy.push_to_hub = false \
  --output_dir=./outputs/train/pi05_<name> \
  --batch_size=32 --steps=100000
```

- Deps and gated-tokenizer requirement identical to Pi0.

## SmolVLA

Compact VLA (~450 M, SmolVLM2 backbone) — the lightest language-conditioned option.

```bash
# [container: cyclo_lab] /workspace/cyclo_lab
lerobot-python -m lerobot.scripts.lerobot_train \
  --dataset.root=./datasets/<name> --dataset.repo_id=local/<name> \
  --policy.path=lerobot/smolvla_base \
  --policy.push_to_hub = false \
  --output_dir=./outputs/train/smolvla_<name> \
  --batch_size=64 --steps=100000
```

- Deps: `lerobot[smolvla]` extras — installed in [`01_setup.md`](01_setup.md).
- `lerobot/smolvla_base` is ungated (HF download, no token needed).

---

## Notes on this embodiment

- If the dataset includes base-velocity axes (`linear_x`, `linear_y`, `angular_z`) that
  are constant for a stationary task (std ~0), training still works; the default
  `MIN_MAX` normalization handles constant channels.

Problems: [`TROUBLESHOOTING`](../docs/TROUBLESHOOTING.md#a3-train)
