# B1. Setup

All of setup runs on the host. Prerequisites: NVIDIA driver, Docker + NVIDIA Container
Toolkit, `git` + `git-lfs`. Verify the GPU is visible to Docker:

```bash
# [host]
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
```

## 1. Clone the Isaac-GR00T Repository

Use the ROBOTIS fork at the tested commit. Do not use the upstream NVIDIA repo.

```bash
# [host] ~
git clone https://github.com/ROBOTIS-GIT/Isaac-GR00T-n1.7.git Isaac-GR00T
cd Isaac-GR00T
git checkout e81d02b
```

The configs and shared scripts used below are in the AIWORKER clone, under
`~/AIWORKER/model_training`. If you do not have it yet:

```bash
# [host] ~
git clone https://github.com/Disniekie01/EKAIWORKER.git AIWORKER
```

## 2. Install uv

```bash
# [host]
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## 3. Build the Docker image

```bash
# [host] ~/Isaac-GR00T
bash docker/build.sh          # x86 / dGPU -> image "gr00t" (~41 GB, tens of minutes)
docker images | grep gr00t    # expect gr00t:latest
```

For Jetson, pass `--profile=thor|orin|spark`.

## 4. Hugging Face: base model + gated backbone

GR00T fine-tunes from HF-hosted weights, so this one-time step is required:

1. Create an HF account and a token (read scope is enough for download).
2. Accept the gated backbone once: <https://huggingface.co/nvidia/Cosmos-Reason2-2B> →
   "Agree and access". Loading `nvidia/GR00T-N1.7-3B` pulls this backbone, so without
   acceptance training fails with a 401.
3. Export the token in the shell used to launch training:

   ```bash
   # [host]
   export HF_TOKEN=<your token>
   ```

Weights download to `~/.cache/huggingface` on first run and are reused after (the cache
is mounted into the container). Always keep `HF_TOKEN` exported — the tokenizer build
calls the Hub, so `HF_HUB_OFFLINE=1` fails even with a full cache.

## 5. Place the modality config

```bash
# [host] ~/AIWORKER/model_training
mkdir -p ~/Isaac-GR00T/examples/CYCLO/ffw_sg2_ltable
cp gr00t/configs/ffw_sg2_ltable_config.py ~/Isaac-GR00T/examples/CYCLO/ffw_sg2_ltable/
cp gr00t/configs/modality.json            ~/Isaac-GR00T/examples/CYCLO/ffw_sg2_ltable/
```

`~/Isaac-GR00T/examples/CYCLO` is mounted into the container in [`03_train.md`](03_train.md),
so edits on the host take effect inside.

File placement (`~/AIWORKER/model_training/` → destination):

| File | Destination | Used in |
|------|-------------|---------|
| `gr00t/configs/ffw_sg2_ltable_config.py` | `~/Isaac-GR00T/examples/CYCLO/ffw_sg2_ltable/` | `--modality-config-path` ([`03_train.md`](03_train.md)) |
| `gr00t/configs/modality.json` | same dir, **and** `<dataset>/meta/modality.json` | [`02_dataset.md`](02_dataset.md) |
| `scripts/unify_resolution.py` | no copy — run from `~/AIWORKER/model_training` | [`02_dataset.md`](02_dataset.md) |

Problems: [`TROUBLESHOOTING`](../docs/TROUBLESHOOTING.md#b1-setup--hugging-face)
