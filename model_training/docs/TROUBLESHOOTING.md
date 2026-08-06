# Troubleshooting

Both tracks, one table per step (A = LeRobot, B = GR00T). Run markers as in the step docs.

## A1 Setup

| Symptom | Fix |
|---------|-----|
| `torch` / `torchvision` version error | `lerobot-pip install --index-url https://download.pytorch.org/whl/cu128 --force-reinstall "torch==2.7.1" "torchvision==0.22.1"` |
| `ensurepip is not available` | `sudo apt-get install -y python3.12-venv` |
| `python3` resolves to 3.11 (isaac alias) | use the full path `/usr/bin/python3.12` for the venv |
| `av.option` / PyAV import error | `lerobot-pip install "av>=15.0.0,<16.0.0"` |

## A2 Dataset

| Symptom | Fix |
|---------|-----|
| `ValueError: ... we expect all image shapes to match` | unify resolution ([`lerobot/02_dataset.md`](../lerobot/02_dataset.md)); `--policy.resize_shape` does not fix it |
| `unify_resolution.py` dies with `FileNotFoundError: 'ffmpeg'` | `apt install ffmpeg`, delete the partial `_uni256` dir, rerun |
| `WARNING ... Unknown fields in DatasetInfo: ['annotation_path']` | harmless |
| `datasets/` files root-owned | `sudo chown -R $USER:$USER ~/AIWORKER/cyclo_lab/datasets` |

## A3 Train

| Symptom | Fix |
|---------|-----|
| `unrecognized arguments: --repo_type=model` | remove it |
| segfault during training (video decode) | `--num_workers=0`, or `apt install ffmpeg` |
| CUDA out of memory | lower `--batch_size` (32 or 16) |
| `No module named 'diffusers'` | `lerobot-pip install "diffusers==0.38.0"` |
| `No module named 'optuna'` (HPO) | `lerobot-pip install "optuna==4.9.0"` |
| only one checkpoint saved | only multiples of `save_freq` (default 20000) are saved — lower it |

## A4 Eval (Isaac Sim rollout)

| Symptom | Fix |
|---------|-----|
| segfault, no message | numpy is 2.x — `$ISAACLAB_PATH/_isaac_sim/python.sh -m pip install "numpy==1.26.0"` |
| `packaging` conflict | `$ISAACLAB_PATH/_isaac_sim/python.sh -m pip install "packaging>=24.2,<26"` |
| `No module named 'pip._vendor.packaging._structures'` | `$ISAACLAB_PATH/_isaac_sim/python.sh -m ensurepip --upgrade`; still broken: run `get-pip.py` with the same interpreter |
| missing image key / state-dim mismatch at `select_action` | policy was not trained on the 19-dim single-`cam_head` sim dataset — no sim eval path for it |
| shape mismatch after policy conversion | ACT / VQ-BeT / Diffusion are confirmed convertible |
| a config field errors during conversion | add it to `_STRIP_KEYS` in `convert_policy_v05_to_v03.py` |

## A5 HPO (Optuna)

| Symptom | Fix |
|---------|-----|
| worker crash, study stalls | rerun the same command (no `--fresh` — it discards the study) |
| OOM with `--n_jobs > 1` | reduce `--n_jobs`, `--steps`, or per-trial batch size |

## B1 Setup / Hugging Face

| Symptom | Fix |
|---------|-----|
| `docker: could not select device driver ... --gpus` | install the NVIDIA Container Toolkit |
| `no space left on device` during the build | image is ~41 GB — free/expand disk |
| `401` / `GatedRepoError` on `Cosmos-Reason2-2B` | accept access at <https://huggingface.co/nvidia/Cosmos-Reason2-2B>, `export HF_TOKEN=...` |
| `OfflineModeIsEnabled` at tokenizer build | `unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE`, keep `HF_TOKEN` set — tokenizer build needs the Hub even fully cached |

## B2 Dataset

| Symptom | Fix |
|---------|-----|
| `ImportError: ... 'load_info'` running the converter | wrong Python — use `scripts/lerobot_conversion/.venv/bin/python` ([`gr00t/02_dataset.md`](../gr00t/02_dataset.md)) |
| `KeyError` in `tasks_map` / `AttributeError: 'int' object has no attribute 'lower'` | `tasks.jsonl` incomplete or `task` left as int by the conversion — map every `task_index` to a string, run the pre-flight check ([`gr00t/02_dataset.md`](../gr00t/02_dataset.md)) |
| `RuntimeError: stack expects each tensor to be equal size` | camera resolutions differ — run [`unify_resolution.py`](../scripts/unify_resolution.py) on the converted dataset |
| `Permission denied` on `~/groot_data/...` | `sudo chown -R $USER:$USER ~/groot_data ~/.cache/huggingface` |
| editing `meta/tasks.parquet` has no effect | loader reads `tasks.jsonl` only |
| `SyntaxError: invalid non-printable character U+00A0` | pasted non-breaking space — use a heredoc (`python - <<'PY'`) |

## B3 Train

| Symptom | Fix |
|---------|-----|
| CUDA out of memory | lower `--global-batch-size`; still OOM: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` + `--no-tune-diffusion-model` (projector-only — don't deploy as final) |
| `unrecognized arguments: --gradient-accmulation-steps` | typo — `--gradient-accumulation-steps` |
| only the config saved, no weights | `--save-steps` never reached — lower it (checkpoints only at multiples) |
| disk fills | checkpoints ~20 GB each — raise `--save-steps`, delete old ones |
| `No module named 'gr00t'` in the container | image not built from the fork — rebuild ([`gr00t/01_setup.md`](../gr00t/01_setup.md)), or mount the repo + `-e PYTHONPATH=/workspace/Isaac-GR00T` |

## B4 Deploy

| Symptom | Fix |
|---------|-----|
| `Unrecognized processing class ... Can't instantiate a processor` at load | top-level `processor_config.json` (~26 KB) missing — copy it from the training checkpoint (`experiment_cfg/final_processor_config.json` is a different file) |
| loads, but action `D` ≠ your action dim | wrong modality pair for the dataset — `D=19` sim/ltable, `D=22` real/ffw_sg2_rev1 |
