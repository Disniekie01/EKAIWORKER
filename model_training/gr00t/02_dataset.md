# B2. Dataset

GR00T reads a LeRobot **v2.1** dataset. Convert a v3.0 dataset to v2.1 first.

## Convert v3.0 to v2.1

### 1. Create and Activate Virtual Environment

```bash
# [host] ~/Isaac-GR00T/scripts/lerobot_conversion
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e . --verbose
# if the lerobot install hits git-lfs errors:
GIT_LFS_SKIP_SMUDGE=1 uv pip install "lerobot @ git+https://github.com/huggingface/lerobot.git@c75455a6de5c818fa1bb69fb2d92423e86c70475"
```

### 2. Run Conversion Script

Downloads the v3.0 dataset `<owner>/<dataset>` from the HF Hub and converts it:

```bash
# [host] ~/Isaac-GR00T  (venv active)
python scripts/lerobot_conversion/convert_v3_to_v2.py \
  --repo-id <owner>/<dataset> \
  --root ~/groot_data
```

## Modality config

Two files, two roles:

- **`modality.json` — dataset side.** Slices the flat state/action array into named
  groups (`start`/`end`) and maps video columns to camera keys. Copy into the dataset:

  ```bash
  # [host] ~/AIWORKER/model_training
  cp gr00t/configs/modality.json ~/groot_data/<owner>/<dataset>/meta/modality.json
  ```

- **`ffw_sg2_ltable_config.py` — model side.** Registers which of those keys the model
  consumes (state/video at t, 16-step action horizon) under `NEW_EMBODIMENT`. Passed via
  `--modality-config-path`; its keys must match `modality.json`.

This pair is for the sim dataset (19-dim, 1 cam). Real-robot 22-dim data: use the fork's
built-in pair `examples/CYCLO/ffw_sg2_rev1/` (3 cams, odometry) the same way.

## Task instruction

The loader resolves `task_index` → task string via `meta/tasks.jsonl` (hardcoded name).
The v3→v2 conversion leaves `task` as a raw index, so always rewrite it:

```bash
# [host] anywhere  (single-task dataset)
echo '{"task_index": 0, "task": "Pick up the box and place it on the table."}' \
  > ~/groot_data/<owner>/<dataset>/meta/tasks.jsonl
```

Multi-task dataset: one line per index — **every** `task_index` in the data, all strings.

Guidance:
- Use a short English imperative (verb + object + destination), not an environment ID.
- Single-task: `--tune-llm` learns nothing from a constant string ([`03_train.md`](03_train.md)).
- The deployment instruction must match the training instruction.

## Multi-camera: unify resolution

Mixed camera resolutions fail at train time; single-camera datasets skip this. Run
**last** (after the meta edits above) so the copy carries them:

```bash
# [host] ~/AIWORKER/model_training  (needs ffmpeg)
python3 scripts/unify_resolution.py --src ~/groot_data/<owner>/<dataset> --size 256
# -> <dataset>_uni256  — use this as the dataset path in 03_train
```

## Pre-flight checklist

1. Dataset root has `meta/ data/ videos/`, under `~/groot_data`.
2. `meta/info.json` video `shape` == actual (resized) resolution.
3. `meta/tasks.jsonl` — every `task_index` mapped, all strings (both lines must print
   `none`):

   ```bash
   # [host] ~/Isaac-GR00T
   DS=~/groot_data/<owner>/<dataset> scripts/lerobot_conversion/.venv/bin/python - <<'PY'
   import glob, json, os
   import pandas as pd
   ds = os.environ["DS"]
   tasks = {}
   for line in open(f"{ds}/meta/tasks.jsonl"):
       d = json.loads(line)
       tasks[d["task_index"]] = d["task"]
   missing = set()
   for p in glob.glob(f"{ds}/data/**/*.parquet", recursive=True):
       missing |= set(pd.read_parquet(p, columns=["task_index"])["task_index"].unique()) - tasks.keys()
   print("missing task indices:", sorted(missing) or "none")
   print("non-string tasks    :", sorted(i for i, t in tasks.items() if not isinstance(t, str)) or "none")
   PY
   ```

4. Config `.py` keys == `modality.json` keys.
5. Gated backbone accepted, `HF_TOKEN` exported ([`01_setup.md`](01_setup.md)).

Problems: [`TROUBLESHOOTING`](../docs/TROUBLESHOOTING.md#b2-dataset)
