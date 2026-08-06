# A2. Dataset

This track assumes a prepared LeRobot **v3.0** dataset on disk (a `meta/`, `data/`,
`videos/` tree). Place it under the mounted working dir so it is visible in the container:

```
~/AIWORKER/cyclo_lab/datasets/<name>/     (host)
= /workspace/cyclo_lab/datasets/<name>/   (container)
```

Training points at it with `--dataset.root=./datasets/<name>` (run from
`/workspace/cyclo_lab`). `--dataset.repo_id` is only a label when `root` is set.

Sanity check it loads:

```bash
# [container: cyclo_lab] /workspace/cyclo_lab
lerobot-python - <<'PY'
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds=LeRobotDataset("local/x", root="./datasets/<name>")
print("episodes", ds.num_episodes, "frames", ds.num_frames)
print("images", [k for k in ds.meta.features if k.startswith("observation.images")])
PY
```

## Camera resolution must match (Diffusion / Pi0 / Pi0.5)

Multi-camera policies stack all camera images into one tensor before the vision encoder,
so **every camera feature must have the same `H×W`**. A dataset mixing, e.g., head
cameras at 672×376 with wrist cameras at 240×424 fails with:

```
ValueError: observation.images...cam_left_wrist does not match ...cam_left_head,
but we expect all image shapes to match.
```

`--policy.resize_shape` does not fix this — the shape check runs on the raw dataset shapes
before any resize.

Unify all cameras with `unify_resolution.py` (re-encodes a copy, updates `meta/info.json`):

```bash
# [container: cyclo_lab] /workspace/cyclo_lab  (needs ffmpeg: apt install ffmpeg)
python3 unify_resolution.py --src ./datasets/<name> --size 256
# -> ./datasets/<name>_uni256  — use this as <name> from here on
```

`scale`/`pad` keep the frame count, so v3.0 timestamps stay valid; image statistics in
`meta/stats.json` are per-channel, so resolution changes do not invalidate them.

Problems: [`TROUBLESHOOTING`](../docs/TROUBLESHOOTING.md#a2-dataset)
