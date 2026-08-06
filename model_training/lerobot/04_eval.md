# A4. Evaluate (Isaac Sim rollout)

Rollout runs the trained policy in the Isaac Sim L-Table task. It supports only policies
trained on the 19-dim, single-`cam_head` sim dataset — the rollout script hard-codes that
embodiment; multi-camera / 22-dim policies have no sim eval path.

The rollout runs under the **isaac Python (3.11 / lerobot 0.3.3)**, not the training venv.
A 0.5.2-trained policy is converted to 0.3.3 first.

## 1. Isaac Python setup (one-time)

```bash
# [container: cyclo_lab]
$ISAACLAB_PATH/_isaac_sim/python.sh -m pip install --no-deps lerobot==0.3.3
$ISAACLAB_PATH/_isaac_sim/python.sh -m pip install \
    draccus deepdiff jsonlines termcolor packaging imageio datasets \
    einops safetensors huggingface_hub
# only when evaluating a Diffusion policy:
$ISAACLAB_PATH/_isaac_sim/python.sh -m pip install diffusers
# REQUIRED: pin numpy back to 1.x (installing datasets pulls it to 2.x)
$ISAACLAB_PATH/_isaac_sim/python.sh -m pip install "numpy==1.26.0"
```

Verify (a wrong numpy causes a silent segfault, not an error message):

```bash
# [container: cyclo_lab]
$ISAACLAB_PATH/_isaac_sim/python.sh -c "
import numpy, lerobot, torch
print('numpy  :', numpy.__version__)     # 1.26.0
print('lerobot:', lerobot.__version__)   # 0.3.3
print('torch  :', torch.__version__)"    # 2.7.0+cu128
```

## 2. Convert the policy 0.5.2 -> 0.3.3

```bash
# [container: cyclo_lab] /workspace/cyclo_lab  (training venv: lerobot-python)
lerobot-python scripts/sim2real/imitation_learning/inference/convert_policy_v05_to_v03.py \
    --policy ./outputs/train/<run_name>/checkpoints/last/pretrained_model \
    --out ./models/<run_name>_v03
```

Weights are unchanged; the script strips/reshapes config and injects normalization stats
so 0.3.3 can load it.

## 3. Run the rollout (isaac Python)

```bash
# [container: cyclo_lab] /workspace/cyclo_lab  (isaac python: plain `python`)
HF_HUB_OFFLINE=1 python scripts/sim2real/imitation_learning/inference/eval_policy_sim.py \
    --task Cyclo-Real-Pick-Place-LTable-FFW-SG2-v0 \
    --robot_type FFW_SG2 \
    --policy ./models/<run_name>_v03 --policy_type <act|vqbet|diffusion> \
    --num_rollouts 10 --enable_cameras \
    --scripted_l_motion
```

- `--policy_type` must match how the policy was trained.
- `--scripted_l_motion` is required for L-Table: base transport is scripted; the policy
  handles grasp/place.
- `HF_HUB_OFFLINE=1` keeps it fully local.
- Force-kill from another terminal: `pkill -9 -f eval_policy_sim.py`.

Problems: [`TROUBLESHOOTING`](../docs/TROUBLESHOOTING.md#a4-eval-isaac-sim-rollout)
