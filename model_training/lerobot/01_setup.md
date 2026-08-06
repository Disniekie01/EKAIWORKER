# A1. Setup

From an empty machine. Prerequisites: NVIDIA driver, Docker + NVIDIA Container Toolkit,
`git`. Verify the GPU is visible to Docker:

```bash
# [host]
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
```

## 1. Clone the AIWORKER stack

`setup.sh` clones the three ROBOTIS repos (`cyclo_lab`, `ai_worker`,
`robotis_applications`) at pinned commits and initializes submodules.

```bash
# [host] ~
git clone https://github.com/Disniekie01/EKAIWORKER.git AIWORKER
cd AIWORKER
./setup.sh ~/AIWORKER
```

These docs and the shared scripts are part of the same clone, under
`~/AIWORKER/model_training`. No second checkout is needed.

## 2. Build and enter the container

```bash
# [host]
cd ~/AIWORKER/cyclo_lab/docker
./container.sh start
./container.sh enter   # drops you into the container
```

The container provides Isaac Sim 5.1.0, Isaac Lab 2.3.0, and the isaac Python 3.11.
The host `~/AIWORKER/cyclo_lab` is mounted at `/workspace/cyclo_lab`.

## 3. Scripts in the mounted tree

`setup.sh` (step 1) rsyncs `overlays/cyclo_lab/` onto `~/AIWORKER/cyclo_lab`, which is
mounted at `/workspace/cyclo_lab`. The HPO and inference scripts ship with the overlay,
so they are already in place — no copy needed:

| Script | Path on host | Used in |
|--------|--------------|---------|
| `hpo_optuna.py`, `hpo_train_shim.py` | `~/AIWORKER/cyclo_lab/` | [`05_hpo.md`](../../overlays/cyclo_lab/05_hpo.md) |
| `scripts/sim2real/.../inference/*.py` | `~/AIWORKER/cyclo_lab/scripts/sim2real/imitation_learning/inference/` | [`04_eval.md`](04_eval.md) |

Only the shared resolution tool lives outside the overlay and has to be copied:

```bash
# [host] ~/AIWORKER/model_training
cp scripts/unify_resolution.py ~/AIWORKER/cyclo_lab/    # used in 02_dataset.md
```

## 4. Training environment (Python 3.12 / lerobot 0.5.2)

`python3` inside the container is the isaac 3.11 alias, so build the venv with the system
3.12 explicitly.

```bash
# [container: cyclo_lab]
/usr/bin/python3.12 -m venv /root/lerobot_env
/root/lerobot_env/bin/pip install --upgrade pip wheel
# lerobot 0.5.2 (pinned commit)
/root/lerobot_env/bin/pip install \
  "lerobot[training,av-dep,pi,smolvla] @ git+https://github.com/ROBOTIS-GIT/lerobot-cyclo.git@2e9cd87bbdb93c23503f7eeca7317bd33027b279" \
  h5py "diffusers==0.38.0" "optuna==4.9.0"
```

Pin torch/torchvision to the tested CUDA 12.8 builds:

```bash
# [container: cyclo_lab]
/root/lerobot_env/bin/pip install --index-url https://download.pytorch.org/whl/cu128 \
  --force-reinstall "torch==2.7.1" "torchvision==0.22.1"
```

Add aliases:

```bash
# [container: cyclo_lab]
cat >> ~/.bashrc <<'EOF'
export LEROBOT_VENV=/root/lerobot_env
alias lerobot-python='${LEROBOT_VENV}/bin/python'
alias lerobot-pip='${LEROBOT_VENV}/bin/pip'
EOF
source ~/.bashrc
lerobot-python -c "import lerobot,torch,torchvision; \
print(lerobot.__version__, torch.__version__, torchvision.__version__)"
# expect: 0.5.2 2.7.1+cu128 0.22.1+cu128
```

The rollout (isaac Python 3.11 / lerobot 0.3.3) is set up later, in
[`04_eval.md`](04_eval.md), only when you evaluate.

Problems: [`TROUBLESHOOTING`](../docs/TROUBLESHOOTING.md#a1-setup)
