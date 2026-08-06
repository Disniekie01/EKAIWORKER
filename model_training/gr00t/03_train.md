# B3. Train

Written for an RTX PRO 6000 Blackwell workstation (96 GB). Launch the container on the
host, then run training inside it.

## 1. Launch the container (host)

```bash
# [host]
export HF_TOKEN=<your token>          # keep exported every run — tokenizer build calls the Hub

docker run -it --rm --gpus all \
  --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -e HF_TOKEN="$HF_TOKEN" \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v ~/groot_data:/workspace/groot_data \
  -v ~/Isaac-GR00T/examples/CYCLO:/workspace/examples/CYCLO \
  gr00t:latest
```

This drops you into `/workspace` inside the container.

## 2. Train (container)

Default: the projector and the **diffusion action head** are trained (defaults on). At
96 GB this fits and is the recommended configuration for deployment — the diffusion head
is what generates robot motion.

```bash
# [container: gr00t] /workspace
python gr00t/experiment/launch_finetune.py \
  --base-model-path nvidia/GR00T-N1.7-3B \
  --dataset-path /workspace/groot_data/<owner>/<dataset> \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path examples/CYCLO/ffw_sg2_ltable/ffw_sg2_ltable_config.py \
  --num-gpus 1 --output-dir /workspace/groot_data/out/<run_name> \
  --max-steps 20000 --save-steps 2000 \
  --global-batch-size 16 --dataloader-num-workers 8
```

Argument names are dashed (`--base-model-path`, not `--base_model_path`).

### Optional: train more of the model

Defaults: `--tune-projector` and `--tune-diffusion-model` on; `--tune-visual` and
`--tune-llm` off. At 96 GB you can also enable the backbone:

`--tune-visual`: when the visual domain differs from pretraining (e.g. sim renders).
`--tune-llm`: only for multi-task datasets with distinct instructions — a constant
instruction can degrade the LLM ([`02_dataset.md`](02_dataset.md)).

```bash
# [container: gr00t] /workspace
  ... \
  --tune-visual \
  --tune-llm
```


Problems: [`TROUBLESHOOTING`](../docs/TROUBLESHOOTING.md#b3-train)
