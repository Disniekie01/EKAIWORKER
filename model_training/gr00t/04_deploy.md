# B4. Deploy (model files)

Deployment itself (engine, UI, robot bring-up) is documented separately by the
deployment owner. This step covers only the model files an inference engine needs.

## Required files

The trained run lands on the host at `~/groot_data/out/<run_name>` (mounted in
[`03_train.md`](03_train.md)). For inference, copy exactly these:

```
config.json
model-00001-of-00003.safetensors
model-00002-of-00003.safetensors
model-00003-of-00003.safetensors
model.safetensors.index.json
experiment_cfg/
processor_config.json        # top-level; required
```

Not needed (training-resume artifacts): `checkpoint-*/`, `optimizer.pt`, `scheduler.pt`,
`rng_state.pth`, `trainer_state.json`, `training_args.bin`, `wandb_config.json`.

**`processor_config.json` (top-level, ~26 KB) is not the same file as
`experiment_cfg/final_processor_config.json` (~2.5 MB).** The loader instantiates the
processor from the top-level file; `experiment_cfg/` must be this model's own (it carries
the normalization statistics).

A successful load logs an action chunk whose `D` equals the modality pair's action dim:
`D=19` (sim / ltable) or `D=22` (real / ffw_sg2_rev1).

Problems: [`TROUBLESHOOTING`](../docs/TROUBLESHOOTING.md#b4-deploy)
