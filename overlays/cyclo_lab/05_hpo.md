# A5. Hyperparameter search (Optuna) — optional

`scripts/hpo_optuna.py` runs an Optuna study over short trials, then optionally trains the
best config to completion. `scripts/hpo_train_shim.py` runs a single trial. Both use the
training venv (`optuna==4.9.0`); there is no separate service.

## Run

```bash
# [container: cyclo_lab] /workspace/cyclo_lab
lerobot-python hpo_optuna.py \
  --policy act \
  --root ./datasets/<name> \
  --n_trials 20 --steps 8000 \
  --then_train --full_steps 100000
```

Key arguments:

| Argument | Meaning |
|----------|---------|
| `--policy` | `act`, `diffusion`, or `vqbet` |
| `--space` | `core` (default) or `full` search space |
| `--root` | local dataset dir |
| `--n_trials` | number of trials |
| `--steps` | steps per trial — keep short |
| `--n_jobs` | parallel trials (needs the VRAM) |
| `--then_train` | after search, train the best config to `--full_steps` |
| `--fresh` | discard any existing study and restart |
| `--report_only` | print current results and exit |

Output of the final train lands in `./outputs/train/`, ready for
[`04_eval.md`](04_eval.md).

Problems: [`TROUBLESHOOTING`](../docs/TROUBLESHOOTING.md#a5-hpo-optuna)
