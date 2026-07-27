#!/usr/bin/env python3
"""Convert a LeRobot 0.5.x (v3.0) policy so it loads in LeRobot 0.3.3.

Why: the rollout runs in Isaac Sim's python (3.11 / lerobot 0.3.3), which cannot
be upgraded to 0.5.x (needs 3.12). A 0.5.x-trained policy fails to load in 0.3.3
because (a) config.json has new fields, and (b) normalization stats moved out of
model.safetensors into separate preprocessor/postprocessor files.

This script:
  1. downloads / copies the policy locally,
  2. strips 0.5.x-only config fields,
  3. injects the normalization buffers (min/max or mean/std) back into
     model.safetensors, matching the mode in config's normalization_mapping.

The model *architecture* weights are unchanged (they already load in 0.3.3);
only the normalization buffers are added.

Run with lerobot-python (0.3.3 venv has safetensors + huggingface_hub):
    lerobot-python scripts/.../convert_policy_v05_to_v03.py \
        --policy JSHNSL/vqbet_ffw_ltable_merge \
        --out ./policies/vqbet_merge_v03
then roll out with  --policy ./policies/vqbet_merge_v03
"""
import argparse
import glob
import json
import os
import shutil

from safetensors.torch import load_file, save_file

# 0.5.x-only config keys that 0.3.3's config dataclasses reject.
# (common) use_peft/pretrained_path/peft ; (diffusion) resize_shape/crop_ratio/
# compile_model/compile_mode. Add more here if a policy reports extra invalid fields.
_STRIP_KEYS = [
    "use_peft", "pretrained_path", "peft",
    "resize_shape", "crop_ratio", "compile_model", "compile_mode",
]
_STATS_FOR_MODE = {"MIN_MAX": ["min", "max"], "MEAN_STD": ["mean", "std"], "IDENTITY": []}


def _feature_type(name: str) -> str:
    if name.startswith("observation.image"):
        return "VISUAL"
    if name == "action":
        return "ACTION"
    return "STATE"


def _buf(name: str) -> str:
    return "buffer_" + name.replace(".", "_")


def convert(src: str, out: str):
    # --- 1. get a local copy ---
    if os.path.isdir(src):
        shutil.copytree(src, out, dirs_exist_ok=True)
        # resolve symlinks (HF cache uses them) into real files
        for f in glob.glob(os.path.join(out, "*")):
            if os.path.islink(f):
                real = os.path.realpath(f)
                os.remove(f)
                shutil.copy(real, f)
    else:
        from huggingface_hub import snapshot_download
        snap = snapshot_download(src)
        os.makedirs(out, exist_ok=True)
        for f in glob.glob(os.path.join(snap, "*")):
            shutil.copy(f, os.path.join(out, os.path.basename(f)))

    # --- 2. strip incompatible config fields ---
    cfg_path = os.path.join(out, "config.json")
    cfg = json.load(open(cfg_path))
    removed = [k for k in _STRIP_KEYS if cfg.pop(k, None) is not None]
    json.dump(cfg, open(cfg_path, "w"), indent=2)
    print(f"[config] removed keys: {removed}")

    norm_map = cfg.get("normalization_mapping", {})  # {"STATE":"MIN_MAX",...}

    # --- 3. inject normalization buffers into model.safetensors ---
    pre_path = glob.glob(os.path.join(out, "*normalizer_processor.safetensors"))
    post_path = glob.glob(os.path.join(out, "*unnormalizer_processor.safetensors"))
    if not pre_path or not post_path:
        print("[norm] no separate normalizer files -> nothing to inject (already 0.3.x-style?)")
        return out

    model = load_file(os.path.join(out, "model.safetensors"))
    pre = load_file(pre_path[0])
    post = load_file(post_path[0])

    add = {}
    # input features present in the preprocessor stats (observation.*)
    input_feats = sorted({k.rsplit(".", 1)[0] for k in pre if k.startswith("observation.")})
    for feat in input_feats:
        mode = norm_map.get(_feature_type(feat), "IDENTITY")
        for stat in _STATS_FOR_MODE.get(mode, []):
            key = f"{feat}.{stat}"
            if key in pre:
                add[f"normalize_inputs.{_buf(feat)}.{stat}"] = pre[key]
    # action (targets + unnormalize outputs)
    amode = norm_map.get("ACTION", "IDENTITY")
    for stat in _STATS_FOR_MODE.get(amode, []):
        if f"action.{stat}" in pre:
            add[f"normalize_targets.buffer_action.{stat}"] = pre[f"action.{stat}"]
        if f"action.{stat}" in post:
            add[f"unnormalize_outputs.buffer_action.{stat}"] = post[f"action.{stat}"]

    model.update(add)
    save_file(model, os.path.join(out, "model.safetensors"))
    print(f"[norm] injected {len(add)} buffers:")
    for k in add:
        print(f"    {k}")
    print(f"\nDone -> {out}")
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Convert a 0.5.x LeRobot policy to load in 0.3.3")
    p.add_argument("--policy", required=True, help="HF repo_id or local dir of the 0.5.x policy")
    p.add_argument("--out", required=True, help="output dir (0.3.3-loadable)")
    args = p.parse_args()
    convert(args.policy, args.out)
