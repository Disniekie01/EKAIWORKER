#!/usr/bin/env python3
"""Unify every camera of a LeRobot dataset to one square resolution.

Re-encodes a copy (aspect-preserving pad, frame count kept), updates meta/info.json.
Works on v3.0 and v2.1 datasets.

Usage: python3 unify_resolution.py --src ./datasets/<name> --size 256
       -> ./datasets/<name>_uni256
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="dataset root (meta/ data/ videos/)")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--dst", default=None, help="default: <src>_uni<size>")
    a = ap.parse_args()

    src = Path(a.src)
    dst = Path(a.dst) if a.dst else src.with_name(f"{src.name}_uni{a.size}")
    if not (src / "meta" / "info.json").exists():
        sys.exit(f"not a dataset root: {src}")
    if dst.exists():
        sys.exit(f"already exists: {dst}")
    shutil.copytree(src, dst)

    s = a.size
    vf = (f"scale={s}:{s}:force_original_aspect_ratio=decrease,"
          f"pad={s}:{s}:(ow-iw)/2:(oh-ih)/2,setsar=1")
    mp4s = sorted(dst.glob("videos/**/*.mp4"))
    for i, f in enumerate(mp4s, 1):
        tmp = f.with_suffix(".tmp.mp4")
        subprocess.run(
            ["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-i", str(f),
             "-vf", vf, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-an", str(tmp)],
            check=True)
        tmp.replace(f)
        print(f"[{i}/{len(mp4s)}] {f.relative_to(dst)}")

    p = dst / "meta" / "info.json"
    info = json.loads(p.read_text())
    for k, ft in info["features"].items():
        if k.startswith("observation.images"):
            shape = ft.get("shape", [])
            ft["shape"] = [d if d == 3 else s for d in shape] if 3 in shape else [3, s, s]
            ft.get("info", {}).update({"video.height": s, "video.width": s})
    p.write_text(json.dumps(info, indent=4))
    print(f"done: {dst}")


if __name__ == "__main__":
    main()