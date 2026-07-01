#!/usr/bin/env bash
# Push EYKAIWORKER overlay files onto a live cyclo_lab checkout.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_TARGET="$(cd "$ROOT_DIR/.." && pwd)/cyclo_lab"
TARGET="${1:-$DEFAULT_TARGET}"

if [[ ! -d "$TARGET" ]]; then
  echo "cyclo_lab target not found: $TARGET" >&2
  echo "Usage: $0 [path/to/cyclo_lab]" >&2
  exit 1
fi

echo "Syncing overlay -> $TARGET"
rsync -a "$ROOT_DIR/overlays/cyclo_lab/" "$TARGET/"
echo "Done. Restart Isaac / Kill + Launch Record if the recorder is running."
