#!/usr/bin/env bash
# Pull cyclo_lab changes back into the EYKAIWORKER overlay (reverse of sync_overlay.sh).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_SOURCE="$(cd "$ROOT_DIR/.." && pwd)/cyclo_lab"
SOURCE="${1:-$DEFAULT_SOURCE}"

if [[ ! -d "$SOURCE" ]]; then
  echo "cyclo_lab source not found: $SOURCE" >&2
  echo "Usage: $0 [path/to/cyclo_lab]" >&2
  exit 1
fi

echo "Pulling overlay <- $SOURCE"
rsync -a \
  --exclude '.git' \
  --exclude 'datasets' \
  --exclude 'docker/.cyclo-lab-docker-history' \
  --exclude '__pycache__' \
  "$SOURCE/" "$ROOT_DIR/overlays/cyclo_lab/"
echo "Done. Commit EYKAIWORKER overlay changes when ready."
