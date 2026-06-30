#!/usr/bin/env bash
set -euo pipefail

INSTALL_DIR="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/workspace}"

cd "$INSTALL_DIR/cyclo_lab"
exec python3 sg2_ltable_dashboard.py
