#!/usr/bin/env bash
# Opt-in experimental LeRobot >=0.4 venv for dataset format v3.0 conversion.
#
# Does NOT replace the production lerobot==0.3.3 venv (lerobot-python → v2.1).
# After setup, use alias lerobot-python-v3 or export LEROBOT_V3_PYTHON.
#
# Usage (inside cyclo_lab container or any host with python3):
#   bash scripts/sim2real/imitation_learning/data_converter/setup_lerobot_v3_env.sh
set -euo pipefail

HOME_DIR="${HOME:-/root}"
VENV="${LEROBOT_V3_VENV:-${HOME_DIR}/lerobot_env_v3}"
PY="${VENV}/bin/python3"
PIP="${VENV}/bin/pip"

echo "[setup_lerobot_v3] creating venv at ${VENV}"
python3 -m venv "${VENV}"
"${PY}" -m pip install --upgrade pip
# Dataset v3.0 lands in lerobot>=0.4. Pin convert-script deps used by v21→v30.
# av>=16 broke lerobot 0.6.x (missing av.option); keep av in [12,16).
"${PIP}" install "lerobot>=0.4.0" h5py jsonlines pandas pyarrow datasets "av>=12,<16"

# Shell aliases (append once)
BASHRC="${HOME_DIR}/.bashrc"
MARKER="# cyclo_lab lerobot_env_v3"
if [[ -f "${BASHRC}" ]] && ! grep -qF "${MARKER}" "${BASHRC}"; then
  {
    echo ""
    echo "${MARKER}"
    echo "export LEROBOT_V3_VENV=${VENV}"
    echo "export LEROBOT_V3_PYTHON=${PY}"
    echo "alias lerobot-python-v3='${PY}'"
    echo "alias lerobot-activate-v3='source ${VENV}/bin/activate'"
  } >> "${BASHRC}"
  echo "[setup_lerobot_v3] appended aliases to ${BASHRC}"
fi

# Optional wrapper on PATH for non-interactive docker exec
BIN_DIR="${HOME_DIR}/.local/bin"
mkdir -p "${BIN_DIR}"
WRAPPER="${BIN_DIR}/lerobot-python-v3"
cat > "${WRAPPER}" <<EOF
#!/usr/bin/env bash
exec "${PY}" "\$@"
EOF
chmod +x "${WRAPPER}"

echo "[setup_lerobot_v3] verifying convert module…"
"${PY}" -c "import lerobot; import lerobot.scripts.convert_dataset_v21_to_v30 as m; print('lerobot', getattr(lerobot,'__version__','?'), 'ok')"

echo "[setup_lerobot_v3] done."
echo "  LEROBOT_V3_PYTHON=${PY}"
echo "  wrapper: ${WRAPPER}"
echo "  Example:"
echo "    lerobot-python scripts/.../isaaclab2lerobot.py --dataset_format v3 --dataset_file ... --task ..."
