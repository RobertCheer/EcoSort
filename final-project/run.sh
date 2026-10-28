#!/usr/bin/env bash
# Start the EcoSort station.
#
# Creates a virtual environment on first run, installs the package, then opens
# the Streamlit app. Safe to re-run: setup is skipped once the environment exists.
set -euo pipefail

cd "$(dirname "$0")"

# The chosen technique is MobileNetV2, which runs on TensorFlow. TensorFlow
# supports Python 3.10-3.13, so prefer the newest interpreter that can install
# it and fall back to plain python3 only if none of them is present — the SVM
# backend still runs there, the app just says so in the About tab.
pick_python() {
  if [ -n "${PYTHON_BIN:-}" ]; then
    echo "$PYTHON_BIN"
    return
  fi
  for candidate in python3.13 python3.12 python3.11 python3.10; do
    if command -v "$candidate" >/dev/null 2>&1; then
      echo "$candidate"
      return
    fi
  done
  echo python3
}

PYTHON_BIN="$(pick_python)"
VENV_DIR="${VENV_DIR:-.venv}"

if [ ! -d "$VENV_DIR" ]; then
  echo "Creating virtual environment in $VENV_DIR with $PYTHON_BIN ..."
  "$PYTHON_BIN" -m venv "$VENV_DIR"
  "$VENV_DIR/bin/python" -m pip install --quiet --upgrade pip

  echo "Installing EcoSort with the TensorFlow backend ..."
  if ! "$VENV_DIR/bin/python" -m pip install --quiet -e ".[cnn]"; then
    echo >&2
    echo "TensorFlow could not be installed in this environment." >&2
    echo "Installing the base package instead; EcoSort will use the SVM backend." >&2
    echo "To get MobileNetV2, re-create the environment with PYTHON_BIN=python3.13 ./run.sh" >&2
    "$VENV_DIR/bin/python" -m pip install --quiet -e .
  fi
fi

if [ ! -f "models/mobilenetv3_224.keras" ] && [ ! -f "models/mobilenetv2_224.keras" ] \
   && [ ! -f "models/svm_hog_glcm.pkl" ]; then
  cat >&2 <<'MSG'
No trained model found in models/.

Run the training notebook (kaggle/train_ecosort_224.ipynb) on Kaggle, then copy
mobilenetv3_224.keras and svm_hog_glcm.pkl into models/. See USER_GUIDE.md.
MSG
  exit 1
fi

echo "Starting EcoSort. The backend actually in use is shown in the About tab."
exec "$VENV_DIR/bin/python" -m streamlit run app.py "$@"
