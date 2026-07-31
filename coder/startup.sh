#!/bin/bash
set -e

echo ">>> CardioScope Coder workspace starting up..."

# --- Clone the project repo (replace with your actual GitHub URL after pushing) ---
if [ ! -d "$HOME/cardioscope" ]; then
  git clone https://github.com/JayendraUpadhyay/cardioscope.git "$HOME/cardioscope"
fi
cd "$HOME/cardioscope"

# --- Preload dataset subset based on the selected parameter ---
if [ "${dataset}" = "sample" ]; then
  echo ">>> Using bundled sample dataset (fast startup)."
  # Sample CSVs are expected to live in data/samples/ in the repo, already
  # small enough to commit directly (unlike the full 626MB ECG file).
else
  echo ">>> Full dataset selected — expecting data/ to be populated separately (not auto-downloaded due to size)."
fi

# --- Python environment (backend + notebooks) ---
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r backend/requirements.txt
pip install jupyterlab

# --- Node environment (frontend) ---
if [ "${component}" = "full-stack" ] || [ "${component}" = "frontend-only" ]; then
  cd frontend
  npm install
  cd ..
fi

# --- Launch the selected component(s) ---
case "${component}" in
  full-stack)
    echo ">>> Launching backend + frontend..."
    (source .venv/bin/activate && uvicorn backend.main:app --host 0.0.0.0 --port 8000 &)
    (cd frontend && npm run dev -- --host 0.0.0.0 --port 5173 &)
    ;;
  backend-only)
    echo ">>> Launching backend only..."
    source .venv/bin/activate && uvicorn backend.main:app --host 0.0.0.0 --port 8000 &
    ;;
  frontend-only)
    echo ">>> Launching frontend only..."
    (cd frontend && npm run dev -- --host 0.0.0.0 --port 5173 &)
    ;;
  notebooks-only)
    echo ">>> Launching JupyterLab..."
    source .venv/bin/activate && jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root &
    ;;
esac

echo ">>> CardioScope workspace ready."
