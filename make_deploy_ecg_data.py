"""
Run this ONCE locally (where the full-resolution ECG data exists) to create
a small, deployable version of the waveform data for the ECG Explorer.

This does NOT change the model, the anomaly scores, or the methodology —
it only creates a smaller display-resolution copy of the raw signal so it
can be committed to GitHub and deployed (the full file is 125MB, over
GitHub's 100MB limit; the frontend only ever renders ~1500 points anyway).

Usage: place this in the project root and run:
    python make_deploy_ecg_data.py
"""

import numpy as np
import json
from pathlib import Path

# Adjust this path if your usable-signal file lives somewhere else
SOURCE_PATH = Path("data/ecg_signals_usable.npy")
OUTPUT_DIR = Path("models")
OUTPUT_DIR.mkdir(exist_ok=True)

TARGET_POINTS_PER_RECORDING = 2000  # matches frontend's display decimation

def decimate(signal: np.ndarray, target_len: int) -> np.ndarray:
    """Simple even-stride decimation preserving overall shape."""
    if len(signal) <= target_len:
        return signal
    indices = np.linspace(0, len(signal) - 1, target_len).astype(int)
    return signal[indices]

def main():
    print(f"Loading full-resolution data from {SOURCE_PATH} ...")
    full = np.load(SOURCE_PATH, allow_pickle=True)
    print(f"Loaded shape: {full.shape if hasattr(full, 'shape') else len(full)}")

    decimated_rows = []
    for i, row in enumerate(full):
        row = np.asarray(row, dtype=np.float32)
        decimated_rows.append(decimate(row, TARGET_POINTS_PER_RECORDING))

    decimated_array = np.array(decimated_rows, dtype=np.float32)
    out_path = OUTPUT_DIR / "ecg_signals_display.npy"
    np.save(out_path, decimated_array)

    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"Saved decimated array with shape {decimated_array.shape} to {out_path}")
    print(f"File size: {size_mb:.2f} MB (should be well under 100MB)")

if __name__ == "__main__":
    main()
