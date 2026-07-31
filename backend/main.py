# =============================================================================
# CardioScope Backend — FastAPI Application
# =============================================================================
# Serves two prediction endpoints:
#   POST /predict/tabular  — Clinical risk prediction with SHAP explainability
#   POST /predict/ecg      — ECG anomaly score (unsupervised reconstruction error)
# =============================================================================

import os
import json
import logging
import numpy as np
from contextlib import asynccontextmanager
from typing import Optional, List

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
try:  # Supports both `uvicorn main:app` from backend/ and `uvicorn backend.main:app` from root.
    from .schemas import ECGInput, ECGPrediction, ShapFactor, TabularInput, TabularPrediction
except ImportError:
    from schemas import ECGInput, ECGPrediction, ShapFactor, TabularInput, TabularPrediction

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BACKEND_DIR, '.env'))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cardioscope")

# =============================================================================
# CONFIGURATION
# =============================================================================
_models_dir = os.getenv("MODELS_DIR", "../models")
MODELS_DIR = _models_dir if os.path.isabs(_models_dir) else os.path.abspath(os.path.join(BACKEND_DIR, _models_dir))
HOST       = os.getenv("HOST", "0.0.0.0")
PORT       = int(os.getenv("PORT", "8000"))

# =============================================================================
# GLOBAL STATE — models loaded at startup
# =============================================================================
state = {
    "tabular_model":     None,
    "tabular_scaler":    None,
    "tabular_features":  None,
    "tabular_ort_sess":  None,
    "ecg_ort_sess":      None,
    "ecg_anomaly_scores": None,
    "shap_top_features": None,
    "ecg_signals": None,
}

# =============================================================================
# LIFESPAN — load models on startup
# =============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading models...")
    _load_models()
    yield
    logger.info("Shutdown complete.")


def _load_models():
    """Load all ONNX models and supporting artifacts into global state."""
    import joblib

    # Tabular sklearn model (for SHAP computation on-the-fly)
    tabular_pkl = os.path.join(MODELS_DIR, 'tabular_model.pkl')
    if os.path.exists(tabular_pkl):
        try:
            state["tabular_model"]  = joblib.load(tabular_pkl)
            state["tabular_scaler"] = joblib.load(os.path.join(MODELS_DIR, 'tabular_scaler.pkl'))
            with open(os.path.join(MODELS_DIR, 'tabular_features.json')) as f:
                state["tabular_features"] = json.load(f)
            logger.info("Tabular sklearn model loaded.")
        except Exception as e:
            logger.warning(f"Could not load tabular sklearn model: {e}")

    # Tabular ONNX model
    tabular_onnx = os.path.join(MODELS_DIR, 'tabular_model.onnx')
    if os.path.exists(tabular_onnx):
        try:
            import onnxruntime as ort
            state["tabular_ort_sess"] = ort.InferenceSession(
                tabular_onnx, providers=['CPUExecutionProvider'])
            logger.info("Tabular ONNX model loaded.")
        except Exception as e:
            logger.warning(f"Could not load tabular ONNX model: {e}")

    # ECG ONNX model
    ecg_onnx = os.path.join(MODELS_DIR, 'ecg_autoencoder.onnx')
    if os.path.exists(ecg_onnx):
        try:
            import onnxruntime as ort
            state["ecg_ort_sess"] = ort.InferenceSession(
                ecg_onnx, providers=['CPUExecutionProvider'])
            logger.info("ECG ONNX model loaded.")
        except Exception as e:
            logger.warning(f"Could not load ECG ONNX model: {e}")

    # ECG anomaly scores
    ecg_scores = os.path.join(MODELS_DIR, 'ecg_anomaly_scores.npy')
    if os.path.exists(ecg_scores):
        state["ecg_anomaly_scores"] = np.load(ecg_scores)
        logger.info(f"ECG anomaly scores loaded: {state['ecg_anomaly_scores'].shape}")

    signals_path = os.path.join(BACKEND_DIR, '..', 'data', 'ecg_signals_usable.npy')
    if os.path.exists(signals_path):
        state["ecg_signals"] = np.load(signals_path, mmap_mode='r')
        logger.info("ECG display signals loaded.")

    # SHAP top features
    shap_json = os.path.join(MODELS_DIR, 'shap_top_features.json')
    if os.path.exists(shap_json):
        with open(shap_json) as f:
            state["shap_top_features"] = json.load(f)
        logger.info("SHAP top features loaded.")


# =============================================================================
# APP FACTORY
# =============================================================================
app = FastAPI(
    title="CardioScope API",
    description=(
        "Multi-modal cardiovascular risk assessment API. "
        "Provides tabular clinical risk prediction with SHAP explainability, "
        "and ECG anomaly scoring via unsupervised 1D-CNN autoencoder."
    ),
    version="1.0.0",
    docs_url="/docs",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://localhost:4173"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# HELPERS
# =============================================================================

def _classify_risk(prob: float) -> str:
    if prob < 0.35:   return "Low"
    elif prob < 0.65: return "Moderate"
    return "High"


def _ecg_risk_level(score: float, all_scores: np.ndarray) -> tuple[str, float]:
    """Classify ECG anomaly score relative to the distribution."""
    pct = float(np.mean(all_scores <= score) * 100)
    if pct < 60:   level = "Normal"
    elif pct < 85: level = "Elevated"
    else:          level = "High"
    return level, pct


def _ecg_display_data(recording_index: int) -> tuple[list[float], list[int], list[float]]:
    """Return display-only decimation and real per-window reconstruction errors."""
    if state["ecg_signals"] is None or state["ecg_ort_sess"] is None:
        raise HTTPException(status_code=503, detail="ECG signal/model data not available.")
    signal = np.asarray(state["ecg_signals"][recording_index], dtype=np.float32)
    lo, hi = signal.min(), signal.max()
    signal = (signal - lo) / (hi - lo) if hi > lo else signal
    window_len, stride = 2000, 1000
    starts = np.arange(0, len(signal) - window_len + 1, stride)
    windows = np.stack([signal[s:s + window_len] for s in starts])[:, None, :]
    input_name = state["ecg_ort_sess"].get_inputs()[0].name
    reconstruction = state["ecg_ort_sess"].run(None, {input_name: windows})[0]
    segment_scores = ((reconstruction - windows) ** 2).mean(axis=(1, 2))
    # Display-only averaging: preserving the min/max pair from every large bucket
    # creates an artificial solid block when this high-frequency signal is rendered
    # in a browser. The model still runs on every original 2,000-sample window.
    buckets = min(900, len(signal))
    edges = np.linspace(0, len(signal), buckets + 1, dtype=int)
    samples, indices = [], []
    for start, end in zip(edges[:-1], edges[1:]):
        chunk = signal[start:end]
        if len(chunk):
            samples.append(float(chunk.mean()))
            indices.append(int((start + end - 1) // 2))
    return samples, indices, segment_scores.astype(float).tolist()


def _predict_tabular(inp: TabularInput):
    """Run tabular risk prediction."""
    # Derive engineered features
    bmi            = inp.weight / (inp.height / 100) ** 2
    pulse_pressure = inp.ap_hi - inp.ap_lo

    features = state["tabular_features"]  # ordered list
    feature_map = {
        'age_years':      inp.age,
        'gender':         inp.gender,
        'height':         inp.height,
        'weight':         inp.weight,
        'bmi':            bmi,
        'ap_hi':          inp.ap_hi,
        'ap_lo':          inp.ap_lo,
        'pulse_pressure': pulse_pressure,
        'cholesterol':    inp.cholesterol,
        'gluc':           inp.gluc,
        'smoke':          inp.smoke,
        'alco':           inp.alco,
        'active':         inp.active,
    }
    x_raw = np.array([[feature_map[f] for f in features]], dtype=np.float32)

    # Try ONNX first
    prob = None
    if state["tabular_ort_sess"] is not None:
        try:
            inp_name = state["tabular_ort_sess"].get_inputs()[0].name
            # Scale if needed (ONNX model may expect raw or scaled)
            if state["tabular_scaler"] is not None:
                x_sc = state["tabular_scaler"].transform(x_raw).astype(np.float32)
            else:
                x_sc = x_raw
            out = state["tabular_ort_sess"].run(None, {inp_name: x_sc})
            # Output may be probabilities array (shape [1,2]) or single value
            out_arr = np.array(out[1]) if len(out) > 1 else np.array(out[0])
            if out_arr.ndim >= 2 and out_arr.shape[-1] == 2:
                prob = float(out_arr[0, 1])
            else:
                prob = float(out_arr.flatten()[0])
        except Exception as e:
            logger.warning(f"ONNX tabular inference failed: {e} — falling back to sklearn")

    # Fallback to sklearn
    if prob is None and state["tabular_model"] is not None:
        x_sc = state["tabular_scaler"].transform(x_raw)
        prob = float(state["tabular_model"].predict_proba(x_sc)[0, 1])

    if prob is None:
        raise HTTPException(status_code=503, detail="Tabular model not available")

    # Build SHAP-style top_factors from precomputed global rankings
    top_factors = []
    if state["shap_top_features"]:
        raw_vals = {f: feature_map.get(f, 0.0) for f in features}
        for item in state["shap_top_features"]["top_features"][:6]:
            feat = item["feature"]
            shap_val = item["mean_abs_shap"]
            # Direction heuristic: positive SHAP = increases risk
            # For display use the precomputed magnitude (not per-sample)
            direction = "increases_risk"  # conservative for static rendering
            top_factors.append(ShapFactor(
                feature=feat,
                raw_value=round(float(raw_vals.get(feat, 0)), 2),
                direction=direction,
                magnitude=round(shap_val, 4),
            ))

    return prob, bmi, pulse_pressure, top_factors


# =============================================================================
# ENDPOINTS
# =============================================================================

@app.get("/health", tags=["System"])
async def health():
    """Health check — returns model loading status."""
    return {
        "status": "ok",
        "tabular_model_loaded":  state["tabular_ort_sess"] is not None
                                 or state["tabular_model"] is not None,
        "ecg_model_loaded":      state["ecg_ort_sess"] is not None,
        "ecg_scores_loaded":     state["ecg_anomaly_scores"] is not None,
        "shap_features_loaded":  state["shap_top_features"] is not None,
    }


@app.post("/predict/tabular", response_model=TabularPrediction, tags=["Prediction"])
async def predict_tabular(inp: TabularInput):
    """
    Predict cardiovascular disease risk from clinical features.
    Returns risk probability, label, and top contributing factors (SHAP-based).
    """
    if state["tabular_model"] is None and state["tabular_ort_sess"] is None:
        raise HTTPException(status_code=503,
                            detail="Tabular model not loaded. Run notebook 02 first.")

    if state["tabular_features"] is None:
        raise HTTPException(status_code=503,
                            detail="tabular_features.json not found in models directory.")

    prob, bmi, pulse_pressure, top_factors = _predict_tabular(inp)

    return TabularPrediction(
        risk_probability=round(prob, 4),
        risk_label=_classify_risk(prob),
        risk_percent=f"{prob*100:.1f}%",
        top_factors=top_factors,
        bmi=round(bmi, 1),
        pulse_pressure=pulse_pressure,
        disclaimer=(
            "This prediction is generated by a machine learning model trained on "
            "population-level data and is NOT a clinical diagnosis. Consult a "
            "qualified healthcare professional for medical advice."
        )
    )


@app.post("/predict/ecg", response_model=ECGPrediction, tags=["Prediction"])
async def predict_ecg(inp: ECGInput):
    """
    Return anomaly score for a specific ECG recording (index 0–186).
    Anomaly score = mean reconstruction error of the 1D-CNN Autoencoder.
    IMPORTANT: No ground-truth labels exist — see limitation_note in response.
    """
    if state["ecg_anomaly_scores"] is None:
        raise HTTPException(status_code=503,
                            detail="ECG anomaly scores not loaded. Run notebook 05 first.")

    idx = inp.recording_index
    if idx >= len(state["ecg_anomaly_scores"]):
        raise HTTPException(status_code=422,
                            detail=f"Recording index {idx} out of range (0–{len(state['ecg_anomaly_scores'])-1})")

    score = float(state["ecg_anomaly_scores"][idx])
    if np.isnan(score):
        raise HTTPException(status_code=422,
                            detail=f"No valid anomaly score for recording {idx}")

    all_scores = state["ecg_anomaly_scores"][~np.isnan(state["ecg_anomaly_scores"])]
    level, pct = _ecg_risk_level(score, all_scores)
    waveform, waveform_sample_indices, segment_scores = _ecg_display_data(idx)

    interpretation_map = {
        "Normal":   "Reconstruction error is within the typical range. No unusual rhythm pattern detected.",
        "Elevated": "Reconstruction error is moderately elevated. The signal contains patterns less typical than most recordings.",
        "High":     "Reconstruction error is in the top 15% of all recordings. The signal has markedly unusual patterns.",
    }

    return ECGPrediction(
        recording_index=idx,
        anomaly_score=round(score, 6),
        risk_level=level,
        interpretation=interpretation_map[level],
        percentile=round(pct, 1),
        limitation_note=(
            "⚠ LIMITATION: This anomaly score is derived from unsupervised reconstruction "
            "error of a 1D-CNN Autoencoder. No ground-truth labels exist for this dataset. "
            "Only 187 of the 528 total recordings contained usable continuous ECG signals. "
            "This output is a RESEARCH PROXY and should NOT be interpreted as a clinical diagnosis."
        )
        , waveform=waveform, waveform_sample_indices=waveform_sample_indices,
        segment_scores=segment_scores
    )


@app.get("/ecg/samples", tags=["ECG"])
async def ecg_samples():
    """Return available ECG recording indices with anomaly scores."""
    if state["ecg_anomaly_scores"] is None:
        raise HTTPException(status_code=503,
                            detail="ECG anomaly scores not loaded.")

    scores = state["ecg_anomaly_scores"]
    valid  = scores[~np.isnan(scores)]
    pct_85 = np.percentile(valid, 85)
    pct_60 = np.percentile(valid, 60)

    samples = []
    for i, s in enumerate(scores):
        if np.isnan(s):
            continue
        level = "High" if s >= pct_85 else ("Elevated" if s >= pct_60 else "Normal")
        samples.append({
            "index": i,
            "anomaly_score": round(float(s), 6),
            "risk_level": level,
        })

    return {
        "total_recordings": len(samples),
        "score_stats": {
            "mean":   round(float(valid.mean()),   6),
            "median": round(float(np.median(valid)), 6),
            "p85":    round(float(pct_85),          6),
        },
        "recordings": samples,
    }


# =============================================================================
# MAIN (for local development)
# =============================================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=HOST, port=PORT, reload=True)
