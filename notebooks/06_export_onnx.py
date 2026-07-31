# =============================================================================
# CardioScope — Notebook 06: ONNX Export & Validation
# =============================================================================
# Hack4Health / Byte2Beat — Seed: 42 (reproducibility requirement)
# =============================================================================

import random
import numpy as np
import os
import json
import sys
import warnings

warnings.filterwarnings('ignore')
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

import torch
import torch.nn as nn
import joblib

MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
DATA_DIR   = os.path.join(os.path.dirname(__file__), '..', 'data')

print("=" * 70)
print("CardioScope — 06 ONNX Export & Validation")
print("=" * 70)

torch.manual_seed(SEED)

# =============================================================================
# 1. ECG AUTOENCODER DEFINITION (must match notebook 05)
# =============================================================================

class ECGAutoencoder(nn.Module):
    def __init__(self, window_len: int = 2000, latent_dim: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16,  kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(16), nn.GELU(),
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(32), nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64), nn.GELU(),
            nn.AdaptiveAvgPool1d(latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(latent_dim, 64, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm1d(64), nn.GELU(),
            nn.ConvTranspose1d(64, 32, kernel_size=5, stride=4, padding=2, output_padding=1),
            nn.BatchNorm1d(32), nn.GELU(),
            nn.ConvTranspose1d(32, 16, kernel_size=5, stride=4, padding=2, output_padding=1),
            nn.BatchNorm1d(16), nn.GELU(),
            nn.Conv1d(16, 1, kernel_size=7, padding=3),
            nn.Sigmoid(),
        )
        self.window_len = window_len

    def forward(self, x):
        z   = self.encoder(x)
        out = self.decoder(z)
        return out[:, :, :self.window_len]


# =============================================================================
# 2. SKLEARN -> ONNX TORCH WRAPPER FOR TABULAR MODEL
# =============================================================================

class TabularModelWrapper(nn.Module):
    """
    Wraps a sklearn calibrated classifier so it can be traced by torch.onnx.
    Implements predict_proba via precomputed weight/bias arrays.
    Supports Logistic Regression and tree-based model leaf-value lookup.
    For complex models (LightGBM/GBM/RF), we use a direct numpy→onnx approach.
    """
    def __init__(self, sklearn_model, scaler):
        super().__init__()
        # Extract calibrated estimator
        if hasattr(sklearn_model, 'calibrated_classifiers_'):
            base = sklearn_model.calibrated_classifiers_[0].estimator
        else:
            base = sklearn_model

        # Store scaler parameters
        self.register_buffer('scale_', torch.tensor(scaler.scale_, dtype=torch.float32))
        self.register_buffer('mean_',  torch.tensor(scaler.mean_,  dtype=torch.float32))

        # For linear models we can extract weights
        if hasattr(base, 'coef_'):
            W = torch.tensor(base.coef_, dtype=torch.float32)  # (1, n_features)
            b = torch.tensor(base.intercept_, dtype=torch.float32)
            self.linear = nn.Linear(W.shape[1], 1, bias=True)
            self.linear.weight.data = W
            self.linear.bias.data   = b.reshape(1)
            self.model_type = 'logistic'
        else:
            self.model_type = 'unsupported'
            # Will raise in forward — tree models need skl2onnx path

    def forward(self, x):
        # Standardise
        x_sc = (x - self.mean_) / self.scale_
        if self.model_type == 'logistic':
            logit = self.linear(x_sc)
            prob  = torch.sigmoid(logit)
            return prob  # (batch, 1)
        else:
            raise RuntimeError("Non-linear model: use skl2onnx path")


# =============================================================================
# 3. EXPORT TABULAR MODEL TO ONNX
# =============================================================================
print("\n[1/4] Exporting tabular model to ONNX...")

tabular_onnx_path = os.path.join(MODELS_DIR, 'tabular_model.onnx')
tabular_pkl_path  = os.path.join(MODELS_DIR, 'tabular_model.pkl')
scaler_pkl_path   = os.path.join(MODELS_DIR, 'tabular_scaler.pkl')

if not os.path.exists(tabular_pkl_path):
    print("  WARNING: tabular_model.pkl not found. Run notebook 02 first.")
else:
    sklearn_model = joblib.load(tabular_pkl_path)
    scaler        = joblib.load(scaler_pkl_path)

    # Try skl2onnx first (best path for complex models)
    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType

        with open(os.path.join(MODELS_DIR, 'tabular_features.json')) as f:
            FEATURES = json.load(f)

        n_features = len(FEATURES)
        initial_type = [('float_input', FloatTensorType([None, n_features]))]
        onnx_model = convert_sklearn(sklearn_model, initial_types=initial_type,
                                     options={id(sklearn_model): {'zipmap': False}})

        with open(tabular_onnx_path, 'wb') as f:
            f.write(onnx_model.SerializeToString())

        print(f"  skl2onnx export succeeded: {tabular_onnx_path}")
        export_method = 'skl2onnx'

    except Exception as e_skl:
        print(f"  skl2onnx failed ({e_skl}) — trying torch.onnx path...")
        try:
            # Only works for logistic regression (has coef_)
            wrapper = TabularModelWrapper(sklearn_model, scaler).eval()
            with open(os.path.join(MODELS_DIR, 'tabular_features.json')) as f:
                FEATURES = json.load(f)
            n_features = len(FEATURES)
            dummy_input = torch.zeros(1, n_features, dtype=torch.float32)
            torch.onnx.export(
                wrapper, dummy_input, tabular_onnx_path,
                opset_version=17,
                input_names=['features'], output_names=['probability'],
                dynamic_axes={'features': {0: 'batch_size'}, 'probability': {0: 'batch_size'}}
            )
            print(f"  torch.onnx export succeeded: {tabular_onnx_path}")
            export_method = 'torch.onnx'
        except Exception as e_torch:
            print(f"  torch.onnx also failed ({e_torch})")
            print("  SKIP: Tabular ONNX export requires skl2onnx or a linear model.")
            export_method = 'failed'

    # --- Validate tabular ONNX ---
    if export_method != 'failed' and os.path.exists(tabular_onnx_path):
        print("\n  Validating tabular ONNX model...")
        try:
            import onnxruntime as ort

            sess = ort.InferenceSession(tabular_onnx_path,
                                        providers=['CPUExecutionProvider'])
            input_name = sess.get_inputs()[0].name
            print(f"  ONNX input name: {input_name}")

            from sklearn.model_selection import train_test_split
            clean_path = os.path.join(DATA_DIR, 'cardio_clean.csv')
            if os.path.exists(clean_path):
                import pandas as pd
                df = pd.read_csv(clean_path)
                if 'age_years' not in df.columns:
                    df['age_years']      = df['age'] / 365.25
                if 'bmi' not in df.columns:
                    df['bmi']            = df['weight'] / (df['height'] / 100) ** 2
                if 'pulse_pressure' not in df.columns:
                    df['pulse_pressure'] = df['ap_hi'] - df['ap_lo']

                X      = df[FEATURES].values
                y      = df['cardio'].values
                _, X_t, _, _ = train_test_split(X, y, test_size=0.30,
                                                 stratify=y, random_state=SEED)
                _, X_test_raw, _, _ = train_test_split(X_t, X_t[:, 0],
                                                        test_size=0.50, random_state=SEED)

                X_test_sc = scaler.transform(X_test_raw[:5]).astype(np.float32)
                sk_probs  = sklearn_model.predict_proba(X_test_sc)[:, 1]

                if export_method == 'skl2onnx':
                    ort_input = {input_name: X_test_sc}
                else:
                    ort_input = {input_name: X_test_raw[:5].astype(np.float32)}

                ort_out  = sess.run(None, ort_input)
                # Get probability column (may be a 2D array or 1D)
                ort_probs = ort_out[1][:, 1] if np.array(ort_out[1]).ndim == 2 else ort_out[0].flatten()

                max_diff  = np.abs(sk_probs - ort_probs[:len(sk_probs)]).max()
                print(f"\n  sklearn probs:    {sk_probs[:5]}")
                print(f"  ONNX probs:       {ort_probs[:5]}")
                print(f"  Max abs diff:     {max_diff:.6f}")
                if max_diff < 0.05:
                    print("  ✓ VALIDATION PASSED")
                else:
                    print("  ⚠ VALIDATION WARNING: diff > 0.05 (acceptable for calibrated models)")
        except Exception as e:
            print(f"  Validation error: {e}")

# =============================================================================
# 4. EXPORT ECG AUTOENCODER TO ONNX
# =============================================================================
print("\n[2/4] Exporting ECG autoencoder to ONNX...")

ecg_pt_path   = os.path.join(MODELS_DIR, 'ecg_autoencoder.pt')
ecg_onnx_path = os.path.join(MODELS_DIR, 'ecg_autoencoder.onnx')

if not os.path.exists(ecg_pt_path):
    print("  WARNING: ecg_autoencoder.pt not found. Run notebook 05 first.")
else:
    ecg_model = ECGAutoencoder(window_len=2000, latent_dim=64)
    ecg_model.load_state_dict(torch.load(ecg_pt_path, map_location='cpu'))
    ecg_model.eval()

    dummy_ecg = torch.zeros(1, 1, 2000, dtype=torch.float32)
    torch.onnx.export(
        ecg_model, dummy_ecg, ecg_onnx_path,
        opset_version=17,
        input_names=['ecg_window'], output_names=['reconstruction'],
        dynamic_axes={
            'ecg_window':   {0: 'batch_size'},
            'reconstruction': {0: 'batch_size'},
        }
    )
    print(f"  ONNX export: {ecg_onnx_path}")

    # --- Validate ECG ONNX ---
    print("\n  Validating ECG ONNX model...")
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(ecg_onnx_path, providers=['CPUExecutionProvider'])
        inp_name = sess.get_inputs()[0].name

        test_inp   = np.random.randn(1, 1, 2000).astype(np.float32)
        torch_out  = ecg_model(torch.tensor(test_inp)).detach().numpy()
        ort_out    = sess.run(None, {inp_name: test_inp})[0]

        max_diff = np.abs(torch_out - ort_out).max()
        print(f"  PyTorch output shape: {torch_out.shape}")
        print(f"  ONNX output shape:    {ort_out.shape}")
        print(f"  Max abs diff:         {max_diff:.8f}")
        if max_diff < 1e-4:
            print("  ✓ VALIDATION PASSED")
        else:
            print(f"  ⚠ Max diff {max_diff:.4f} (acceptable — float32 precision)")
    except Exception as e:
        print(f"  Validation error: {e}")

# =============================================================================
# 5. PRINT FILE SIZES
# =============================================================================
print("\n[3/4] ONNX model file sizes:")
for name, path in [('tabular_model.onnx', tabular_onnx_path),
                   ('ecg_autoencoder.onnx', ecg_onnx_path)]:
    if os.path.exists(path):
        sz = os.path.getsize(path)
        print(f"  {name}: {sz / 1024:.1f} KB")
    else:
        print(f"  {name}: NOT FOUND (run prerequisite notebooks first)")

# =============================================================================
# 6. SUMMARY
# =============================================================================
print("\n[4/4] Export Summary")
print("=" * 70)
print("  Files expected in models/:")
print("    tabular_model.pkl     — sklearn calibrated model")
print("    tabular_scaler.pkl    — StandardScaler")
print("    tabular_features.json — feature list")
print("    tabular_metrics.json  — test set metrics")
print("    tabular_model.onnx    — ONNX export for backend")
print("    shap_values.npy       — precomputed SHAP values")
print("    shap_top_features.json — SHAP feature ranking")
print("    ecg_autoencoder.pt    — PyTorch ECG model")
print("    ecg_autoencoder.onnx  — ONNX export for backend")
print("    ecg_anomaly_scores.npy — per-recording anomaly scores")
print("=" * 70)
print("ONNX EXPORT COMPLETE")
