# =============================================================================
# CardioScope — Notebook 03: SHAP Explainability
# =============================================================================
# Hack4Health / Byte2Beat — Seed: 42 (reproducibility requirement)
# =============================================================================

import random
import numpy as np
import pandas as pd
import joblib
import json
import os
import sys
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("WARNING: shap not installed. Run: pip install shap")

DATA_DIR   = os.path.join(os.path.dirname(__file__), '..', 'data')
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
FIG_DIR    = os.path.join(os.path.dirname(__file__), '..', 'reports', 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

print("=" * 70)
print("CardioScope — 03 SHAP Explainability")
print("=" * 70)

# =============================================================================
# 1. LOAD ARTIFACTS
# =============================================================================
print("\n[1/6] Loading saved model artifacts...")
model  = joblib.load(os.path.join(MODELS_DIR, 'tabular_model.pkl'))
scaler = joblib.load(os.path.join(MODELS_DIR, 'tabular_scaler.pkl'))

with open(os.path.join(MODELS_DIR, 'tabular_features.json')) as f:
    FEATURES = json.load(f)

print(f"  Model loaded: {type(model).__name__}")
print(f"  Features ({len(FEATURES)}): {FEATURES}")

# =============================================================================
# 2. RECONSTRUCT TEST SET
# =============================================================================
print("\n[2/6] Reconstructing test split (same seed=42, 15% test)...")
from sklearn.model_selection import train_test_split

clean_path = os.path.join(DATA_DIR, 'cardio_clean.csv')
if os.path.exists(clean_path):
    df = pd.read_csv(clean_path)
else:
    df = pd.read_csv(os.path.join(DATA_DIR, 'cardio_base.csv'), sep=';')
    mask_bad = (
        (df['ap_hi'] < 60) | (df['ap_hi'] > 250) |
        (df['ap_lo'] < 30) | (df['ap_lo'] > 200) |
        (df['ap_hi'] <= df['ap_lo']) |
        (df['height'] < 100) | (df['height'] > 220) |
        (df['weight'] < 20)  | (df['weight'] > 200)
    )
    df = df[~mask_bad].reset_index(drop=True)

if 'age_years' not in df.columns:
    df['age_years']      = df['age'] / 365.25
if 'bmi' not in df.columns:
    df['bmi']            = df['weight'] / (df['height'] / 100) ** 2
if 'pulse_pressure' not in df.columns:
    df['pulse_pressure'] = df['ap_hi'] - df['ap_lo']

X = df[FEATURES].values
y = df['cardio'].values

# Same splits as notebook 02
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=SEED)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=SEED)

X_train_sc = scaler.transform(X_train)
X_test_sc = scaler.transform(X_test)
print(f"  Test set size: {len(X_test):,}")

# =============================================================================
# 3. COMPUTE SHAP VALUES
# =============================================================================
if not HAS_SHAP:
    print("SHAP not available — skipping explainability computation")
else:
    print("\n[3/6] Computing SHAP values...")

    # Use a sample for speed (500 test samples)
    np.random.seed(SEED)
    sample_idx = np.random.choice(len(X_test_sc), size=min(500, len(X_test_sc)), replace=False)
    X_sample   = X_test_sc[sample_idx]

    # Build SHAP explainer on the underlying estimator
    # CalibratedClassifierCV wraps the base model; access inner estimator
    base_estimator = model.calibrated_classifiers_[0].estimator if hasattr(model, 'calibrated_classifiers_') else model

    # The fitted estimator consumes standardized features, so the SHAP
    # background must be in the same feature space as the explained samples.
    background = X_train_sc[:500]
    explainer = shap.Explainer(base_estimator, background)
    shap_values_all = explainer(X_sample)
    # For binary classification, SHAP returns explanation for class 1
    if hasattr(shap_values_all, 'values'):
        sv = shap_values_all.values
        if sv.ndim == 3:
            sv = sv[:, :, 1]  # class 1
    else:
        sv = shap_values_all

    print(f"  SHAP values shape: {sv.shape}")

    # --- Mean |SHAP| per feature ---
    mean_abs_shap = np.abs(sv).mean(axis=0)
    feature_importance_shap = dict(zip(FEATURES, mean_abs_shap.tolist()))
    sorted_features = sorted(feature_importance_shap.items(),
                              key=lambda x: x[1], reverse=True)

    print("\n  TOP 5 RISK FACTORS (by mean |SHAP|):")
    for feat, imp in sorted_features[:5]:
        print(f"    {feat:<20}: mean |SHAP| = {imp:.4f}")

    top_feat = sorted_features[0]
    print(f"\n  KEY FINDING: '{top_feat[0]}' is the strongest predictor "
          f"with mean |SHAP| = {top_feat[1]:.4f}")

    # ==========================================================================
    # 4. GLOBAL SUMMARY PLOT
    # ==========================================================================
    print("\n[4/6] Generating SHAP plots...")

    # Beeswarm
    shap_exp = shap.Explanation(
        values=sv,
        base_values=explainer.expected_value if hasattr(explainer, 'expected_value') else 0,
        data=X_sample,
        feature_names=FEATURES
    )

    plt.figure(figsize=(10, 7))
    shap.plots.beeswarm(shap_exp, show=False, max_display=13)
    plt.title('SHAP Global Feature Importance (Beeswarm)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'shap_global_summary.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: reports/figures/shap_global_summary.png")

    # Bar chart
    plt.figure(figsize=(9, 6))
    feat_names_sorted = [f for f, _ in sorted_features]
    imp_sorted        = [v for _, v in sorted_features]
    colors = ['tomato' if i < 5 else 'steelblue' for i in range(len(feat_names_sorted))]
    plt.barh(feat_names_sorted[::-1], imp_sorted[::-1], color=colors[::-1], alpha=0.85)
    plt.xlabel('Mean |SHAP Value|', fontsize=12)
    plt.title('SHAP Feature Importance (Mean |SHAP|)', fontsize=13, fontweight='bold')
    plt.grid(True, axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'shap_feature_importance.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: reports/figures/shap_feature_importance.png")

    # Waterfall plots for 3 individual predictions
    for i in range(3):
        plt.figure(figsize=(9, 5))
        shap.plots.waterfall(shap_exp[i], show=False)
        plt.title(f'SHAP Waterfall — Sample {i+1}', fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, f'shap_waterfall_sample_{i+1}.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()
    print("  Saved: reports/figures/shap_waterfall_sample_1/2/3.png")

    # ==========================================================================
    # 5. SAVE SHAP ARTIFACTS
    # ==========================================================================
    print("\n[5/6] Saving SHAP artifacts...")
    np.save(os.path.join(MODELS_DIR, 'shap_values.npy'), sv)
    np.save(os.path.join(MODELS_DIR, 'shap_background.npy'), background)

    top_features_dict = {
        'top_features': [
            {'feature': f, 'mean_abs_shap': round(v, 6)}
            for f, v in sorted_features
        ]
    }
    with open(os.path.join(MODELS_DIR, 'shap_top_features.json'), 'w') as f:
        json.dump(top_features_dict, f, indent=2)

    print("  Saved: models/shap_values.npy")
    print("  Saved: models/shap_background.npy")
    print("  Saved: models/shap_top_features.json")

    # ==========================================================================
    # 6. PER-PREDICTION SHAP FUNCTION (used by backend)
    # ==========================================================================
    print("\n[6/6] Defining per-prediction SHAP function...")

    def get_shap_explanation(input_dict, model=model, scaler=scaler,
                              features=FEATURES, explainer=explainer):
        """
        Compute SHAP explanation for a single new patient.
        
        Args:
            input_dict: dict with keys matching FEATURES list
            
        Returns:
            dict with keys: risk_probability, shap_contributions
        """
        # Assemble feature vector
        x_raw = np.array([[input_dict.get(f, 0.0) for f in features]])
        x_sc  = scaler.transform(x_raw)
        
        # Prediction
        risk_prob = model.predict_proba(x_sc)[0, 1]
        
        # SHAP for this single sample
        shap_exp_single = explainer(x_sc)
        if hasattr(shap_exp_single, 'values'):
            sv_single = shap_exp_single.values[0]
            if sv_single.ndim == 2:
                sv_single = sv_single[:, 1]
        else:
            sv_single = np.zeros(len(features))
        
        contributions = [
            {'feature': f, 'shap_value': float(sv_single[i]),
             'direction': 'increases_risk' if sv_single[i] > 0 else 'decreases_risk'}
            for i, f in enumerate(features)
        ]
        contributions.sort(key=lambda x: abs(x['shap_value']), reverse=True)
        
        return {'risk_probability': float(risk_prob), 'shap_contributions': contributions}

    # Test it
    sample_patient = {
        'age_years': 55, 'gender': 2, 'height': 175, 'weight': 85,
        'bmi': 27.8, 'ap_hi': 140, 'ap_lo': 90, 'pulse_pressure': 50,
        'cholesterol': 2, 'gluc': 1, 'smoke': 0, 'alco': 0, 'active': 1
    }
    result = get_shap_explanation(sample_patient)
    print(f"\n  Test explanation for sample patient:")
    print(f"  Risk probability: {result['risk_probability']:.3f}")
    print(f"  Top 3 contributing factors:")
    for c in result['shap_contributions'][:3]:
        print(f"    {c['feature']:<20}: SHAP={c['shap_value']:+.4f} ({c['direction']})")

print("\n" + "=" * 70)
print("EXPLAINABILITY COMPLETE")
print("\n  Next: Notebook 04 — ECG EDA")
print("=" * 70)
