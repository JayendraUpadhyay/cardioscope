# =============================================================================
# CardioScope — Notebook 02: Tabular Model Training
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

# --- Reproducibility ---
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

# --- Sklearn ---
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, brier_score_loss, classification_report,
    confusion_matrix, roc_curve, average_precision_score
)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

# Try LightGBM (preferred), fall back to GradientBoosting
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
    print("LightGBM available — using LGBMClassifier as primary gradient boosting model")
except ImportError:
    HAS_LIGHTGBM = False
    print("LightGBM not available — using sklearn GradientBoostingClassifier")

# --- Paths ---
DATA_DIR   = os.path.join(os.path.dirname(__file__), '..', 'data')
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
FIG_DIR    = os.path.join(os.path.dirname(__file__), '..', 'reports', 'figures')
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

print("=" * 70)
print("CardioScope — 02 Tabular Model Training")
print("=" * 70)

# =============================================================================
# 1. LOAD & PREPARE DATA
# =============================================================================
print("\n[1/9] Loading and preparing data...")

clean_path = os.path.join(DATA_DIR, 'cardio_clean.csv')
if not os.path.exists(clean_path):
    # Fall back to raw dataset and apply basic cleaning
    print("  cardio_clean.csv not found, loading cardio_base.csv and cleaning...")
    df = pd.read_csv(os.path.join(DATA_DIR, 'cardio_base.csv'), sep=';')
    df['age_years']      = df['age'] / 365.25
    df['bmi']            = df['weight'] / (df['height'] / 100) ** 2
    df['pulse_pressure'] = df['ap_hi'] - df['ap_lo']
    mask_bad = (
        (df['ap_hi'] < 60)  | (df['ap_hi'] > 250)  |
        (df['ap_lo'] < 30)  | (df['ap_lo'] > 200)  |
        (df['ap_hi'] <= df['ap_lo'])                |
        (df['height'] < 100) | (df['height'] > 220) |
        (df['weight'] < 20)  | (df['weight'] > 200) |
        (df['bmi'] < 10)     | (df['bmi'] > 60)
    )
    df = df[~mask_bad].reset_index(drop=True)
else:
    df = pd.read_csv(clean_path)
    if 'age_years' not in df.columns:
        df['age_years']      = df['age'] / 365.25
    if 'bmi' not in df.columns:
        df['bmi']            = df['weight'] / (df['height'] / 100) ** 2
    if 'pulse_pressure' not in df.columns:
        df['pulse_pressure'] = df['ap_hi'] - df['ap_lo']

print(f"  Dataset shape: {df.shape}")

# --- Feature matrix and target ---
FEATURES = ['age_years', 'gender', 'height', 'weight', 'bmi',
            'ap_hi', 'ap_lo', 'pulse_pressure',
            'cholesterol', 'gluc', 'smoke', 'alco', 'active']

X = df[FEATURES].values
y = df['cardio'].values
print(f"  Feature count: {len(FEATURES)}")
print(f"  Class balance: 0={np.sum(y==0):,}, 1={np.sum(y==1):,}")

# =============================================================================
# 2. TRAIN / VAL / TEST SPLIT
# =============================================================================
print("\n[2/9] Splitting data (70% train / 15% val / 15% test, stratified)...")

# First: 70% train, 30% temp
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=SEED)

# Then: 50% of temp = val (15% total), 50% = test (15% total)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=SEED)

print(f"  Train: {len(X_train):,}  Val: {len(X_val):,}  Test: {len(X_test):,}")

# --- Scale features ---
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_val_sc   = scaler.transform(X_val)
X_test_sc  = scaler.transform(X_test)

# =============================================================================
# 3. DEFINE MODELS
# =============================================================================
print("\n[3/9] Defining models...")

models = {
    'Logistic Regression': LogisticRegression(
        C=0.1, max_iter=1000, random_state=SEED, class_weight='balanced'),
    'Random Forest': RandomForestClassifier(
        n_estimators=200, max_depth=10, random_state=SEED,
        class_weight='balanced', n_jobs=-1),
}

if HAS_LIGHTGBM:
    models['LightGBM'] = lgb.LGBMClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        num_leaves=63, random_state=SEED, class_weight='balanced',
        n_jobs=-1, verbose=-1)
else:
    models['Gradient Boosting'] = GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        random_state=SEED)

# =============================================================================
# 4. TRAIN AND EVALUATE ON VAL SET
# =============================================================================
print("\n[4/9] Training models and evaluating on validation set...")

results = {}
fitted_models = {}

for name, model in models.items():
    print(f"\n  Training {name}...")
    model.fit(X_train_sc, y_train)

    y_val_prob = model.predict_proba(X_val_sc)[:, 1]
    y_val_pred = (y_val_prob >= 0.5).astype(int)

    metrics = {
        'accuracy':  accuracy_score(y_val, y_val_pred),
        'precision': precision_score(y_val, y_val_pred),
        'recall':    recall_score(y_val, y_val_pred),
        'f1':        f1_score(y_val, y_val_pred),
        'roc_auc':   roc_auc_score(y_val, y_val_prob),
        'pr_auc':    average_precision_score(y_val, y_val_prob),
        'brier':     brier_score_loss(y_val, y_val_prob),
    }
    results[name]       = metrics
    fitted_models[name] = model

    print(f"  {'Accuracy':<12}: {metrics['accuracy']:.4f}")
    print(f"  {'Precision':<12}: {metrics['precision']:.4f}")
    print(f"  {'Recall':<12}: {metrics['recall']:.4f}")
    print(f"  {'F1':<12}: {metrics['f1']:.4f}")
    print(f"  {'ROC-AUC':<12}: {metrics['roc_auc']:.4f}")
    print(f"  {'PR-AUC':<12}: {metrics['pr_auc']:.4f}")
    print(f"  {'Brier Score':<12}: {metrics['brier']:.4f}")

# --- Comparison table ---
print("\n" + "=" * 70)
print("MODEL COMPARISON (Validation Set)")
print("=" * 70)
print(f"{'Model':<22} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} "
      f"{'F1':>7} {'ROC-AUC':>9} {'Brier':>7}")
print("-" * 70)
for name, m in results.items():
    print(f"{name:<22} {m['accuracy']:>9.4f} {m['precision']:>10.4f} "
          f"{m['recall']:>8.4f} {m['f1']:>7.4f} {m['roc_auc']:>9.4f} "
          f"{m['brier']:>7.4f}")

# =============================================================================
# 5. SELECT BEST MODEL BY ROC-AUC
# =============================================================================
print("\n[5/9] Selecting best model by ROC-AUC...")
best_name = max(results, key=lambda n: results[n]['roc_auc'])
best_model = fitted_models[best_name]
print(f"  Best model: {best_name}  (ROC-AUC = {results[best_name]['roc_auc']:.4f})")

# =============================================================================
# 6. CALIBRATE BEST MODEL
# =============================================================================
print("\n[6/9] Calibrating best model (Platt sigmoid)...")
# Use an explicit seeded cross-validation calibration step. This replaces the
# removed cv='prefit' API in current scikit-learn and produces a standard
# CalibratedClassifierCV artifact that skl2onnx can export.
calibrated_model = CalibratedClassifierCV(best_model, method='sigmoid', cv=5)
calibrated_model.fit(X_val_sc, y_val)
print("  Calibration complete.")

# =============================================================================
# 7. EVALUATE CALIBRATED MODEL ON TEST SET
# =============================================================================
print("\n[7/9] Evaluating calibrated model on test set...")
y_test_prob = calibrated_model.predict_proba(X_test_sc)[:, 1]
y_test_pred = (y_test_prob >= 0.5).astype(int)

test_metrics = {
    'accuracy':  accuracy_score(y_test, y_test_pred),
    'precision': precision_score(y_test, y_test_pred),
    'recall':    recall_score(y_test, y_test_pred),
    'f1':        f1_score(y_test, y_test_pred),
    'roc_auc':   roc_auc_score(y_test, y_test_prob),
    'pr_auc':    average_precision_score(y_test, y_test_prob),
    'brier':     brier_score_loss(y_test, y_test_prob),
}

print(f"\n  TEST SET RESULTS ({best_name} + Calibration)")
print(f"  {'Accuracy':<12}: {test_metrics['accuracy']:.4f}")
print(f"  {'Precision':<12}: {test_metrics['precision']:.4f}")
print(f"  {'Recall':<12}: {test_metrics['recall']:.4f}")
print(f"  {'F1':<12}: {test_metrics['f1']:.4f}")
print(f"  {'ROC-AUC':<12}: {test_metrics['roc_auc']:.4f}")
print(f"  {'PR-AUC':<12}: {test_metrics['pr_auc']:.4f}")
print(f"  {'Brier Score':<12}: {test_metrics['brier']:.4f}")

print("\n  Classification Report:")
print(classification_report(y_test, y_test_pred,
                             target_names=['No CVD', 'CVD']))
print("  Confusion Matrix:")
cm = confusion_matrix(y_test, y_test_pred)
print(cm)

# =============================================================================
# 8. VISUALIZATIONS
# =============================================================================
print("\n[8/9] Generating visualizations...")

# --- ROC Curves ---
fig, ax = plt.subplots(figsize=(8, 6))
for name, model in fitted_models.items():
    y_prob = model.predict_proba(X_val_sc)[:, 1]
    fpr, tpr, _ = roc_curve(y_val, y_prob)
    auc_val = results[name]['roc_auc']
    ax.plot(fpr, tpr, linewidth=2, label=f'{name} (AUC={auc_val:.3f})')
ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curves — All Models (Validation Set)', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'roc_curves.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: reports/figures/roc_curves.png")

# --- Calibration Curve ---
fig, ax = plt.subplots(figsize=(7, 6))
fraction_of_positives, mean_predicted = calibration_curve(
    y_test, y_test_prob, n_bins=10, strategy='uniform')
ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Perfect calibration')
ax.plot(mean_predicted, fraction_of_positives, 'o-',
        color='steelblue', linewidth=2, markersize=8, label=f'{best_name} (calibrated)')
ax.set_xlabel('Mean Predicted Probability', fontsize=12)
ax.set_ylabel('Fraction of Positives', fontsize=12)
ax.set_title('Calibration Curve (Test Set)', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'calibration_curve.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: reports/figures/calibration_curve.png")

# --- Feature Importance ---
tree_model_name = next((n for n in ['LightGBM', 'Gradient Boosting', 'Random Forest']
                        if n in fitted_models), None)
if tree_model_name:
    tree_model = fitted_models[tree_model_name]
    # Get importances
    if hasattr(tree_model, 'feature_importances_'):
        importances = tree_model.feature_importances_
    else:
        importances = np.zeros(len(FEATURES))

    sorted_idx = np.argsort(importances)[::-1]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh([FEATURES[i] for i in sorted_idx], importances[sorted_idx],
            color='steelblue', alpha=0.85)
    ax.set_xlabel('Feature Importance (MDI)', fontsize=12)
    ax.set_title(f'Feature Importances — {tree_model_name}', fontsize=13, fontweight='bold')
    ax.grid(True, axis='x', alpha=0.3)
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'feature_importance.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: reports/figures/feature_importance.png")

# =============================================================================
# 9. SAVE ARTIFACTS
# =============================================================================
print("\n[9/9] Saving model artifacts...")

joblib.dump(calibrated_model, os.path.join(MODELS_DIR, 'tabular_model.pkl'))
joblib.dump(scaler,           os.path.join(MODELS_DIR, 'tabular_scaler.pkl'))

with open(os.path.join(MODELS_DIR, 'tabular_features.json'), 'w') as f:
    json.dump(FEATURES, f, indent=2)

with open(os.path.join(MODELS_DIR, 'tabular_metrics.json'), 'w') as f:
    json.dump({'best_model': best_name, 'test_metrics': test_metrics}, f, indent=2)

print(f"  Saved: models/tabular_model.pkl")
print(f"  Saved: models/tabular_scaler.pkl")
print(f"  Saved: models/tabular_features.json")
print(f"  Saved: models/tabular_metrics.json")

print("\n" + "=" * 70)
print("TRAINING COMPLETE")
print(f"  Best model:    {best_name}")
print(f"  Test ROC-AUC:  {test_metrics['roc_auc']:.4f}")
print(f"  Test Recall:   {test_metrics['recall']:.4f}  (critical metric for medical screening)")
print(f"  Test Brier:    {test_metrics['brier']:.4f}  (lower = better calibrated)")
print("\n  Next: Notebook 03 — SHAP Explainability")
print("=" * 70)
