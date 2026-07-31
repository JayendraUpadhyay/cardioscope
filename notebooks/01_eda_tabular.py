# =============================================================================
# CardioScope — Notebook 01: Tabular EDA (cardio_base.csv)
# =============================================================================
# Hack4Health / Byte2Beat Competition — Seed: 42 (reproducibility requirement)
# =============================================================================

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os
import sys
import warnings

warnings.filterwarnings('ignore')
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# --- Reproducibility ---
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# --- Paths ---
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
FIG_DIR  = os.path.join(os.path.dirname(__file__), '..', 'reports', 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

print("=" * 70)
print("CardioScope — 01 Tabular EDA")
print("=" * 70)

# =============================================================================
# 1. LOAD DATA
# =============================================================================
print("\n[1/8] Loading cardio_base.csv...")
df = pd.read_csv(os.path.join(DATA_DIR, 'cardio_base.csv'), sep=';')
print(f"  Shape: {df.shape}")
print(f"  Columns: {df.columns.tolist()}")
print(f"  Dtypes:\n{df.dtypes}")
print(f"\n  Head:\n{df.head(3)}")

# =============================================================================
# 2. BASIC STATISTICS
# =============================================================================
print("\n[2/8] Basic Statistics...")
print(df.describe().to_string())
print(f"\n  Null counts:\n{df.isnull().sum()}")

# =============================================================================
# 3. FEATURE ENGINEERING
# =============================================================================
print("\n[3/8] Feature Engineering...")

# Age in years (age is stored in days)
df['age_years'] = df['age'] / 365.25

# Body Mass Index
df['bmi'] = df['weight'] / (df['height'] / 100) ** 2

# Pulse Pressure (systolic - diastolic)
df['pulse_pressure'] = df['ap_hi'] - df['ap_lo']

print(f"  age_years: min={df.age_years.min():.1f}, max={df.age_years.max():.1f}, "
      f"mean={df.age_years.mean():.1f}")
print(f"  bmi:       min={df.bmi.min():.1f}, max={df.bmi.max():.1f}, "
      f"mean={df.bmi.mean():.1f}")
print(f"  pulse_pressure: min={df.pulse_pressure.min()}, max={df.pulse_pressure.max()}, "
      f"mean={df.pulse_pressure.mean():.1f}")

# =============================================================================
# 4. CLASS BALANCE
# =============================================================================
print("\n[4/8] Class Balance...")
vc = df['cardio'].value_counts()
print(f"  cardio=0 (no CVD): {vc[0]} ({vc[0]/len(df)*100:.1f}%)")
print(f"  cardio=1 (CVD):    {vc[1]} ({vc[1]/len(df)*100:.1f}%)")
print(f"  Imbalance ratio: {vc[0]/vc[1]:.3f}")

# =============================================================================
# 5. PHYSIOLOGICAL OUTLIER DETECTION
# =============================================================================
print("\n[5/8] Physiological Outlier Detection...")

outlier_masks = {
    'ap_hi < 60 (impossible low systolic)':    df['ap_hi'] < 60,
    'ap_hi > 250 (impossible high systolic)':  df['ap_hi'] > 250,
    'ap_lo < 30 (impossible low diastolic)':   df['ap_lo'] < 30,
    'ap_lo > 200 (impossible high diastolic)': df['ap_lo'] > 200,
    'ap_hi <= ap_lo (systolic ≤ diastolic)':   df['ap_hi'] <= df['ap_lo'],
    'height < 100 cm':                         df['height'] < 100,
    'height > 220 cm':                         df['height'] > 220,
    'weight < 20 kg':                          df['weight'] < 20,
    'weight > 200 kg':                         df['weight'] > 200,
    'bmi < 10 (impossible)':                   df['bmi'] < 10,
    'bmi > 60 (extreme)':                      df['bmi'] > 60,
    'age_years < 20':                          df['age_years'] < 20,
    'age_years > 80':                          df['age_years'] > 80,
    'pulse_pressure < 0':                      df['pulse_pressure'] < 0,
}

total_outlier_mask = pd.Series(False, index=df.index)
for desc, mask in outlier_masks.items():
    count = mask.sum()
    total_outlier_mask = total_outlier_mask | mask
    if count > 0:
        print(f"  ⚠  {desc}: {count} rows ({count/len(df)*100:.2f}%)")

total_bad = total_outlier_mask.sum()
print(f"\n  Total rows with at least one physiological anomaly: {total_bad} "
      f"({total_bad/len(df)*100:.2f}%)")

# =============================================================================
# 6. CLEAN DATASET
# =============================================================================
print("\n[6/8] Removing outliers and saving clean dataset...")
df_before = df.copy()
df_clean = df[~total_outlier_mask].reset_index(drop=True)
print(f"  Before: {len(df_before)} rows")
print(f"  After:  {len(df_clean)} rows  (removed {len(df_before)-len(df_clean)} rows)")
print(f"\n  Clean dataset class balance:")
vc_clean = df_clean['cardio'].value_counts()
print(f"  cardio=0: {vc_clean[0]} ({vc_clean[0]/len(df_clean)*100:.1f}%)")
print(f"  cardio=1: {vc_clean[1]} ({vc_clean[1]/len(df_clean)*100:.1f}%)")

df_clean.to_csv(os.path.join(DATA_DIR, 'cardio_clean.csv'), index=False)
print(f"  Saved to data/cardio_clean.csv")

# =============================================================================
# 7. DISTRIBUTION PLOTS
# =============================================================================
print("\n[7/8] Generating distribution plots...")

# --- Age, BMI, BP distributions ---
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
fig.suptitle('CardioScope — Tabular Feature Distributions', fontsize=16, fontweight='bold')

plot_cols = [
    ('age_years',      'Age (years)',              'steelblue'),
    ('bmi',            'BMI',                      'darkorange'),
    ('ap_hi',          'Systolic BP (mmHg)',        'tomato'),
    ('ap_lo',          'Diastolic BP (mmHg)',       'salmon'),
    ('pulse_pressure', 'Pulse Pressure (mmHg)',     'mediumpurple'),
    ('height',         'Height (cm)',               'teal'),
    ('weight',         'Weight (kg)',               'goldenrod'),
    ('cholesterol',    'Cholesterol (ordinal)',     'forestgreen'),
    ('gluc',           'Glucose (ordinal)',         'cornflowerblue'),
]

for ax, (col, label, color) in zip(axes.flat, plot_cols):
    if col in ['cholesterol', 'gluc']:
        df_clean[col].value_counts().sort_index().plot(kind='bar', ax=ax, color=color, alpha=0.8)
        ax.set_xlabel(label)
        ax.set_ylabel('Count')
    else:
        ax.hist(df_clean[col], bins=50, color=color, alpha=0.8, edgecolor='white', linewidth=0.5)
        ax.set_xlabel(label)
        ax.set_ylabel('Count')
    ax.set_title(f'{label} Distribution')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'feature_distributions.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: reports/figures/feature_distributions.png")

# --- Correlation heatmap ---
numeric_cols = ['age_years', 'bmi', 'ap_hi', 'ap_lo', 'pulse_pressure',
                'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'cardio']
corr_matrix = df_clean[numeric_cols].corr()

fig, ax = plt.subplots(figsize=(11, 9))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, vmin=-1, vmax=1, ax=ax, square=True,
            linewidths=0.5, annot_kws={'size': 9})
ax.set_title('Feature Correlation Heatmap (Clean Dataset)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'correlation_heatmap.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: reports/figures/correlation_heatmap.png")

# --- Risk factor comparison by cardio label ---
fig, axes = plt.subplots(1, 4, figsize=(16, 5))
fig.suptitle('Key Risk Factors by CVD Status', fontsize=14, fontweight='bold')

compare_cols = ['age_years', 'bmi', 'ap_hi', 'cholesterol']
labels_str   = ['Age (years)', 'BMI', 'Systolic BP', 'Cholesterol (ordinal)']

for ax, col, lbl in zip(axes, compare_cols, labels_str):
    for label_val, color, name in [(0, 'steelblue', 'No CVD'), (1, 'tomato', 'CVD')]:
        vals = df_clean[df_clean['cardio'] == label_val][col]
        ax.hist(vals, bins=30, alpha=0.6, color=color, label=name,
                density=True, edgecolor='none')
    ax.set_title(lbl)
    ax.set_xlabel(lbl)
    ax.set_ylabel('Density')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'risk_factor_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: reports/figures/risk_factor_comparison.png")

# =============================================================================
# 8. SUMMARY
# =============================================================================
print("\n[8/8] EDA Summary")
print("=" * 70)
print(f"  Original dataset:     {len(df_before):,} rows")
print(f"  Physiological anomalies removed: {len(df_before)-len(df_clean):,} rows")
print(f"  Clean dataset:        {len(df_clean):,} rows")
print(f"  Class balance (clean): No CVD={vc_clean[0]:,}, CVD={vc_clean[1]:,}")
print(f"  Age range:            {df_clean.age_years.min():.0f}–{df_clean.age_years.max():.0f} years")
print(f"  BMI range:            {df_clean.bmi.min():.1f}–{df_clean.bmi.max():.1f}")
print(f"  Systolic BP range:    {df_clean.ap_hi.min()}–{df_clean.ap_hi.max()} mmHg")
print("\n  KEY FINDINGS:")
print("  - Near-perfect class balance (~50/50), no resampling needed")
print("  - Age and BP show strongest separation between CVD/no-CVD groups")
print("  - Cholesterol ordinal feature skewed toward category 1 (normal)")
print("  - BMI distribution right-skewed (expected in population data)")
print("  - Physiological impossibilities removed (negative pulse pressure, etc.)")
print("\n  Next: Notebook 02 — Model Training")
print("=" * 70)
