"""
Karachi AQI Predictor - SHAP Feature Importance Analysis
==========================================================
Phase 6: Interpret the trained Random Forest model using SHAP values.

Charts produced:
  1. shap_summary.png             - Beeswarm / summary plot
  2. shap_bar.png                 - Bar plot of top 10 features
  3. shap_dependence_pm25.png     - Dependence plot for pm2_5

Usage:
    python notebooks/shap_analysis.py

Author: 10Pearls Internship Project
"""

import os
import sys
import glob
import warnings
import pickle

# ---------------------------------------------------------------------------
# Set working directory to project root
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for chart saving
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import shap

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Style configuration
# ---------------------------------------------------------------------------
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_context("talk")

CHARTS_DIR = os.path.join("notebooks", "charts")
os.makedirs(CHARTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Training features (must match model training)
# ---------------------------------------------------------------------------
FEATURE_COLS = [
    "pm2_5", "pm10", "no2", "so2", "o3", "co",
    "temperature", "humidity", "wind_speed", "wind_deg", "pressure",
    "hour", "day_of_week", "month",
    "is_weekend", "is_rush_hour", "season",
    "pm25_rolling_3h", "pm25_rolling_6h", "pm25_rolling_24h",
    "aqi_rolling_3h",
]
TARGET_COL = "aqi_next_24h"

# ---------------------------------------------------------------------------
# 1. Load the trained model
# ---------------------------------------------------------------------------
print("=" * 60)
print("  SHAP ANALYSIS — Karachi AQI Predictor")
print("=" * 60)

model_files = sorted(glob.glob(os.path.join("models", "aqi_best_model_RandomForest_*.pkl")))
if not model_files:
    print("ERROR: No model .pkl file found in models/ directory.")
    sys.exit(1)

model_path = model_files[-1]  # most recent
print(f"\n📦 Loading model: {model_path}")

with open(model_path, "rb") as f:
    model = pickle.load(f)

print(f"   Model type: {type(model).__name__}")
if hasattr(model, "n_estimators"):
    print(f"   Trees: {model.n_estimators}, max_depth: {model.max_depth}")

# ---------------------------------------------------------------------------
# 2. Load and prepare data
# ---------------------------------------------------------------------------
DATA_PATH = os.path.join("data", "aqi_features_karachi.csv")
print(f"\n📂 Loading data: {DATA_PATH}")
df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])

# Filter rows where target is valid
df = df.dropna(subset=[TARGET_COL])
df = df.dropna(subset=FEATURE_COLS)
print(f"   Valid rows (target + features): {len(df):,}")

# Sort by time for a faithful time-based split
df = df.sort_values("timestamp").reset_index(drop=True)

X = df[FEATURE_COLS]
y = df[TARGET_COL]

# ---------------------------------------------------------------------------
# 3. Time-based 80/20 split (same as training)
# ---------------------------------------------------------------------------
split_idx = int(len(df) * 0.8)
X_train = X.iloc[:split_idx]
X_test = X.iloc[split_idx:]
y_test = y.iloc[split_idx:]

print(f"   Train size: {len(X_train):,}")
print(f"   Test size:  {len(X_test):,}")

# ---------------------------------------------------------------------------
# 4. Compute SHAP values
# ---------------------------------------------------------------------------
print("\n⏳ Computing SHAP values (TreeExplainer) ...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

print(f"   SHAP values shape: {np.array(shap_values).shape}")

# ===================================================================
# Chart 1 – SHAP Summary (Beeswarm) Plot
# ===================================================================
print("\n[1/3] Generating shap_summary.png ...")
fig, ax = plt.subplots(figsize=(12, 8))
shap.summary_plot(shap_values, X_test, show=False, max_display=21)
plt.title("SHAP Feature Importance — Beeswarm Plot", fontsize=14, fontweight="bold", pad=15)
plt.tight_layout()
plt.savefig(os.path.join(CHARTS_DIR, "shap_summary.png"), dpi=150, bbox_inches="tight")
plt.close("all")

# ===================================================================
# Chart 2 – SHAP Bar Plot (Top 10 Features)
# ===================================================================
print("[2/3] Generating shap_bar.png ...")
fig, ax = plt.subplots(figsize=(10, 7))
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False, max_display=10)
plt.title("Top 10 Features by Mean |SHAP Value|", fontsize=14, fontweight="bold", pad=15)
plt.tight_layout()
plt.savefig(os.path.join(CHARTS_DIR, "shap_bar.png"), dpi=150, bbox_inches="tight")
plt.close("all")

# ===================================================================
# Chart 3 – SHAP Dependence Plot for PM2.5
# ===================================================================
print("[3/3] Generating shap_dependence_pm25.png ...")
fig, ax = plt.subplots(figsize=(10, 7))
shap.dependence_plot("pm2_5", shap_values, X_test, show=False, ax=ax)
ax.set_title("SHAP Dependence Plot — PM2.5", fontsize=14, fontweight="bold")
plt.tight_layout()
fig.savefig(os.path.join(CHARTS_DIR, "shap_dependence_pm25.png"), dpi=150, bbox_inches="tight")
plt.close("all")

# ===================================================================
# Top 5 Features by mean |SHAP value|
# ===================================================================
mean_abs_shap = np.abs(shap_values).mean(axis=0)
feature_importance = pd.Series(mean_abs_shap, index=FEATURE_COLS).sort_values(ascending=False)

print("\n" + "=" * 60)
print("  TOP 5 MOST IMPORTANT FEATURES (Mean |SHAP Value|)")
print("=" * 60)
for rank, (feat, val) in enumerate(feature_importance.head(5).items(), 1):
    bar_len = int(val / feature_importance.max() * 30)
    bar = "█" * bar_len
    print(f"  {rank}. {feat:25s}  {val:.4f}  {bar}")

print(f"\n  Total features analysed: {len(FEATURE_COLS)}")
print(f"  Test samples used:      {len(X_test):,}")

# Full ranking
print("\n  Full feature ranking:")
for rank, (feat, val) in enumerate(feature_importance.items(), 1):
    print(f"    {rank:2d}. {feat:25s}  {val:.4f}")

print(f"\n✅ All 3 SHAP charts saved to: {os.path.abspath(CHARTS_DIR)}")
print("=" * 60)
