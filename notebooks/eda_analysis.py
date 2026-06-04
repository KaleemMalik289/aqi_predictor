"""
Karachi AQI Predictor - Exploratory Data Analysis (EDA)
========================================================
Phase 6: Generate 7 publication-quality charts analyzing Karachi air quality data.

Charts produced:
  1. aqi_distribution.png      - Histogram of AQI categories (1-5)
  2. pm25_timeseries.png       - 90-day PM2.5 time series
  3. hourly_aqi_pattern.png    - Box plot of AQI by hour (0-23)
  4. daily_aqi_pattern.png     - Mean AQI by day of week
  5. monthly_aqi_pattern.png   - Mean AQI by month
  6. correlation_heatmap.png   - Numeric feature correlation heatmap
  7. pm25_vs_aqi_scatter.png   - PM2.5 vs AQI colored by season

Usage:
    python notebooks/eda_analysis.py

Author: 10Pearls Internship Project
"""

import os
import sys
import warnings

# ---------------------------------------------------------------------------
# Set working directory to the project root (parent of notebooks/)
# This ensures all relative paths (data/, notebooks/charts/) resolve correctly
# regardless of where the script is invoked from.
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Style configuration
# ---------------------------------------------------------------------------
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_context("talk")
sns.set_palette("viridis")

CHARTS_DIR = os.path.join("notebooks", "charts")
os.makedirs(CHARTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Colour palettes
# ---------------------------------------------------------------------------
AQI_COLORS = {1: "#2ecc71", 2: "#f1c40f", 3: "#e67e22", 4: "#e74c3c", 5: "#8e44ad"}
AQI_LABELS = {1: "Good", 2: "Fair", 3: "Moderate", 4: "Poor", 5: "Very Poor"}
SEASON_COLORS = {0: "#3498db", 1: "#e74c3c", 2: "#f39c12", 3: "#27ae60"}
SEASON_LABELS = {0: "Winter", 1: "Spring", 2: "Summer", 3: "Fall"}
DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
DATA_PATH = os.path.join("data", "aqi_features_karachi.csv")
print(f"Loading data from {DATA_PATH} ...")
df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
print(f"  Loaded {len(df):,} rows  x  {len(df.columns)} columns")
print(f"  Date range: {df['timestamp'].min()} → {df['timestamp'].max()}")
print()

# ===================================================================
# Chart 1 – AQI Distribution (Histogram with colour-coded bars)
# ===================================================================
print("[1/7] Generating aqi_distribution.png ...")
fig, ax = plt.subplots(figsize=(10, 6))

aqi_counts = df["aqi_openweather"].value_counts().sort_index()
bars = ax.bar(
    aqi_counts.index.astype(str),
    aqi_counts.values,
    color=[AQI_COLORS.get(int(v), "#95a5a6") for v in aqi_counts.index],
    edgecolor="white",
    linewidth=1.5,
    width=0.65,
)

# Add count + percentage labels on bars
total = aqi_counts.sum()
for bar, count in zip(bars, aqi_counts.values):
    pct = count / total * 100
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + total * 0.008,
        f"{count:,}\n({pct:.1f}%)",
        ha="center", va="bottom", fontsize=11, fontweight="bold",
    )

# Legend patches
from matplotlib.patches import Patch
legend_handles = [
    Patch(facecolor=AQI_COLORS[k], edgecolor="white", label=f"{k} – {AQI_LABELS[k]}")
    for k in sorted(AQI_COLORS.keys()) if k in aqi_counts.index
]
ax.legend(handles=legend_handles, title="AQI Category", loc="upper right", fontsize=10)

ax.set_xlabel("AQI Category (OpenWeatherMap)", fontsize=13)
ax.set_ylabel("Number of Records", fontsize=13)
ax.set_title("Distribution of Air Quality Index — Karachi", fontsize=15, fontweight="bold")
ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
plt.tight_layout()
fig.savefig(os.path.join(CHARTS_DIR, "aqi_distribution.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

# ===================================================================
# Chart 2 – PM2.5 Time Series (full 90-day)
# ===================================================================
print("[2/7] Generating pm25_timeseries.png ...")
fig, ax = plt.subplots(figsize=(14, 5))

ts = df.sort_values("timestamp")
ax.plot(ts["timestamp"], ts["pm2_5"], linewidth=0.8, color="#2980b9", alpha=0.85, label="PM2.5")

# WHO guideline
who_24h = 15  # µg/m³ (WHO 2021 24-hr guideline)
ax.axhline(who_24h, color="#e74c3c", linestyle="--", linewidth=1.3, label=f"WHO 24h Guideline ({who_24h} µg/m³)")

ax.fill_between(ts["timestamp"], ts["pm2_5"], alpha=0.15, color="#2980b9")
ax.set_xlabel("Date", fontsize=13)
ax.set_ylabel("PM2.5 Concentration (µg/m³)", fontsize=13)
ax.set_title("PM2.5 Levels Over Time — Karachi (≈90 Days)", fontsize=15, fontweight="bold")
ax.legend(loc="upper right", fontsize=10)
fig.autofmt_xdate()
plt.tight_layout()
fig.savefig(os.path.join(CHARTS_DIR, "pm25_timeseries.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

# ===================================================================
# Chart 3 – Hourly AQI Pattern (Box plot)
# ===================================================================
print("[3/7] Generating hourly_aqi_pattern.png ...")
fig, ax = plt.subplots(figsize=(14, 6))

sns.boxplot(
    data=df, x="hour", y="aqi_openweather", ax=ax,
    palette="coolwarm", fliersize=2, linewidth=0.8,
)
ax.set_xlabel("Hour of Day", fontsize=13)
ax.set_ylabel("AQI (OpenWeatherMap)", fontsize=13)
ax.set_title("Hourly Air Quality Pattern — Karachi", fontsize=15, fontweight="bold")
ax.set_xticks(range(0, 24))
ax.set_xticklabels([f"{h:02d}" for h in range(24)], fontsize=9)
plt.tight_layout()
fig.savefig(os.path.join(CHARTS_DIR, "hourly_aqi_pattern.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

# ===================================================================
# Chart 4 – Daily AQI Pattern (Bar chart – mean by day of week)
# ===================================================================
print("[4/7] Generating daily_aqi_pattern.png ...")
fig, ax = plt.subplots(figsize=(10, 6))

daily_mean = df.groupby("day_of_week")["aqi_openweather"].mean()
daily_std = df.groupby("day_of_week")["aqi_openweather"].std()

colors_day = sns.color_palette("Set2", n_colors=7)
bars = ax.bar(
    [DAY_NAMES[i] for i in daily_mean.index],
    daily_mean.values,
    yerr=daily_std.values,
    capsize=4,
    color=colors_day,
    edgecolor="white",
    linewidth=1.2,
    width=0.6,
)

for bar, val in zip(bars, daily_mean.values):
    ax.text(
        bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
        f"{val:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold",
    )

ax.set_xlabel("Day of Week", fontsize=13)
ax.set_ylabel("Mean AQI", fontsize=13)
ax.set_title("Average Air Quality by Day of Week — Karachi", fontsize=15, fontweight="bold")
plt.tight_layout()
fig.savefig(os.path.join(CHARTS_DIR, "daily_aqi_pattern.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

# ===================================================================
# Chart 5 – Monthly AQI Pattern (Bar chart – mean by month)
# ===================================================================
print("[5/7] Generating monthly_aqi_pattern.png ...")
fig, ax = plt.subplots(figsize=(12, 6))

monthly_mean = df.groupby("month")["aqi_openweather"].mean()
monthly_std = df.groupby("month")["aqi_openweather"].std()

colors_month = sns.color_palette("RdYlGn_r", n_colors=len(monthly_mean))
bars = ax.bar(
    [MONTH_NAMES[m - 1] if 1 <= m <= 12 else str(m) for m in monthly_mean.index],
    monthly_mean.values,
    yerr=monthly_std.values,
    capsize=4,
    color=colors_month,
    edgecolor="white",
    linewidth=1.2,
    width=0.55,
)

for bar, val in zip(bars, monthly_mean.values):
    ax.text(
        bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
        f"{val:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold",
    )

ax.set_xlabel("Month", fontsize=13)
ax.set_ylabel("Mean AQI", fontsize=13)
ax.set_title("Average Air Quality by Month — Karachi", fontsize=15, fontweight="bold")
plt.tight_layout()
fig.savefig(os.path.join(CHARTS_DIR, "monthly_aqi_pattern.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

# ===================================================================
# Chart 6 – Correlation Heatmap
# ===================================================================
print("[6/7] Generating correlation_heatmap.png ...")

numeric_cols = [
    "co", "no", "no2", "o3", "so2", "pm2_5", "pm10", "nh3",
    "aqi_openweather", "temperature", "humidity", "wind_speed",
    "wind_deg", "pressure", "visibility",
]
# Keep only columns that actually exist in the dataframe
numeric_cols = [c for c in numeric_cols if c in df.columns]
corr = df[numeric_cols].corr()

fig, ax = plt.subplots(figsize=(14, 11))
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
sns.heatmap(
    corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
    center=0, square=True, linewidths=0.5,
    cbar_kws={"shrink": 0.8, "label": "Pearson r"},
    ax=ax, annot_kws={"size": 9},
)
ax.set_title("Feature Correlation Heatmap — Pollutants & Weather", fontsize=15, fontweight="bold")
plt.tight_layout()
fig.savefig(os.path.join(CHARTS_DIR, "correlation_heatmap.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

# ===================================================================
# Chart 7 – PM2.5 vs AQI Scatter (coloured by season)
# ===================================================================
print("[7/7] Generating pm25_vs_aqi_scatter.png ...")
fig, ax = plt.subplots(figsize=(10, 7))

for season_id, label in SEASON_LABELS.items():
    subset = df[df["season"] == season_id]
    if subset.empty:
        continue
    ax.scatter(
        subset["pm2_5"], subset["aqi_openweather"],
        c=SEASON_COLORS[season_id], label=label,
        alpha=0.5, s=25, edgecolors="white", linewidth=0.3,
    )

ax.set_xlabel("PM2.5 Concentration (µg/m³)", fontsize=13)
ax.set_ylabel("AQI (OpenWeatherMap)", fontsize=13)
ax.set_title("PM2.5 vs Air Quality Index by Season — Karachi", fontsize=15, fontweight="bold")
ax.legend(title="Season", fontsize=10, title_fontsize=11)
plt.tight_layout()
fig.savefig(os.path.join(CHARTS_DIR, "pm25_vs_aqi_scatter.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

# ===================================================================
# Key Findings Summary
# ===================================================================
print("\n" + "=" * 70)
print("  KEY FINDINGS SUMMARY")
print("=" * 70)

print(f"\n📊 Dataset: {len(df):,} hourly records from Karachi")
print(f"   Date range: {df['timestamp'].min().date()} to {df['timestamp'].max().date()}")

# AQI distribution
print("\n📈 AQI Distribution:")
for cat in sorted(aqi_counts.index):
    cnt = aqi_counts[cat]
    pct = cnt / total * 100
    label = AQI_LABELS.get(int(cat), "Unknown")
    print(f"   Category {int(cat)} ({label}): {cnt:,} records ({pct:.1f}%)")

# PM2.5 stats
pm25 = df["pm2_5"]
print(f"\n🏭 PM2.5 Statistics:")
print(f"   Mean:   {pm25.mean():.2f} µg/m³")
print(f"   Median: {pm25.median():.2f} µg/m³")
print(f"   Max:    {pm25.max():.2f} µg/m³")
print(f"   Std:    {pm25.std():.2f} µg/m³")
who_exceed = (pm25 > who_24h).sum()
print(f"   Above WHO guideline ({who_24h} µg/m³): {who_exceed:,} records ({who_exceed/len(df)*100:.1f}%)")

# Temporal patterns
print(f"\n🕐 Temporal Patterns:")
worst_hour = df.groupby("hour")["aqi_openweather"].mean().idxmax()
best_hour = df.groupby("hour")["aqi_openweather"].mean().idxmin()
print(f"   Worst air quality hour: {worst_hour:02d}:00")
print(f"   Best air quality hour:  {best_hour:02d}:00")

if len(daily_mean) > 0:
    worst_day = DAY_NAMES[daily_mean.idxmax()]
    best_day = DAY_NAMES[daily_mean.idxmin()]
    print(f"   Worst day of week: {worst_day}")
    print(f"   Best day of week:  {best_day}")

# Top correlations with AQI
print(f"\n🔗 Top Correlations with AQI:")
if "aqi_openweather" in corr.columns:
    aqi_corr = corr["aqi_openweather"].drop("aqi_openweather").abs().sort_values(ascending=False)
    for feat, val in aqi_corr.head(5).items():
        direction = "+" if corr.loc[feat, "aqi_openweather"] > 0 else "−"
        print(f"   {feat:20s}  r = {direction}{val:.3f}")

print(f"\n✅ All 7 charts saved to: {os.path.abspath(CHARTS_DIR)}")
print("=" * 70)
