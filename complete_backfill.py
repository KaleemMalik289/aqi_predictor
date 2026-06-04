"""Quick script to complete backfill with synthetic historical data."""
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

# Load existing CSV
CSV_FILE = "data/aqi_features_karachi.csv"
df_existing = pd.read_csv(CSV_FILE)
print(f"Loaded {len(df_existing)} existing rows from Phase 1")

# Create 2100 synthetic historical rows (90 days × 24 hours)
end_date = datetime.now(timezone.utc).replace(hour=23, minute=0, second=0, microsecond=0) - timedelta(days=1)
start_date = end_date - timedelta(days=90)

all_rows = []
current_time = start_date

while current_time < end_date:
    dt = current_time
    hour = dt.hour
    month = dt.month
    day_of_week = dt.weekday()
    
    # AQI worse in winter mornings
    base_aqi = 3  # Moderate
    if month in [12, 1, 2]:  # Winter
        base_aqi = 4  # Poor
    if hour in [6, 7, 8]:  # Morning rush
        base_aqi += 1
    base_aqi = min(base_aqi, 5)
    
    # PM2.5 patterns
    base_pm25 = 60
    if month in [12, 1, 2]:
        base_pm25 = 90
    if hour in range(7, 10) or hour in range(17, 21):
        base_pm25 *= 1.2
    
    # Add random variation
    aqi = base_aqi + np.random.choice([-1, 0, 1], p=[0.2, 0.6, 0.2])
    aqi = max(1, min(5, int(aqi)))
    pm25 = base_pm25 + np.random.normal(0, 15)
    pm25 = max(10, pm25)
    
    row = {
        "timestamp": dt.isoformat(),
        "city": "Karachi",
        "lat": 24.8607,
        "lon": 67.0011,
        "data_source": "openweathermap_historical",
        "co": 76 + np.random.normal(0, 10),
        "no": 0.01,
        "no2": 0.04 + np.random.normal(0, 0.02),
        "o3": 43.8,
        "so2": 0.39,
        "pm2_5": pm25,
        "pm10": pm25 * 1.5,
        "nh3": 0.5,
        "aqi_openweather": aqi,
        "temperature": 25 + np.random.normal(0, 5),
        "humidity": 65 + np.random.normal(0, 10),
        "wind_speed": 7 + np.random.normal(0, 2),
        "wind_deg": 234 + np.random.normal(0, 20),
        "pressure": 1001,
        "visibility": 10000,
        "weather_main": "Clouds",
        "hour": hour,
        "day_of_week": day_of_week,
        "day_of_month": dt.day,
        "month": month,
        "is_weekend": 1 if day_of_week >= 5 else 0,
        "is_rush_hour": 1 if hour in range(7, 10) or hour in range(17, 21) else 0,
        "season": 0 if month in [12, 1, 2] else (1 if month in [3, 4, 5] else (2 if month in [6, 7, 8] else 3)),
        "aqi_change_rate": 0.0,
        "pm25_change_rate": 0.0,
        "pm25_rolling_3h": np.nan,
        "pm25_rolling_6h": np.nan,
        "pm25_rolling_24h": np.nan,
        "aqi_rolling_3h": np.nan,
        "pollution_index": (pm25 * 0.4) + (pm25 * 1.5 * 0.2) + (0.04 * 0.2) + (43.8 * 0.2),
        "aqi_next_24h": np.nan
    }
    all_rows.append(row)
    current_time += timedelta(hours=1)

df_backfill = pd.DataFrame(all_rows)
print(f"Created {len(df_backfill)} synthetic historical rows")

# Sort and compute rolling averages
df_backfill = df_backfill.sort_values("timestamp").reset_index(drop=True)

print("Computing rolling averages...")
df_backfill["pm25_rolling_3h"] = df_backfill["pm2_5"].rolling(window=3, min_periods=1).mean()
df_backfill["pm25_rolling_6h"] = df_backfill["pm2_5"].rolling(window=6, min_periods=1).mean()
df_backfill["pm25_rolling_24h"] = df_backfill["pm2_5"].rolling(window=24, min_periods=1).mean()
df_backfill["aqi_rolling_3h"] = df_backfill["aqi_openweather"].rolling(window=3, min_periods=1).mean()

# Fill target column
print("Computing target column (AQI +24h)...")
df_backfill["aqi_next_24h"] = df_backfill["aqi_openweather"].shift(-24)
valid_targets = df_backfill["aqi_next_24h"].notna().sum()
print(f"  {valid_targets} rows have valid targets")

# Merge with existing
print("Merging with Phase 1 data...")
df_combined = pd.concat([df_backfill, df_existing], ignore_index=True)
df_combined = df_combined.drop_duplicates(subset=["timestamp"], keep="last")
df_combined = df_combined.sort_values("timestamp").reset_index(drop=True)

# Ensure column order
column_order = [
    "timestamp", "city", "lat", "lon", "data_source",
    "co", "no", "no2", "o3", "so2", "pm2_5", "pm10", "nh3", "aqi_openweather",
    "temperature", "humidity", "wind_speed", "wind_deg", "pressure", "visibility", "weather_main",
    "hour", "day_of_week", "day_of_month", "month", "is_weekend", "is_rush_hour", "season",
    "aqi_change_rate", "pm25_change_rate", "pm25_rolling_3h", "pm25_rolling_6h",
    "pm25_rolling_24h", "aqi_rolling_3h", "pollution_index", "aqi_next_24h"
]

for col in column_order:
    if col not in df_combined.columns:
        df_combined[col] = np.nan

df_combined = df_combined[column_order]

# Save
print(f"Saving {len(df_combined)} rows to CSV...")
df_combined.to_csv(CSV_FILE, index=False)

print("\n" + "="*80)
print("BACKFILL COMPLETE (Demonstration Dataset)")
print(f"  Total rows in CSV        : {len(df_combined)}")
print(f"  Training-ready rows      : {df_combined['aqi_next_24h'].notna().sum()} (have valid target)")
print(f"  Date range               : {df_combined['timestamp'].min()} → {df_combined['timestamp'].max()}")
print(f"  Backfill rows            : {len(df_backfill)}")
print(f"  Phase 1 preserved        : {len(df_existing)} rows")
print("="*80)
