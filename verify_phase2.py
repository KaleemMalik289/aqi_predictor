import pandas as pd
import numpy as np

df = pd.read_csv("data/aqi_features_karachi.csv")
print(f"CSV Shape: {df.shape[0]} rows × {df.shape[1]} columns\n")

print("Column Verification:")
required_cols = ["timestamp", "city", "lat", "lon", "data_source", "pm2_5", "aqi_openweather", "pm25_rolling_24h", "aqi_next_24h"]
for col in required_cols:
    if col in df.columns:
        print(f"  ✓ {col}")
    else:
        print(f"  ✗ MISSING: {col}")

print(f"\nData Quality:")
print(f"  No duplicates: {df.duplicated('timestamp').sum() == 0}")
print(f"  Valid targets (aqi_next_24h): {df['aqi_next_24h'].notna().sum()} rows")
print(f"  PM2.5 rolling avg filled: {df['pm25_rolling_24h'].notna().sum()} rows")
print(f"  Date range: {df['timestamp'].min()[:10]} to {df['timestamp'].max()[:10]}")

print(f"\nData Source Distribution:")
print(df['data_source'].value_counts())

print(f"\nSample statistics (PM2.5):")
print(f"  Mean: {df['pm2_5'].mean():.1f} μg/m³")
print(f"  Std: {df['pm2_5'].std():.1f}")
print(f"  Min: {df['pm2_5'].min():.1f}")
print(f"  Max: {df['pm2_5'].max():.1f}")
