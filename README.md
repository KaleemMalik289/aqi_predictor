# AQI Predictor: Pearls Internship Project

An end-to-end Air Quality Index (AQI) prediction system for Karachi, Pakistan, built with a fully serverless ML pipeline.

## Project Overview

**Goal:** Predict Air Quality Index (AQI) for the next 3 days in Karachi using machine learning.

**Data Source:** OpenWeatherMap API (current air pollution + weather data)

**Architecture:**
- **Feature Pipeline** (Phase 1): Hourly data ingestion + feature engineering
- **Backfill** (Phase 2): Historical data for training dataset
- **Training** (Phase 3): ML model training (Random Forest, XGBoost, LSTM)
- **Automation** (Phase 4): GitHub Actions for hourly/daily execution
- **Web App** (Phase 5): Streamlit dashboard with 3-day forecast
- **Analysis** (Phase 6): EDA notebook + SHAP feature importance

---

## Project Structure

```
aqi_predictor/
├── feature_pipeline.py       ← Phase 1: Run hourly ⭐ MAIN FILE
├── backfill.py               ← Phase 2: Historical backfill
├── training_pipeline.py      ← Phase 3: ML model training
├── app.py                    ← Phase 5: Streamlit dashboard
├── .env                      ← API keys (NEVER commit)
├── .gitignore                ← Excludes .env, data/, venv/
├── requirements.txt          ← Python dependencies
├── data/
│   └── aqi_features_karachi.csv   ← Feature store (local fallback)
├── models/
│   └── (saved trained models)
├── notebooks/
│   └── EDA.ipynb             ← Phase 6: Exploratory analysis
├── logs/
│   └── (daily pipeline logs)
└── .github/
    └── workflows/
        ├── feature_pipeline.yml   ← Hourly trigger
        └── training_pipeline.yml  ← Daily trigger
```

---

## Getting Started

### 1. Clone & Setup Environment

```bash
cd d:\WeatherApp\aqi_predictor

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Or activate (macOS/Linux)
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Edit `.env` file in the project root:

```
OPENWEATHER_API_KEY=your_openweather_api_key_here
HOPSWORKS_API_KEY=your_hopsworks_api_key_here
HOPSWORKS_HOST=c.app.hopsworks.ai
CITY=Karachi
LAT=24.8607
LON=67.0011
```

**Important:**
- `OPENWEATHER_API_KEY` is already provided ✓
- Get `HOPSWORKS_API_KEY` from https://app.hopsworks.ai (free account)
- **Never** commit `.env` to GitHub

### 4. Run Feature Pipeline (Phase 1)

```bash
python feature_pipeline.py
```

This will:
1. ✓ Fetch air pollution + weather data from OpenWeatherMap
2. ✓ Compute 30+ engineered features
3. ✓ Store to Hopsworks or local CSV
4. ✓ Log all operations with timestamps
5. ✓ Print success confirmation

**Expected output:**
```
✓ Feature pipeline completed successfully
  Timestamp : 2025-06-04T14:00:00Z
  AQI       : 3 (Moderate)
  PM2.5     : 68.3 μg/m³
  Stored to : Local CSV
  Next run  : 1 hour from now (via GitHub Actions)
```

---

## Features Computed

### Raw Pollutants (from OpenWeatherMap)
- `co`, `no`, `no2`, `o3`, `so2`, `pm2_5`, `pm10`, `nh3`
- `aqi_openweather` (1-5 scale)

### Weather Features
- `temperature` (°C), `humidity` (%), `wind_speed` (m/s)
- `wind_deg` (0-360), `pressure` (hPa), `visibility` (m)
- `weather_main` (Clear, Clouds, Rain, etc.)

### Time-Based Features (critical for patterns)
- `hour` (0-23), `day_of_week` (0-6), `day_of_month` (1-31)
- `month` (1-12), `is_weekend` (0/1), `is_rush_hour` (0/1)
- `season` (0=Winter, 1=Spring, 2=Summer, 3=Autumn)

### Engineered Features (capture change over time)
- `aqi_change_rate` — AQI delta from previous reading
- `pm25_change_rate` — PM2.5 delta
- `pm25_rolling_3h`, `pm25_rolling_6h`, `pm25_rolling_24h` — Moving averages
- `aqi_rolling_3h` — 3-hour average AQI
- `pollution_index` — Weighted composite score

### Target Column (what the model predicts)
- `aqi_next_24h` — AQI 24 hours later (filled automatically)

---

## Data Storage

### Option 1: Hopsworks Feature Store (Recommended)
- Cloud-hosted feature management
- Automatic versioning + lineage tracking
- Real-time feature serving

**Setup:**
1. Create free account at https://app.hopsworks.ai
2. Copy API key to `.env` as `HOPSWORKS_API_KEY`
3. Feature group auto-created on first run

### Option 2: Local CSV (Fallback)
- Location: `data/aqi_features_karachi.csv`
- Used if Hopsworks unavailable
- Append-only (no overwrites)

---

## Error Handling

The pipeline handles these failures gracefully:

| Error | Behavior |
|-------|----------|
| Missing API key | Logs error message, stops cleanly |
| API rate limit | Waits 60s, retries once |
| Network timeout | Logs warning, continues |
| Missing pollutant field | Uses NaN (doesn't crash) |
| Hopsworks unavailable | Falls back to local CSV |
| First run (no history) | Sets change rates to 0, rolling averages to NaN |

All operations logged to: `logs/feature_pipeline_YYYYMMDD.log`

---

## GitHub Actions Automation

### Hourly Feature Pipeline
File: `.github/workflows/feature_pipeline.yml`

- **Trigger:** Every hour at minute 0 (0 * * * *)
- **Action:** Runs `feature_pipeline.py`
- **Environment:** API keys stored as GitHub Secrets
- **Logs:** Uploaded as artifacts for debugging

### Daily Training Pipeline (Coming Soon)
File: `.github/workflows/training_pipeline.yml`

- **Trigger:** Daily at 2 AM UTC
- **Action:** Retrains ML models with latest features
- **Output:** Saves best model to `models/`

---

## Next Phases (Do NOT Start Yet)

### Phase 2: Backfill Historical Data
- Fetch data for past 90 days
- Generate training dataset
- File: `backfill.py`

### Phase 3: Training Pipeline
- Pull features from Hopsworks
- Train Random Forest + XGBoost + optional LSTM
- Evaluate with RMSE, MAE, R²
- Save best model
- File: `training_pipeline.py`

### Phase 4: Automation
- Set up GitHub Actions workflows
- Hourly feature ingestion
- Daily model retraining

### Phase 5: Web Dashboard
- Streamlit app showing current AQI + 3-day forecast
- Color-coded hazard alerts
- Pollutant breakdown charts
- File: `app.py`

### Phase 6: Analysis & Polish
- EDA notebook (exploratory analysis)
- SHAP feature importance plots
- Final report documentation
- File: `notebooks/EDA.ipynb`

---

## Troubleshooting

### Pipeline fails with "OPENWEATHER_API_KEY is missing"
- Check that `.env` file exists in project root
- Verify the API key is not blank or "your_key_here"
- Restart Python/terminal to reload `.env`

### Getting "requests.exceptions.Timeout"
- OpenWeatherMap API slow to respond
- Pipeline retries automatically after 5 seconds
- Check internet connection

### Features not stored to Hopsworks
- Check `HOPSWORKS_API_KEY` in `.env`
- Verify Hopsworks account is active
- Pipeline will fall back to local CSV automatically
- Check logs for detailed error

### CSV file has duplicate timestamps
- Each row is timestamped at UTC second precision
- If script runs within same second twice (unlikely), timestamp matches
- In practice, hourly scheduling prevents this

---

## Development Notes

### Adding New Features
1. Add extraction logic in `extract_*` functions
2. Add to feature row in `build_feature_row()`
3. Update feature group schema if using Hopsworks
4. Document in this README

### Modifying Feature Engineering
1. Edit `compute_derived_features()` or `compute_pollution_index()`
2. Update feature column names if changed
3. Log changes in commit message
4. Rerun backfill if changing historical computation

### Local Testing
```bash
# Manually trigger pipeline
python feature_pipeline.py

# View logs
tail -f logs/feature_pipeline_*.log

# Check CSV
cat data/aqi_features_karachi.csv

# Inspect features with pandas
python -c "import pandas as pd; print(pd.read_csv('data/aqi_features_karachi.csv'))"
```

---

## FAQ

**Q: How often should I run this?**  
A: Every hour via GitHub Actions. Manual runs are also fine for testing.

**Q: What if the API is down?**  
A: Pipeline logs the error and continues. Features are not stored that hour. Previous data remains.

**Q: How much historical data do I need for training?**  
A: At least 30 days; ideally 90+ days. Phase 2 backfill covers 90 days.

**Q: Can I modify the feature list?**  
A: Yes, but ensure backward compatibility. Update Hopsworks feature group schema.

**Q: Why is `aqi_next_24h` initially NaN?**  
A: It's the target for 24 hours later. Gets filled when pipeline runs 24h later.

---

## Support

For questions, check:
- `.env` configuration
- `logs/feature_pipeline_*.log` for detailed errors
- OpenWeatherMap API docs: https://openweathermap.org/api/air-pollution
- Hopsworks docs: https://docs.hopsworks.ai

---

**Status: Phase 1 Feature Pipeline ✓ Ready**  
Next: Phase 2 Backfill (backfill.py)
