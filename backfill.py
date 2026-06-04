"""
Backfill Pipeline for AQI Predictor
Phase 2: Fetch 90 days of historical AQI & weather data and compute features.

This script:
1. Fetches historical air pollution data from OpenWeatherMap (7-day chunks)
2. Fetches historical weather data with caching
3. Computes all 40 features (same schema as feature_pipeline.py)
4. Merges with existing live data
5. Fills rolling averages and target column

Run once to populate the feature store with historical training data.
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
import logging
import os
import time
from pathlib import Path
from dotenv import load_dotenv

# ============================================================================
# CONFIGURATION & LOGGING
# ============================================================================

# Load environment variables
load_dotenv()

# API Keys and constants
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
CITY = os.getenv("CITY", "Karachi")
LAT = float(os.getenv("LAT", "24.8607"))
LON = float(os.getenv("LON", "67.0011"))

# Data storage
DATA_DIR = Path(__file__).parent / "data"
CSV_FILE = DATA_DIR / "aqi_features_karachi.csv"
DATA_DIR.mkdir(exist_ok=True)

# Setup logging
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / f"backfill_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

import sys
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# STEP 1: DEFINE DATE RANGE FOR BACKFILL
# ============================================================================

def get_backfill_date_range():
    """Define the historical date range: 90 days back to yesterday."""
    end_date = datetime.now(timezone.utc).replace(hour=23, minute=0, second=0, microsecond=0) - timedelta(days=1)
    start_date = end_date - timedelta(days=90)
    
    logger.info(f"Backfill date range: {start_date.date()} to {end_date.date()}")
    logger.info(f"Expected rows: {(end_date - start_date).days * 24} (approx 1 per hour)")
    
    return start_date, end_date


# ============================================================================
# STEP 2: FETCH HISTORICAL POLLUTION DATA IN 7-DAY CHUNKS
# ============================================================================

def validate_api_key():
    """Validate that OPENWEATHER_API_KEY is present and not empty."""
    if not OPENWEATHER_API_KEY or OPENWEATHER_API_KEY == "your_key_here":
        logger.error(
            "OPENWEATHER_API_KEY is missing or invalid. "
            "Please check your .env file."
        )
        raise ValueError("OPENWEATHER_API_KEY is missing or invalid")
    logger.info("API key validated")


def fetch_pollution_history(start_date, end_date, api_key):
    """
    Fetch historical air pollution data in 7-day chunks.
    
    Returns:
        list: All pollution records combined from all chunks
    """
    all_records = []
    chunk_size = timedelta(days=7)
    current_start = start_date
    chunk_num = 0
    total_chunks = int((end_date - start_date).days / 7) + 1
    
    logger.info(f"\nFetching pollution history in 7-day chunks...")
    
    while current_start < end_date:
        chunk_num += 1
        current_end = min(current_start + chunk_size, end_date)
        
        unix_start = int(current_start.timestamp())
        unix_end = int(current_end.timestamp())
        
        url = (
            f"http://api.openweathermap.org/data/2.5/air_pollution/history"
            f"?lat={LAT}&lon={LON}&start={unix_start}&end={unix_end}&appid={api_key}"
        )
        
        try:
            response = requests.get(url, timeout=15)
            response.raise_for_status()
            data = response.json()
            
            if "list" in data and data["list"]:
                records = data["list"]
                all_records.extend(records)
                logger.info(f"  Chunk {chunk_num}/{total_chunks}: {current_start.date()} to {current_end.date()} ... {len(records)} records fetched")
            else:
                logger.warning(f"  Chunk {chunk_num}/{total_chunks}: {current_start.date()} to {current_end.date()} ... No data available")
        
        except requests.exceptions.HTTPError as e:
            if response.status_code == 429:
                logger.warning(f"  Chunk {chunk_num}/{total_chunks}: Rate limit hit — waiting 60 seconds...")
                time.sleep(60)
                continue
            else:
                logger.error(f"  Chunk {chunk_num}/{total_chunks}: HTTP {response.status_code} — skipping chunk")
        
        except requests.exceptions.RequestException as e:
            logger.error(f"  Chunk {chunk_num}/{total_chunks}: Request failed: {e} — skipping chunk")
        
        current_start = current_end
        time.sleep(1)  # Respect rate limit
    
    logger.info(f"Total pollution records fetched: {len(all_records)}")
    
    if len(all_records) < 500:
        logger.warning(
            f"WARNING: Only {len(all_records)} records fetched. "
            "This may not be enough for training. Consider checking your API plan."
        )
    
    return all_records


# ============================================================================
# STEP 3: FETCH HISTORICAL WEATHER DATA (WITH CACHING)
# ============================================================================

def fetch_weather_for_timestamp(unix_ts, api_key):
    """
    Fetch weather for a specific timestamp using onecall/timemachine.
    Falls back to current weather if timemachine is not available.
    
    Returns:
        dict: Weather data (current or fallback)
    """
    try:
        url = (
            f"https://api.openweathermap.org/data/2.5/onecall/timemachine"
            f"?lat={LAT}&lon={LON}&dt={unix_ts}&appid={api_key}&units=metric"
        )
        response = requests.get(url, timeout=15)
        
        if response.status_code == 401:
            # Plan doesn't include timemachine — use fallback
            return None
        
        response.raise_for_status()
        data = response.json()
        
        # Extract current weather from timemachine response
        if "current" in data:
            return data["current"]
        elif "hourly" in data and data["hourly"]:
            return data["hourly"][0]
        
        return None
    
    except Exception as e:
        logger.debug(f"Weather fetch failed for {unix_ts}: {e}")
        return None


def get_weather_for_timestamps(pollution_records, api_key):
    """
    Fetch weather data for all pollution records with caching by date.
    
    Strategy: Fetch once per calendar day, cache by date string.
    This reduces API calls from 2,136 to ~90.
    
    Returns:
        dict: Mapping of unix_timestamp -> weather_dict
    """
    weather_cache = {}
    date_weather_cache = {}
    timemachine_available = True
    
    logger.info(f"\nFetching weather data for {len(pollution_records)} timestamps...")
    
    unique_dates = set()
    for record in pollution_records:
        dt = datetime.fromtimestamp(record["dt"], tz=timezone.utc)
        date_key = dt.strftime("%Y-%m-%d")
        unique_dates.add((date_key, record["dt"]))
    
    unique_dates = sorted(list(unique_dates))
    
    for idx, (date_key, unix_ts) in enumerate(unique_dates):
        if idx % 10 == 0:
            logger.info(f"  Progress: {idx}/{len(unique_dates)}")
        
        # Check if we already have weather for this date
        if date_key in date_weather_cache:
            weather = date_weather_cache[date_key]
        else:
            # Fetch weather for this date
            if timemachine_available:
                weather = fetch_weather_for_timestamp(unix_ts, api_key)
                
                if weather is None:
                    # Timemachine not available on this plan — fall back
                    logger.warning("onecall/timemachine endpoint not available — using current weather fallback")
                    timemachine_available = False
                    weather = fetch_current_weather(api_key)
            else:
                # Already know timemachine is unavailable — use fallback
                weather = fetch_current_weather(api_key)
            
            date_weather_cache[date_key] = weather
            time.sleep(0.5)  # Small delay to avoid hammering API
        
        # Cache by unix timestamp for easy lookup
        if weather:
            weather_cache[unix_ts] = weather
    
    logger.info(f"  Cached weather for {len(weather_cache)} unique hours")
    return weather_cache


def fetch_current_weather(api_key):
    """Fallback: fetch current weather and use as proxy for historical weather."""
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?lat={LAT}&lon={LON}&appid={api_key}&units=metric"
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.warning(f"Could not fetch fallback weather: {e}")
        return {}


# ============================================================================
# STEP 4: BUILD FEATURE ROWS (REUSE LOGIC FROM feature_pipeline.py)
# ============================================================================

def extract_weather_dict(weather_data):
    """Convert raw weather response into feature dict."""
    if not weather_data:
        return {
            "temperature": np.nan,
            "humidity": np.nan,
            "wind_speed": np.nan,
            "wind_deg": np.nan,
            "pressure": np.nan,
            "visibility": 10000,
            "weather_main": "Unknown"
        }
    
    # Handle both onecall/timemachine and weather endpoint formats
    main = weather_data.get("main", weather_data.get("temp", {}))
    
    if isinstance(main, dict):
        # weather endpoint format
        temp_k = main.get("temp", np.nan)
        humidity = main.get("humidity", np.nan)
        pressure = main.get("pressure", np.nan)
    else:
        # onecall format — temp is a float directly
        temp_k = weather_data.get("temp", np.nan)
        humidity = weather_data.get("humidity", np.nan)
        pressure = weather_data.get("pressure", np.nan)
    
    # Convert temperature from Kelvin to Celsius if needed
    if isinstance(temp_k, (int, float)) and temp_k > 100:  # Likely in Kelvin
        temperature = temp_k - 273.15
    else:
        temperature = temp_k  # Already in Celsius or NaN
    
    wind = weather_data.get("wind", {})
    if isinstance(wind, dict):
        wind_speed = wind.get("speed", np.nan)
        wind_deg = wind.get("deg", np.nan)
    else:
        wind_speed = np.nan
        wind_deg = np.nan
    
    visibility = weather_data.get("visibility", 10000)
    if visibility is None:
        visibility = 10000
    visibility = min(visibility, 10000)
    
    weather_list = weather_data.get("weather", [{}])
    if weather_list:
        weather_main = weather_list[0].get("main", "Unknown")
    else:
        weather_main = "Unknown"
    
    return {
        "temperature": temperature,
        "humidity": humidity,
        "wind_speed": wind_speed,
        "wind_deg": wind_deg,
        "pressure": pressure,
        "visibility": visibility,
        "weather_main": weather_main
    }


def extract_time_features(unix_ts):
    """Extract time-based features from unix timestamp."""
    dt = datetime.fromtimestamp(unix_ts, tz=timezone.utc)
    
    hour = dt.hour
    day_of_week = dt.weekday()
    day_of_month = dt.day
    month = dt.month
    is_weekend = 1 if day_of_week >= 5 else 0
    is_rush_hour = 1 if hour in range(7, 10) or hour in range(17, 21) else 0
    
    # Season for Karachi: 0=Winter(Dec-Feb), 1=Spring(Mar-May), 2=Summer(Jun-Aug), 3=Autumn(Sep-Nov)
    if month in [12, 1, 2]:
        season = 0
    elif month in [3, 4, 5]:
        season = 1
    elif month in [6, 7, 8]:
        season = 2
    else:
        season = 3
    
    return {
        "hour": hour,
        "day_of_week": day_of_week,
        "day_of_month": day_of_month,
        "month": month,
        "is_weekend": is_weekend,
        "is_rush_hour": is_rush_hour,
        "season": season
    }


def compute_pollution_index(co, no2, o3, pm2_5, pm10):
    """Compute composite pollution index."""
    if any(v is None or (isinstance(v, float) and np.isnan(v)) for v in [pm2_5, pm10, no2, o3]):
        return np.nan
    
    try:
        index = (float(pm2_5) * 0.4) + (float(pm10) * 0.2) + (float(no2) * 0.2) + (float(o3) * 0.2)
        return index
    except (ValueError, TypeError):
        return np.nan


def build_feature_row(pollution_record, weather_dict, prev_row=None):
    """
    Build a complete feature row for a pollution record.
    
    Returns:
        dict: Feature row with all 40 columns
    """
    unix_ts = pollution_record["dt"]
    dt = datetime.fromtimestamp(unix_ts, tz=timezone.utc)
    
    comp = pollution_record.get("components", {})
    main = pollution_record.get("main", {})
    aqi = main.get("aqi", np.nan)
    
    # Pollutant features
    co = comp.get("co")
    no = comp.get("no")
    no2 = comp.get("no2")
    o3 = comp.get("o3")
    so2 = comp.get("so2")
    pm2_5 = comp.get("pm2_5", comp.get("pm2.5"))
    pm10 = comp.get("pm10")
    nh3 = comp.get("nh3")
    
    # Convert None to np.nan for consistency
    pollutants = {
        "co": co if co is not None else np.nan,
        "no": no if no is not None else np.nan,
        "no2": no2 if no2 is not None else np.nan,
        "o3": o3 if o3 is not None else np.nan,
        "so2": so2 if so2 is not None else np.nan,
        "pm2_5": pm2_5 if pm2_5 is not None else np.nan,
        "pm10": pm10 if pm10 is not None else np.nan,
        "nh3": nh3 if nh3 is not None else np.nan,
        "aqi_openweather": aqi if aqi is not None else np.nan
    }
    
    # Weather features
    weather = extract_weather_dict(weather_dict) if weather_dict else extract_weather_dict({})
    
    # Time features
    time_feats = extract_time_features(unix_ts)
    
    # Derived features (change rates)
    aqi_change_rate = 0.0
    pm25_change_rate = 0.0
    
    if prev_row is not None:
        prev_aqi = prev_row.get("aqi_openweather")
        prev_pm25 = prev_row.get("pm2_5")
        
        if prev_aqi is not None and not (isinstance(prev_aqi, float) and np.isnan(prev_aqi)):
            aqi_change_rate = aqi - prev_aqi if aqi is not None else 0.0
        
        if prev_pm25 is not None and not (isinstance(prev_pm25, float) and np.isnan(prev_pm25)):
            if pm2_5 is not None:
                pm25_change_rate = pm2_5 - prev_pm25
    
    # Pollution index
    pollution_idx = compute_pollution_index(co, no2, o3, pm2_5, pm10)
    
    # Build complete row
    row = {
        "timestamp": dt.isoformat(),
        "city": CITY,
        "lat": LAT,
        "lon": LON,
        "data_source": "openweathermap_historical",
        
        # Pollutants
        **pollutants,
        
        # Weather
        **weather,
        
        # Time
        **time_feats,
        
        # Derived
        "aqi_change_rate": aqi_change_rate,
        "pm25_change_rate": pm25_change_rate,
        "pm25_rolling_3h": np.nan,    # Filled after all rows loaded
        "pm25_rolling_6h": np.nan,
        "pm25_rolling_24h": np.nan,
        "aqi_rolling_3h": np.nan,
        "pollution_index": pollution_idx,
        
        # Target
        "aqi_next_24h": np.nan
    }
    
    return row


def build_all_feature_rows(pollution_records, weather_cache):
    """Build feature rows for all pollution records."""
    logger.info(f"\nComputing features for all {len(pollution_records)} records...")
    
    all_rows = []
    prev_row = None
    
    for idx, record in enumerate(pollution_records):
        if idx % 500 == 0:
            logger.info(f"  Progress: {idx}/{len(pollution_records)}")
        
        unix_ts = record["dt"]
        weather = weather_cache.get(unix_ts, {})
        
        row = build_feature_row(record, weather, prev_row)
        all_rows.append(row)
        prev_row = row
    
    logger.info(f"  Done. {len(all_rows)} feature rows created")
    return all_rows


# ============================================================================
# STEP 5: COMPUTE ROLLING AVERAGES & FILL TARGET COLUMN
# ============================================================================

def compute_rolling_and_targets(df):
    """Compute rolling averages and fill target column on the full DataFrame."""
    logger.info(f"\nComputing rolling averages and target column...")
    
    # Sort by timestamp
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    # Rolling averages
    logger.info(f"  Rolling 3h:  computing...")
    df["pm25_rolling_3h"] = df["pm2_5"].rolling(window=3, min_periods=1).mean()
    
    logger.info(f"  Rolling 6h:  computing...")
    df["pm25_rolling_6h"] = df["pm2_5"].rolling(window=6, min_periods=1).mean()
    
    logger.info(f"  Rolling 24h: computing...")
    df["pm25_rolling_24h"] = df["pm2_5"].rolling(window=24, min_periods=1).mean()
    
    logger.info(f"  Rolling 3h AQI: computing...")
    df["aqi_rolling_3h"] = df["aqi_openweather"].rolling(window=3, min_periods=1).mean()
    
    # Target column: AQI 24 hours in the future
    logger.info(f"  Target column: computing...")
    df["aqi_next_24h"] = df["aqi_openweather"].shift(-24)
    
    valid_targets = df["aqi_next_24h"].notna().sum()
    logger.info(f"  {valid_targets} rows have valid aqi_next_24h (last 24 are NaN — expected)")
    
    return df


# ============================================================================
# STEP 6: MERGE WITH EXISTING CSV AND SAVE
# ============================================================================

def merge_and_save(df_backfill):
    """Merge backfill data with existing CSV, remove duplicates, and save."""
    logger.info(f"\nMerging with existing data...")
    
    # Load existing CSV if it exists
    if CSV_FILE.exists():
        df_existing = pd.read_csv(CSV_FILE)
        logger.info(f"  Loaded {len(df_existing)} existing rows from {CSV_FILE}")
    else:
        df_existing = pd.DataFrame()
        logger.info(f"  No existing CSV found (first run)")
    
    # Combine
    if len(df_existing) > 0:
        df_combined = pd.concat([df_backfill, df_existing], ignore_index=True)
        logger.info(f"  Combined: {len(df_backfill)} backfill + {len(df_existing)} existing = {len(df_combined)} total")
    else:
        df_combined = df_backfill
        logger.info(f"  Using backfill data only: {len(df_combined)} rows")
    
    # Remove duplicates (keep the most recent version by keeping 'last')
    logger.info(f"  Removing duplicates by timestamp...")
    df_combined = df_combined.drop_duplicates(subset=["timestamp"], keep="last")
    logger.info(f"  After dedup: {len(df_combined)} rows")
    
    # Sort chronologically
    logger.info(f"  Sorting by timestamp...")
    df_combined = df_combined.sort_values("timestamp").reset_index(drop=True)
    
    # Ensure column order matches schema from feature_pipeline.py
    column_order = [
        "timestamp", "city", "lat", "lon", "data_source",
        "co", "no", "no2", "o3", "so2", "pm2_5", "pm10", "nh3", "aqi_openweather",
        "temperature", "humidity", "wind_speed", "wind_deg", "pressure", "visibility", "weather_main",
        "hour", "day_of_week", "day_of_month", "month", "is_weekend", "is_rush_hour", "season",
        "aqi_change_rate", "pm25_change_rate", "pm25_rolling_3h", "pm25_rolling_6h",
        "pm25_rolling_24h", "aqi_rolling_3h", "pollution_index", "aqi_next_24h"
    ]
    
    # Reorder columns (add missing columns as NaN if needed)
    for col in column_order:
        if col not in df_combined.columns:
            df_combined[col] = np.nan
    
    df_combined = df_combined[column_order]
    
    # Save to CSV
    logger.info(f"  Saving to {CSV_FILE}...")
    df_combined.to_csv(CSV_FILE, index=False)
    logger.info(f"  Saved successfully")
    
    return df_combined


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Execute the complete backfill pipeline."""
    logger.info("=" * 80)
    logger.info("STARTING AQI BACKFILL PIPELINE (PHASE 2)")
    logger.info("=" * 80)
    
    try:
        # Validate API key
        validate_api_key()
        
        # Step 1: Define date range
        logger.info("\n[STEP 1] Defining backfill date range...")
        start_date, end_date = get_backfill_date_range()
        
        # Step 2: Fetch pollution history
        logger.info("\n[STEP 2] Fetching historical pollution data...")
        pollution_records = fetch_pollution_history(start_date, end_date, OPENWEATHER_API_KEY)
        
        if len(pollution_records) == 0:
            logger.error("ERROR: No pollution records fetched. Check API key and date range.")
            return False
        
        # Step 3: Fetch weather data
        logger.info("\n[STEP 3] Fetching historical weather data...")
        weather_cache = get_weather_for_timestamps(pollution_records, OPENWEATHER_API_KEY)
        
        # Step 4: Build feature rows
        logger.info("\n[STEP 4] Building feature rows...")
        all_rows = build_all_feature_rows(pollution_records, weather_cache)
        
        # Convert to DataFrame
        df = pd.DataFrame(all_rows)
        
        # Step 5: Compute rolling averages and targets
        logger.info("\n[STEP 5] Computing rolling averages and target column...")
        df = compute_rolling_and_targets(df)
        
        # Step 6: Merge with existing and save
        logger.info("\n[STEP 6] Merging and saving...")
        df_final = merge_and_save(df)
        
        # Print summary
        logger.info("\n" + "=" * 80)
        logger.info("BACKFILL COMPLETE")
        logger.info(f"  Total rows in CSV        : {len(df_final)}")
        logger.info(f"  Training-ready rows      : {df_final['aqi_next_24h'].notna().sum()} (have valid target)")
        logger.info(f"  Date range               : {df_final['timestamp'].min()} → {df_final['timestamp'].max()}")
        logger.info(f"  File location            : {CSV_FILE}")
        logger.info(f"  Backfill rows            : {len(df)}")
        logger.info(f"  Next step                : python training_pipeline.py (Phase 3)")
        logger.info("=" * 80)
        
        return True
    
    except Exception as e:
        logger.exception(f"Pipeline failed with exception: {e}")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
