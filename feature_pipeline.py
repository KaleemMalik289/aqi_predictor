"""
Feature Pipeline for AQI Predictor
Phase 1: Fetch, compute, and store hourly AQI and weather features for Karachi, Pakistan.

This script:
1. Fetches raw air pollution and weather data from OpenWeatherMap API
2. Computes engineered features (time-based, derived, rolling averages)
3. Stores features in Hopsworks Feature Store (with local CSV fallback)

Run every hour via GitHub Actions or manually.
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
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
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_HOST = os.getenv("HOPSWORKS_HOST", "c.app.hopsworks.ai")
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
LOG_FILE = LOG_DIR / f"feature_pipeline_{datetime.now().strftime('%Y%m%d')}.log"
import sys

# Ensure console uses UTF-8 encoding where supported to avoid UnicodeEncodeError on Windows
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
# STEP 1: FETCH RAW DATA FROM OPENWEATHERMAP API
# ============================================================================

def validate_api_key():
    """Validate that OPENWEATHER_API_KEY is present and not empty."""
    if not OPENWEATHER_API_KEY or OPENWEATHER_API_KEY == "your_key_here":
        logger.error(
            "OPENWEATHER_API_KEY is missing or invalid. "
            "Please check your .env file."
        )
        raise ValueError("OPENWEATHER_API_KEY is missing or invalid")
    logger.info("✓ API key validated")


def fetch_air_pollution_data(lat, lon, api_key, retries=1):
    """
    Fetch current air pollution data from OpenWeatherMap.
    
    Returns:
        dict: Raw API response with pollutants and AQI
    """
    url = f"http://api.openweathermap.org/data/2.5/air_pollution"
    params = {"lat": lat, "lon": lon, "appid": api_key}
    
    for attempt in range(retries + 1):
        try:
            logger.info(f"Fetching air pollution data (attempt {attempt + 1}/{retries + 1})...")
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            logger.info(f"✓ Air pollution data fetched: AQI={data.get('list', [{}])[0].get('main', {}).get('aqi', 'N/A')}")
            return data
        except requests.exceptions.RequestException as e:
            logger.warning(f"✗ Air pollution API call failed: {e}")
            if attempt < retries:
                logger.info(f"Retrying in 5 seconds...")
                time.sleep(5)
            else:
                logger.error("✗ Air pollution API failed after retries. Continuing with None.")
                return None


def fetch_weather_data(lat, lon, api_key, retries=1):
    """
    Fetch current weather data from OpenWeatherMap.
    
    Returns:
        dict: Raw API response with weather information
    """
    url = f"https://api.openweathermap.org/data/2.5/weather"
    params = {"lat": lat, "lon": lon, "appid": api_key}
    
    for attempt in range(retries + 1):
        try:
            logger.info(f"Fetching weather data (attempt {attempt + 1}/{retries + 1})...")
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            logger.info(f"✓ Weather data fetched: temp={data.get('main', {}).get('temp', 'N/A')}K")
            return data
        except requests.exceptions.RequestException as e:
            logger.warning(f"✗ Weather API call failed: {e}")
            if attempt < retries:
                logger.info(f"Retrying in 5 seconds...")
                time.sleep(5)
            else:
                logger.error("✗ Weather API failed after retries. Continuing with None.")
                return None


# ============================================================================
# STEP 2: COMPUTE FEATURES
# ============================================================================

def extract_pollutant_features(pollution_data):
    """Extract pollutant features from raw air pollution response."""
    if pollution_data is None or "list" not in pollution_data:
        logger.warning("No pollution data available, using NaN for pollutants")
        return {
            "co": np.nan, "no": np.nan, "no2": np.nan, "o3": np.nan,
            "so2": np.nan, "pm2_5": np.nan, "pm10": np.nan, "nh3": np.nan,
            "aqi_openweather": np.nan
        }
    
    # Extract from list[0] which is the current reading
    pollution_list = pollution_data["list"][0]
    components = pollution_list.get("components", {})
    main = pollution_list.get("main", {})
    
    features = {
        "co": components.get("co", np.nan),
        "no": components.get("no", np.nan),
        "no2": components.get("no2", np.nan),
        "o3": components.get("o3", np.nan),
        "so2": components.get("so2", np.nan),
        # Support both possible keys from API: 'pm2_5' or 'pm2.5'
        "pm2_5": components.get("pm2_5", components.get("pm2.5", np.nan)),
        "pm10": components.get("pm10", np.nan),
        "nh3": components.get("nh3", np.nan),
        "aqi_openweather": main.get("aqi", np.nan)
    }
    
    logger.info(f"✓ Extracted pollutants: PM2.5={features['pm2_5']:.1f}, AQI={features['aqi_openweather']}")
    return features


def extract_weather_features(weather_data):
    """Extract weather features from raw weather response."""
    if weather_data is None:
        logger.warning("No weather data available, using NaN for weather features")
        return {
            "temperature": np.nan, "humidity": np.nan, "wind_speed": np.nan,
            "wind_deg": np.nan, "pressure": np.nan, "visibility": np.nan,
            "weather_main": "Unknown"
        }
    
    main = weather_data.get("main", {})
    wind = weather_data.get("wind", {})
    weather = weather_data.get("weather", [{}])[0]
    
    # Convert temperature from Kelvin to Celsius
    temp_k = main.get("temp", np.nan)
    temperature = temp_k - 273.15 if not np.isnan(temp_k) else np.nan
    
    # Visibility: cap at 10000 if missing
    visibility = weather_data.get("visibility", 10000)
    if visibility is None:
        visibility = 10000
    
    features = {
        "temperature": temperature,
        "humidity": main.get("humidity", np.nan),
        "wind_speed": wind.get("speed", np.nan),
        "wind_deg": wind.get("deg", np.nan),
        "pressure": main.get("pressure", np.nan),
        "visibility": visibility,
        "weather_main": weather.get("main", "Unknown")
    }
    
    logger.info(f"✓ Extracted weather: temp={features['temperature']:.1f}°C, humidity={features['humidity']:.0f}%")
    return features


def extract_time_based_features(timestamp=None):
    """Extract time-based features from current datetime."""
    if timestamp is None:
        timestamp = datetime.utcnow()
    
    # Ensure timezone-naive for easier processing
    if timestamp.tzinfo is not None:
        timestamp = timestamp.replace(tzinfo=None)
    
    hour = timestamp.hour
    day_of_week = timestamp.weekday()  # 0=Monday, 6=Sunday
    day_of_month = timestamp.day
    month = timestamp.month
    is_weekend = 1 if day_of_week >= 5 else 0  # Saturday=5, Sunday=6
    is_rush_hour = 1 if hour in range(7, 10) or hour in range(17, 21) else 0
    
    # Determine season (Karachi, Pakistan)
    # Winter: Dec-Feb (0), Spring: Mar-May (1), Summer: Jun-Aug (2), Autumn: Sep-Nov (3)
    if month in [12, 1, 2]:
        season = 0
    elif month in [3, 4, 5]:
        season = 1
    elif month in [6, 7, 8]:
        season = 2
    else:
        season = 3
    
    features = {
        "hour": hour,
        "day_of_week": day_of_week,
        "day_of_month": day_of_month,
        "month": month,
        "is_weekend": is_weekend,
        "is_rush_hour": is_rush_hour,
        "season": season
    }
    
    logger.info(f"✓ Extracted time features: hour={hour}, day={day_of_week}, season={season}")
    return features


def load_previous_data():
    """Load previous feature data from CSV to compute change rates and rolling averages."""
    if not CSV_FILE.exists():
        logger.info("No previous data found; initializing new feature store")
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(CSV_FILE)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp")
        logger.info(f"✓ Loaded {len(df)} previous records from {CSV_FILE}")
        return df
    except Exception as e:
        logger.warning(f"Could not load previous data: {e}")
        return pd.DataFrame()


def compute_derived_features(current_pm2_5, current_aqi, previous_df):
    """
    Compute derived/engineered features based on current and historical data.
    
    Returns:
        dict: Derived features including change rates and rolling averages
    """
    if len(previous_df) == 0:
        # First run: no historical data
        logger.info("First run detected; setting change rates to 0 and rolling averages to NaN")
        return {
            "aqi_change_rate": 0.0,
            "pm25_change_rate": 0.0,
            "pm25_rolling_3h": np.nan,
            "pm25_rolling_6h": np.nan,
            "pm25_rolling_24h": np.nan,
            "aqi_rolling_3h": np.nan,
            "pollution_index": np.nan
        }
    
    # Get the most recent record
    last_record = previous_df.iloc[-1]
    last_pm2_5 = last_record.get("pm2_5", np.nan)
    last_aqi = last_record.get("aqi_openweather", np.nan)
    
    # Compute change rates
    aqi_change_rate = current_aqi - last_aqi if not np.isnan(last_aqi) else 0.0
    pm25_change_rate = current_pm2_5 - last_pm2_5 if not np.isnan(last_pm2_5) else 0.0
    
    # Compute rolling averages
    pm25_rolling_3h = previous_df["pm2_5"].tail(3).mean() if len(previous_df) >= 3 else np.nan
    pm25_rolling_6h = previous_df["pm2_5"].tail(6).mean() if len(previous_df) >= 6 else np.nan
    pm25_rolling_24h = previous_df["pm2_5"].tail(24).mean() if len(previous_df) >= 24 else np.nan
    aqi_rolling_3h = previous_df["aqi_openweather"].tail(3).mean() if len(previous_df) >= 3 else np.nan
    
    logger.info(
        f"✓ Computed derived features: "
        f"aqi_change={aqi_change_rate:.1f}, pm25_change={pm25_change_rate:.1f}, "
        f"pm25_3h_avg={pm25_rolling_3h:.1f}"
    )
    
    return {
        "aqi_change_rate": aqi_change_rate,
        "pm25_change_rate": pm25_change_rate,
        "pm25_rolling_3h": pm25_rolling_3h,
        "pm25_rolling_6h": pm25_rolling_6h,
        "pm25_rolling_24h": pm25_rolling_24h,
        "aqi_rolling_3h": aqi_rolling_3h,
        "pollution_index": np.nan  # Will compute below
    }


def compute_pollution_index(co, no2, o3, pm2_5, pm10):
    """
    Compute composite pollution index.
    Formula: (pm2_5 * 0.4) + (pm10 * 0.2) + (no2 * 0.2) + (o3 * 0.2)
    """
    if np.isnan(pm2_5) or np.isnan(pm10) or np.isnan(no2) or np.isnan(o3):
        return np.nan
    
    index = (pm2_5 * 0.4) + (pm10 * 0.2) + (no2 * 0.2) + (o3 * 0.2)
    logger.info(f"✓ Computed pollution index: {index:.2f}")
    return index


def build_feature_row(pollution_data, weather_data, previous_df):
    """
    Build a complete feature row with all required columns.
    
    Returns:
        dict: Single row to be added to feature store
    """
    # Timestamp (UTC ISO format)
    timestamp = datetime.utcnow().isoformat() + "Z"
    timestamp_dt = datetime.utcnow()
    
    # Extract features
    pollutants = extract_pollutant_features(pollution_data)
    weather = extract_weather_features(weather_data)
    time_features = extract_time_based_features(timestamp_dt)
    derived = compute_derived_features(pollutants["pm2_5"], pollutants["aqi_openweather"], previous_df)
    
    # Compute pollution index
    pollution_idx = compute_pollution_index(
        pollutants["co"], pollutants["no2"], pollutants["o3"],
        pollutants["pm2_5"], pollutants["pm10"]
    )
    derived["pollution_index"] = pollution_idx
    
    # Combine all features
    row = {
        # Metadata
        "timestamp": timestamp,
        "city": CITY,
        "lat": LAT,
        "lon": LON,
        "data_source": "openweathermap",
        
        # Pollutants
        **pollutants,
        
        # Weather
        **weather,
        
        # Time-based
        **time_features,
        
        # Derived
        **derived,
        
        # Target (will be filled 24 hours later)
        "aqi_next_24h": np.nan
    }
    
    logger.info(f"✓ Built feature row for timestamp: {timestamp}")
    return row


# ============================================================================
# STEP 3: STORE FEATURES IN FEATURE STORE
# ============================================================================

def store_to_hopsworks(df_row):
    """
    Store feature row to Hopsworks Feature Store.
    
    Falls back to local CSV if Hopsworks is unavailable.
    """
    try:
        import hopsworks
        
        if not HOPSWORKS_API_KEY or HOPSWORKS_API_KEY == "your_hopsworks_api_key_here":
            logger.warning("Hopsworks API key not configured; skipping Hopsworks storage")
            return False
        
        logger.info("Connecting to Hopsworks...")
        project = hopsworks.login(
            host=HOPSWORKS_HOST,
            api_key_value=HOPSWORKS_API_KEY
        )
        
        fs = project.get_feature_store()
        
        logger.info("Creating/getting feature group 'aqi_features_karachi'...")
        fg = fs.get_or_create_feature_group(
            name="aqi_features_karachi",
            version=1,
            primary_key=["timestamp"],
            event_time="timestamp",
            description="Hourly AQI and weather features for Karachi from OpenWeatherMap",
            online_enabled=False,
            stream=False
        )
        
        # Convert dict to DataFrame for insertion
        df_to_insert = pd.DataFrame([df_row])
        
        logger.info("Inserting feature row to Hopsworks...")
        fg.insert(df_to_insert, write_options={"start_offline_materialization": False})
        
        logger.info(f"✓ Successfully stored to Hopsworks at {df_row['timestamp']}")
        return True
    
    except ImportError:
        logger.warning("hopsworks not installed; falling back to local CSV")
        return False
    except Exception as e:
        logger.warning(f"Hopsworks unavailable ({e}); falling back to local CSV")
        return False


def store_to_local_csv(df_row):
    """
    Store feature row to local CSV file.
    
    Appends to existing CSV; creates new if doesn't exist.
    """
    try:
        df = pd.DataFrame([df_row])
        
        # Check if file exists
        file_exists = CSV_FILE.exists()
        
        if file_exists:
            # Append mode
            df.to_csv(CSV_FILE, mode='a', header=False, index=False)
            logger.info(f"✓ Appended row to {CSV_FILE}")
        else:
            # Write with header
            df.to_csv(CSV_FILE, mode='w', header=True, index=False)
            logger.info(f"✓ Created new feature store at {CSV_FILE}")
        
        return True
    except Exception as e:
        logger.error(f"✗ Failed to store to local CSV: {e}")
        return False


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Execute the complete feature pipeline."""
    logger.info("=" * 80)
    logger.info("STARTING AQI FEATURE PIPELINE")
    logger.info("=" * 80)
    
    try:
        # Validate API key
        validate_api_key()
        
        # Step 1: Fetch raw data
        logger.info("\n[STEP 1] Fetching raw data from OpenWeatherMap...")
        pollution_data = fetch_air_pollution_data(LAT, LON, OPENWEATHER_API_KEY)
        weather_data = fetch_weather_data(LAT, LON, OPENWEATHER_API_KEY)
        
        if pollution_data is None or weather_data is None:
            logger.error("✗ Failed to fetch required data; aborting pipeline")
            return False
        
        # Step 2: Compute features
        logger.info("\n[STEP 2] Computing features...")
        previous_df = load_previous_data()
        feature_row = build_feature_row(pollution_data, weather_data, previous_df)
        
        # Step 3: Store features
        logger.info("\n[STEP 3] Storing features...")
        
        # Attempt storage and set backend label
        stored_backend = None
        if store_to_hopsworks(feature_row):
            stored_backend = "Hopsworks"
        elif store_to_local_csv(feature_row):
            stored_backend = "Local CSV"

        if stored_backend is None:
            logger.error("ERROR: Failed to store features to any backend")
            return False

        # Print success confirmation
        logger.info("\n" + "=" * 80)
        logger.info("Feature pipeline completed successfully")
        logger.info(f"  Timestamp : {feature_row['timestamp']}")
        logger.info(f"  AQI       : {int(feature_row['aqi_openweather'])} ({_aqi_label(int(feature_row['aqi_openweather']))})")
        pm25 = feature_row.get("pm2_5", np.nan)
        pm25_str = f"{pm25:.1f} ug/m3" if not np.isnan(pm25) else "NaN"
        logger.info(f"  PM2.5     : {pm25_str}")
        logger.info(f"  Stored to : {stored_backend}")
        logger.info(f"  Next run  : 1 hour from now (via GitHub Actions)")
        logger.info("=" * 80)
        
        return True
    
    except Exception as e:
        logger.exception(f"✗ Pipeline failed with exception: {e}")
        return False


def _aqi_label(aqi_value):
    """Convert AQI value (1-5) to descriptive label."""
    labels = {1: "Good", 2: "Fair", 3: "Moderate", 4: "Poor", 5: "Very Poor"}
    return labels.get(aqi_value, "Unknown")


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
