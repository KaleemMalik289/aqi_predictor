"""
Karachi AQI Predictor — Premium Streamlit Dashboard
====================================================
A glassmorphism dark-theme dashboard that displays live air-quality data for
Karachi, makes 3-day AQI forecasts with a trained RandomForest model, and
visualises historical trends and pollutant breakdowns.
"""

import streamlit as st

# ── Page config MUST be the very first Streamlit call ──────────────────────
st.set_page_config(
    page_title="Karachi AQI Predictor",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded",
)

import os
import glob
import json
import datetime as dt
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import requests
import joblib
from dotenv import load_dotenv

warnings.filterwarnings("ignore")

# ── Resolve base directory (works from any cwd) ───────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                        CONSTANTS & CONFIGURATION                       ║
# ╚══════════════════════════════════════════════════════════════════════════╝

API_KEY = os.getenv("OPENWEATHER_API_KEY", "")
LAT = float(os.getenv("LAT", 24.8607))
LON = float(os.getenv("LON", 67.0011))
CITY = os.getenv("CITY", "Karachi")

AQI_LABELS = {1: "Good", 2: "Fair", 3: "Moderate", 4: "Poor", 5: "Very Poor"}
AQI_COLORS = {
    1: "#00e676",
    2: "#ffeb3b",
    3: "#ff9800",
    4: "#f44336",
    5: "#9c27b0",
}
AQI_EMOJIS = {1: "🟢", 2: "🟡", 3: "🟠", 4: "🔴", 5: "🟣"}
AQI_HEALTH = {
    1: "Air quality is satisfactory. Enjoy outdoor activities!",
    2: "Acceptable quality. Unusually sensitive people should limit prolonged outdoor exertion.",
    3: "Sensitive groups may experience health effects. Reduce prolonged outdoor exertion.",
    4: "⚠️ Health alert: Everyone may experience health effects. Avoid outdoor exertion.",
    5: "🚨 EMERGENCY: Serious health effects for everyone. Stay indoors with air purification.",
}

FEATURE_ORDER = [
    "pm2_5", "pm10", "no2", "so2", "o3", "co",
    "temperature", "humidity", "wind_speed", "wind_deg", "pressure",
    "hour", "day_of_week", "month", "is_weekend", "is_rush_hour",
    "season", "pm25_rolling_3h", "pm25_rolling_6h", "pm25_rolling_24h",
    "aqi_rolling_3h",
]

SEASON_MAP = {12: 1, 1: 1, 2: 1, 3: 2, 4: 2, 5: 2,
              6: 3, 7: 3, 8: 3, 9: 4, 10: 4, 11: 4}

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                          CUSTOM CSS INJECTION                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝

CUSTOM_CSS = """
<style>
    /* ── Google Fonts ───────────────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    /* ── Root Variables ────────────────────────────────────────────── */
    :root {
        --bg-primary:    #0e1117;
        --bg-secondary:  #1a1f2e;
        --bg-card:       rgba(26, 31, 46, 0.65);
        --border-glass:  rgba(255, 255, 255, 0.08);
        --text-primary:  #e8eaed;
        --text-secondary:#9aa0a6;
        --accent-blue:   #7c4dff;
        --accent-cyan:   #00e5ff;
        --gradient-main: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }

    /* ── Global ────────────────────────────────────────────────────── */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif !important;
    }
    .stApp {
        background: var(--bg-primary);
    }

    /* ── Sidebar ───────────────────────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #12162b 0%, #1a1f2e 100%) !important;
        border-right: 1px solid var(--border-glass);
    }

    /* ── Glassmorphism Card ─────────────────────────────────────────── */
    .glass-card {
        background: var(--bg-card);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid var(--border-glass);
        border-radius: 16px;
        padding: 24px;
        transition: transform 0.25s ease, box-shadow 0.25s ease;
    }
    .glass-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 40px rgba(124, 77, 255, 0.15);
    }

    /* ── Metric Card ───────────────────────────────────────────────── */
    .metric-card {
        background: var(--bg-card);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid var(--border-glass);
        border-radius: 16px;
        padding: 20px 18px;
        text-align: center;
        transition: transform 0.25s ease, box-shadow 0.25s ease;
        min-height: 160px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 40px rgba(124, 77, 255, 0.15);
    }
    .metric-label {
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        color: var(--text-secondary);
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        line-height: 1.1;
        margin-bottom: 4px;
    }
    .metric-unit {
        font-size: 0.75rem;
        color: var(--text-secondary);
    }

    /* ── Hero Header ───────────────────────────────────────────────── */
    .hero-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        border-radius: 20px;
        padding: 40px 48px;
        margin-bottom: 24px;
        position: relative;
        overflow: hidden;
    }
    .hero-header::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -20%;
        width: 400px;
        height: 400px;
        background: radial-gradient(circle, rgba(255,255,255,0.08) 0%, transparent 70%);
        border-radius: 50%;
    }
    .hero-title {
        font-size: 2.6rem;
        font-weight: 900;
        color: #fff;
        margin: 0;
        text-shadow: 0 2px 12px rgba(0,0,0,0.2);
    }
    .hero-subtitle {
        font-size: 1.05rem;
        font-weight: 400;
        color: rgba(255,255,255,0.85);
        margin-top: 6px;
    }
    .live-dot {
        display: inline-block;
        width: 10px;
        height: 10px;
        background: #00e676;
        border-radius: 50%;
        margin-right: 8px;
        animation: pulse-dot 1.5s ease-in-out infinite;
        vertical-align: middle;
    }

    /* ── Hazard Alert ──────────────────────────────────────────────── */
    .hazard-alert {
        border-radius: 14px;
        padding: 20px 28px;
        margin-bottom: 24px;
        animation: pulse-alert 2s ease-in-out infinite;
        display: flex;
        align-items: center;
        gap: 16px;
    }
    .hazard-poor {
        background: linear-gradient(135deg, rgba(244,67,54,0.25) 0%, rgba(244,67,54,0.10) 100%);
        border: 1px solid rgba(244,67,54,0.45);
    }
    .hazard-verypoor {
        background: linear-gradient(135deg, rgba(156,39,176,0.30) 0%, rgba(244,67,54,0.15) 100%);
        border: 1px solid rgba(156,39,176,0.55);
    }
    .hazard-icon {
        font-size: 2.2rem;
    }
    .hazard-text h3 {
        margin: 0 0 4px 0;
        font-weight: 700;
        font-size: 1.15rem;
    }
    .hazard-text p {
        margin: 0;
        font-size: 0.9rem;
        color: var(--text-secondary);
    }

    /* ── Forecast Card ─────────────────────────────────────────────── */
    .forecast-card {
        background: var(--bg-card);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid var(--border-glass);
        border-radius: 16px;
        padding: 22px 18px;
        text-align: center;
        transition: transform 0.25s ease, box-shadow 0.25s ease;
    }
    .forecast-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 40px rgba(124, 77, 255, 0.15);
    }
    .forecast-day {
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: var(--text-secondary);
        margin-bottom: 10px;
    }
    .forecast-aqi {
        font-size: 3rem;
        font-weight: 900;
        line-height: 1;
        margin-bottom: 6px;
    }
    .forecast-label {
        font-size: 0.85rem;
        font-weight: 600;
        padding: 4px 14px;
        border-radius: 20px;
        display: inline-block;
        margin-top: 6px;
    }

    /* ── Section Title ─────────────────────────────────────────────── */
    .section-title {
        font-size: 1.35rem;
        font-weight: 700;
        color: var(--text-primary);
        margin-bottom: 18px;
        padding-bottom: 8px;
        border-bottom: 2px solid var(--border-glass);
        display: flex;
        align-items: center;
        gap: 10px;
    }

    /* ── Sidebar Styles ────────────────────────────────────────────── */
    .sidebar-section {
        background: rgba(26, 31, 46, 0.5);
        border: 1px solid var(--border-glass);
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 16px;
    }
    .sidebar-title {
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: var(--text-secondary);
        margin-bottom: 12px;
    }
    .aqi-legend-item {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 5px 0;
        font-size: 0.88rem;
    }
    .aqi-dot {
        width: 12px;
        height: 12px;
        border-radius: 50%;
        display: inline-block;
        flex-shrink: 0;
    }

    /* ── Keyframes ─────────────────────────────────────────────────── */
    @keyframes pulse-dot {
        0%, 100% { opacity: 1; box-shadow: 0 0 0 0 rgba(0,230,118,0.5); }
        50%      { opacity: 0.7; box-shadow: 0 0 0 6px rgba(0,230,118,0); }
    }
    @keyframes pulse-alert {
        0%, 100% { opacity: 1; }
        50%      { opacity: 0.88; }
    }

    /* ── Hide default Streamlit chrome ──────────────────────────────── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header[data-testid="stHeader"] {background: transparent;}

    /* ── Plotly chart container override ────────────────────────────── */
    .stPlotlyChart {
        background: transparent !important;
        border-radius: 16px;
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                            DATA LOADING                                ║
# ╚══════════════════════════════════════════════════════════════════════════╝

@st.cache_resource(show_spinner=False)
def load_model():
    """Load the latest RandomForest model from the models/ directory."""
    pattern = os.path.join(BASE_DIR, "models", "aqi_best_model_RandomForest_*.pkl")
    files = sorted(glob.glob(pattern))
    if not files:
        return None
    return joblib.load(files[-1])


@st.cache_resource(show_spinner=False)
def load_model_metadata():
    """Load model metadata JSON."""
    pattern = os.path.join(BASE_DIR, "models", "aqi_model_metadata_*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        return {}
    with open(files[-1], "r") as f:
        return json.load(f)


@st.cache_data(show_spinner=False, ttl=300)
def load_csv():
    """Load historical CSV data."""
    path = os.path.join(BASE_DIR, "data", "aqi_features_karachi.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601", utc=True)
    return df


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                        OPENWEATHERMAP API CALLS                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝

@st.cache_data(show_spinner=False, ttl=600)
def fetch_current_weather():
    """Fetch current weather from OpenWeatherMap."""
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {"lat": LAT, "lon": LON, "appid": API_KEY, "units": "metric"}
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


@st.cache_data(show_spinner=False, ttl=600)
def fetch_air_pollution():
    """Fetch current air pollution data from OpenWeatherMap."""
    url = "https://api.openweathermap.org/data/2.5/air_pollution"
    params = {"lat": LAT, "lon": LON, "appid": API_KEY}
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                           HELPER FUNCTIONS                             ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def parse_live_data(weather_data, pollution_data):
    """Parse API responses into a flat dict of current conditions."""
    info = {}
    if weather_data:
        main = weather_data.get("main", {})
        wind = weather_data.get("wind", {})
        info["temperature"] = main.get("temp", 0)
        info["humidity"] = main.get("humidity", 0)
        info["pressure"] = main.get("pressure", 1013)
        info["wind_speed"] = wind.get("speed", 0)
        info["wind_deg"] = wind.get("deg", 0)
        info["visibility"] = weather_data.get("visibility", 10000)
        weather_list = weather_data.get("weather", [{}])
        info["weather_main"] = weather_list[0].get("main", "Clear") if weather_list else "Clear"

    if pollution_data:
        plist = pollution_data.get("list", [{}])
        comp = plist[0].get("components", {}) if plist else {}
        pm = plist[0].get("main", {}) if plist else {}
        info["aqi"] = pm.get("aqi", 1)
        info["pm2_5"] = comp.get("pm2_5", 0)
        info["pm10"] = comp.get("pm10", 0)
        info["no2"] = comp.get("no2", 0)
        info["so2"] = comp.get("so2", 0)
        info["o3"] = comp.get("o3", 0)
        info["co"] = comp.get("co", 0)
        info["no"] = comp.get("no", 0)
        info["nh3"] = comp.get("nh3", 0)
    else:
        info.setdefault("aqi", 1)
        info.setdefault("pm2_5", 0)
        info.setdefault("pm10", 0)

    return info


def build_feature_row(live: dict, hour_offset: int = 0) -> pd.DataFrame:
    """Build a single-row DataFrame with the 21 features expected by the model."""
    # Cloud servers run in UTC. Karachi is UTC+5.
    now = dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=5 + hour_offset)
    hour = now.hour
    dow = now.weekday()
    month = now.month
    is_weekend = 1 if dow >= 5 else 0
    is_rush_hour = 1 if hour in (7, 8, 9, 17, 18, 19) else 0
    season = SEASON_MAP.get(month, 1)

    pm25 = live.get("pm2_5", 0)
    aqi_val = live.get("aqi", 1)

    row = {
        "pm2_5": pm25,
        "pm10": live.get("pm10", 0),
        "no2": live.get("no2", 0),
        "so2": live.get("so2", 0),
        "o3": live.get("o3", 0),
        "co": live.get("co", 0),
        "temperature": live.get("temperature", 30),
        "humidity": live.get("humidity", 50),
        "wind_speed": live.get("wind_speed", 3),
        "wind_deg": live.get("wind_deg", 0),
        "pressure": live.get("pressure", 1013),
        "hour": hour,
        "day_of_week": dow,
        "month": month,
        "is_weekend": is_weekend,
        "is_rush_hour": is_rush_hour,
        "season": season,
        "pm25_rolling_3h": pm25,
        "pm25_rolling_6h": pm25,
        "pm25_rolling_24h": pm25,
        "aqi_rolling_3h": float(aqi_val),
    }
    return pd.DataFrame([row], columns=FEATURE_ORDER)


def predict_aqi(model, live: dict, hour_offset: int = 0) -> int:
    """Return a clamped AQI prediction (1-5)."""
    if model is None:
        return live.get("aqi", 1)
    X = build_feature_row(live, hour_offset)
    pred = model.predict(X)[0]
    return int(np.clip(round(pred), 1, 5))


def aqi_color(level: int) -> str:
    return AQI_COLORS.get(level, "#e0e0e0")


def aqi_label(level: int) -> str:
    return AQI_LABELS.get(level, "Unknown")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                              SIDEBAR                                   ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def render_sidebar(metadata: dict, live: dict):
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center; padding:12px 0 4px 0;">
            <span style="font-size:2.4rem;">🌫️</span>
            <h2 style="margin:4px 0 0 0; font-weight:800; background:linear-gradient(135deg,#667eea,#764ba2);
            -webkit-background-clip:text; -webkit-text-fill-color:transparent;">AQI Predictor</h2>
            <p style="color:#9aa0a6; font-size:0.82rem; margin-top:2px;">Karachi Air Quality Intelligence</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # ── Refresh button ─────────────────────────────────────────────
        if st.button("🔄  Refresh Data", width="stretch", type="primary"):
            st.cache_data.clear()
            st.rerun()

        # ── Model Info ─────────────────────────────────────────────────
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-title">📊 Model Information</div>', unsafe_allow_html=True)

        model_type = metadata.get("model_type", "RandomForest")
        rmse = metadata.get("rmse", 0)
        mae = metadata.get("mae", 0)
        r2 = metadata.get("r2", 0)
        n_features = metadata.get("features_used", 21)

        st.markdown(f"""
        | Metric | Value |
        |--------|-------|
        | **Model** | {model_type} |
        | **RMSE** | {rmse:.4f} |
        | **MAE** | {mae:.4f} |
        | **R²** | {r2:.4f} |
        | **Features** | {n_features} |
        """)
        st.markdown('</div>', unsafe_allow_html=True)

        # ── AQI Legend ─────────────────────────────────────────────────
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-title">🎨 AQI Scale Legend</div>', unsafe_allow_html=True)
        for level in range(1, 6):
            st.markdown(
                f'<div class="aqi-legend-item">'
                f'<span class="aqi-dot" style="background:{AQI_COLORS[level]};"></span>'
                f'<span>{level} — {AQI_LABELS[level]}</span></div>',
                unsafe_allow_html=True,
            )
        st.markdown('</div>', unsafe_allow_html=True)

        # ── Location Info ──────────────────────────────────────────────
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-title">📍 Location</div>', unsafe_allow_html=True)
        st.markdown(f"""
        | | |
        |--|--|
        | **City** | {CITY} |
        | **Lat** | {LAT} |
        | **Lon** | {LON} |
        | **Source** | OpenWeatherMap |
        """)
        st.markdown('</div>', unsafe_allow_html=True)

        # ── About ──────────────────────────────────────────────────────
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-title">ℹ️ About</div>', unsafe_allow_html=True)
        st.markdown(
            "<p style='font-size:0.82rem; color:#9aa0a6; line-height:1.55;'>"
            "Real-time air-quality monitoring and ML-powered AQI forecasting "
            "for Karachi, Pakistan. Data sourced from OpenWeatherMap APIs. "
            "Model trained on historical pollutant and meteorological features."
            "</p>",
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                         PANEL RENDERERS                                ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# ── 1. Hero Header ────────────────────────────────────────────────────────

def render_hero():
    # Cloud servers run in UTC. Karachi is UTC+5.
    now = dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=5)
    date_str = now.strftime("%A, %B %d, %Y • %I:%M %p")
    st.markdown(f"""
    <div class="hero-header">
        <div class="hero-title">🌫️ {CITY} Air Quality</div>
        <div class="hero-subtitle">
            <span class="live-dot"></span> Live Monitoring &nbsp;•&nbsp; {date_str}
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── 2. Hazard Alert Banner ───────────────────────────────────────────────

def render_hazard_alert(aqi_level: int):
    if aqi_level < 4:
        return
    cls = "hazard-verypoor" if aqi_level == 5 else "hazard-poor"
    icon = "🚨" if aqi_level == 5 else "⚠️"
    title_text = "HAZARDOUS AIR QUALITY" if aqi_level == 5 else "POOR AIR QUALITY ALERT"
    advice = AQI_HEALTH.get(aqi_level, "")
    st.markdown(f"""
    <div class="hazard-alert {cls}">
        <div class="hazard-icon">{icon}</div>
        <div class="hazard-text">
            <h3 style="color:{aqi_color(aqi_level)};">{title_text} — AQI Level {aqi_level} ({aqi_label(aqi_level)})</h3>
            <p>{advice}</p>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── 3. Current Conditions ────────────────────────────────────────────────

def render_current_conditions(live: dict):
    st.markdown('<div class="section-title">📡 Current Conditions</div>', unsafe_allow_html=True)

    aqi_level = live.get("aqi", 1)
    metrics = [
        ("AQI", aqi_level, aqi_label(aqi_level), aqi_color(aqi_level)),
        ("PM2.5", f"{live.get('pm2_5', 0):.1f}", "µg/m³", "#00e5ff"),
        ("PM10", f"{live.get('pm10', 0):.1f}", "µg/m³", "#7c4dff"),
        ("Temperature", f"{live.get('temperature', 0):.1f}", "°C", "#ff9800"),
        ("Humidity", f"{live.get('humidity', 0):.0f}", "%", "#29b6f6"),
        ("Wind Speed", f"{live.get('wind_speed', 0):.1f}", "m/s", "#66bb6a"),
    ]

    cols = st.columns(6)
    for col, (label, value, unit, color) in zip(cols, metrics):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value" style="color:{color};">{value}</div>
                <div class="metric-unit">{unit}</div>
            </div>
            """, unsafe_allow_html=True)


# ── 4. 3-Day Forecast ───────────────────────────────────────────────────

def render_forecast(model, live: dict):
    st.markdown('<div class="section-title">🔮 3-Day AQI Forecast</div>', unsafe_allow_html=True)

    forecasts = [
        ("Now", 0),
        ("Tomorrow", 24),
        ("Day 2", 48),
        ("Day 3", 72),
    ]

    cols = st.columns(4)
    for col, (label, offset) in zip(cols, forecasts):
        pred = predict_aqi(model, live, offset)
        color = aqi_color(pred)
        tag = aqi_label(pred)
        with col:
            st.markdown(f"""
            <div class="forecast-card">
                <div class="forecast-day">{label}</div>
                <div class="forecast-aqi" style="color:{color};">{pred}</div>
                <div class="forecast-label" style="background:{color}22; color:{color}; border:1px solid {color}44;">
                    {tag}
                </div>
            </div>
            """, unsafe_allow_html=True)


# ── 5. Historical AQI Chart ─────────────────────────────────────────────

def render_historical_chart(df: pd.DataFrame):
    st.markdown('<div class="section-title">📈 7-Day Historical AQI Trend</div>', unsafe_allow_html=True)

    if df.empty:
        st.info("No historical data available.")
        return

    df = df.copy()
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=7)
    week = df[df["timestamp"] >= cutoff].sort_values("timestamp")

    if week.empty:
        week = df.sort_values("timestamp").tail(168)  # fallback: last 168 rows (~7d hourly)

    fig = go.Figure()

    # AQI zone bands
    zones = [
        (0, 1.5, "Good", "rgba(0,230,118,0.08)"),
        (1.5, 2.5, "Fair", "rgba(255,235,59,0.08)"),
        (2.5, 3.5, "Moderate", "rgba(255,152,0,0.08)"),
        (3.5, 4.5, "Poor", "rgba(244,67,54,0.08)"),
        (4.5, 5.5, "Very Poor", "rgba(156,39,176,0.08)"),
    ]
    for y0, y1, name, fill in zones:
        fig.add_hrect(
            y0=y0, y1=y1, fillcolor=fill,
            line_width=0,
            annotation_text=name, annotation_position="top left",
            annotation_font=dict(size=10, color="rgba(255,255,255,0.35)"),
        )

    fig.add_trace(go.Scatter(
        x=week["timestamp"],
        y=week["aqi_openweather"],
        mode="lines+markers",
        line=dict(color="#7c4dff", width=2.5, shape="spline"),
        marker=dict(size=4, color="#7c4dff"),
        name="AQI",
        hovertemplate="<b>%{x|%b %d, %H:%M}</b><br>AQI: %{y}<extra></extra>",
    ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=30, b=20),
        height=370,
        yaxis=dict(
            title="AQI Level",
            range=[0.5, 5.5],
            gridcolor="rgba(255,255,255,0.05)",
            dtick=1,
        ),
        xaxis=dict(
            gridcolor="rgba(255,255,255,0.05)",
            title="",
        ),
        showlegend=False,
        font=dict(family="Inter", color="#e0e0e0"),
        hoverlabel=dict(
            bgcolor="#1a1f2e",
            font_size=13,
            font_family="Inter",
        ),
    )

    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


# ── 6. Pollutant Breakdown ───────────────────────────────────────────────

def render_pollutant_breakdown(live: dict):
    st.markdown('<div class="section-title">🧪 Pollutant Breakdown</div>', unsafe_allow_html=True)

    pollutants = {
        "CO": live.get("co", 0),
        "NO": live.get("no", 0),
        "NO₂": live.get("no2", 0),
        "O₃": live.get("o3", 0),
        "SO₂": live.get("so2", 0),
        "PM2.5": live.get("pm2_5", 0),
        "PM10": live.get("pm10", 0),
        "NH₃": live.get("nh3", 0),
    }

    names = list(pollutants.keys())
    values = list(pollutants.values())

    gradient_colors = [
        "#667eea", "#764ba2", "#f093fb", "#00e5ff",
        "#7c4dff", "#ff6e40", "#ffab40", "#69f0ae",
    ]

    fig = go.Figure(go.Bar(
        x=names,
        y=values,
        marker=dict(
            color=gradient_colors[: len(names)],
            line=dict(width=0),
            cornerradius=6,
        ),
        hovertemplate="<b>%{x}</b><br>%{y:.2f} µg/m³<extra></extra>",
    ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=20, b=20),
        height=360,
        yaxis=dict(title="Concentration (µg/m³)", gridcolor="rgba(255,255,255,0.05)"),
        xaxis=dict(title="", gridcolor="rgba(255,255,255,0.05)"),
        font=dict(family="Inter", color="#e0e0e0"),
        showlegend=False,
        hoverlabel=dict(
            bgcolor="#1a1f2e",
            font_size=13,
            font_family="Inter",
        ),
    )

    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                              MAIN APP                                  ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def main():
    # Load resources
    model = load_model()
    metadata = load_model_metadata()
    df = load_csv()

    # Fetch live data
    weather_data = fetch_current_weather()
    pollution_data = fetch_air_pollution()
    live = parse_live_data(weather_data, pollution_data)

    aqi_level = live.get("aqi", 1)

    # ── Sidebar ────────────────────────────────────────────────────────
    render_sidebar(metadata, live)

    # ── 1. Hero Header ────────────────────────────────────────────────
    render_hero()

    # ── 2. Hazard Alert Banner ────────────────────────────────────────
    render_hazard_alert(aqi_level)

    # ── 3. Current Conditions ─────────────────────────────────────────
    render_current_conditions(live)

    st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)

    # ── 4. 3-Day Forecast ─────────────────────────────────────────────
    render_forecast(model, live)

    st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)

    # ── 5 & 6. Charts side-by-side ────────────────────────────────────
    col_left, col_right = st.columns([1.3, 1])
    with col_left:
        render_historical_chart(df)
    with col_right:
        render_pollutant_breakdown(live)

    # ── Footer ────────────────────────────────────────────────────────
    st.markdown("""
    <div style="text-align:center; padding:32px 0 16px 0; border-top:1px solid rgba(255,255,255,0.06); margin-top:40px;">
        <p style="color:#555; font-size:0.78rem;">
            Built with ❤️ using Streamlit &amp; OpenWeatherMap &nbsp;•&nbsp;
            Karachi AQI Predictor &nbsp;•&nbsp; © 2026
        </p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
