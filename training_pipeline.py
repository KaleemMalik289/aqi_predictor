"""
Training pipeline for AQI prediction models.
Phase 3: Train Random Forest and XGBoost models on backfilled feature data.

SPECIFICATION:
- Load feature CSV (2100+ rows with targets)
- Split: 80% train, 20% test (time-based to prevent leakage)
- Train: Random Forest + XGBoost for PM2.5 prediction
- Evaluate: RMSE, MAE, R², prediction error distribution
- Save: Best model to models/; summary to logs/
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import json
import joblib
import logging
from pathlib import Path

# Fix Unicode encoding on Windows
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

# Configure logging
def setup_logging():
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"training_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    return logger, log_file

logger, log_file = setup_logging()

def load_feature_data():
    """Load CSV and validate structure."""
    logger.info("="*80)
    logger.info("PHASE 3: TRAINING PIPELINE")
    logger.info("="*80)
    logger.info("\n[STEP 1] Loading feature data...")
    
    csv_file = Path("data/aqi_features_karachi.csv")
    if not csv_file.exists():
        logger.error(f"Feature CSV not found: {csv_file}")
        return None
    
    try:
        df = pd.read_csv(csv_file)
        logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
        
        # Check target column
        if 'aqi_next_24h' not in df.columns:
            logger.error("Missing target column: aqi_next_24h")
            return None
        
        # Filter to rows with valid targets (supervised learning)
        df_train = df[df['aqi_next_24h'].notna()].copy()
        logger.info(f"Rows with valid targets: {len(df_train)}")
        
        if len(df_train) < 100:
            logger.error(f"Insufficient training data: {len(df_train)} rows")
            return None
        
        return df_train
    
    except Exception as e:
        logger.error(f"Failed to load CSV: {e}")
        return None

def select_features(df):
    """Select input features for model."""
    logger.info("\n[STEP 2] Selecting input features...")
    
    feature_groups = {
        "Pollutants (µg/m³)": ["pm2_5", "pm10", "no2", "so2", "o3", "co"],
        "Weather": ["temperature", "humidity", "wind_speed", "wind_deg", "pressure"],
        "Time-based": ["hour", "day_of_week", "month", "is_weekend", "is_rush_hour", "season"],
        "Derived": ["pm25_rolling_3h", "pm25_rolling_6h", "pm25_rolling_24h", "aqi_rolling_3h"]
    }
    
    features = []
    for group_name, cols in feature_groups.items():
        available = [c for c in cols if c in df.columns]
        features.extend(available)
        logger.info(f"  {group_name}: {len(available)}/{len(cols)} available")
    
    logger.info(f"  Total features: {len(features)}")
    return features

def split_data(df, features, test_size=0.2):
    """Time-based split to prevent data leakage."""
    logger.info(f"\n[STEP 3] Splitting data (time-based, {100*(1-test_size):.0f}% train / {100*test_size:.0f}% test)...")
    
    # Sort by timestamp (should already be sorted)
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Time-based split
    split_idx = int(len(df) * (1 - test_size))
    df_train = df[:split_idx].copy()
    df_test = df[split_idx:].copy()
    
    X_train = df_train[features].fillna(df_train[features].mean())
    y_train = df_train['aqi_next_24h']
    X_test = df_test[features].fillna(df_train[features].mean())  # Use train mean for imputation
    y_test = df_test['aqi_next_24h']
    
    logger.info(f"  Train: {len(X_train)} samples")
    logger.info(f"  Test:  {len(X_test)} samples")
    logger.info(f"  Feature matrix shape: {X_train.shape}")
    
    return X_train, X_test, y_train, y_test, df_train, df_test

def train_models(X_train, y_train, X_test, y_test):
    """Train Random Forest and XGBoost models."""
    logger.info("\n[STEP 4] Training models...")
    
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    results = {}
    
    # Random Forest
    logger.info("  Training Random Forest (n_estimators=100)...")
    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    
    y_pred_rf = rf_model.predict(X_test)
    rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
    mae_rf = mean_absolute_error(y_test, y_pred_rf)
    r2_rf = r2_score(y_test, y_pred_rf)
    
    results['RandomForest'] = {
        'model': rf_model,
        'rmse': rmse_rf,
        'mae': mae_rf,
        'r2': r2_rf,
        'predictions': y_pred_rf
    }
    logger.info(f"    RMSE: {rmse_rf:.3f}, MAE: {mae_rf:.3f}, R²: {r2_rf:.3f}")
    
    # XGBoost
    try:
        import xgboost as xgb
        logger.info("  Training XGBoost (n_estimators=100)...")
        xgb_model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1
        )
        xgb_model.fit(X_train, y_train)
        
        y_pred_xgb = xgb_model.predict(X_test)
        rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
        mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
        r2_xgb = r2_score(y_test, y_pred_xgb)
        
        results['XGBoost'] = {
            'model': xgb_model,
            'rmse': rmse_xgb,
            'mae': mae_xgb,
            'r2': r2_xgb,
            'predictions': y_pred_xgb
        }
        logger.info(f"    RMSE: {rmse_xgb:.3f}, MAE: {mae_xgb:.3f}, R²: {r2_xgb:.3f}")
    
    except ImportError:
        logger.warning("  XGBoost not available (install: pip install xgboost)")
    
    return results

def save_models(results, X_test, y_test, df_test):
    """Save best model and analysis."""
    logger.info("\n[STEP 5] Saving models...")
    
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    
    # Select best model
    best_model_name = min(results, key=lambda k: results[k]['rmse'])
    best_result = results[best_model_name]
    best_model = best_result['model']
    
    # Save model
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    model_file = model_dir / f"aqi_best_model_{best_model_name}_{timestamp}.pkl"
    joblib.dump(best_model, model_file)
    logger.info(f"  Saved: {model_file}")
    
    # Save metadata
    metadata = {
        'model_type': best_model_name,
        'timestamp': timestamp,
        'rmse': float(best_result['rmse']),
        'mae': float(best_result['mae']),
        'r2': float(best_result['r2']),
        'train_samples': len(X_test) - len(df_test),
        'test_samples': len(df_test),
        'features_used': len(X_test.columns)
    }
    
    metadata_file = model_dir / f"aqi_model_metadata_{timestamp}.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"  Metadata: {metadata_file}")
    
    return best_model_name, model_file

def evaluate_model(results, y_test, df_test):
    """Generate evaluation report."""
    logger.info("\n[STEP 6] Model Evaluation Report...")
    
    logger.info("\nComparison:")
    logger.info(f"{'Model':<12} {'RMSE':<8} {'MAE':<8} {'R²':<8}")
    logger.info("-" * 40)
    
    for name, result in sorted(results.items(), key=lambda x: x[1]['rmse']):
        logger.info(f"{name:<12} {result['rmse']:<8.3f} {result['mae']:<8.3f} {result['r2']:<8.3f}")
    
    # Prediction error analysis
    best_name = min(results, key=lambda k: results[k]['rmse'])
    y_pred = results[best_name]['predictions']
    errors = np.abs(y_test - y_pred)
    
    logger.info(f"\nPrediction Error Distribution ({best_name}):")
    logger.info(f"  Mean error:   {errors.mean():.3f} AQI points")
    logger.info(f"  Std error:    {errors.std():.3f}")
    logger.info(f"  Min error:    {errors.min():.3f}")
    logger.info(f"  Max error:    {errors.max():.3f}")
    logger.info(f"  Median error: {np.median(errors):.3f}")
    
    # Accuracy by range
    perfect = (errors < 0.5).sum()
    good = ((errors >= 0.5) & (errors < 1.0)).sum()
    fair = ((errors >= 1.0) & (errors < 1.5)).sum()
    poor = (errors >= 1.5).sum()
    
    logger.info(f"\nAccuracy Distribution:")
    logger.info(f"  Perfect (<0.5 AQI):  {perfect:4d} ({100*perfect/len(errors):.1f}%)")
    logger.info(f"  Good (0.5-1.0):      {good:4d} ({100*good/len(errors):.1f}%)")
    logger.info(f"  Fair (1.0-1.5):      {fair:4d} ({100*fair/len(errors):.1f}%)")
    logger.info(f"  Poor (>1.5 AQI):     {poor:4d} ({100*poor/len(errors):.1f}%)")

def main():
    """Main training pipeline."""
    try:
        # Load and validate data
        df = load_feature_data()
        if df is None:
            logger.error("Pipeline aborted: data loading failed")
            return False
        
        # Feature engineering
        features = select_features(df)
        
        # Train/test split
        X_train, X_test, y_train, y_test, df_train, df_test = split_data(df, features)
        
        # Train models
        results = train_models(X_train, y_train, X_test, y_test)
        
        if not results:
            logger.error("Pipeline aborted: model training failed")
            return False
        
        # Save best model
        best_model_name, model_file = save_models(results, X_test, y_test, df_test)
        
        # Evaluation
        evaluate_model(results, y_test, df_test)
        
        logger.info("\n" + "="*80)
        logger.info(f"PHASE 3 TRAINING — COMPLETE")
        logger.info(f"  Best model: {best_model_name}")
        logger.info(f"  Model file: {model_file}")
        logger.info(f"  Log file: {log_file}")
        logger.info("="*80)
        
        return True
    
    except Exception as e:
        logger.error(f"Pipeline error: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
