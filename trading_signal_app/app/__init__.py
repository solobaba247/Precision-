# app/__init__.py

import os
from flask import Flask
import joblib
import pandas as pd

def create_app():
    app = Flask(__name__)

    # --- Configuration ---
    APP_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(APP_DIR)
    ML_MODELS_FOLDER = os.path.join(PROJECT_ROOT, 'ml_models/')

    # --- Asset Classes Configuration ---
    app.ASSET_CLASSES = {
        'Forex': ['EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'USDCHF=X', 'AUDUSD=X', 'USDCAD=X', 'NZDUSD=X'],
        'Crypto': ['BTC-USD', 'ETH-USD', 'LTC-USD', 'XRP-USD', 'ADA-USD', 'DOT-USD', 'LINK-USD'],
        'Stocks': ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM', 'JNJ', 'V'],
        'Indices': ['^GSPC', '^DJI', '^IXIC', '^RUT', '^VIX', '^TNX']
    }

    # --- Timeframes Configuration ---
    app.TIMEFRAMES = {
        '1 Hour': '1h',
        '4 Hours': '4h',
        '1 Day': '1d',
        '1 Week': '1wk'
    }

    # --- Load Model Artifacts ---
    print("\n--- Initializing Model Loading ---")
    try:
        model_path = os.path.join(ML_MODELS_FOLDER, 'model.joblib')
        scaler_path = os.path.join(ML_MODELS_FOLDER, 'scaler.joblib')
        features_path = os.path.join(ML_MODELS_FOLDER, 'feature_columns.csv')

        # Check if files exist
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
        if not os.path.exists(features_path):
            raise FileNotFoundError(f"Features file not found: {features_path}")

        app.model = joblib.load(model_path)
        print("✅ SUCCESS: Model loaded.")

        app.scaler = joblib.load(scaler_path)
        print("✅ SUCCESS: Scaler loaded.")

        app.feature_columns = pd.read_csv(features_path)['feature_name'].tolist()
        print(f"✅ SUCCESS: Feature columns ({len(app.feature_columns)}) loaded.")

        # Set models loaded flag
        app.models_loaded = True

    except Exception as e:
        app.model = None
        app.scaler = None
        app.feature_columns = None
        app.models_loaded = False
        print(f"🔥🔥🔥 CRITICAL ERROR loading ML artifacts: {e}")

    # Register routes
    with app.app_context():
        from . import routes

    return app
