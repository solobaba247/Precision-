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

    # This is the standard and more robust way to handle Flask configuration.
    app.config['ASSET_CLASSES'] = {
        "Forex": [
            # Majors
            "EURUSD=X", "GBPUSD=X", "USDJPY=X", "USDCHF=X", "AUDUSD=X", "USDCAD=X", "NZDUSD=X",
            # Crosses (EUR)
            "EURGBP=X", "EURJPY=X", "EURAUD=X", "EURCAD=X", "EURCHF=X",
            # Crosses (JPY)
            "GBPJPY=X", "AUDJPY=X", "CADJPY=X", "CHFJPY=X", "NZDJPY=X",
            # Crosses (Other)
            "GBPAUD=X", "GBPCAD=X", "GBPCHF=X",
            # Exotics (Note: data can be less clean than majors)
            "USDZAR=X", "USDMXN=X", "USDTRY=X", "USDSGD=X"
        ],
        "Crypto": [
            # Large Caps
            "BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "BNB-USD", "ADA-USD", "AVAX-USD",
            # Popular Altcoins
            "DOGE-USD", "DOT-USD", "MATIC-USD", "SHIB-USD", "TRX-USD", "LTC-USD", "LINK-USD", "BCH-USD"
        ],
        "Stocks": [
            # Tech
            "AAPL", "GOOGL", "MSFT", "AMZN", "NVDA", "META", "TSLA", "AMD", "INTC", "CRM",
            # Finance
            "JPM", "BAC", "WFC", "GS", "MS", "V", "MA",
            # Consumer & Retail
            "WMT", "COST", "HD", "NKE", "MCD", "KO",
            # Healthcare
            "JNJ", "PFE", "MRNA", "UNH",
            # Energy & Industrial
            "XOM", "CVX", "BA"
        ],
        "Indices": [
            # US
            "^GSPC", # S&P 500
            "^DJI",  # Dow Jones Industrial Average
            "^IXIC", # NASDAQ Composite
            "^RUT",  # Russell 2000
            # Europe
            "^FTSE", # FTSE 100 (London)
            "^GDAXI",# DAX (Germany)
            "^FCHI", # CAC 40 (France)
            # Asia
            "^N225", # Nikkei 225 (Japan)
            "^HSI",  # Hang Seng (Hong Kong)
            # Volatility & Bonds
            "^VIX",   # CBOE Volatility Index
            "^TNX"    # 10-Year Treasury Yield
        ]
    }

    app.config['TIMEFRAMES'] = {
        "1 Hour": "1h",
        "4 Hours": "4h",
        "1 Day": "1d",
        "1 Week": "1wk"
    }

    # --- Load Model Artifacts ---
    print("\n--- Initializing Model Loading ---")
    try:
        model_path = os.path.join(ML_MODELS_FOLDER, 'model.joblib')
        scaler_path = os.path.join(ML_MODELS_FOLDER, 'scaler.joblib')
        features_path = os.path.join(ML_MODELS_FOLDER, 'feature_columns.csv')

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

        app.config['MODELS_LOADED'] = True

    except Exception as e:
        app.model = None
        app.scaler = None
        app.feature_columns = None
        app.config['MODELS_LOADED'] = False
        print(f"🔥🔥🔥 CRITICAL ERROR loading ML artifacts: {e}")

    # Register routes
    with app.app_context():
        from . import routes

    return app
