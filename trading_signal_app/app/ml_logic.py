# app/ml_logic.py

import pandas as pd
import numpy as np
import requests
import pandas_ta as ta
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

def fetch_yfinance_data(symbol, period='90d', interval='1h'):
    """Fetches data from the fuzzy-happiness API."""
    print(f"--- Starting API fetch for {symbol} ---")
    
    BASE_URL = "https://fuzzy-happiness-wr4569rv69r9hvj6p-8000.app.github.dev/api/v1"
    
    endpoint = ""
    api_symbol_value = symbol

    if "=X" in symbol:
        endpoint = "/forex/ohlc"
        api_symbol_value = symbol.replace('=X', '')
    elif "-USD" in symbol:
        endpoint = "/crypto/ohlc"
        api_symbol_value = symbol
    elif symbol.startswith('^'):
        endpoint = "/index/ohlc"
        api_symbol_value = symbol.replace('^', '')
    else:
        endpoint = "/stock/ohlc"
        api_symbol_value = symbol

    params = {'symbol': api_symbol_value, 'period': period, 'interval': interval}
    url = f"{BASE_URL}{endpoint}"

    try:
        response = requests.get(url, params=params, timeout=60)
        response.raise_for_status()
        data = response.json()
        
        if isinstance(data, dict) and 'error' in data:
            print(f"   ⚠️ API returned an error for {symbol}: {data['error']}")
            return None
        if not data or not isinstance(data, list):
            print(f"   ⚠️ API returned no data or an invalid format for {symbol}")
            return None
            
        df = pd.DataFrame(data)
        if df.empty:
            print(f"   ⚠️ API returned no data for {symbol}")
            return None
            
        date_col_found = False
        for col_name in ['date', 'Date', 'datetime', 'Datetime', 'timestamp', 'Timestamp']:
            if col_name in df.columns:
                df[col_name] = pd.to_datetime(df[col_name], format='mixed', utc=True)
                df = df.set_index(col_name)
                df.index.name = 'Datetime'
                date_col_found = True
                break
        
        if not date_col_found:
            print(f"   ⚠️ No recognizable date/time column found for {symbol}. Cannot process.")
            return None

        df.columns = df.columns.str.lower()
        df = df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'})
        
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required_cols):
             print(f"   ⚠️ Missing one or more OHLCV columns for {symbol}. Found: {df.columns.tolist()}")
             return None

        if 'Adj Close' in df.columns: df = df.drop('Adj Close', axis=1)
        df = df[required_cols].dropna()
        
        if not df.empty:
            print(f"   ✅ Success with API for {symbol}! Got {len(df)} rows")
            return df
        else:
            print(f"   ⚠️ No data after cleaning for {symbol}")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"   ❌ API fetch failed for {symbol}: {e}")
        return None
    except Exception as e:
        print(f"   ❌ Unexpected error processing API response for {symbol}: {e}")
        return None

def create_features_for_prediction(data, feature_columns_list):
    """
    Creates all necessary features for the model from raw price data.
    This is a realistic implementation based on the feature_columns.csv file.
    """
    df = data.copy()
    if df.empty or len(df) < 20:  # Need enough data for rolling calculations
        return pd.DataFrame()

    try:
        # --- Standard Technical Indicators ---
        df.ta.rsi(length=14, append=True)
        df.ta.ema(length=200, append=True)
        df.ta.atr(length=14, append=True)
        df.rename(columns={'RSI_14': 'rsi_14', 'ATRr_14': 'atr_14'}, inplace=True)
        
        # --- Channel Features (using a 20-period Donchian channel as a proxy) ---
        channel_period = 20
        df['channel_high'] = df['High'].rolling(window=channel_period).max()
        df['channel_low'] = df['Low'].rolling(window=channel_period).min()
        df['channel_mid'] = (df['channel_high'] + df['channel_low']) / 2
        
        # Linear regression for channel slope
        def get_slope(array):
            y = np.array(array)
            x = np.arange(len(y))
            slope, _ = np.polyfit(x, y, 1)
            return slope
        
        df['channel_slope'] = df['channel_mid'].rolling(window=channel_period).apply(get_slope, raw=False)
        df['channel_width'] = df['channel_high'] - df['channel_low']
        df['channel_width_atr'] = df['channel_width'] / df['atr_14'].replace(0, 1)
        df['bars_outside_zone'] = (df['Close'] > df['channel_high']).rolling(10).sum() + \
                                  (df['Close'] < df['channel_low']).rolling(10).sum()
        
        # --- Breakout Features ---
        df['breakout_distance'] = df['Close'] - df['channel_mid']
        df['breakout_distance_norm'] = df['breakout_distance'] / df['atr_14'].replace(0, 1)
        df['breakout_candle_body_ratio'] = abs(df['Close'] - df['Open']) / (df['High'] - df['Low']).replace(0, 1)
        
        # --- Relative Features ---
        df['price_vs_ema200'] = df['Close'] / df['EMA_200'].replace(0, 1)
        volume_ma = df['Volume'].rolling(20, min_periods=1).mean()
        df['volume_ratio'] = df['Volume'] / volume_ma.replace(0, 1)
        
        # --- Time-based Features ---
        df['hour_of_day'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter
        df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
        
        # --- Assumed/Fixed Features (placeholders based on model training) ---
        df['risk_reward_ratio'] = 2.0  # Common default
        df['stop_loss_in_atrs'] = 1.5  # Common default
        df['entry_pos_in_channel_norm'] = (df['Close'] - df['channel_low']) / df['channel_width'].replace(0, 1)

        # --- Historical Lagged Features ---
        channel_dev = (df['Close'] - df['channel_mid']) / df['channel_width'].replace(0, 1)
        for i in range(24):
            df[f'hist_close_channel_dev_t_minus_{i}'] = channel_dev.shift(i)

        # --- Interaction and Composite Features ---
        df['volume_rsi_interaction'] = df['volume_ratio'] * df['rsi_14']
        df['breakout_strength'] = df['breakout_distance_norm'] * df['volume_ratio']
        df['channel_efficiency'] = df['channel_slope'] / df['channel_width_atr'].replace(0, 1)

        # --- Boolean/Categorical Features ---
        df['rsi_overbought'] = (df['rsi_14'] > 70).astype(int)
        df['rsi_oversold'] = (df['rsi_14'] < 30).astype(int)
        df['price_above_ema'] = (df['Close'] > df['EMA_200']).astype(int)
        df['high_risk_trade'] = ((df['rsi_14'] > 75) | (df['rsi_14'] < 25)).astype(int)
        
        # This will be set later, but we create the column here
        df['trade_type_encoded'] = 0
        
        # Ensure all required columns are present, filling any missing ones with 0
        for col in feature_columns_list:
            if col not in df.columns:
                df[col] = 0.0
        
        df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # Keep track of a few key values needed for the response
        required_cols = feature_columns_list + ['Close', 'atr_14']
        available_cols = [col for col in required_cols if col in df.columns]
        
        return df[available_cols]
        
    except Exception as e:
        print(f"Error creating features: {e}")
        return pd.DataFrame()

def get_model_prediction(data, model, scaler, feature_columns):
    """Generates a prediction and returns key data for response formatting."""
    if data is None or data.empty:
        return {"error": "Cannot generate prediction, input data is missing."}
    
    try:
        features_df = create_features_for_prediction(data, feature_columns)
        if features_df.empty:
            return {"error": "Could not create features for prediction."}

        latest_features = features_df.iloc[-1].copy()
        last_price = latest_features['Close']
        latest_atr = latest_features.get('atr_14', 0) # Get latest ATR for dynamic SL/TP

        buy_features = latest_features.copy()
        buy_features['trade_type_encoded'] = 0 # 0 for BUY
        sell_features = latest_features.copy()
        sell_features['trade_type_encoded'] = 1 # 1 for SELL

        buy_df = pd.DataFrame([buy_features])[feature_columns]
        sell_df = pd.DataFrame([sell_features])[feature_columns]

        buy_scaled = scaler.transform(buy_df)
        sell_scaled = scaler.transform(sell_df)
        
        buy_prob = model.predict_proba(buy_scaled)[0][1]
        sell_prob = model.predict_proba(sell_scaled)[0][1]

        confidence_threshold = 0.55
        signal_type = "HOLD"
        confidence = max(buy_prob, sell_prob)

        if buy_prob > sell_prob and buy_prob > confidence_threshold:
            signal_type = "BUY"
            confidence = buy_prob
        elif sell_prob > buy_prob and sell_prob > confidence_threshold:
            signal_type = "SELL"
            confidence = sell_prob
        else:
            confidence = 0.5 # Default confidence for HOLD

        return {
            "signal": signal_type,
            "confidence": confidence,
            "latest_price": last_price,
            "latest_atr": latest_atr, # Pass ATR to the routes
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}

def fetch_data_via_proxies(symbol, period='90d', interval='1h'):
    """Alias for fetch_yfinance_data for backward compatibility in helpers.py."""
    return fetch_yfinance_data(symbol, period, interval)
