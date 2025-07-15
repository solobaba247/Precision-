# app/ml_logic.py

import pandas as pd
import numpy as np
import requests
import pandas_ta as ta
import warnings

warnings.filterwarnings('ignore')

def fetch_yfinance_data(symbol, period='90d', interval='1h'):
    """Fetches data from the custom finance API instead of yfinance."""
    print(f"--- Starting API fetch for {symbol} ---")
    
    BASE_URL = "https://my-finance-appi.onrender.com/api"
    endpoint = ""
    api_symbol_param = 'symbol' 
    api_symbol_value = symbol

    # Determine the correct endpoint and symbol format based on conventions
    if "=X" in symbol:
        endpoint = "/forex/ohlc"
        api_symbol_param = 'pair'
        api_symbol_value = symbol.replace('=X', '')
    elif "-USD" in symbol:
        endpoint = "/stock/ohlc"
        api_symbol_value = symbol
    elif symbol.startswith('^'):
        endpoint = "/index/ohlc"
        api_symbol_value = symbol.replace('^', '')
    else: # Default to stock
        endpoint = "/stock/ohlc"
        api_symbol_value = symbol

    params = {
        api_symbol_param: api_symbol_value,
        'period': period, 
        'interval': interval
    }
    url = f"{BASE_URL}{endpoint}"

    try:
        response = requests.get(url, params=params, timeout=45)
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
            
        # --- Data Cleaning and Standardization ---
        date_col_found = False
        for col_name in ['date', 'Date', 'datetime', 'Datetime', 'timestamp', 'Timestamp']:
            if col_name in df.columns:
                # --- FINAL FIX: Added format='mixed' to handle inconsistent date strings from the API ---
                df[col_name] = pd.to_datetime(df[col_name], format='mixed', utc=True)
                df = df.set_index(col_name)
                df.index.name = 'Datetime'
                date_col_found = True
                break
        
        if not date_col_found:
            print(f"   ⚠️ No recognizable date/time column found for {symbol}. Cannot process.")
            return None

        # Standardize OHLCV column names to TitleCase
        df.columns = df.columns.str.lower()
        df = df.rename(columns={
            'open': 'Open', 'high': 'High', 'low': 'Low',
            'close': 'Close', 'volume': 'Volume'
        })
        
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required_cols):
             print(f"   ⚠️ Missing one or more OHLCV columns for {symbol}. Found: {df.columns.tolist()}")
             return None

        if 'Adj Close' in df.columns:
            df = df.drop('Adj Close', axis=1)
        
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
    """Creates all necessary features for the model from raw price data."""
    df = data.copy()
    if df.empty:
        return pd.DataFrame()

    try:
        # --- Standard Technical Indicators ---
        df.ta.rsi(length=14, append=True)
        df.ta.ema(length=200, append=True)
        df.ta.atr(length=14, append=True)
        
        # --- Feature Creation (Matching feature_columns.csv) ---
        df['channel_slope'] = 0.0
        df['channel_width_atr'] = 1.0
        df['bars_outside_zone'] = 0
        df['breakout_distance_norm'] = 0.0
        df['breakout_candle_body_ratio'] = 0.5
        df['rsi_14'] = df.get('RSI_14', 50.0)
        ema_200 = df.get('EMA_200', df['Close'])
        df['price_vs_ema200'] = df['Close'] / ema_200
        volume_ma = df['Volume'].rolling(20, min_periods=1).mean()
        df['volume_ratio'] = df['Volume'] / volume_ma
        df['hour_of_day'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter
        df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
        df['risk_reward_ratio'] = 2.0
        df['stop_loss_in_atrs'] = 1.5
        df['entry_pos_in_channel_norm'] = 0.5
        for i in range(24):
            df[f'hist_close_channel_dev_t_minus_{i}'] = 0.0
        df['volume_rsi_interaction'] = df['volume_ratio'] * df['rsi_14']
        df['breakout_strength'] = 0.0
        df['channel_efficiency'] = 0.0
        df['rsi_overbought'] = (df['rsi_14'] > 70).astype(int)
        df['rsi_oversold'] = (df['rsi_14'] < 30).astype(int)
        df['price_above_ema'] = (df['Close'] > ema_200).astype(int)
        df['high_risk_trade'] = 0
        df['trade_type_encoded'] = 0
        
        for col in feature_columns_list:
            if col not in df.columns:
                df[col] = 0.0
        
        df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        required_cols = feature_columns_list + ['Close']
        available_cols = [col for col in required_cols if col in df.columns]
        
        return df[available_cols]
        
    except Exception as e:
        print(f"Error creating features: {e}")
        return pd.DataFrame()

def get_model_prediction(data, model, scaler, feature_columns):
    """Generates a prediction for a single asset."""
    if data is None or data.empty:
        return {"error": "Cannot generate prediction, input data is missing."}
    
    try:
        features_df = create_features_for_prediction(data, feature_columns)
        if features_df.empty:
            return {"error": "Could not create features for prediction."}

        latest_features = features_df.iloc[-1].copy()
        last_price = latest_features['Close']

        buy_features = latest_features.copy()
        buy_features['trade_type_encoded'] = 0
        sell_features = latest_features.copy()
        sell_features['trade_type_encoded'] = 1

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
            confidence = 0.5

        return {
            "signal": signal_type,
            "confidence": confidence,
            "latest_price": last_price,
            "buy_prob": buy_prob,
            "sell_prob": sell_prob,
            "timestamp": pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}

def fetch_data_via_proxies(symbol, period='90d', interval='1h'):
    """Alias for fetch_yfinance_data for backward compatibility."""
    return fetch_yfinance_data(symbol, period, interval)
