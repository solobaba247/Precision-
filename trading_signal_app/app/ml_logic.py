# app/ml_logic.py

import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
import warnings

warnings.filterwarnings('ignore')

def fetch_yfinance_data(symbol, period='90d', interval='1h'):
    """Fetches data directly using the yfinance library."""
    print(f"--- Starting yfinance fetch for {symbol} ---")
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval, auto_adjust=False)
        if df.empty:
            print(f"   ⚠️ yfinance returned no data for {symbol}")
            return None
        
        # Ensure proper column names
        df.columns = df.columns.str.title()
        if 'Adj Close' in df.columns:
            df = df.drop('Adj Close', axis=1)
        
        # Keep only OHLCV data
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
        
        if not df.empty:
            print(f"   ✅ Success with yfinance for {symbol}! Got {len(df)} rows")
            return df
        else:
            print(f"   ⚠️ No data after cleaning for {symbol}")
            return None
            
    except Exception as e:
        print(f"   ❌ yfinance fetch failed for {symbol}: {e}")
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
        # Initialize all features with default values
        df['channel_slope'] = 0.0
        df['channel_width_atr'] = 1.0
        df['bars_outside_zone'] = 0
        df['breakout_distance_norm'] = 0.0
        df['breakout_candle_body_ratio'] = 0.5
        
        # RSI features
        df['rsi_14'] = df.get('RSI_14', 50.0)
        
        # Price vs EMA
        ema_200 = df.get('EMA_200', df['Close'])
        df['price_vs_ema200'] = df['Close'] / ema_200
        
        # Volume features
        volume_ma = df['Volume'].rolling(20, min_periods=1).mean()
        df['volume_ratio'] = df['Volume'] / volume_ma
        
        # Time features
        df['hour_of_day'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter
        df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
        
        # Risk management features
        df['risk_reward_ratio'] = 2.0
        df['stop_loss_in_atrs'] = 1.5
        df['entry_pos_in_channel_norm'] = 0.5
        
        # Historical features (placeholders)
        for i in range(24):
            df[f'hist_close_channel_dev_t_minus_{i}'] = 0.0
        
        # Interaction features
        df['volume_rsi_interaction'] = df['volume_ratio'] * df['rsi_14']
        df['breakout_strength'] = 0.0
        df['channel_efficiency'] = 0.0
        
        # Boolean features
        df['rsi_overbought'] = (df['rsi_14'] > 70).astype(int)
        df['rsi_oversold'] = (df['rsi_14'] < 30).astype(int)
        df['price_above_ema'] = (df['Close'] > ema_200).astype(int)
        df['high_risk_trade'] = 0
        
        # Trade type (will be set during prediction)
        df['trade_type_encoded'] = 0
        
        # Ensure all required columns exist
        for col in feature_columns_list:
            if col not in df.columns:
                df[col] = 0.0
        
        # Fill NaN values
        df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # Return with required columns + Close price
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
        # 1. Create features
        features_df = create_features_for_prediction(data, feature_columns)
        if features_df.empty:
            return {"error": "Could not create features for prediction."}

        latest_features = features_df.iloc[-1].copy()
        last_price = latest_features['Close']

        # 2. Prepare for both BUY and SELL scenarios
        buy_features = latest_features.copy()
        buy_features['trade_type_encoded'] = 0
        sell_features = latest_features.copy()
        sell_features['trade_type_encoded'] = 1

        # Create DataFrames with only the required feature columns
        buy_df = pd.DataFrame([buy_features])[feature_columns]
        sell_df = pd.DataFrame([sell_features])[feature_columns]

        # 3. Scale and Predict
        buy_scaled = scaler.transform(buy_df)
        sell_scaled = scaler.transform(sell_df)
        
        buy_prob = model.predict_proba(buy_scaled)[0][1]
        sell_prob = model.predict_proba(sell_scaled)[0][1]

        # 4. Determine Signal
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
            confidence = 0.5  # Neutral confidence for HOLD

        # 5. Format result
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

# Remove the async functions since we're using ThreadPoolExecutor instead
def fetch_data_via_proxies(symbol, period='90d', interval='1h'):
    """Alias for fetch_yfinance_data for backward compatibility."""
    return fetch_yfinance_data(symbol, period, interval)
