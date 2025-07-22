# app/ml_logic.py

import pandas as pd
import numpy as np
import requests
import pandas_ta as ta
import warnings

warnings.filterwarnings('ignore')

def fetch_yfinance_data(symbol, period='90d', interval='1h'):
    """Fetches data from the new fuzzy-happiness API.""" # <-- (Optional) Updated docstring
    print(f"--- Starting API fetch for {symbol} ---")
    
    # --- CHANGED: Updated BASE_URL to the new API endpoint ---
    BASE_URL = "https://fuzzy-happiness-wr4569rv69r9hvj6p-8000.app.github.dev/api/v1"
    
    endpoint = ""
    api_symbol_value = symbol

    # --- CHANGED: Updated logic to match the new API's endpoint structure ---
    if "=X" in symbol:
        endpoint = "/forex/ohlc"
        api_symbol_value = symbol.replace('=X', '')
    elif "-USD" in symbol:
        # --- NEW: Added dedicated endpoint for Crypto assets ---
        endpoint = "/crypto/ohlc"
        api_symbol_value = symbol # The API accepts "BTC-USD" directly
    elif symbol.startswith('^'):
        endpoint = "/index/ohlc"
        api_symbol_value = symbol.replace('^', '')
    else: # Default to stock
        endpoint = "/stock/ohlc"
        api_symbol_value = symbol

    # --- CHANGED: Simplified parameters, as the new API consistently uses 'symbol' ---
    params = {
        'symbol': api_symbol_value,
        'period': period, 
        'interval': interval
    }
    url = f"{BASE_URL}{endpoint}"

    try:
        # The timeout is increased to handle potentially slower development servers.
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
            
        # --- Data Cleaning and Standardization (No changes needed below this line) ---
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

# The rest of the file (create_features_for_prediction, get_model_prediction, etc.) remains the same.
# ...
