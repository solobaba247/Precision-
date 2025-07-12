# app/helpers.py

import pandas as pd
import pandas_ta as ta
from flask import jsonify
from .ml_logic import fetch_data_via_proxies

def calculate_stop_loss_value(symbol, entry_price, sl_price):
    price_diff = abs(entry_price - sl_price)
    currency_map = {'USD': '$', 'JPY': '¥', 'GBP': '£', 'EUR': '€', 'CHF': 'Fr.'}
    try:
        if "=X" in symbol:
            value = price_diff * 1000
            quote_currency = symbol[3:6]
            currency_symbol = currency_map.get(quote_currency, quote_currency + ' ')
            return f"({currency_symbol}{value:,.2f})"
        elif "-USD" in symbol:
            value = price_diff * 0.01
            return f"(~${value:,.2f})"
        else:
            value = price_diff * 1
            return f"(~${value:,.2f})"
    except Exception:
        return ""

def get_latest_price(symbol):
    if not symbol: return jsonify({"error": "Symbol parameter is required."}), 400
    data = fetch_data_via_proxies(symbol, period='1d', interval='1m')
    if data is None or data.empty: return jsonify({"error": f"Could not fetch latest price for {symbol}."}), 500
    latest_price = data['Close'].iloc[-1]
    return jsonify({"symbol": symbol, "price": latest_price})

def get_technical_indicators(symbol, timeframe):
    if not symbol: return jsonify({"error": "Symbol parameter is required."}), 400
    data = fetch_data_via_proxies(symbol, period='90d', interval=timeframe)
    if data is None or len(data) < 20: return jsonify({"error": f"Could not fetch sufficient historical data for {symbol}."}), 500

    data.ta.rsi(append=True)
    data.ta.macd(append=True)
    data.ta.bbands(append=True)
    latest = data.iloc[-1]
    results = {}

    rsi_val = latest.get('RSI_14')
    if pd.notna(rsi_val):
        summary = f"{rsi_val:.2f}"
        if rsi_val > 70: summary += " (Overbought)"
        elif rsi_val < 30: summary += " (Oversold)"
        else: summary += " (Neutral)"
        results['RSI (14)'] = summary
    if pd.notna(latest.get('MACD_12_26_9')) and pd.notna(latest.get('MACDs_12_26_9')):
        summary = f"MACD: {latest.get('MACD_12_26_9'):.5f}, Signal: {latest.get('MACDs_12_26_9'):.5f}"
        if latest.get('MACD_12_26_9') > latest.get('MACDs_12_26_9'): summary += " (Bullish)"
        else: summary += " (Bearish)"
        results['MACD (12, 26, 9)'] = summary
    if all(pd.notna(latest.get(c)) for c in ['BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0', 'Close']):
        summary = f"Upper: {latest.get('BBU_20_2.0'):.4f}, Middle: {latest.get('BBM_20_2.0'):.4f}, Lower: {latest.get('BBL_20_2.0'):.4f}"
        if latest.get('Close') > latest.get('BBU_20_2.0'): summary += " (Trending Strong Up)"
        elif latest.get('Close') < latest.get('BBL_20_2.0'): summary += " (Trending Strong Down)"
        results['Bollinger Bands (20, 2)'] = summary
    results['Latest Close'] = f"{latest.get('Close'):.5f}"
    return jsonify(results)
