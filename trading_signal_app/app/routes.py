# app/routes.py

from flask import current_app, render_template, request, jsonify
import pandas as pd
import concurrent.futures
from .ml_logic import get_model_prediction, fetch_fmp_data
from .helpers import calculate_stop_loss_value, get_latest_price, get_technical_indicators

def _get_and_format_signal(symbol, timeframe):
    """
    Internal helper to fetch data, generate a prediction, and format the full response.
    This consolidates logic for both single asset and market scan routes.
    """
    try:
        data = fetch_fmp_data(symbol, period='90d', interval=timeframe)
        if data is None or len(data) < 50:
            return {"error": f"Insufficient data for {symbol}. Need at least 50 data points."}

        prediction = get_model_prediction(
            data,
            current_app.model,
            current_app.scaler,
            current_app.feature_columns
        )
        if "error" in prediction:
            return prediction

        signal = prediction['signal']
        confidence = prediction['confidence']
        latest_price = prediction['latest_price']
        latest_atr = prediction['latest_atr']
        
        atr_multiplier_sl = 1.5
        atr_multiplier_tp = 3.0
        
        if signal == "BUY":
            entry_price = latest_price
            stop_loss = latest_price - (atr_multiplier_sl * latest_atr)
            exit_price = latest_price + (atr_multiplier_tp * latest_atr)
        elif signal == "SELL":
            entry_price = latest_price
            stop_loss = latest_price + (atr_multiplier_sl * latest_atr)
            exit_price = latest_price - (atr_multiplier_tp * latest_atr)
        else:  # HOLD
            entry_price, stop_loss, exit_price = latest_price, latest_price, latest_price

        return {
            "symbol": symbol,
            "signal": signal,
            "confidence": f"{confidence:.2%}",
            "entry_price": f"{entry_price:.5f}",
            "exit_price": f"{exit_price:.5f}",
            "stop_loss": f"{stop_loss:.5f}",
            "stop_loss_value": calculate_stop_loss_value(symbol, entry_price, stop_loss),
            "timestamp": prediction['timestamp']
        }
    except Exception as e:
        print(f"Error in _get_and_format_signal for {symbol}: {e}")
        return {"error": f"Failed to generate signal for {symbol}: {str(e)}"}


@current_app.route('/')
def index():
    """Main page route with template variables."""
    return render_template('index.html',
                         asset_classes=current_app.config.get('ASSET_CLASSES', {}),
                         timeframes=current_app.config.get('TIMEFRAMES', {}))

@current_app.route('/api/check_model_status')
def check_model_status():
    """Health check endpoint for model loading status."""
    if current_app.config.get('MODELS_LOADED', False):
        return jsonify({"status": "ok", "models_loaded": True, "message": "Models are loaded and ready."}), 200
    else:
        return jsonify({"status": "error", "models_loaded": False, "message": "Models failed to load."}), 503

@current_app.route('/api/generate_signal')
def generate_signal_route():
    """Generate trading signal for a single asset."""
    symbol = request.args.get('symbol')
    timeframe = request.args.get('timeframe', '1h')

    if not symbol:
        return jsonify({"error": "Symbol parameter is required."}), 400
    if not current_app.config.get('MODELS_LOADED', False):
        return jsonify({"error": "Models are not loaded."}), 503

    response = _get_and_format_signal(symbol, timeframe)
    
    if "error" in response:
        return jsonify(response), 500
    
    return jsonify(response)

def get_prediction_for_symbol_sync(symbol, timeframe):
    """Synchronous wrapper for concurrent execution that filters for actionable signals."""
    result = _get_and_format_signal(symbol, timeframe)
    
    if result and "error" not in result and result.get("signal") in ["BUY", "SELL"]:
        return result
    return None

@current_app.route('/api/scan_market', methods=['POST'])
def scan_market_route():
    """Scan multiple assets concurrently for trading signals."""
    try:
        data = request.get_json()
        asset_type = data.get('asset_type')
        timeframe = data.get('timeframe', '1h')
        
        asset_classes = current_app.config.get('ASSET_CLASSES', {})
        if not asset_type or asset_type not in asset_classes:
            return jsonify({"error": "Invalid asset type"}), 400
        
        if not current_app.config.get('MODELS_LOADED', False):
            return jsonify({"error": "Models are not loaded."}), 503
        
        symbols_to_scan = asset_classes[asset_type]
        results = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            future_to_symbol = {
                executor.submit(get_prediction_for_symbol_sync, symbol, timeframe): symbol for symbol in symbols_to_scan
            }
            for future in concurrent.futures.as_completed(future_to_symbol):
                symbol_name = future_to_symbol[future]
                try:
                    result = future.result(timeout=60)
                    if result:
                        results.append(result)
                except concurrent.futures.TimeoutError:
                    print(f"⏰ Timeout processing {symbol_name} after 60 seconds. Skipping.")
                except Exception as e:
                    print(f"Error processing {symbol_name}: {e}")

        results.sort(key=lambda x: float(x['confidence'].strip('%')), reverse=True)
        
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": f"Failed to scan market: {str(e)}"}), 500

@current_app.route('/api/latest_price')
def latest_price_route():
    return get_latest_price(request.args.get('symbol'))

@current_app.route('/api/technical_indicators')
def technical_indicators_route():
    symbol, timeframe = request.args.get('symbol'), request.args.get('timeframe', '1h')
    return get_technical_indicators(symbol, timeframe)
