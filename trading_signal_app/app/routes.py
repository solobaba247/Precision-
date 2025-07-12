# app/routes.py

from flask import current_app, render_template, request, jsonify
import pandas as pd
import asyncio
import concurrent.futures
from .ml_logic import get_model_prediction, fetch_yfinance_data
from .helpers import calculate_stop_loss_value, get_latest_price, get_technical_indicators

@current_app.route('/')
def index():
    """Main page route with template variables."""
    return render_template('index.html', 
                         asset_classes=current_app.ASSET_CLASSES,
                         timeframes=current_app.TIMEFRAMES)

@current_app.route('/api/check_model_status')
def check_model_status():
    """Health check endpoint for model loading status."""
    models_loaded = getattr(current_app, 'models_loaded', False)
    
    if models_loaded and current_app.model is not None and current_app.scaler is not None:
        return jsonify({
            "status": "ok", 
            "models_loaded": True,
            "message": "Models are loaded and ready."
        }), 200
    else:
        return jsonify({
            "status": "error", 
            "models_loaded": False,
            "message": "Models failed to load or are not available."
        }), 503

@current_app.route('/api/generate_signal')
def generate_signal_route():
    """Generate trading signal for a single asset."""
    symbol = request.args.get('symbol')
    timeframe = request.args.get('timeframe', '1h')
    
    if not symbol:
        return jsonify({"error": "Symbol parameter is required."}), 400
    
    # Check if models are loaded
    if not getattr(current_app, 'models_loaded', False):
        return jsonify({"error": "Models are not loaded. Please wait for initialization."}), 503
    
    try:
        # Fetch data
        data = fetch_yfinance_data(symbol, period='90d', interval=timeframe)
        
        if data is None or len(data) < 50:
            return jsonify({"error": f"Insufficient data for {symbol}. Need at least 50 data points."}), 400
        
        # Get prediction
        prediction = get_model_prediction(
            data, 
            current_app.model, 
            current_app.scaler, 
            current_app.feature_columns
        )
        
        if "error" in prediction:
            return jsonify(prediction), 500
        
        # Format response
        latest_price = prediction['latest_price']
        signal = prediction['signal']
        confidence = prediction['confidence']
        
        # Calculate entry, stop loss, and take profit
        if signal == "BUY":
            entry_price = latest_price
            stop_loss = latest_price * 0.99  # 1% stop loss
            exit_price = latest_price * 1.02  # 2% take profit
        elif signal == "SELL":
            entry_price = latest_price
            stop_loss = latest_price * 1.01  # 1% stop loss
            exit_price = latest_price * 0.98  # 2% take profit
        else:  # HOLD
            entry_price = latest_price
            stop_loss = latest_price
            exit_price = latest_price
        
        response = {
            "symbol": symbol,
            "signal": signal,
            "confidence": f"{confidence:.2%}",
            "entry_price": f"{entry_price:.5f}",
            "exit_price": f"{exit_price:.5f}",
            "stop_loss": f"{stop_loss:.5f}",
            "stop_loss_value": calculate_stop_loss_value(symbol, entry_price, stop_loss),
            "timestamp": prediction['timestamp']
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({"error": f"Failed to generate signal: {str(e)}"}), 500

def get_prediction_for_symbol_sync(symbol, timeframe, model, scaler, feature_columns):
    """Synchronous version of prediction function for concurrent execution."""
    try:
        # Fetch data
        data = fetch_yfinance_data(symbol, period='90d', interval=timeframe)
        
        if data is None or len(data) < 50:
            return None
        
        # Get prediction
        prediction = get_model_prediction(data, model, scaler, feature_columns)
        
        if "error" in prediction:
            return None
        
        signal = prediction['signal']
        confidence = prediction['confidence']
        latest_price = prediction['latest_price']
        
        # Only return BUY/SELL signals for scanner
        if signal == "HOLD":
            return None
        
        # Calculate prices
        if signal == "BUY":
            entry_price = latest_price
            stop_loss = latest_price * 0.99
            exit_price = latest_price * 1.02
        else:  # SELL
            entry_price = latest_price
            stop_loss = latest_price * 1.01
            exit_price = latest_price * 0.98
        
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
        print(f"Error processing {symbol}: {e}")
        return None

@current_app.route('/api/scan_market', methods=['POST'])
def scan_market_route():
    """Scan multiple assets concurrently for trading signals."""
    try:
        data = request.get_json()
        asset_type = data.get('asset_type')
        timeframe = data.get('timeframe', '1h')
        
        if not asset_type or asset_type not in current_app.ASSET_CLASSES:
            return jsonify({"error": "Invalid asset type"}), 400
        
        # Check if models are loaded
        if not getattr(current_app, 'models_loaded', False):
            return jsonify({"error": "Models are not loaded. Please wait for initialization."}), 503
        
        symbols_to_scan = current_app.ASSET_CLASSES[asset_type]
        
        # Use ThreadPoolExecutor for concurrent processing
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            # Submit all tasks
            future_to_symbol = {
                executor.submit(
                    get_prediction_for_symbol_sync,
                    symbol,
                    timeframe,
                    current_app.model,
                    current_app.scaler,
                    current_app.feature_columns
                ): symbol for symbol in symbols_to_scan
            }
            
            # Collect results
            for future in concurrent.futures.as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    result = future.result(timeout=30)  # 30 second timeout per symbol
                    if result is not None:
                        results.append(result)
                except Exception as e:
                    print(f"Error processing {symbol}: {e}")
                    continue
        
        return jsonify(results)
        
    except Exception as e:
        return jsonify({"error": f"Failed to scan market: {str(e)}"}), 500

@current_app.route('/api/latest_price')
def latest_price_route():
    """Get latest price for a symbol."""
    return get_latest_price(request.args.get('symbol'))

@current_app.route('/api/technical_indicators')
def technical_indicators_route():
    """Get technical indicators for a symbol."""
    symbol = request.args.get('symbol')
    timeframe = request.args.get('timeframe', '1h')
    return get_technical_indicators(symbol, timeframe)
