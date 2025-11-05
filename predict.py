"""
Prediction service for stock price prediction
Makes predictions using trained models
"""

import argparse
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.utils import load_config, setup_logging, load_model
from src.data_loader import load_stock_data
from src.feature_engineering import FeatureEngineer
from src.visualize import StockVisualizer

logger = logging.getLogger(__name__)


def predict(
    model_path: str,
    config_path: str = "config/config.yaml",
    symbol: str = None,
    days_ahead: int = 1,
    plot: bool = True
):
    """
    Make predictions using trained model

    Args:
        model_path: Path to saved model
        config_path: Path to configuration file
        symbol: Stock symbol (overrides config)
        days_ahead: Number of days ahead to predict
        plot: Whether to plot results
    """
    # Load configuration
    config = load_config(config_path)

    # Setup logging
    setup_logging(log_level=config.get('logging', {}).get('level', 'INFO'))

    logger.info("=" * 80)
    logger.info("STOCK PRICE PREDICTION SERVICE")
    logger.info("=" * 80)

    # Load model
    logger.info(f"\n📦 Loading model from {model_path}...")
    model_wrapper = load_model(model_path)

    # Extract model components
    if isinstance(model_wrapper, dict):
        model = model_wrapper.get('model')
        scaler = model_wrapper.get('scaler')
        model_name = model_wrapper.get('model_name', 'Unknown')
    else:
        # If it's a model object directly
        from src.models import get_model
        model_obj = model_wrapper
        model_name = model_obj.model_name if hasattr(model_obj, 'model_name') else 'Unknown'

    logger.info(f"Model loaded: {model_name}")

    # Determine symbol
    symbol = symbol or config['data']['symbol']

    # Load data
    logger.info(f"\n📊 Loading recent data for {symbol}...")

    # Get recent data (last year for feature calculation)
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")

    data = load_stock_data(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        validate=True,
        clean=True
    )

    logger.info(f"Loaded {len(data)} records")
    logger.info(f"Latest date: {data.index[-1]}")

    # Feature engineering
    logger.info("\n🔧 Engineering features...")
    feature_engineer = FeatureEngineer(data)
    featured_data = feature_engineer.create_all_features(config['features'])

    # Get latest data point
    latest_data = featured_data.iloc[-1:]

    # Remove target if it exists
    feature_cols = [col for col in latest_data.columns if col not in ['target', 'Close']]
    X_latest = latest_data[feature_cols]

    logger.info(f"Latest data point: {latest_data.index[0]}")
    logger.info(f"Current price: ${data['Close'].iloc[-1]:.2f}")

    # Make prediction
    logger.info(f"\n🎯 Making prediction for {days_ahead} day(s) ahead...")

    # For single day prediction
    if isinstance(model_wrapper, dict):
        # Manual prediction using saved scaler and model
        X_scaled = scaler.transform(X_latest)
        prediction = model.predict(X_scaled)[0]
    else:
        # Use model object's predict method
        prediction = model_wrapper.predict(X_latest)[0]

    current_price = data['Close'].iloc[-1]
    predicted_price = prediction
    price_change = predicted_price - current_price
    pct_change = (price_change / current_price) * 100

    # Display results
    logger.info("\n" + "=" * 80)
    logger.info("PREDICTION RESULTS")
    logger.info("=" * 80)
    logger.info(f"\n📅 Date: {datetime.now().strftime('%Y-%m-%d')}")
    logger.info(f"📈 Symbol: {symbol}")
    logger.info(f"💰 Current Price: ${current_price:.2f}")
    logger.info(f"🔮 Predicted Price ({days_ahead} day): ${predicted_price:.2f}")
    logger.info(f"📊 Expected Change: ${price_change:.2f} ({pct_change:+.2f}%)")

    if pct_change > 0:
        signal = "🟢 BUY SIGNAL"
    elif pct_change < -0.5:
        signal = "🔴 SELL SIGNAL"
    else:
        signal = "🟡 HOLD SIGNAL"

    logger.info(f"🎯 Trading Signal: {signal}")
    logger.info("=" * 80)

    # Plot recent prices and prediction
    if plot:
        logger.info("\n📊 Creating visualization...")

        visualizer = StockVisualizer()

        # Get last 60 days for plotting
        plot_data = data.tail(60)

        # Create future date
        future_date = plot_data.index[-1] + pd.Timedelta(days=days_ahead)

        # Combine historical and predicted
        dates = plot_data.index.tolist() + [future_date]
        prices = plot_data['Close'].tolist() + [predicted_price]

        # Create figure
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(14, 6))

        # Plot historical
        ax.plot(plot_data.index, plot_data['Close'],
               label='Historical', linewidth=2, color='blue')

        # Plot prediction
        ax.plot([plot_data.index[-1], future_date],
               [plot_data['Close'].iloc[-1], predicted_price],
               label='Predicted', linewidth=2, color='red',
               linestyle='--', marker='o', markersize=8)

        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Price ($)', fontsize=12)
        ax.set_title(f'{symbol} Stock Price Prediction', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    return {
        'symbol': symbol,
        'current_price': current_price,
        'predicted_price': predicted_price,
        'price_change': price_change,
        'pct_change': pct_change,
        'signal': signal,
        'date': datetime.now().strftime('%Y-%m-%d')
    }


def batch_predict(
    model_path: str,
    config_path: str = "config/config.yaml",
    symbols: list = None,
    output_file: str = "predictions.csv"
):
    """
    Make predictions for multiple stocks

    Args:
        model_path: Path to saved model
        config_path: Path to configuration file
        symbols: List of stock symbols
        output_file: Output CSV file path
    """
    if symbols is None:
        symbols = ['NVDA', 'AMD', 'TSM', 'INTC']

    logger.info(f"Making predictions for {len(symbols)} stocks...")

    results = []

    for symbol in symbols:
        try:
            logger.info(f"\nProcessing {symbol}...")
            result = predict(
                model_path=model_path,
                config_path=config_path,
                symbol=symbol,
                plot=False
            )
            results.append(result)

        except Exception as e:
            logger.error(f"Error processing {symbol}: {str(e)}")
            continue

    # Create DataFrame
    df = pd.DataFrame(results)

    # Save to file
    df.to_csv(output_file, index=False)
    logger.info(f"\n📁 Predictions saved to {output_file}")

    # Display results
    print("\n" + "=" * 80)
    print("BATCH PREDICTION RESULTS")
    print("=" * 80)
    print(df.to_string(index=False))
    print("=" * 80)

    return df


def interactive_predict(config_path: str = "config/config.yaml"):
    """
    Interactive prediction mode

    Args:
        config_path: Path to configuration file
    """
    print("=" * 80)
    print("INTERACTIVE STOCK PRICE PREDICTION")
    print("=" * 80)

    # List available models
    config = load_config(config_path)
    models_dir = Path(config['paths']['models_dir'])

    if not models_dir.exists() or not any(models_dir.iterdir()):
        print("\n❌ No trained models found. Please run train.py first.")
        return

    model_files = list(models_dir.glob("*.joblib"))

    print("\n📦 Available models:")
    for i, model_file in enumerate(model_files, 1):
        print(f"  {i}. {model_file.name}")

    # Get user input
    while True:
        try:
            choice = int(input("\nSelect model (number): "))
            if 1 <= choice <= len(model_files):
                model_path = model_files[choice - 1]
                break
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Please enter a valid number.")

    # Get symbol
    symbol = input("\nEnter stock symbol (default: NVDA): ").strip().upper()
    if not symbol:
        symbol = "NVDA"

    # Get days ahead
    days_ahead = input("\nDays ahead to predict (default: 1): ").strip()
    days_ahead = int(days_ahead) if days_ahead else 1

    # Make prediction
    predict(
        model_path=str(model_path),
        config_path=config_path,
        symbol=symbol,
        days_ahead=days_ahead,
        plot=True
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Stock price prediction service')
    parser.add_argument('--model', type=str, help='Path to trained model')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--symbol', type=str, help='Stock symbol')
    parser.add_argument('--days', type=int, default=1,
                       help='Days ahead to predict')
    parser.add_argument('--batch', action='store_true',
                       help='Batch prediction mode')
    parser.add_argument('--symbols', nargs='+',
                       help='List of symbols for batch prediction')
    parser.add_argument('--interactive', action='store_true',
                       help='Interactive mode')
    parser.add_argument('--no-plot', action='store_true',
                       help='Disable plotting')

    args = parser.parse_args()

    if args.interactive:
        interactive_predict(config_path=args.config)
    elif args.batch:
        if not args.model:
            print("Error: --model is required for batch prediction")
        else:
            batch_predict(
                model_path=args.model,
                config_path=args.config,
                symbols=args.symbols
            )
    else:
        if not args.model:
            print("Error: --model is required. Use --interactive for interactive mode.")
        else:
            predict(
                model_path=args.model,
                config_path=args.config,
                symbol=args.symbol,
                days_ahead=args.days,
                plot=not args.no_plot
            )
