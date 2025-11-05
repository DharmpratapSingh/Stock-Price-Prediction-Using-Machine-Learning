"""
Main training pipeline for stock price prediction
"""

import argparse
import logging
from pathlib import Path
import pandas as pd
import numpy as np

from src.utils import load_config, setup_logging, time_series_split, save_model, save_results
from src.data_loader import load_stock_data
from src.feature_engineering import FeatureEngineer, create_target_variable
from src.models import get_model
from src.evaluation import ModelEvaluator, compare_models
from src.backtesting import Backtester, WalkForwardBacktester, print_backtest_results
from src.visualize import StockVisualizer

logger = logging.getLogger(__name__)


def main(config_path: str = "config/config.yaml", model_name: str = None):
    """
    Main training function

    Args:
        config_path: Path to configuration file
        model_name: Name of model to train (if None, trains all models)
    """
    # Load configuration
    config = load_config(config_path)

    # Setup logging
    setup_logging(
        log_level=config.get('logging', {}).get('level', 'INFO'),
        log_file=f"{config['paths']['logs_dir']}/training.log"
    )

    logger.info("=" * 80)
    logger.info("STOCK PRICE PREDICTION TRAINING PIPELINE")
    logger.info("=" * 80)

    # Create directories
    Path(config['paths']['data_dir']).mkdir(exist_ok=True)
    Path(config['paths']['models_dir']).mkdir(exist_ok=True)
    Path(config['paths']['results_dir']).mkdir(exist_ok=True)
    Path(config['paths']['logs_dir']).mkdir(exist_ok=True)

    # Load data
    logger.info(f"\n📊 Loading data for {config['data']['symbol']}...")
    data = load_stock_data(
        symbol=config['data']['symbol'],
        start_date=config['data']['start_date'],
        end_date=config['data']['end_date'],
        validate=True,
        clean=True
    )

    logger.info(f"Data loaded: {len(data)} records")
    logger.info(f"Date range: {data.index[0]} to {data.index[-1]}")

    # Feature engineering
    logger.info("\n🔧 Engineering features...")
    feature_engineer = FeatureEngineer(data)
    featured_data = feature_engineer.create_all_features(config['features'])

    logger.info(f"Created {len(featured_data.columns)} features")

    # Create target variable
    logger.info("\n🎯 Creating target variable...")
    final_data = create_target_variable(featured_data, target_type='price', horizon=1)

    logger.info(f"Final dataset shape: {final_data.shape}")

    # Split data (time series split)
    logger.info("\n✂️ Splitting data...")
    train_data, val_data, test_data = time_series_split(
        final_data,
        test_size=config['data']['test_size'],
        validation_size=config['data']['validation_size']
    )

    logger.info(f"Train: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_data)}")

    # Prepare feature and target columns
    feature_cols = [col for col in final_data.columns if col not in ['target', 'Close']]
    target_col = 'target'

    X_train = train_data[feature_cols]
    y_train = train_data[target_col]
    X_val = val_data[feature_cols]
    y_val = val_data[target_col]
    X_test = test_data[feature_cols]
    y_test = test_data[target_col]

    logger.info(f"Number of features: {len(feature_cols)}")

    # Train models
    logger.info("\n🤖 Training models...")

    models_to_train = [model_name] if model_name else ['linear_regression', 'random_forest', 'xgboost']

    trained_models = {}
    results = {}

    for model_type in models_to_train:
        logger.info(f"\n{'='*60}")
        logger.info(f"Training {model_type.upper()}")
        logger.info('='*60)

        try:
            # Get model
            model = get_model(model_type)

            # Train
            logger.info("Training...")
            model.fit(X_train, y_train)

            # Validate
            logger.info("Validating...")
            val_predictions = model.predict(X_val)

            val_evaluator = ModelEvaluator(y_val.values, val_predictions)
            val_metrics = val_evaluator.calculate_all_metrics()

            logger.info(f"Validation R²: {val_metrics['R2']:.4f}")
            logger.info(f"Validation RMSE: {val_metrics['RMSE']:.4f}")

            # Test
            logger.info("Testing...")
            test_predictions = model.predict(X_test)

            test_evaluator = ModelEvaluator(y_test.values, test_predictions)
            test_metrics = test_evaluator.calculate_all_metrics()

            logger.info(f"Test R²: {test_metrics['R2']:.4f}")
            logger.info(f"Test RMSE: {test_metrics['RMSE']:.4f}")

            # Store results
            trained_models[model_type] = model
            results[model_type] = {
                'val_predictions': val_predictions,
                'test_predictions': test_predictions,
                'val_metrics': val_metrics,
                'test_metrics': test_metrics
            }

            # Save model
            model_path = save_model(model, model_type, config['paths']['models_dir'])
            logger.info(f"Model saved to {model_path}")

        except Exception as e:
            logger.error(f"Error training {model_type}: {str(e)}")
            continue

    # Compare models
    if len(results) > 1:
        logger.info("\n" + "="*80)
        logger.info("MODEL COMPARISON")
        logger.info("="*80)

        test_results = {name: (y_test.values, res['test_predictions'])
                       for name, res in results.items()}

        comparison_df = compare_models(test_results)
        print("\n", comparison_df)

        # Find best model
        best_model_name = comparison_df['R2'].idxmax()
        logger.info(f"\n🏆 Best model: {best_model_name}")
    else:
        best_model_name = list(results.keys())[0]

    # Detailed evaluation of best model
    logger.info(f"\n{'='*80}")
    logger.info(f"DETAILED EVALUATION: {best_model_name.upper()}")
    logger.info('='*80)

    best_result = results[best_model_name]
    best_evaluator = ModelEvaluator(
        y_test.values,
        best_result['test_predictions'],
        prices=test_data['Close'].values if 'Close' in test_data.columns else None
    )
    best_evaluator.print_metrics()

    # Backtesting
    logger.info(f"\n{'='*80}")
    logger.info("BACKTESTING")
    logger.info('='*80)

    backtester = Backtester(
        initial_capital=config['backtesting']['initial_capital'],
        commission=config['backtesting']['commission'],
        slippage=config['backtesting']['slippage']
    )

    # Strategy backtest
    strategy_results = backtester.simple_strategy(
        predictions=best_result['test_predictions'],
        actuals=y_test.values,
        dates=test_data.index,
        threshold=0.01
    )

    print_backtest_results(strategy_results)

    # Buy and hold comparison
    buy_hold_results = backtester.buy_and_hold_strategy(
        prices=y_test.values,
        dates=test_data.index
    )

    logger.info("\n📊 Buy & Hold Strategy:")
    print_backtest_results(buy_hold_results)

    # Compare strategies
    comparison = backtester.compare_strategies(
        {'ML Strategy': strategy_results},
        buy_hold_results
    )

    print("\n", comparison)

    # Visualizations
    logger.info("\n📊 Creating visualizations...")

    visualizer = StockVisualizer(
        figsize=tuple(config['visualization']['figure_size']),
        dpi=config['visualization']['dpi']
    )

    # Plot predictions
    visualizer.plot_predictions(
        dates=test_data.index,
        actual=y_test.values,
        predicted=best_result['test_predictions'],
        title=f"{best_model_name.upper()} - Actual vs Predicted",
        save_path=f"{config['paths']['results_dir']}/{best_model_name}_predictions.png"
    )

    # Plot equity curves
    visualizer.plot_equity_curve(
        dates=test_data.index[:len(strategy_results['equity_curve'])],
        equity_curve=strategy_results['equity_curve'],
        benchmark=buy_hold_results['equity_curve'][:len(strategy_results['equity_curve'])],
        title="Backtesting Results",
        save_path=f"{config['paths']['results_dir']}/{best_model_name}_equity_curve.png"
    )

    # Plot feature importance (if available)
    if hasattr(trained_models[best_model_name].model, 'feature_importances_'):
        importance_df = trained_models[best_model_name].get_feature_importance(feature_cols)
        visualizer.plot_feature_importance(
            importance_df,
            top_n=20,
            title=f"{best_model_name.upper()} - Feature Importance",
            save_path=f"{config['paths']['results_dir']}/{best_model_name}_feature_importance.png"
        )

    # Save results
    logger.info("\n💾 Saving results...")

    all_results = {
        'config': config,
        'models': results,
        'best_model': best_model_name,
        'comparison': comparison,
        'feature_cols': feature_cols
    }

    results_path = save_results(all_results, 'training_results', config['paths']['results_dir'])
    logger.info(f"Results saved to {results_path}")

    logger.info("\n" + "="*80)
    logger.info("✅ TRAINING PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train stock price prediction models')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--model', type=str, default=None,
                       help='Specific model to train (default: train all)')

    args = parser.parse_args()

    main(config_path=args.config, model_name=args.model)
