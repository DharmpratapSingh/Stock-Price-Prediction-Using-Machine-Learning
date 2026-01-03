"""
Main training pipeline for stock price prediction
Enhanced with hyperparameter tuning, feature selection, and ensemble support
"""

import argparse
import logging
from pathlib import Path
import pandas as pd
import numpy as np

from src.utils import load_config, setup_logging, time_series_split, save_model, save_results
from src.data_loader import load_stock_data
from src.feature_engineering import FeatureEngineer, create_target_variable
from src.feature_selection import FeatureSelector, analyze_feature_correlation
from src.models import get_model, ModelTuner
from src.ensemble import ModelEnsemble, create_ensemble
from src.evaluation import ModelEvaluator, compare_models, walk_forward_validation
from src.backtesting import Backtester, WalkForwardBacktester, print_backtest_results
from src.visualize import StockVisualizer

logger = logging.getLogger(__name__)


def main(config_path: str = "config/config.yaml", model_name: str = None):
    """
    Main training function with enhanced features

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
    logger.info("STOCK PRICE PREDICTION TRAINING PIPELINE (ENHANCED)")
    logger.info("=" * 80)

    # Create directories
    Path(config['paths']['data_dir']).mkdir(exist_ok=True)
    Path(config['paths']['models_dir']).mkdir(exist_ok=True)
    Path(config['paths']['results_dir']).mkdir(exist_ok=True)
    Path(config['paths']['logs_dir']).mkdir(exist_ok=True)
    
    # Create cache directory if caching enabled
    if config.get('cache', {}).get('enabled', True):
        Path(config.get('cache', {}).get('cache_dir', 'cache')).mkdir(exist_ok=True)

    # Load data
    logger.info(f"\n📊 Loading data for {config['data']['symbol']}...")
    use_cache = config.get('cache', {}).get('enabled', True)
    data = load_stock_data(
        symbol=config['data']['symbol'],
        start_date=config['data']['start_date'],
        end_date=config['data']['end_date'],
        validate=True,
        clean=True,
        use_cache=use_cache
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

    # Feature selection
    feature_cols = [col for col in final_data.columns if col not in ['target', 'Close', 'Open', 'High', 'Low']]
    
    if config.get('training', {}).get('use_feature_selection', True):
        logger.info("\n🔍 Performing feature selection...")
        selection_config = config.get('feature_selection', {})
        
        if selection_config.get('enabled', True):
            selector = FeatureSelector(
                method=selection_config.get('method', 'correlation'),
                threshold=selection_config.get('correlation_threshold', 0.95)
            )
            
            top_k = selection_config.get('top_k', None)
            selected_features = selector.select_features(
                final_data,
                target_col='target',
                top_k=top_k
            )
            
            if len(selected_features) > 0:
                feature_cols = selected_features
                logger.info(f"Selected {len(feature_cols)} features (from {len([col for col in final_data.columns if col not in ['target', 'Close', 'Open', 'High', 'Low']])} total)")
                
                # Analyze feature correlation with target
                corr_analysis = analyze_feature_correlation(final_data, target_col='target', top_n=20)
                if not corr_analysis.empty:
                    logger.info("\nTop features by correlation with target:")
                    logger.info(corr_analysis.to_string(index=False))
            else:
                logger.warning("Feature selection returned no features, using all features")
        else:
            logger.info("Feature selection disabled in config")
    else:
        logger.info("Feature selection skipped")

    logger.info(f"Final number of features: {len(feature_cols)}")

    # Split data (time series split)
    logger.info("\n✂️ Splitting data...")
    train_data, val_data, test_data = time_series_split(
        final_data,
        test_size=config['data']['test_size'],
        validation_size=config['data']['validation_size']
    )

    logger.info(f"Train: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_data)}")

    # Prepare feature and target columns
    target_col = 'target'

    X_train = train_data[feature_cols]
    y_train = train_data[target_col]
    X_val = val_data[feature_cols]
    y_val = val_data[target_col]
    X_test = test_data[feature_cols]
    y_test = test_data[target_col]

    # Check if using walk-forward validation
    use_walk_forward = config.get('training', {}).get('use_walk_forward', False)
    use_ensemble = config.get('ensemble', {}).get('enabled', False)
    use_tuning = config.get('training', {}).get('use_hyperparameter_tuning', True)

    if use_walk_forward:
        logger.info("\n🔄 Using walk-forward validation...")
        # This is a more realistic but slower approach
        # For now, we'll use regular training but note this option
        logger.info("Walk-forward validation is available but using standard split for this run")
        logger.info("Use WalkForwardBacktester for full walk-forward backtesting")

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
            # Hyperparameter tuning
            if use_tuning and model_type in config.get('models', {}):
                logger.info("🔧 Performing hyperparameter tuning...")
                
                tuning_config = config.get('tuning', {})
                model_config = config['models'].get(model_type, {})
                
                if model_config:
                    # Convert config lists to proper format for sklearn
                    param_grid = {}
                    for param, values in model_config.items():
                        # Handle None values
                        if isinstance(values, list) and None in values:
                            param_grid[param] = [v for v in values if v is not None] + [None]
                        else:
                            param_grid[param] = values
                    
                    # Get model class
                    from src.models import get_model
                    model_class = type(get_model(model_type))
                    
                    # Create tuner
                    tuner = ModelTuner(
                        model_class=model_class,
                        param_grid=param_grid,
                        cv_splits=tuning_config.get('cv_folds', 5),
                        n_iter=tuning_config.get('n_iter', 20),
                        method=tuning_config.get('method', 'random_search').replace('_search', '')
                    )
                    
                    # Tune model
                    model = tuner.tune(X_train, y_train)
                    logger.info(f"Tuning completed. Best params: {tuner.best_params}")
                    # Model is already fitted by tuner
                else:
                    # No tuning config, use default model
                    model = get_model(model_type)
                    model.fit(X_train, y_train)
            else:
                # No tuning, train normally
                model = get_model(model_type)
                logger.info("Training with default parameters...")
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
            logger.error(f"Error training {model_type}: {str(e)}", exc_info=True)
            continue

    if len(results) == 0:
        logger.error("No models were successfully trained. Exiting.")
        return

    # Ensemble model (if enabled)
    if use_ensemble and len(trained_models) > 1:
        logger.info("\n" + "="*80)
        logger.info("CREATING ENSEMBLE MODEL")
        logger.info("="*80)
        
        ensemble_config = config.get('ensemble', {})
        ensemble_method = ensemble_config.get('method', 'average')
        ensemble_model_names = ensemble_config.get('models', list(trained_models.keys()))
        
        # Filter to only include trained models
        ensemble_models = [trained_models[name] for name in ensemble_model_names if name in trained_models]
        
        if len(ensemble_models) > 1:
            ensemble = ModelEnsemble(ensemble_models, method=ensemble_method)
            
            # For weighted or stacking, set weights from validation performance
            if ensemble_method in ['weighted', 'stacking']:
                ensemble.set_weights_from_performance(X_val, y_val)
                if ensemble_method == 'stacking':
                    ensemble.fit_stacking(X_train, y_train, X_val, y_val)
            
            # Evaluate ensemble
            ensemble_val_pred = ensemble.predict(X_val)
            ensemble_test_pred = ensemble.predict(X_test)
            
            ensemble_val_eval = ModelEvaluator(y_val.values, ensemble_val_pred)
            ensemble_test_eval = ModelEvaluator(y_test.values, ensemble_test_pred)
            
            ensemble_val_metrics = ensemble_val_eval.calculate_all_metrics()
            ensemble_test_metrics = ensemble_test_eval.calculate_all_metrics()
            
            logger.info(f"Ensemble Validation R²: {ensemble_val_metrics['R2']:.4f}")
            logger.info(f"Ensemble Test R²: {ensemble_test_metrics['R2']:.4f}")
            
            # Add ensemble to results
            trained_models['ensemble'] = ensemble
            results['ensemble'] = {
                'val_predictions': ensemble_val_pred,
                'test_predictions': ensemble_test_pred,
                'val_metrics': ensemble_val_metrics,
                'test_metrics': ensemble_test_metrics
            }
            
            # Save ensemble
            ensemble_path = save_model(ensemble, 'ensemble', config['paths']['models_dir'])
            logger.info(f"Ensemble saved to {ensemble_path}")

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
    best_model_obj = trained_models[best_model_name]
    if hasattr(best_model_obj, 'get_feature_importance'):
        try:
            importance_df = best_model_obj.get_feature_importance(feature_cols)
            visualizer.plot_feature_importance(
                importance_df,
                top_n=20,
                title=f"{best_model_name.upper()} - Feature Importance",
                save_path=f"{config['paths']['results_dir']}/{best_model_name}_feature_importance.png"
            )
        except Exception as e:
            logger.warning(f"Could not plot feature importance: {str(e)}")

    # Save results
    logger.info("\n💾 Saving results...")

    all_results = {
        'config': config,
        'models': results,
        'best_model': best_model_name,
        'comparison': comparison,
        'feature_cols': feature_cols,
        'n_features_original': len([col for col in final_data.columns if col not in ['target', 'Close', 'Open', 'High', 'Low']]),
        'n_features_selected': len(feature_cols)
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
