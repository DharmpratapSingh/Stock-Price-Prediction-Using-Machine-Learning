"""
Model ensemble module for stock price prediction
Implements ensemble methods like voting, stacking, and blending
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
import logging
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from src.models import StockPriceModel, get_model

logger = logging.getLogger(__name__)


class ModelEnsemble:
    """
    Ensemble of multiple models for stock price prediction
    """

    def __init__(
        self,
        models: List[StockPriceModel],
        method: str = 'average',
        weights: Optional[List[float]] = None
    ):
        """
        Initialize ensemble

        Args:
            models: List of trained models
            method: Ensemble method ('average', 'weighted', 'stacking', 'voting')
            weights: Weights for weighted average (optional)
        """
        self.models = models
        self.method = method
        self.weights = weights
        self.meta_model = None
        self.scaler = StandardScaler()

    def predict_average(self, X: pd.DataFrame) -> np.ndarray:
        """
        Average predictions from all models

        Args:
            X: Features

        Returns:
            Averaged predictions
        """
        predictions = []
        for model in self.models:
            pred = model.predict(X)
            predictions.append(pred)

        predictions = np.array(predictions)
        return np.mean(predictions, axis=0)

    def predict_weighted(self, X: pd.DataFrame) -> np.ndarray:
        """
        Weighted average of predictions

        Args:
            X: Features

        Returns:
            Weighted predictions
        """
        if self.weights is None:
            # Equal weights
            self.weights = [1.0 / len(self.models)] * len(self.models)

        predictions = []
        for model in self.models:
            pred = model.predict(X)
            predictions.append(pred)

        predictions = np.array(predictions)
        weights = np.array(self.weights).reshape(-1, 1)

        return np.sum(predictions * weights, axis=0)

    def fit_stacking(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ):
        """
        Fit stacking ensemble using meta-learner

        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features (optional)
            y_val: Validation target (optional)
        """
        logger.info("Fitting stacking ensemble")

        # Generate base model predictions
        base_predictions = []
        for model in self.models:
            pred = model.predict(X_train)
            base_predictions.append(pred)

        base_predictions = np.array(base_predictions).T

        # Train meta-learner
        self.meta_model = LinearRegression()
        self.meta_model.fit(base_predictions, y_train)

        logger.info("Stacking ensemble fitted")

    def predict_stacking(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict using stacking ensemble

        Args:
            X: Features

        Returns:
            Stacked predictions
        """
        if self.meta_model is None:
            raise ValueError("Stacking ensemble not fitted. Call fit_stacking() first.")

        # Get base predictions
        base_predictions = []
        for model in self.models:
            pred = model.predict(X)
            base_predictions.append(pred)

        base_predictions = np.array(base_predictions).T

        # Meta-learner prediction
        return self.meta_model.predict(base_predictions)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make ensemble prediction

        Args:
            X: Features

        Returns:
            Ensemble predictions
        """
        if self.method == 'average':
            return self.predict_average(X)
        elif self.method == 'weighted':
            return self.predict_weighted(X)
        elif self.method == 'stacking':
            return self.predict_stacking(X)
        else:
            raise ValueError(f"Unknown ensemble method: {self.method}")

    def get_model_weights_from_performance(
        self,
        X_val: pd.DataFrame,
        y_val: pd.Series
    ) -> List[float]:
        """
        Calculate model weights based on validation performance

        Args:
            X_val: Validation features
            y_val: Validation target

        Returns:
            List of weights
        """
        from src.evaluation import ModelEvaluator

        performances = []
        for model in self.models:
            pred = model.predict(X_val)
            evaluator = ModelEvaluator(y_val.values, pred)
            metrics = evaluator.calculate_all_metrics()
            # Use R² as performance metric
            performances.append(metrics.get('R2', 0.0))

        # Convert to weights (normalize)
        performances = np.array(performances)
        # Handle negative R²
        performances = np.maximum(performances, 0)
        
        if performances.sum() == 0:
            # Equal weights if all models perform poorly
            return [1.0 / len(self.models)] * len(self.models)

        weights = performances / performances.sum()
        return weights.tolist()

    def set_weights_from_performance(
        self,
        X_val: pd.DataFrame,
        y_val: pd.Series
    ):
        """
        Set weights based on validation performance

        Args:
            X_val: Validation features
            y_val: Validation target
        """
        self.weights = self.get_model_weights_from_performance(X_val, y_val)
        logger.info(f"Set ensemble weights: {self.weights}")


def create_ensemble(
    model_names: List[str],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    method: str = 'average',
    use_tuning: bool = False,
    config: Optional[Dict] = None
) -> ModelEnsemble:
    """
    Create and train an ensemble of models

    Args:
        model_names: List of model names to include
        X_train: Training features
        y_train: Training target
        method: Ensemble method
        use_tuning: Whether to use hyperparameter tuning
        config: Configuration dictionary

    Returns:
        Trained ModelEnsemble
    """
    logger.info(f"Creating ensemble with models: {model_names}")

    models = []
    for model_name in model_names:
        try:
            model = get_model(model_name)
            model.fit(X_train, y_train)
            models.append(model)
            logger.info(f"Trained {model_name} for ensemble")
        except Exception as e:
            logger.error(f"Error training {model_name}: {str(e)}")
            continue

    if len(models) == 0:
        raise ValueError("No models successfully trained for ensemble")

    ensemble = ModelEnsemble(models, method=method)
    logger.info(f"Ensemble created with {len(models)} models")

    return ensemble

