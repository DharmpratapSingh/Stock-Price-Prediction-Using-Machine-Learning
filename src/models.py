"""
Model implementations for stock price prediction
Includes traditional ML and deep learning models
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
import logging

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import joblib

logger = logging.getLogger(__name__)

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    logger.warning("LightGBM not available")

try:
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Bidirectional
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.optimizers import Adam
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logger.warning("TensorFlow not available")


class StockPriceModel:
    """
    Base class for stock price prediction models
    """

    def __init__(self, model_name: str, model_params: Optional[Dict] = None):
        """
        Initialize model

        Args:
            model_name: Name of the model
            model_params: Model parameters
        """
        self.model_name = model_name
        self.model_params = model_params or {}
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        """
        Fit the model

        Args:
            X_train: Training features
            y_train: Training target
        """
        raise NotImplementedError

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions

        Args:
            X: Features

        Returns:
            Predictions
        """
        raise NotImplementedError

    def save(self, filepath: str):
        """
        Save model to disk

        Args:
            filepath: Path to save model
        """
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'model_name': self.model_name,
            'model_params': self.model_params
        }, filepath)
        logger.info(f"Model saved to {filepath}")

    def load(self, filepath: str):
        """
        Load model from disk

        Args:
            filepath: Path to load model from
        """
        data = joblib.load(filepath)
        self.model = data['model']
        self.scaler = data['scaler']
        self.model_name = data['model_name']
        self.model_params = data['model_params']
        self.is_fitted = True
        logger.info(f"Model loaded from {filepath}")


class LinearRegressionModel(StockPriceModel):
    """
    Linear Regression model
    """

    def __init__(self, model_params: Optional[Dict] = None):
        super().__init__("Linear Regression", model_params)
        self.model = LinearRegression(**self.model_params)

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        X_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_scaled, y_train)
        self.is_fitted = True
        logger.info(f"{self.model_name} fitted")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)


class RandomForestModel(StockPriceModel):
    """
    Random Forest Regressor
    """

    def __init__(self, model_params: Optional[Dict] = None):
        default_params = {
            'n_estimators': 100,
            'max_depth': 20,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42,
            'n_jobs': -1
        }
        params = {**default_params, **(model_params or {})}
        super().__init__("Random Forest", params)
        self.model = RandomForestRegressor(**self.model_params)

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        X_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_scaled, y_train)
        self.is_fitted = True
        logger.info(f"{self.model_name} fitted")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def get_feature_importance(self, feature_names: list) -> pd.DataFrame:
        """
        Get feature importance

        Args:
            feature_names: List of feature names

        Returns:
            DataFrame with feature importance
        """
        importance = pd.DataFrame({
            'feature': feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        return importance


class XGBoostModel(StockPriceModel):
    """
    XGBoost Regressor
    """

    def __init__(self, model_params: Optional[Dict] = None):
        default_params = {
            'n_estimators': 100,
            'max_depth': 7,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1
        }
        params = {**default_params, **(model_params or {})}
        super().__init__("XGBoost", params)
        self.model = XGBRegressor(**self.model_params)

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        X_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_scaled, y_train)
        self.is_fitted = True
        logger.info(f"{self.model_name} fitted")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def get_feature_importance(self, feature_names: list) -> pd.DataFrame:
        """
        Get feature importance

        Args:
            feature_names: List of feature names

        Returns:
            DataFrame with feature importance
        """
        importance = pd.DataFrame({
            'feature': feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        return importance


class LightGBMModel(StockPriceModel):
    """
    LightGBM Regressor
    """

    def __init__(self, model_params: Optional[Dict] = None):
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM is not installed")

        default_params = {
            'n_estimators': 100,
            'max_depth': 7,
            'learning_rate': 0.05,
            'num_leaves': 31,
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1
        }
        params = {**default_params, **(model_params or {})}
        super().__init__("LightGBM", params)
        self.model = lgb.LGBMRegressor(**self.model_params)

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        X_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_scaled, y_train)
        self.is_fitted = True
        logger.info(f"{self.model_name} fitted")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)


class LSTMModel(StockPriceModel):
    """
    LSTM Neural Network for time series prediction
    """

    def __init__(self, model_params: Optional[Dict] = None):
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is not installed")

        default_params = {
            'sequence_length': 60,
            'lstm_units': 50,
            'dropout': 0.2,
            'epochs': 100,
            'batch_size': 32,
            'learning_rate': 0.001
        }
        params = {**default_params, **(model_params or {})}
        super().__init__("LSTM", params)
        self.sequence_length = params['sequence_length']

    def create_sequences(self, X: np.ndarray, y: np.ndarray = None) -> Tuple:
        """
        Create sequences for LSTM

        Args:
            X: Features
            y: Target (optional)

        Returns:
            Tuple of (X_seq, y_seq) or just X_seq
        """
        X_seq = []
        y_seq = [] if y is not None else None

        for i in range(self.sequence_length, len(X)):
            X_seq.append(X[i - self.sequence_length:i])
            if y is not None:
                y_seq.append(y[i])

        X_seq = np.array(X_seq)

        if y is not None:
            y_seq = np.array(y_seq)
            return X_seq, y_seq

        return X_seq

    def build_model(self, input_shape: Tuple) -> Sequential:
        """
        Build LSTM model architecture

        Args:
            input_shape: Input shape (sequence_length, n_features)

        Returns:
            Keras model
        """
        model = Sequential([
            LSTM(
                units=self.model_params['lstm_units'],
                return_sequences=True,
                input_shape=input_shape
            ),
            Dropout(self.model_params['dropout']),
            LSTM(units=self.model_params['lstm_units'] // 2, return_sequences=False),
            Dropout(self.model_params['dropout']),
            Dense(units=25, activation='relu'),
            Dense(units=1)
        ])

        model.compile(
            optimizer=Adam(learning_rate=self.model_params['learning_rate']),
            loss='mse',
            metrics=['mae']
        )

        return model

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        """
        Fit LSTM model

        Args:
            X_train: Training features
            y_train: Training target
        """
        # Scale data
        X_scaled = self.scaler.fit_transform(X_train)
        y_array = y_train.values

        # Create sequences
        X_seq, y_seq = self.create_sequences(X_scaled, y_array)

        # Build model
        self.model = self.build_model((self.sequence_length, X_train.shape[1]))

        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )

        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7
        )

        # Train
        logger.info(f"Training {self.model_name} with {len(X_seq)} sequences")

        self.model.fit(
            X_seq, y_seq,
            epochs=self.model_params['epochs'],
            batch_size=self.model_params['batch_size'],
            validation_split=0.1,
            callbacks=[early_stopping, reduce_lr],
            verbose=0
        )

        self.is_fitted = True
        logger.info(f"{self.model_name} fitted")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions

        Args:
            X: Features

        Returns:
            Predictions
        """
        X_scaled = self.scaler.transform(X)
        X_seq = self.create_sequences(X_scaled)

        predictions = self.model.predict(X_seq, verbose=0)

        # Pad predictions to match original length
        padding = np.full((self.sequence_length,), np.nan)
        predictions = np.concatenate([padding, predictions.flatten()])

        return predictions


class ModelTuner:
    """
    Hyperparameter tuning for models
    """

    def __init__(
        self,
        model_class: type,
        param_grid: Dict,
        cv_splits: int = 5,
        n_iter: int = 20,
        method: str = 'random'
    ):
        """
        Initialize model tuner

        Args:
            model_class: Model class to tune
            param_grid: Parameter grid for search
            cv_splits: Number of CV splits
            n_iter: Number of iterations for random search
            method: Search method ('random' or 'grid')
        """
        self.model_class = model_class
        self.param_grid = param_grid
        self.cv_splits = cv_splits
        self.n_iter = n_iter
        self.method = method
        self.best_params = None
        self.best_model = None

    def tune(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict:
        """
        Perform hyperparameter tuning

        Args:
            X_train: Training features
            y_train: Training target

        Returns:
            Best parameters
        """
        logger.info(f"Starting hyperparameter tuning using {self.method} search")

        # Create base model
        base_model = self.model_class().model

        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=self.cv_splits)

        # Choose search method
        if self.method == 'random':
            search = RandomizedSearchCV(
                base_model,
                self.param_grid,
                n_iter=self.n_iter,
                cv=tscv,
                scoring='neg_mean_squared_error',
                n_jobs=-1,
                random_state=42,
                verbose=1
            )
        else:
            search = GridSearchCV(
                base_model,
                self.param_grid,
                cv=tscv,
                scoring='neg_mean_squared_error',
                n_jobs=-1,
                verbose=1
            )

        # Fit
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)

        search.fit(X_scaled, y_train)

        self.best_params = search.best_params_
        self.best_model = search.best_estimator_

        logger.info(f"Best parameters: {self.best_params}")
        logger.info(f"Best score: {-search.best_score_:.4f}")

        return self.best_params


def get_model(model_name: str, model_params: Optional[Dict] = None) -> StockPriceModel:
    """
    Factory function to get model by name

    Args:
        model_name: Name of the model
        model_params: Model parameters

    Returns:
        Model instance
    """
    models = {
        'linear_regression': LinearRegressionModel,
        'random_forest': RandomForestModel,
        'xgboost': XGBoostModel,
        'lightgbm': LightGBMModel,
        'lstm': LSTMModel
    }

    model_class = models.get(model_name.lower())

    if model_class is None:
        raise ValueError(f"Unknown model: {model_name}")

    return model_class(model_params)
