"""
Feature selection module for stock price prediction
Handles correlation analysis, feature importance, and feature selection
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging
from sklearn.feature_selection import (
    SelectKBest, f_regression, mutual_info_regression,
    RFE, SelectFromModel
)
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class FeatureSelector:
    """
    Feature selection for stock prediction models
    """

    def __init__(self, method: str = 'correlation', threshold: float = 0.95):
        """
        Initialize feature selector

        Args:
            method: Selection method ('correlation', 'importance', 'mutual_info', 'rfe', 'model_based')
            threshold: Threshold for correlation removal (for correlation method)
        """
        self.method = method
        self.threshold = threshold
        self.selected_features = []
        self.feature_scores = {}

    def remove_correlated_features(
        self,
        data: pd.DataFrame,
        target_col: str = 'target',
        threshold: float = 0.95
    ) -> List[str]:
        """
        Remove highly correlated features

        Args:
            data: DataFrame with features
            target_col: Target column name
            threshold: Correlation threshold

        Returns:
            List of selected feature names
        """
        logger.info(f"Removing correlated features (threshold: {threshold})")

        # Get feature columns (exclude target and price columns)
        feature_cols = [col for col in data.columns 
                       if col not in [target_col, 'Close', 'Open', 'High', 'Low']]

        if len(feature_cols) == 0:
            return []

        # Calculate correlation matrix
        corr_matrix = data[feature_cols].corr().abs()

        # Find highly correlated pairs
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        # Find features to drop
        to_drop = [column for column in upper_triangle.columns
                  if any(upper_triangle[column] > threshold)]

        selected = [col for col in feature_cols if col not in to_drop]

        logger.info(f"Removed {len(to_drop)} highly correlated features")
        logger.info(f"Selected {len(selected)} features")

        return selected

    def select_by_importance(
        self,
        data: pd.DataFrame,
        target_col: str = 'target',
        top_k: int = 50,
        model: Optional[object] = None
    ) -> List[str]:
        """
        Select features based on importance from a model

        Args:
            data: DataFrame with features
            target_col: Target column name
            top_k: Number of top features to select
            model: Trained model with feature_importances_ (optional)

        Returns:
            List of selected feature names
        """
        logger.info(f"Selecting top {top_k} features by importance")

        feature_cols = [col for col in data.columns 
                       if col not in [target_col, 'Close', 'Open', 'High', 'Low']]

        if len(feature_cols) == 0:
            return []

        X = data[feature_cols]
        y = data[target_col]

        # Train a model if not provided
        if model is None:
            model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            model.fit(X_scaled, y)

        # Get feature importance
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        else:
            # Use mutual information as fallback
            importances = mutual_info_regression(X, y, random_state=42)

        # Create importance DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_cols,
            'importance': importances
        }).sort_values('importance', ascending=False)

        # Select top k
        selected = importance_df.head(top_k)['feature'].tolist()
        self.feature_scores = dict(zip(importance_df['feature'], importance_df['importance']))

        logger.info(f"Selected {len(selected)} features by importance")

        return selected

    def select_by_mutual_info(
        self,
        data: pd.DataFrame,
        target_col: str = 'target',
        top_k: int = 50
    ) -> List[str]:
        """
        Select features using mutual information

        Args:
            data: DataFrame with features
            target_col: Target column name
            top_k: Number of top features to select

        Returns:
            List of selected feature names
        """
        logger.info(f"Selecting top {top_k} features by mutual information")

        feature_cols = [col for col in data.columns 
                       if col not in [target_col, 'Close', 'Open', 'High', 'Low']]

        if len(feature_cols) == 0:
            return []

        X = data[feature_cols]
        y = data[target_col]

        # Calculate mutual information
        mi_scores = mutual_info_regression(X, y, random_state=42)

        # Create scores DataFrame
        scores_df = pd.DataFrame({
            'feature': feature_cols,
            'score': mi_scores
        }).sort_values('score', ascending=False)

        # Select top k
        selected = scores_df.head(top_k)['feature'].tolist()
        self.feature_scores = dict(zip(scores_df['feature'], scores_df['score']))

        logger.info(f"Selected {len(selected)} features by mutual information")

        return selected

    def select_by_rfe(
        self,
        data: pd.DataFrame,
        target_col: str = 'target',
        top_k: int = 50,
        estimator: Optional[object] = None
    ) -> List[str]:
        """
        Select features using Recursive Feature Elimination

        Args:
            data: DataFrame with features
            target_col: Target column name
            top_k: Number of features to select
            estimator: Base estimator (optional)

        Returns:
            List of selected feature names
        """
        logger.info(f"Selecting {top_k} features using RFE")

        feature_cols = [col for col in data.columns 
                       if col not in [target_col, 'Close', 'Open', 'High', 'Low']]

        if len(feature_cols) == 0:
            return []

        X = data[feature_cols]
        y = data[target_col]

        # Use default estimator if not provided
        if estimator is None:
            estimator = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # RFE
        rfe = RFE(estimator=estimator, n_features_to_select=min(top_k, len(feature_cols)))
        rfe.fit(X_scaled, y)

        # Get selected features
        selected = [feature_cols[i] for i in range(len(feature_cols)) if rfe.support_[i]]

        logger.info(f"Selected {len(selected)} features using RFE")

        return selected

    def select_features(
        self,
        data: pd.DataFrame,
        target_col: str = 'target',
        top_k: Optional[int] = None,
        model: Optional[object] = None
    ) -> List[str]:
        """
        Select features using specified method

        Args:
            data: DataFrame with features
            target_col: Target column name
            top_k: Number of features to select (optional)
            model: Trained model (for importance-based selection)

        Returns:
            List of selected feature names
        """
        logger.info(f"Selecting features using method: {self.method}")

        if self.method == 'correlation':
            # First remove highly correlated features
            selected = self.remove_correlated_features(data, target_col, self.threshold)
            
            # If top_k is specified, further reduce by importance
            if top_k and len(selected) > top_k:
                selected = self.select_by_importance(data[selected + [target_col]], 
                                                     target_col, top_k, model)

        elif self.method == 'importance':
            top_k = top_k or 50
            selected = self.select_by_importance(data, target_col, top_k, model)

        elif self.method == 'mutual_info':
            top_k = top_k or 50
            selected = self.select_by_mutual_info(data, target_col, top_k)

        elif self.method == 'rfe':
            top_k = top_k or 50
            selected = self.select_by_rfe(data, target_col, top_k)

        elif self.method == 'model_based':
            # Use SelectFromModel
            feature_cols = [col for col in data.columns 
                           if col not in [target_col, 'Close', 'Open', 'High', 'Low']]
            
            if model is None:
                model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            
            X = data[feature_cols]
            y = data[target_col]
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            model.fit(X_scaled, y)
            
            selector = SelectFromModel(model, prefit=True)
            selected = [feature_cols[i] for i in range(len(feature_cols)) 
                       if selector.get_support()[i]]
            
            logger.info(f"Selected {len(selected)} features using model-based selection")

        else:
            logger.warning(f"Unknown method: {self.method}, returning all features")
            selected = [col for col in data.columns 
                       if col not in [target_col, 'Close', 'Open', 'High', 'Low']]

        self.selected_features = selected
        logger.info(f"Final selection: {len(selected)} features")

        return selected

    def get_correlation_matrix(
        self,
        data: pd.DataFrame,
        feature_cols: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Get correlation matrix for features

        Args:
            data: DataFrame with features
            feature_cols: List of feature columns (optional)

        Returns:
            Correlation matrix DataFrame
        """
        if feature_cols is None:
            feature_cols = [col for col in data.columns 
                          if col not in ['target', 'Close', 'Open', 'High', 'Low']]

        return data[feature_cols].corr()

    def get_feature_scores(self) -> Dict[str, float]:
        """
        Get feature scores from selection

        Returns:
            Dictionary of feature scores
        """
        return self.feature_scores.copy()


def analyze_feature_correlation(
    data: pd.DataFrame,
    target_col: str = 'target',
    top_n: int = 20
) -> pd.DataFrame:
    """
    Analyze correlation between features and target

    Args:
        data: DataFrame with features and target
        target_col: Target column name
        top_n: Number of top features to return

    Returns:
        DataFrame with feature correlations
    """
    feature_cols = [col for col in data.columns 
                   if col not in [target_col, 'Close', 'Open', 'High', 'Low']]

    if target_col not in data.columns:
        logger.warning(f"Target column {target_col} not found")
        return pd.DataFrame()

    correlations = data[feature_cols + [target_col]].corr()[target_col].abs()
    correlations = correlations.drop(target_col).sort_values(ascending=False)

    result = pd.DataFrame({
        'feature': correlations.index,
        'correlation': correlations.values
    }).head(top_n)

    return result

