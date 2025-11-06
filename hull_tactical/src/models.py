#!/usr/bin/env python3
"""
Model Training Module for Hull Tactical Market Prediction
Ensemble of XGBoost, LightGBM, and Neural Networks
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
import joblib


class MarketPredictor:
    """
    Ensemble model for market return prediction
    """

    def __init__(self, config: Dict = None):
        self.config = config or self._default_config()
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.validation_scores = {}

    def _default_config(self) -> Dict:
        """Default configuration"""
        return {
            'use_xgboost': True,
            'use_lightgbm': True,
            'use_neural_net': True,
            'use_ridge': True,
            'n_folds': 5,
            'random_state': 42,
        }

    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """
        Train ensemble of models
        """
        print("🚀 Training ensemble models...")

        results = {}

        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=self.config['n_folds'])

        # Train each model type
        if self.config['use_xgboost']:
            print("\n  Training XGBoost...")
            self.models['xgboost'], results['xgboost'] = self._train_xgboost(X, y, tscv)

        if self.config['use_lightgbm']:
            print("\n  Training LightGBM...")
            self.models['lightgbm'], results['lightgbm'] = self._train_lightgbm(X, y, tscv)

        if self.config['use_neural_net']:
            print("\n  Training Neural Network...")
            self.models['neural_net'], results['neural_net'] = self._train_neural_net(X, y, tscv)

        if self.config['use_ridge']:
            print("\n  Training Ridge Regression...")
            self.models['ridge'], results['ridge'] = self._train_ridge(X, y, tscv)

        # Calculate ensemble weights based on validation performance
        self._calculate_ensemble_weights(results)

        print("\n✅ Training complete!")
        self._print_results(results)

        return results

    def _train_xgboost(self, X: pd.DataFrame, y: pd.Series, cv) -> Tuple:
        """Train XGBoost model"""
        try:
            import xgboost as xgb
        except ImportError:
            print("    ⚠️  XGBoost not available, skipping...")
            return None, {'rmse': np.inf, 'r2': -np.inf}

        params = {
            'objective': 'reg:squarederror',
            'learning_rate': 0.01,
            'max_depth': 6,
            'min_child_weight': 3,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0.1,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': self.config['random_state'],
            'n_estimators': 1000,
            'early_stopping_rounds': 50,
        }

        cv_scores = []
        for fold, (train_idx, val_idx) in enumerate(cv.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            model = xgb.XGBRegressor(**params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )

            y_pred = model.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            r2 = r2_score(y_val, y_pred)
            cv_scores.append({'rmse': rmse, 'r2': r2})

        # Train final model on all data
        final_model = xgb.XGBRegressor(**params)
        final_model.fit(X, y, verbose=False)

        # Store feature importance
        self.feature_importance['xgboost'] = dict(zip(
            X.columns,
            final_model.feature_importances_
        ))

        avg_scores = {
            'rmse': np.mean([s['rmse'] for s in cv_scores]),
            'r2': np.mean([s['r2'] for s in cv_scores])
        }

        return final_model, avg_scores

    def _train_lightgbm(self, X: pd.DataFrame, y: pd.Series, cv) -> Tuple:
        """Train LightGBM model"""
        try:
            import lightgbm as lgb
        except ImportError:
            print("    ⚠️  LightGBM not available, skipping...")
            return None, {'rmse': np.inf, 'r2': -np.inf}

        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'learning_rate': 0.01,
            'num_leaves': 31,
            'max_depth': 6,
            'min_data_in_leaf': 20,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'lambda_l1': 0.1,
            'lambda_l2': 1.0,
            'random_state': self.config['random_state'],
            'n_estimators': 1000,
            'verbose': -1,
        }

        cv_scores = []
        for fold, (train_idx, val_idx) in enumerate(cv.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            model = lgb.LGBMRegressor(**params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(50, verbose=False)]
            )

            y_pred = model.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            r2 = r2_score(y_val, y_pred)
            cv_scores.append({'rmse': rmse, 'r2': r2})

        # Train final model on all data
        final_model = lgb.LGBMRegressor(**params)
        final_model.fit(X, y)

        # Store feature importance
        self.feature_importance['lightgbm'] = dict(zip(
            X.columns,
            final_model.feature_importances_
        ))

        avg_scores = {
            'rmse': np.mean([s['rmse'] for s in cv_scores]),
            'r2': np.mean([s['r2'] for s in cv_scores])
        }

        return final_model, avg_scores

    def _train_neural_net(self, X: pd.DataFrame, y: pd.Series, cv) -> Tuple:
        """Train Neural Network model"""
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
        except ImportError:
            print("    ⚠️  PyTorch not available, skipping...")
            return None, {'rmse': np.inf, 'r2': -np.inf}

        # Scale features for neural network
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self.scalers['neural_net'] = scaler

        class NeuralNet(nn.Module):
            def __init__(self, input_dim):
                super(NeuralNet, self).__init__()
                self.network = nn.Sequential(
                    nn.Linear(input_dim, 256),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(64, 32),
                    nn.ReLU(),
                    nn.Linear(32, 1)
                )

            def forward(self, x):
                return self.network(x).squeeze()

        cv_scores = []
        for fold, (train_idx, val_idx) in enumerate(cv.split(X)):
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train, y_val = y.iloc[train_idx].values, y.iloc[val_idx].values

            model = NeuralNet(X.shape[1])
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

            # Convert to tensors
            X_train_t = torch.FloatTensor(X_train)
            y_train_t = torch.FloatTensor(y_train)
            X_val_t = torch.FloatTensor(X_val)

            # Training
            epochs = 100
            batch_size = 256
            for epoch in range(epochs):
                model.train()
                for i in range(0, len(X_train_t), batch_size):
                    batch_X = X_train_t[i:i+batch_size]
                    batch_y = y_train_t[i:i+batch_size]

                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()

            # Evaluation
            model.eval()
            with torch.no_grad():
                y_pred = model(X_val_t).numpy()

            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            r2 = r2_score(y_val, y_pred)
            cv_scores.append({'rmse': rmse, 'r2': r2})

        # Train final model on all data
        final_model = NeuralNet(X.shape[1])
        criterion = nn.MSELoss()
        optimizer = optim.Adam(final_model.parameters(), lr=0.001, weight_decay=1e-5)

        X_train_t = torch.FloatTensor(X_scaled)
        y_train_t = torch.FloatTensor(y.values)

        epochs = 100
        batch_size = 256
        for epoch in range(epochs):
            final_model.train()
            for i in range(0, len(X_train_t), batch_size):
                batch_X = X_train_t[i:i+batch_size]
                batch_y = y_train_t[i:i+batch_size]

                optimizer.zero_grad()
                outputs = final_model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

        avg_scores = {
            'rmse': np.mean([s['rmse'] for s in cv_scores]),
            'r2': np.mean([s['r2'] for s in cv_scores])
        }

        return final_model, avg_scores

    def _train_ridge(self, X: pd.DataFrame, y: pd.Series, cv) -> Tuple:
        """Train Ridge regression model"""
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self.scalers['ridge'] = scaler

        cv_scores = []
        for fold, (train_idx, val_idx) in enumerate(cv.split(X)):
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            model = Ridge(alpha=1.0)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            r2 = r2_score(y_val, y_pred)
            cv_scores.append({'rmse': rmse, 'r2': r2})

        # Train final model on all data
        final_model = Ridge(alpha=1.0)
        final_model.fit(X_scaled, y)

        avg_scores = {
            'rmse': np.mean([s['rmse'] for s in cv_scores]),
            'r2': np.mean([s['r2'] for s in cv_scores])
        }

        return final_model, avg_scores

    def _calculate_ensemble_weights(self, results: Dict):
        """Calculate ensemble weights based on validation performance"""
        # Use inverse RMSE as weights
        weights = {}
        total_weight = 0

        for model_name, scores in results.items():
            if scores['rmse'] < np.inf:
                weight = 1 / (scores['rmse'] + 1e-10)
                weights[model_name] = weight
                total_weight += weight

        # Normalize weights
        self.ensemble_weights = {
            name: weight / total_weight
            for name, weight in weights.items()
        }

        print("\n📊 Ensemble Weights:")
        for name, weight in self.ensemble_weights.items():
            print(f"    {name}: {weight:.4f}")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using ensemble
        """
        predictions = []
        weights = []

        for model_name, model in self.models.items():
            if model is None:
                continue

            try:
                # Apply scaling if needed
                if model_name in self.scalers:
                    X_scaled = self.scalers[model_name].transform(X)
                    if model_name == 'neural_net':
                        import torch
                        model.eval()
                        with torch.no_grad():
                            X_tensor = torch.FloatTensor(X_scaled)
                            pred = model(X_tensor).numpy()
                    else:
                        pred = model.predict(X_scaled)
                else:
                    pred = model.predict(X)

                predictions.append(pred)
                weights.append(self.ensemble_weights.get(model_name, 0))
            except Exception as e:
                print(f"    ⚠️  Warning: {model_name} prediction failed: {e}")
                continue

        if not predictions:
            raise ValueError("No models available for prediction")

        # Weighted ensemble
        predictions = np.array(predictions)
        weights = np.array(weights)
        weights = weights / weights.sum()  # Normalize

        ensemble_pred = np.average(predictions, axis=0, weights=weights)

        return ensemble_pred

    def _print_results(self, results: Dict):
        """Print training results"""
        print("\n" + "="*60)
        print("MODEL PERFORMANCE SUMMARY")
        print("="*60)

        for model_name, scores in results.items():
            print(f"\n{model_name.upper()}:")
            print(f"  RMSE: {scores['rmse']:.6f}")
            print(f"  R²:   {scores['r2']:.6f}")

        print("="*60)

    def save(self, filepath: str):
        """Save trained models"""
        joblib.dump({
            'models': self.models,
            'scalers': self.scalers,
            'ensemble_weights': self.ensemble_weights,
            'feature_importance': self.feature_importance,
            'config': self.config
        }, filepath)
        print(f"✅ Models saved to {filepath}")

    @classmethod
    def load(cls, filepath: str):
        """Load trained models"""
        data = joblib.load(filepath)
        predictor = cls(config=data['config'])
        predictor.models = data['models']
        predictor.scalers = data['scalers']
        predictor.ensemble_weights = data['ensemble_weights']
        predictor.feature_importance = data['feature_importance']
        print(f"✅ Models loaded from {filepath}")
        return predictor


if __name__ == "__main__":
    print("Model Training Module")
    print("=" * 50)
    print("Ensemble of XGBoost, LightGBM, Neural Network, and Ridge")
