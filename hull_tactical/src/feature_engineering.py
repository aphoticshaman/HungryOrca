#!/usr/bin/env python3
"""
Feature Engineering Module for Hull Tactical Market Prediction
Advanced feature creation for S&P 500 return prediction
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class FeatureEngineer:
    """
    Comprehensive feature engineering for market prediction
    """

    def __init__(self):
        self.feature_names = []
        self.feature_importance = {}

    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create all features in one pass
        """
        print("🔧 Engineering features...")

        # Make a copy to avoid modifying original
        data = df.copy()

        # 1. Basic features
        data = self._create_lag_features(data)

        # 2. Technical indicators
        data = self._create_technical_indicators(data)

        # 3. Momentum features
        data = self._create_momentum_features(data)

        # 4. Volatility features
        data = self._create_volatility_features(data)

        # 5. Regime features
        data = self._create_regime_features(data)

        # 6. Interaction features
        data = self._create_interaction_features(data)

        # 7. Time-based features
        data = self._create_time_features(data)

        # 8. Statistical features
        data = self._create_statistical_features(data)

        print(f"✅ Created {len(data.columns) - len(df.columns)} new features")

        return data

    def _create_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create lagged features"""
        data = df.copy()

        # Get all feature columns (M*, E*, I*, P*, V*, S*, MOM*, D*)
        feature_cols = [col for col in df.columns
                       if col.startswith(('M', 'E', 'I', 'P', 'V', 'S', 'MOM', 'D'))]

        # Create lags for key features
        for lag in [1, 2, 3, 5, 10, 20]:
            for col in feature_cols[:10]:  # Top 10 features to avoid explosion
                data[f'{col}_lag_{lag}'] = df[col].shift(lag)

        return data

    def _create_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create technical indicators"""
        data = df.copy()

        # Use forward_returns as a proxy for price (if available in train)
        if 'forward_returns' in df.columns:
            price_proxy = (1 + df['forward_returns'].fillna(0)).cumprod()

            # Moving averages
            for window in [5, 10, 20, 50, 200]:
                data[f'SMA_{window}'] = price_proxy.rolling(window).mean()
                data[f'EMA_{window}'] = price_proxy.ewm(span=window).mean()

            # RSI - Relative Strength Index
            delta = price_proxy.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            data['RSI_14'] = 100 - (100 / (1 + rs))

            # MACD
            exp1 = price_proxy.ewm(span=12).mean()
            exp2 = price_proxy.ewm(span=26).mean()
            data['MACD'] = exp1 - exp2
            data['MACD_signal'] = data['MACD'].ewm(span=9).mean()
            data['MACD_diff'] = data['MACD'] - data['MACD_signal']

            # Bollinger Bands
            for window in [20]:
                sma = price_proxy.rolling(window).mean()
                std = price_proxy.rolling(window).std()
                data[f'BB_upper_{window}'] = sma + (std * 2)
                data[f'BB_lower_{window}'] = sma - (std * 2)
                data[f'BB_width_{window}'] = data[f'BB_upper_{window}'] - data[f'BB_lower_{window}']
                data[f'BB_position_{window}'] = (price_proxy - data[f'BB_lower_{window}']) / data[f'BB_width_{window}']

        return data

    def _create_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create momentum-based features"""
        data = df.copy()

        # Get momentum columns if they exist
        mom_cols = [col for col in df.columns if col.startswith('MOM')]

        for col in mom_cols:
            # Rate of change
            for period in [1, 5, 10, 20]:
                data[f'{col}_roc_{period}'] = df[col].pct_change(period)

            # Rolling z-score
            for window in [20, 60]:
                mean = df[col].rolling(window).mean()
                std = df[col].rolling(window).std()
                data[f'{col}_zscore_{window}'] = (df[col] - mean) / std

        return data

    def _create_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create volatility features"""
        data = df.copy()

        # Get volatility columns
        vol_cols = [col for col in df.columns if col.startswith('V')]

        for col in vol_cols:
            # Rolling statistics
            for window in [5, 10, 20, 60]:
                data[f'{col}_mean_{window}'] = df[col].rolling(window).mean()
                data[f'{col}_std_{window}'] = df[col].rolling(window).std()
                data[f'{col}_min_{window}'] = df[col].rolling(window).min()
                data[f'{col}_max_{window}'] = df[col].rolling(window).max()

        # Realized volatility if forward_returns available
        if 'forward_returns' in df.columns:
            for window in [5, 10, 20, 60]:
                data[f'realized_vol_{window}'] = df['forward_returns'].rolling(window).std() * np.sqrt(252)

        return data

    def _create_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create market regime features"""
        data = df.copy()

        if 'forward_returns' in df.columns:
            returns = df['forward_returns']

            # Trend regime (based on moving average)
            for window in [20, 60, 200]:
                cumret = (1 + returns.fillna(0)).cumprod()
                ma = cumret.rolling(window).mean()
                data[f'trend_regime_{window}'] = (cumret > ma).astype(int)

            # Volatility regime (high/low vol)
            for window in [20, 60]:
                vol = returns.rolling(window).std()
                vol_ma = vol.rolling(window).mean()
                data[f'vol_regime_{window}'] = (vol > vol_ma).astype(int)

            # Bull/Bear regime
            for window in [60, 120]:
                ret_sum = returns.rolling(window).sum()
                data[f'bull_regime_{window}'] = (ret_sum > 0).astype(int)

        return data

    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between different categories"""
        data = df.copy()

        # Get feature categories
        market_cols = [col for col in df.columns if col.startswith('M')][:5]
        econ_cols = [col for col in df.columns if col.startswith('E')][:5]
        vol_cols = [col for col in df.columns if col.startswith('V')][:5]

        # Market × Economic
        for m_col in market_cols:
            for e_col in econ_cols:
                if m_col in df.columns and e_col in df.columns:
                    data[f'{m_col}_x_{e_col}'] = df[m_col] * df[e_col]

        # Market × Volatility
        for m_col in market_cols:
            for v_col in vol_cols:
                if m_col in df.columns and v_col in df.columns:
                    data[f'{m_col}_x_{v_col}'] = df[m_col] * df[v_col]

        return data

    def _create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features"""
        data = df.copy()

        # If date_id is numeric, create cyclical features
        if 'date_id' in df.columns:
            # Day of week proxy (assuming 5 trading days)
            data['day_of_week'] = df['date_id'] % 5
            data['day_of_week_sin'] = np.sin(2 * np.pi * data['day_of_week'] / 5)
            data['day_of_week_cos'] = np.cos(2 * np.pi * data['day_of_week'] / 5)

            # Week of month
            data['week_of_month'] = (df['date_id'] % 20) // 5

            # Month of quarter
            data['month_of_quarter'] = (df['date_id'] % 60) // 20

        return data

    def _create_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create statistical aggregation features"""
        data = df.copy()

        # Get all numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_cols
                       if col.startswith(('M', 'E', 'I', 'P', 'V', 'S', 'MOM', 'D'))][:20]

        # Rolling percentiles
        for col in feature_cols:
            for window in [20, 60]:
                data[f'{col}_pct25_{window}'] = df[col].rolling(window).quantile(0.25)
                data[f'{col}_pct75_{window}'] = df[col].rolling(window).quantile(0.75)
                data[f'{col}_iqr_{window}'] = (
                    data[f'{col}_pct75_{window}'] - data[f'{col}_pct25_{window}']
                )

        # Rolling skewness and kurtosis
        for col in feature_cols[:10]:
            for window in [20, 60]:
                data[f'{col}_skew_{window}'] = df[col].rolling(window).skew()
                data[f'{col}_kurt_{window}'] = df[col].rolling(window).kurt()

        return data

    def get_feature_names(self, df: pd.DataFrame) -> List[str]:
        """Get list of all feature names"""
        exclude = ['date_id', 'forward_returns', 'risk_free_rate',
                   'market_forward_excess_returns', 'is_scored',
                   'lagged_forward_returns', 'lagged_risk_free_rate',
                   'lagged_market_forward_excess_returns']

        return [col for col in df.columns if col not in exclude]

    def handle_missing_values(self, df: pd.DataFrame, method: str = 'forward_fill') -> pd.DataFrame:
        """Handle missing values intelligently"""
        data = df.copy()

        if method == 'forward_fill':
            # Forward fill then backward fill
            data = data.fillna(method='ffill').fillna(method='bfill')
        elif method == 'median':
            # Fill with rolling median
            for col in data.columns:
                if data[col].isna().any():
                    data[col] = data[col].fillna(data[col].rolling(20, min_periods=1).median())
        elif method == 'zero':
            data = data.fillna(0)

        return data

    def select_top_features(self, df: pd.DataFrame, target: str, n_features: int = 200) -> List[str]:
        """Select top features based on correlation with target"""
        feature_cols = self.get_feature_names(df)

        if target not in df.columns:
            return feature_cols[:n_features]

        # Calculate correlation with target
        correlations = {}
        for col in feature_cols:
            if col in df.columns:
                corr = abs(df[col].corr(df[target]))
                if not np.isnan(corr):
                    correlations[col] = corr

        # Sort by correlation
        sorted_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)

        # Return top n features
        top_features = [feat for feat, _ in sorted_features[:n_features]]

        print(f"Selected top {len(top_features)} features")
        print(f"Top 10 features by correlation:")
        for feat, corr in sorted_features[:10]:
            print(f"  {feat}: {corr:.4f}")

        return top_features


def create_features(train_df: pd.DataFrame, test_df: pd.DataFrame = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Main function to create features for train and test sets
    """
    engineer = FeatureEngineer()

    # Create features for train
    print("Creating features for training set...")
    train_features = engineer.create_all_features(train_df)
    train_features = engineer.handle_missing_values(train_features)

    # Create features for test if provided
    test_features = None
    if test_df is not None:
        print("Creating features for test set...")
        test_features = engineer.create_all_features(test_df)
        test_features = engineer.handle_missing_values(test_features)

    return train_features, test_features


if __name__ == "__main__":
    # Example usage
    print("Feature Engineering Module")
    print("=" * 50)
    print("This module creates comprehensive features for market prediction")
    print("\nFeature Categories:")
    print("  1. Lag features - Historical values")
    print("  2. Technical indicators - RSI, MACD, Bollinger Bands")
    print("  3. Momentum features - Rate of change, z-scores")
    print("  4. Volatility features - Rolling vol, realized vol")
    print("  5. Regime features - Bull/bear, high/low vol")
    print("  6. Interaction features - Feature combinations")
    print("  7. Time features - Cyclical time patterns")
    print("  8. Statistical features - Percentiles, skew, kurtosis")
