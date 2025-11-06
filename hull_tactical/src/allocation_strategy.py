#!/usr/bin/env python3
"""
Allocation Strategy Module for Hull Tactical Market Prediction
Volatility-aware position sizing with Kelly Criterion and risk management
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')


class AllocationStrategy:
    """
    Sophisticated allocation strategy with volatility targeting
    """

    def __init__(self, config: Dict = None):
        self.config = config or self._default_config()
        self.allocation_history = []
        self.volatility_history = []
        self.current_regime = 'normal'

    def _default_config(self) -> Dict:
        """Default configuration"""
        return {
            'min_allocation': 0.0,
            'max_allocation': 2.0,
            'target_volatility': 0.20,  # 20% annualized vol target
            'max_volatility_ratio': 1.20,  # Max 120% of market vol
            'kelly_fraction': 0.25,  # Use 1/4 Kelly
            'lookback_window': 60,  # Days for vol estimation
            'confidence_threshold': 0.55,  # Minimum confidence to allocate
            'use_kelly': True,
            'use_volatility_targeting': True,
            'use_regime_detection': True,
        }

    def calculate_allocation(
        self,
        predicted_return: float,
        predicted_volatility: float,
        current_volatility: float,
        confidence: float = 0.5,
        historical_volatility: float = None
    ) -> float:
        """
        Calculate optimal allocation based on prediction and risk constraints

        Args:
            predicted_return: Predicted excess return
            predicted_volatility: Predicted market volatility
            current_volatility: Current realized volatility
            confidence: Confidence in prediction (0-1)
            historical_volatility: Historical volatility for comparison

        Returns:
            Allocation amount (0 to 2)
        """
        # Base allocation from prediction
        base_allocation = self._base_allocation_from_prediction(predicted_return, confidence)

        # Apply Kelly Criterion if enabled
        if self.config['use_kelly']:
            kelly_allocation = self._kelly_criterion(
                predicted_return,
                predicted_volatility,
                confidence
            )
            base_allocation = min(base_allocation, kelly_allocation)

        # Apply volatility targeting if enabled
        if self.config['use_volatility_targeting']:
            vol_adjusted_allocation = self._volatility_targeting(
                base_allocation,
                current_volatility,
                predicted_volatility
            )
        else:
            vol_adjusted_allocation = base_allocation

        # Apply regime-based adjustments
        if self.config['use_regime_detection'] and historical_volatility:
            regime = self._detect_regime(current_volatility, historical_volatility)
            regime_adjusted_allocation = self._regime_adjustment(
                vol_adjusted_allocation,
                regime
            )
        else:
            regime_adjusted_allocation = vol_adjusted_allocation

        # Apply hard constraints
        final_allocation = self._apply_constraints(
            regime_adjusted_allocation,
            current_volatility,
            predicted_volatility
        )

        # Record allocation
        self.allocation_history.append({
            'predicted_return': predicted_return,
            'allocation': final_allocation,
            'confidence': confidence,
            'volatility': current_volatility
        })

        return final_allocation

    def _base_allocation_from_prediction(self, predicted_return: float, confidence: float) -> float:
        """
        Calculate base allocation from predicted return

        Simple approach:
        - Positive prediction → long (0.5 to 2.0)
        - Negative prediction → reduce exposure (0 to 0.5)
        - Scale by confidence
        """
        # Confidence threshold
        if confidence < self.config['confidence_threshold']:
            return 1.0  # Market neutral if not confident

        # Scale allocation by predicted return magnitude
        if predicted_return > 0:
            # Bull case: increase exposure
            # Map predicted return to allocation
            # Assuming predicted returns are small (e.g., -0.05 to 0.05)
            allocation = 1.0 + np.clip(predicted_return * 20, 0, 1.0)
        else:
            # Bear case: reduce exposure
            allocation = 1.0 + np.clip(predicted_return * 20, -1.0, 0)

        # Scale by confidence
        # If confidence is high, use more of the signal
        confidence_scaled = 1.0 + (allocation - 1.0) * confidence

        return np.clip(confidence_scaled, 0.0, 2.0)

    def _kelly_criterion(
        self,
        expected_return: float,
        volatility: float,
        confidence: float
    ) -> float:
        """
        Kelly Criterion for optimal position sizing

        Kelly% = (expected_return / variance) * confidence
        We use fractional Kelly to be conservative
        """
        if volatility <= 0:
            return 1.0

        variance = volatility ** 2

        # Kelly fraction
        kelly_pct = (expected_return / variance) * confidence

        # Apply Kelly fraction (e.g., 1/4 Kelly for safety)
        fractional_kelly = kelly_pct * self.config['kelly_fraction']

        # Convert to allocation (1.0 = market weight)
        allocation = 1.0 + fractional_kelly

        return np.clip(allocation, 0.0, 2.0)

    def _volatility_targeting(
        self,
        base_allocation: float,
        current_volatility: float,
        predicted_volatility: float
    ) -> float:
        """
        Adjust allocation to target specific volatility level

        If market is more volatile than target, reduce exposure
        If market is less volatile, can increase exposure (up to limits)
        """
        target_vol = self.config['target_volatility']

        # Use predicted or current volatility (prefer current as more recent)
        effective_vol = current_volatility if current_volatility > 0 else predicted_volatility

        if effective_vol <= 0:
            return base_allocation

        # Volatility scaling factor
        vol_scalar = target_vol / effective_vol

        # Adjust allocation
        adjusted_allocation = base_allocation * vol_scalar

        return np.clip(adjusted_allocation, 0.0, 2.0)

    def _detect_regime(self, current_volatility: float, historical_volatility: float) -> str:
        """
        Detect market regime based on volatility
        """
        vol_ratio = current_volatility / (historical_volatility + 1e-10)

        if vol_ratio > 1.5:
            return 'high_volatility'
        elif vol_ratio > 1.2:
            return 'elevated_volatility'
        elif vol_ratio < 0.8:
            return 'low_volatility'
        else:
            return 'normal'

    def _regime_adjustment(self, allocation: float, regime: str) -> float:
        """
        Adjust allocation based on market regime
        """
        self.current_regime = regime

        if regime == 'high_volatility':
            # Reduce exposure in high vol
            return allocation * 0.7
        elif regime == 'elevated_volatility':
            # Slightly reduce exposure
            return allocation * 0.85
        elif regime == 'low_volatility':
            # Can increase exposure slightly
            return allocation * 1.1
        else:
            # Normal regime - no adjustment
            return allocation

    def _apply_constraints(
        self,
        allocation: float,
        current_volatility: float,
        predicted_volatility: float
    ) -> float:
        """
        Apply hard constraints on allocation
        """
        # Constraint 1: Min/Max allocation
        allocation = np.clip(
            allocation,
            self.config['min_allocation'],
            self.config['max_allocation']
        )

        # Constraint 2: Volatility ratio constraint
        # If predicted vol exceeds max ratio of market vol, reduce exposure
        effective_vol = current_volatility if current_volatility > 0 else predicted_volatility
        market_vol = 0.16  # Assume typical S&P 500 vol ~16%

        if effective_vol > market_vol * self.config['max_volatility_ratio']:
            # Scale down allocation proportionally
            vol_penalty = (market_vol * self.config['max_volatility_ratio']) / effective_vol
            allocation = allocation * vol_penalty

        # Final clip
        allocation = np.clip(
            allocation,
            self.config['min_allocation'],
            self.config['max_allocation']
        )

        return allocation

    def calculate_position_size(
        self,
        allocation: float,
        portfolio_value: float,
        current_price: float
    ) -> int:
        """
        Convert allocation to number of shares/contracts

        Args:
            allocation: Allocation weight (0-2)
            portfolio_value: Total portfolio value
            current_price: Current asset price

        Returns:
            Number of shares to hold
        """
        position_value = portfolio_value * allocation
        shares = int(position_value / current_price)

        return shares

    def get_statistics(self) -> Dict:
        """Get allocation statistics"""
        if not self.allocation_history:
            return {}

        allocations = [a['allocation'] for a in self.allocation_history]
        returns = [a['predicted_return'] for a in self.allocation_history]

        return {
            'mean_allocation': np.mean(allocations),
            'std_allocation': np.std(allocations),
            'min_allocation': np.min(allocations),
            'max_allocation': np.max(allocations),
            'mean_predicted_return': np.mean(returns),
            'current_regime': self.current_regime,
            'n_allocations': len(allocations)
        }


class RiskManager:
    """
    Risk management and portfolio monitoring
    """

    def __init__(self, config: Dict = None):
        self.config = config or self._default_config()
        self.drawdown_history = []
        self.risk_breaches = []

    def _default_config(self) -> Dict:
        """Default risk configuration"""
        return {
            'max_drawdown': 0.20,  # 20% max drawdown
            'volatility_limit': 0.30,  # 30% vol limit
            'var_confidence': 0.95,  # 95% VaR
            'concentration_limit': 2.0,  # Max position size
        }

    def check_risk_limits(
        self,
        allocation: float,
        current_volatility: float,
        portfolio_returns: pd.Series
    ) -> Tuple[bool, List[str]]:
        """
        Check if allocation violates risk limits

        Returns:
            (is_safe, list_of_warnings)
        """
        warnings = []
        is_safe = True

        # Check 1: Volatility limit
        if current_volatility > self.config['volatility_limit']:
            warnings.append(f"Volatility {current_volatility:.2%} exceeds limit {self.config['volatility_limit']:.2%}")
            is_safe = False

        # Check 2: Concentration limit
        if allocation > self.config['concentration_limit']:
            warnings.append(f"Allocation {allocation:.2f} exceeds concentration limit {self.config['concentration_limit']:.2f}")
            is_safe = False

        # Check 3: Drawdown limit
        if len(portfolio_returns) > 0:
            current_drawdown = self._calculate_drawdown(portfolio_returns)
            if current_drawdown > self.config['max_drawdown']:
                warnings.append(f"Drawdown {current_drawdown:.2%} exceeds limit {self.config['max_drawdown']:.2%}")
                is_safe = False

        if not is_safe:
            self.risk_breaches.append({
                'allocation': allocation,
                'volatility': current_volatility,
                'warnings': warnings
            })

        return is_safe, warnings

    def _calculate_drawdown(self, returns: pd.Series) -> float:
        """Calculate current drawdown"""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max

        return abs(drawdown.iloc[-1]) if len(drawdown) > 0 else 0.0

    def calculate_var(self, returns: pd.Series, confidence: float = 0.95) -> float:
        """Calculate Value at Risk"""
        if len(returns) < 2:
            return 0.0

        return np.percentile(returns, (1 - confidence) * 100)

    def calculate_expected_shortfall(self, returns: pd.Series, confidence: float = 0.95) -> float:
        """Calculate Conditional VaR (Expected Shortfall)"""
        var = self.calculate_var(returns, confidence)
        return returns[returns <= var].mean()

    def get_risk_metrics(self, returns: pd.Series) -> Dict:
        """Get comprehensive risk metrics"""
        if len(returns) < 2:
            return {}

        return {
            'volatility': returns.std() * np.sqrt(252),
            'sharpe_ratio': (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0,
            'max_drawdown': self._calculate_drawdown(returns),
            'var_95': self.calculate_var(returns, 0.95),
            'cvar_95': self.calculate_expected_shortfall(returns, 0.95),
            'skewness': returns.skew(),
            'kurtosis': returns.kurtosis(),
        }


if __name__ == "__main__":
    print("Allocation Strategy Module")
    print("=" * 50)
    print("Volatility-aware position sizing with Kelly Criterion")
    print("\nKey Features:")
    print("  • Kelly Criterion for optimal sizing")
    print("  • Volatility targeting")
    print("  • Regime detection")
    print("  • Risk limit monitoring")
    print("  • Drawdown protection")
