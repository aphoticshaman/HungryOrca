#!/usr/bin/env python3
"""
Evaluation Metric and Backtesting Module
Implements the competition's Sharpe ratio variant metric
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')


class CompetitionMetric:
    """
    Implements the competition evaluation metric
    A Sharpe ratio variant that penalizes excess volatility
    """

    def __init__(self, max_vol_ratio: float = 1.20):
        """
        Args:
            max_vol_ratio: Maximum allowed volatility ratio (120% of market vol)
        """
        self.max_vol_ratio = max_vol_ratio

    def calculate_score(
        self,
        allocations: np.ndarray,
        forward_returns: np.ndarray,
        risk_free_rate: np.ndarray = None
    ) -> Dict:
        """
        Calculate competition score

        Args:
            allocations: Array of allocation decisions (0-2)
            forward_returns: Array of market forward returns
            risk_free_rate: Array of risk-free rates (optional)

        Returns:
            Dictionary with score and metrics
        """
        # Calculate strategy returns
        strategy_returns = allocations * forward_returns

        # Calculate excess returns
        if risk_free_rate is not None:
            strategy_excess_returns = strategy_returns - risk_free_rate
            market_excess_returns = forward_returns - risk_free_rate
        else:
            strategy_excess_returns = strategy_returns
            market_excess_returns = forward_returns

        # Calculate metrics
        mean_strategy_return = np.mean(strategy_excess_returns)
        std_strategy_return = np.std(strategy_excess_returns)
        mean_market_return = np.mean(market_excess_returns)
        std_market_return = np.std(market_excess_returns)

        # Sharpe ratio
        sharpe_ratio = (mean_strategy_return / std_strategy_return) * np.sqrt(252) if std_strategy_return > 0 else 0

        # Volatility ratio
        vol_ratio = std_strategy_return / std_market_return if std_market_return > 0 else 1.0

        # Apply penalty if volatility exceeds max ratio
        if vol_ratio > self.max_vol_ratio:
            penalty = (vol_ratio - self.max_vol_ratio) ** 2
            score = sharpe_ratio - penalty
        else:
            score = sharpe_ratio

        # Additional penalty if underperforming market
        cumulative_strategy = np.cumsum(strategy_excess_returns)
        cumulative_market = np.cumsum(market_excess_returns)
        final_strategy_return = cumulative_strategy[-1] if len(cumulative_strategy) > 0 else 0
        final_market_return = cumulative_market[-1] if len(cumulative_market) > 0 else 0

        if final_strategy_return < final_market_return:
            underperformance = final_market_return - final_strategy_return
            score = score - underperformance

        return {
            'score': score,
            'sharpe_ratio': sharpe_ratio,
            'mean_return': mean_strategy_return * 252,  # Annualized
            'volatility': std_strategy_return * np.sqrt(252),  # Annualized
            'vol_ratio': vol_ratio,
            'max_drawdown': self._calculate_max_drawdown(strategy_excess_returns),
            'total_return': final_strategy_return,
            'market_return': final_market_return,
        }

    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown"""
        cumulative = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max)
        max_dd = np.min(drawdown) if len(drawdown) > 0 else 0
        return abs(max_dd)


class Backtester:
    """
    Backtesting framework for strategy evaluation
    """

    def __init__(self, initial_capital: float = 100000.0):
        self.initial_capital = initial_capital
        self.results = None

    def backtest(
        self,
        allocations: np.ndarray,
        forward_returns: np.ndarray,
        risk_free_rate: np.ndarray = None,
        dates: pd.Series = None
    ) -> Dict:
        """
        Run backtest on allocation strategy

        Args:
            allocations: Array of allocation decisions
            forward_returns: Array of market returns
            risk_free_rate: Array of risk-free rates
            dates: Date index

        Returns:
            Dictionary with backtest results
        """
        # Initialize
        portfolio_value = [self.initial_capital]
        portfolio_returns = []
        market_value = [self.initial_capital]
        market_returns = []

        # Simulate trading
        for i in range(len(allocations)):
            # Strategy return
            strat_ret = allocations[i] * forward_returns[i]
            portfolio_returns.append(strat_ret)

            # Update portfolio value
            new_value = portfolio_value[-1] * (1 + strat_ret)
            portfolio_value.append(new_value)

            # Market return (buy and hold)
            mkt_ret = forward_returns[i]
            market_returns.append(mkt_ret)

            # Update market value
            market_value.append(market_value[-1] * (1 + mkt_ret))

        # Convert to arrays
        portfolio_value = np.array(portfolio_value)
        portfolio_returns = np.array(portfolio_returns)
        market_value = np.array(market_value)
        market_returns = np.array(market_returns)

        # Calculate metrics
        total_return = (portfolio_value[-1] - self.initial_capital) / self.initial_capital
        market_total_return = (market_value[-1] - self.initial_capital) / self.initial_capital

        # Annualized returns
        n_periods = len(portfolio_returns)
        years = n_periods / 252  # Assuming daily data
        annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        market_annualized_return = (1 + market_total_return) ** (1 / years) - 1 if years > 0 else 0

        # Volatility
        volatility = np.std(portfolio_returns) * np.sqrt(252)
        market_volatility = np.std(market_returns) * np.sqrt(252)

        # Sharpe ratio
        excess_returns = portfolio_returns
        if risk_free_rate is not None:
            excess_returns = portfolio_returns - risk_free_rate

        sharpe = (np.mean(excess_returns) / np.std(excess_returns)) * np.sqrt(252) if np.std(excess_returns) > 0 else 0

        # Max drawdown
        cumulative = np.cumsum(portfolio_returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max)
        max_drawdown = abs(np.min(drawdown)) if len(drawdown) > 0 else 0

        # Win rate
        win_rate = np.sum(portfolio_returns > 0) / len(portfolio_returns) if len(portfolio_returns) > 0 else 0

        # Sortino ratio (downside deviation)
        downside_returns = portfolio_returns[portfolio_returns < 0]
        downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 0.0001
        sortino = (np.mean(excess_returns) / downside_std) * np.sqrt(252)

        # Calmar ratio (return / max drawdown)
        calmar = annualized_return / max_drawdown if max_drawdown > 0 else 0

        # Create results dataframe
        if dates is not None:
            results_df = pd.DataFrame({
                'date': dates,
                'allocation': allocations,
                'forward_return': forward_returns,
                'strategy_return': portfolio_returns,
                'market_return': market_returns,
                'portfolio_value': portfolio_value[:-1],
                'market_value': market_value[:-1]
            })
        else:
            results_df = pd.DataFrame({
                'allocation': allocations,
                'forward_return': forward_returns,
                'strategy_return': portfolio_returns,
                'market_return': market_returns,
                'portfolio_value': portfolio_value[:-1],
                'market_value': market_value[:-1]
            })

        self.results = results_df

        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'market_total_return': market_total_return,
            'market_annualized_return': market_annualized_return,
            'excess_return': total_return - market_total_return,
            'volatility': volatility,
            'market_volatility': market_volatility,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'calmar_ratio': calmar,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'final_value': portfolio_value[-1],
            'market_final_value': market_value[-1],
            'n_trades': n_periods,
        }

    def plot_results(self):
        """Plot backtest results"""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("Matplotlib not available for plotting")
            return

        if self.results is None:
            print("No backtest results to plot. Run backtest() first.")
            return

        fig, axes = plt.subplots(3, 1, figsize=(12, 10))

        # Portfolio value
        axes[0].plot(self.results['portfolio_value'], label='Strategy', linewidth=2)
        axes[0].plot(self.results['market_value'], label='Market (S&P 500)', linewidth=2, alpha=0.7)
        axes[0].set_title('Portfolio Value Over Time')
        axes[0].set_ylabel('Value ($)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Returns
        axes[1].plot(np.cumsum(self.results['strategy_return']), label='Strategy Cumulative Return', linewidth=2)
        axes[1].plot(np.cumsum(self.results['market_return']), label='Market Cumulative Return', linewidth=2, alpha=0.7)
        axes[1].set_title('Cumulative Returns')
        axes[1].set_ylabel('Cumulative Return')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # Allocations
        axes[2].plot(self.results['allocation'], label='Allocation', linewidth=2)
        axes[2].axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Market Weight')
        axes[2].set_title('Allocation Over Time')
        axes[2].set_ylabel('Allocation (0-2)')
        axes[2].set_xlabel('Time Period')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('hull_tactical/backtest_results.png', dpi=150, bbox_inches='tight')
        print("✅ Plot saved to hull_tactical/backtest_results.png")

    def get_summary(self, metrics: Dict) -> str:
        """Get formatted summary of backtest results"""
        summary = f"""
{'='*60}
BACKTEST SUMMARY
{'='*60}

PERFORMANCE METRICS:
  Total Return:              {metrics['total_return']:.2%}
  Annualized Return:         {metrics['annualized_return']:.2%}
  Market Return:             {metrics['market_total_return']:.2%}
  Market Annualized Return:  {metrics['market_annualized_return']:.2%}
  Excess Return:             {metrics['excess_return']:.2%}

RISK METRICS:
  Volatility (Ann.):         {metrics['volatility']:.2%}
  Market Volatility:         {metrics['market_volatility']:.2%}
  Max Drawdown:              {metrics['max_drawdown']:.2%}

RISK-ADJUSTED RETURNS:
  Sharpe Ratio:              {metrics['sharpe_ratio']:.3f}
  Sortino Ratio:             {metrics['sortino_ratio']:.3f}
  Calmar Ratio:              {metrics['calmar_ratio']:.3f}

TRADING STATISTICS:
  Win Rate:                  {metrics['win_rate']:.2%}
  Number of Trades:          {metrics['n_trades']}
  Final Portfolio Value:     ${metrics['final_value']:,.2f}
  Market Final Value:        ${metrics['market_final_value']:,.2f}

{'='*60}
"""
        return summary


if __name__ == "__main__":
    print("Evaluation Metric Module")
    print("=" * 50)
    print("Implements competition Sharpe ratio variant metric")
    print("\nKey Features:")
    print("  • Sharpe ratio calculation")
    print("  • Volatility penalty for excess risk")
    print("  • Underperformance penalty")
    print("  • Comprehensive backtesting")
    print("  • Performance visualization")
