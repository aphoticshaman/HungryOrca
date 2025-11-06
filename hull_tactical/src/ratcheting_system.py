#!/usr/bin/env python3
"""
Asymmetric Gain Ratcheting System for Market Prediction
Ensures monotonic improvement - models can only get better, never worse
Adapted from ARC Prize breakthrough insights
"""

import numpy as np
import pandas as pd
import hashlib
import json
import pickle
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


@dataclass
class ModelCommit:
    """
    Git-style commit of model performance
    Only improvements are accepted - creates evolutionary pressure
    """
    hash: str                           # SHA256 of model state
    parent: Optional[str]               # Previous commit hash
    timestamp: datetime                 # When committed
    sharpe_ratio: float                # Primary metric
    total_return: float                # Total return achieved
    max_drawdown: float                # Maximum drawdown
    volatility: float                  # Portfolio volatility
    win_rate: float                    # Percentage of winning trades
    delta: float                       # Improvement over parent
    model_config: Dict[str, Any]       # Model configuration
    feature_importance: Dict[str, float]  # Top features
    regime_performance: Dict[str, float]  # Performance by market regime
    metadata: Dict[str, Any]           # Additional metadata

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        d = asdict(self)
        d['timestamp'] = self.timestamp.isoformat()
        return d

    @classmethod
    def from_dict(cls, d: Dict) -> 'ModelCommit':
        """Create from dictionary"""
        d['timestamp'] = datetime.fromisoformat(d['timestamp'])
        return cls(**d)


class AsymmetricRatchet:
    """
    Ratcheting system that ONLY accepts improvements

    Key Insight: By rejecting regressions, we create evolutionary pressure
    toward optimal solutions while preventing catastrophic forgetting.
    """

    def __init__(self, persistence_path: str = 'hull_tactical/models/ratchet_history.pkl'):
        self.persistence_path = persistence_path
        self.best_sharpe = -np.inf
        self.commit_history: List[ModelCommit] = []
        self.locked_capabilities: Dict[str, Any] = {}
        self.rejected_attempts: List[Dict] = []

        # Load existing history if available
        self._load_history()

    def attempt_commit(
        self,
        model: Any,
        performance_metrics: Dict[str, float],
        model_config: Dict[str, Any],
        feature_importance: Dict[str, float] = None,
        regime_performance: Dict[str, float] = None
    ) -> Tuple[bool, str]:
        """
        Attempt to commit a new model

        Returns:
            (accepted, message)
        """
        sharpe_ratio = performance_metrics.get('sharpe_ratio', -np.inf)

        # Calculate improvement delta
        delta = sharpe_ratio - self.best_sharpe

        # CRITICAL: Only accept improvements
        if sharpe_ratio > self.best_sharpe:
            # Create commit
            commit = self._create_commit(
                model=model,
                performance_metrics=performance_metrics,
                model_config=model_config,
                feature_importance=feature_importance or {},
                regime_performance=regime_performance or {},
                delta=delta
            )

            # Lock in the improvement
            self.commit_history.append(commit)
            self.best_sharpe = sharpe_ratio

            # Save immediately
            self._save_history()

            message = f"✅ COMMIT ACCEPTED: Sharpe {sharpe_ratio:.4f} (Δ +{delta:.4f})"
            print(f"\n{'='*60}")
            print(f"🔒 CAPABILITY LOCKED")
            print(f"{'='*60}")
            print(message)
            print(f"Commit Hash: {commit.hash[:12]}")
            print(f"Total Return: {commit.total_return:.2%}")
            print(f"Max Drawdown: {commit.max_drawdown:.2%}")
            print(f"Win Rate: {commit.win_rate:.2%}")
            print(f"Commits: {len(self.commit_history)}")
            print(f"{'='*60}\n")

            return True, message
        else:
            # Reject regression
            self.rejected_attempts.append({
                'timestamp': datetime.now(),
                'sharpe': sharpe_ratio,
                'delta': delta,
                'reason': 'No improvement'
            })

            message = f"❌ COMMIT REJECTED: Sharpe {sharpe_ratio:.4f} (Δ {delta:.4f}) - No improvement"
            print(f"\n⚠️  {message}")
            print(f"   Best remains: {self.best_sharpe:.4f}")
            print(f"   Rejected attempts: {len(self.rejected_attempts)}\n")

            return False, message

    def _create_commit(
        self,
        model: Any,
        performance_metrics: Dict[str, float],
        model_config: Dict[str, Any],
        feature_importance: Dict[str, float],
        regime_performance: Dict[str, float],
        delta: float
    ) -> ModelCommit:
        """Create a model commit"""

        # Generate hash of model state
        model_hash = self._hash_model(model, model_config)

        # Get parent hash
        parent_hash = self.commit_history[-1].hash if self.commit_history else None

        commit = ModelCommit(
            hash=model_hash,
            parent=parent_hash,
            timestamp=datetime.now(),
            sharpe_ratio=performance_metrics['sharpe_ratio'],
            total_return=performance_metrics.get('total_return', 0.0),
            max_drawdown=performance_metrics.get('max_drawdown', 0.0),
            volatility=performance_metrics.get('volatility', 0.0),
            win_rate=performance_metrics.get('win_rate', 0.0),
            delta=delta,
            model_config=model_config,
            feature_importance=feature_importance,
            regime_performance=regime_performance,
            metadata={
                'rejection_count': len(self.rejected_attempts),
                'commit_number': len(self.commit_history) + 1
            }
        )

        return commit

    def _hash_model(self, model: Any, config: Dict) -> str:
        """Generate SHA256 hash of model state"""
        # Combine model config and timestamp for unique hash
        state_str = json.dumps(config, sort_keys=True) + str(datetime.now())
        return hashlib.sha256(state_str.encode()).hexdigest()

    def get_commit_history(self) -> pd.DataFrame:
        """Get commit history as DataFrame"""
        if not self.commit_history:
            return pd.DataFrame()

        data = []
        for commit in self.commit_history:
            data.append({
                'hash': commit.hash[:12],
                'timestamp': commit.timestamp,
                'sharpe': commit.sharpe_ratio,
                'delta': commit.delta,
                'return': commit.total_return,
                'drawdown': commit.max_drawdown,
                'volatility': commit.volatility,
                'win_rate': commit.win_rate,
                'commit_num': commit.metadata['commit_number']
            })

        return pd.DataFrame(data)

    def get_best_model_info(self) -> Optional[ModelCommit]:
        """Get information about the best model"""
        if not self.commit_history:
            return None
        return self.commit_history[-1]  # Latest commit is best

    def plot_evolution(self):
        """Plot model evolution over time"""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("Matplotlib not available for plotting")
            return

        if not self.commit_history:
            print("No commits to plot")
            return

        df = self.get_commit_history()

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Sharpe ratio evolution
        axes[0, 0].plot(df['commit_num'], df['sharpe'], marker='o', linewidth=2)
        axes[0, 0].set_title('Sharpe Ratio Evolution (Asymmetric Ratcheting)', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Commit Number')
        axes[0, 0].set_ylabel('Sharpe Ratio')
        axes[0, 0].grid(True, alpha=0.3)

        # Delta improvements
        axes[0, 1].bar(df['commit_num'], df['delta'], color='green', alpha=0.7)
        axes[0, 1].set_title('Improvement Delta per Commit', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Commit Number')
        axes[0, 1].set_ylabel('Δ Sharpe Ratio')
        axes[0, 1].grid(True, alpha=0.3)

        # Return vs Drawdown
        axes[1, 0].scatter(df['drawdown'], df['return'], c=df['sharpe'], cmap='viridis', s=100, alpha=0.7)
        axes[1, 0].set_title('Return vs Drawdown (colored by Sharpe)', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Max Drawdown')
        axes[1, 0].set_ylabel('Total Return')
        axes[1, 0].grid(True, alpha=0.3)

        # Win rate evolution
        axes[1, 1].plot(df['commit_num'], df['win_rate'], marker='s', linewidth=2, color='purple')
        axes[1, 1].set_title('Win Rate Evolution', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Commit Number')
        axes[1, 1].set_ylabel('Win Rate')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('hull_tactical/models/ratchet_evolution.png', dpi=150, bbox_inches='tight')
        print("✅ Evolution plot saved to hull_tactical/models/ratchet_evolution.png")

    def _save_history(self):
        """Persist commit history to disk"""
        try:
            with open(self.persistence_path, 'wb') as f:
                pickle.dump({
                    'commits': [c.to_dict() for c in self.commit_history],
                    'best_sharpe': self.best_sharpe,
                    'rejected_attempts': self.rejected_attempts
                }, f)
            print(f"💾 History saved ({len(self.commit_history)} commits)")
        except Exception as e:
            print(f"⚠️  Warning: Could not save history: {e}")

    def _load_history(self):
        """Load commit history from disk"""
        try:
            import os
            if os.path.exists(self.persistence_path):
                with open(self.persistence_path, 'rb') as f:
                    data = pickle.load(f)
                    self.commit_history = [ModelCommit.from_dict(c) for c in data['commits']]
                    self.best_sharpe = data['best_sharpe']
                    self.rejected_attempts = data.get('rejected_attempts', [])
                print(f"📂 Loaded history: {len(self.commit_history)} commits, best Sharpe: {self.best_sharpe:.4f}")
        except Exception as e:
            print(f"ℹ️  No previous history found (starting fresh)")


class CapabilityLock:
    """
    Lock in specific capabilities once achieved
    Ensures learned behaviors persist
    """

    def __init__(self):
        self.locked_capabilities: Dict[str, Dict] = {}

    def lock_capability(self, capability_name: str, performance: float, metadata: Dict = None):
        """Lock a capability"""
        if capability_name not in self.locked_capabilities or \
           performance > self.locked_capabilities[capability_name]['performance']:

            self.locked_capabilities[capability_name] = {
                'performance': performance,
                'locked_at': datetime.now(),
                'metadata': metadata or {}
            }
            print(f"🔒 Locked capability: {capability_name} (performance: {performance:.4f})")

    def is_locked(self, capability_name: str) -> bool:
        """Check if capability is locked"""
        return capability_name in self.locked_capabilities

    def get_locked_performance(self, capability_name: str) -> Optional[float]:
        """Get locked performance level"""
        if capability_name in self.locked_capabilities:
            return self.locked_capabilities[capability_name]['performance']
        return None


if __name__ == "__main__":
    print("Asymmetric Gain Ratcheting System")
    print("=" * 50)
    print("Monotonic improvement engine - models can only get better!")
    print("\nKey Features:")
    print("  • Git-style versioning of model performance")
    print("  • Automatic rejection of regressions")
    print("  • Evolutionary pressure toward optimality")
    print("  • Prevention of catastrophic forgetting")
    print("  • Complete commit history tracking")
