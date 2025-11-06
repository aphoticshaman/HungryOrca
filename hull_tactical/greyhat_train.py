#!/usr/bin/env python3
"""
GREY HAT QUANTUM TRAINING PIPELINE - XYZA Framework
X (Design) → Y (Implementation) → Z (Test) → A (Alpha)

Integrates:
- Quick baseline models (Ridge, RF, GB)
- Quantum market exploiter
- Game Genie recommendations
- Full grey hat exploitation

Expected: +10-15% competitive advantage over naive approaches
"""

import numpy as np
import pandas as pd
import pickle
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from quantum_market_exploiter import QuantumMarketExploiter, initialize_grey_hat_system
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

print("\n" + "="*70)
print("⚡ GREY HAT QUANTUM TRAINING PIPELINE")
print("="*70)
print("XYZA Framework: X→Y→Z→A")
print("="*70 + "\n")

# =============================================================================
# X PHASE: DESIGN
# =============================================================================
"""
PSEUDOCODE:

1. LOAD DATA:
   - Training: 9,021 days
   - Game Genie recommendations
   - 94 baseline features

2. FEATURE ENGINEERING:
   - Baseline features (from data)
   - Orthogonal features (from quantum exploiter)
   - Vulnerability exploits (from scanner)

3. MODEL TRAINING:
   - Ridge (Healer) - Conservative
   - Random Forest (Tank) - Exploration
   - Gradient Boosting (DPS) - Precision

4. QUANTUM INTEGRATION:
   - Quantum ensemble for entanglement detection
   - Attractor basin mapping
   - SPDM for self-discovery

5. VALIDATION:
   - Time-series split
   - Backtest with quantum exploitation
   - Track entanglement gains

6. PRODUCTION:
   - Save models + quantum state
   - Ready for submission
"""

# =============================================================================
# Y PHASE: IMPLEMENTATION
# =============================================================================

print("📦 Loading data...")
train = pd.read_csv('data/train.csv')
print(f"  Loaded {len(train):,} trading days\n")

# Load Game Genie recommendations
try:
    with open('data/game_genie_recommendations.pkl', 'rb') as f:
        game_genie = pickle.load(f)
    print("✅ Game Genie recommendations loaded")
    print(f"  Regime winners: {len(game_genie['regime_winners'])} regimes")
    print(f"  Agreement threshold: {game_genie['agreement_threshold_high']:.4f}\n")
except:
    game_genie = None
    print("⚠️  No Game Genie data, using defaults\n")

# Prepare features
feature_cols = [c for c in train.columns if c not in [
    'date_id', 'forward_returns', 'risk_free_rate', 'market_forward_excess_returns'
]]

X = train[feature_cols].fillna(0)
y = train['market_forward_excess_returns'].fillna(0)
forward_returns = train['forward_returns'].fillna(0)

print(f"📊 Features: {len(feature_cols)}")
print(f"   Samples: {len(X)}\n")

# Initialize grey hat system
print("🎮 Initializing grey hat quantum system...")
quantum_exploiter = initialize_grey_hat_system()

# Feature engineering with orthogonal synthesis
print("🔧 Engineering orthogonal features...")
X_orthogonal = quantum_exploiter.orthogonal_synth.synthesize_perpendicular_features(X)
print(f"  Created {len(X_orthogonal.columns) - len(X.columns)} orthogonal features")

# Scan for vulnerabilities
print("\n🔍 Scanning information vulnerabilities...")
vulns = quantum_exploiter.vuln_scanner.scan_feature_space(X)
print(f"  Correlation clusters: {len(vulns['correlation_clusters'])}")
print(f"  Sparse features: {len(vulns['sparse_features'])}")
print(f"  Temporal dependencies: {len(vulns['temporal_dependencies'])}")

# Exploit vulnerabilities
X_exploited = quantum_exploiter.vuln_scanner.exploit_vulnerabilities(X_orthogonal, vulns)
print(f"  Created {len(X_exploited.columns) - len(X_orthogonal.columns)} exploit features")

# Handle NaN values from feature engineering
print(f"\n🔧 Handling NaN values...")
print(f"  NaN before: {X_exploited.isna().sum().sum()}")
X_exploited = X_exploited.fillna(0)  # Simple imputation
print(f"  NaN after: {X_exploited.isna().sum().sum()}")

# Time-series split
split_idx = int(len(X) * 0.8)
X_train, X_val = X_exploited.iloc[:split_idx], X_exploited.iloc[split_idx:]
y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
returns_train, returns_val = forward_returns.iloc[:split_idx], forward_returns.iloc[split_idx:]

print(f"📊 Splits: Train={len(X_train)}, Val={len(X_val)}\n")

# Train specialist models
print("🎯 Training specialist ensemble...")

# 1. Healer (Ridge)
print("  Training Healer (Ridge)...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

healer = Ridge(alpha=1.0)
healer.fit(X_train_scaled, y_train)
healer_pred_val = healer.predict(X_val_scaled)

# 2. Tank (Random Forest)
print("  Training Tank (Random Forest)...")
tank = RandomForestRegressor(n_estimators=150, max_depth=10, min_samples_split=20, n_jobs=-1, random_state=42)
tank.fit(X_train, y_train)
tank_pred_val = tank.predict(X_val)

# 3. DPS (Gradient Boosting)
print("  Training DPS (Gradient Boosting)...")
dps = GradientBoostingRegressor(n_estimators=150, learning_rate=0.06, max_depth=5, subsample=0.8, random_state=42)
dps.fit(X_train, y_train)
dps_pred_val = dps.predict(X_val)

print("✅ All specialists trained\n")

# =============================================================================
# Z PHASE: TESTING
# =============================================================================

print("="*70)
print("📈 QUANTUM EXPLOITATION BACKTEST")
print("="*70 + "\n")

# Backtest with quantum exploitation
strategy_returns = []
allocations = []
exploit_reports = []

for i in range(len(X_val)):
    # Get specialist predictions
    predictions = [
        healer_pred_val[i],
        tank_pred_val[i],
        dps_pred_val[i]
    ]

    # Quantum exploitation
    allocation, report = quantum_exploiter.exploit(
        predictions=predictions,
        features=X_val.iloc[[i]],
        returns_history=returns_val.iloc[:i+1] if i > 0 else returns_val.iloc[[0]]
    )

    allocations.append(allocation)
    exploit_reports.append(report)

    # Calculate return
    actual_return = returns_val.iloc[i]
    strat_return = allocation * actual_return
    strategy_returns.append(strat_return)

allocations = np.array(allocations)
strategy_returns = np.array(strategy_returns)

# Calculate metrics
sharpe = (strategy_returns.mean() / strategy_returns.std()) * np.sqrt(252) if strategy_returns.std() > 0 else 0
total_return = np.sum(strategy_returns)
win_rate = np.mean(strategy_returns > 0)
volatility = strategy_returns.std() * np.sqrt(252)

# Baseline (buy-and-hold)
baseline_returns = returns_val.values
baseline_sharpe = (baseline_returns.mean() / baseline_returns.std()) * np.sqrt(252) if baseline_returns.std() > 0 else 0

print("QUANTUM EXPLOITATION RESULTS:")
print("-"*70)
print(f"  Sharpe Ratio:        {sharpe:.4f}  (Baseline: {baseline_sharpe:.4f})")
print(f"  Improvement:         {((sharpe/baseline_sharpe - 1) * 100):.1f}%")
print(f"  Total Return:        {total_return:.4%}")
print(f"  Win Rate:            {win_rate:.2%}")
print(f"  Volatility (Ann.):   {volatility:.2%}")
print(f"  Avg Allocation:      {allocations.mean():.4f}")

# Quantum exploitation stats
exploit_stats = quantum_exploiter.get_exploitation_stats()
print(f"\n⚛️  QUANTUM STATISTICS:")
print(f"  Mean Exploit Gain:   {exploit_stats['mean_exploit_gain']:.4f}x")
print(f"  Max Exploit Gain:    {exploit_stats['max_exploit_gain']:.4f}x")
print(f"  Mean Entanglement:   {exploit_stats['mean_entanglement']:.4f}")
print(f"  Total Exploits:      {exploit_stats['n_exploits']}")

# SPDM: Discover problems
print(f"\n🧠 SELF-DISCOVERING PROBLEMS...")
all_predictions = (healer_pred_val + tank_pred_val + dps_pred_val) / 3
problems = quantum_exploiter.spdm.discover_problems(all_predictions, y_val.values)
print(f"  Discovered {len(problems)} systematic problems:")
for p in problems:
    print(f"    • {p['type']}: {p['description']}")
    print(f"      → Solution: {p['solution']}")

print("\n" + "="*70)

# =============================================================================
# A PHASE: ALPHA (Production)
# =============================================================================

print("\n💾 SAVING PRODUCTION MODELS...")

# Save everything for submission
production_bundle = {
    'models': {
        'healer': healer,
        'tank': tank,
        'dps': dps
    },
    'scaler': scaler,
    'quantum_exploiter': quantum_exploiter,
    'game_genie': game_genie,
    'feature_cols': feature_cols,
    'performance': {
        'sharpe': sharpe,
        'total_return': total_return,
        'win_rate': win_rate,
        'exploit_gain': exploit_stats['mean_exploit_gain']
    }
}

with open('models/greyhat_production.pkl', 'wb') as f:
    pickle.dump(production_bundle, f)

print("✅ Production bundle saved to models/greyhat_production.pkl")
print(f"   Bundle size: {Path('models/greyhat_production.pkl').stat().st_size / 1024 / 1024:.2f} MB")

print("\n" + "="*70)
print("⚡ GREY HAT TRAINING COMPLETE")
print("="*70)
print(f"\n🏆 Final Performance:")
print(f"  Sharpe: {sharpe:.4f} ({((sharpe/baseline_sharpe - 1) * 100):+.1f}% vs baseline)")
print(f"  Quantum Gain: {exploit_stats['mean_exploit_gain']:.2f}x")
print(f"\n✅ Ready for submission deployment")
print("="*70 + "\n")
