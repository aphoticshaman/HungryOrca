#!/usr/bin/env python3
"""
Create Kaggle submission file from trained quantum models
"""

import os
import sys
import pickle
import pandas as pd
import numpy as np
from pathlib import Path

# Set working directory
script_dir = Path(__file__).parent
os.chdir(script_dir)

print("="*70)
print("🚀 QUANTUM SUBMISSION GENERATOR")
print("="*70)
print()

# =============================================================================
# 1. LOAD TRAINED MODELS
# =============================================================================

print("📦 Loading trained quantum models...")
with open('models/greyhat_production.pkl', 'rb') as f:
    production_bundle = pickle.load(f)

scaler = production_bundle['scaler']
healer = production_bundle['models']['healer']
tank = production_bundle['models']['tank']
dps = production_bundle['models']['dps']
exploiter = production_bundle['quantum_exploiter']
feature_cols = production_bundle['feature_cols']
exploited_feature_cols = production_bundle['exploited_feature_cols']

print(f"  ✅ Loaded 3 specialist models + quantum exploiter")
print(f"  Performance (validation): Sharpe {production_bundle['performance']['sharpe']:.4f}")
print(f"  Expected features after exploitation: {len(exploited_feature_cols)}")
print()

# =============================================================================
# 2. LOAD TEST DATA
# =============================================================================

print("📊 Loading test data...")
test_df = pd.read_csv('data/test.csv')
print(f"  Test samples: {len(test_df)}")
print(f"  Date range: {test_df['date_id'].min()} - {test_df['date_id'].max()}")
print()

# Extract features (use same feature_cols as training)
X_test = test_df[feature_cols].copy()
print(f"  Features: {len(feature_cols)}")

# =============================================================================
# 3. EXPLOIT QUANTUM FEATURES
# =============================================================================

print("⚛️  Applying quantum exploitation...")

# Synthesize orthogonal features
X_test = exploiter.orthogonal_synth.synthesize_perpendicular_features(X_test)
print(f"  + Orthogonal features: {X_test.shape[1] - len(feature_cols)}")

# Scan for vulnerabilities
vulns = exploiter.vuln_scanner.scan_feature_space(X_test)
X_test = exploiter.vuln_scanner.exploit_vulnerabilities(X_test, vulns)
print(f"  + Exploit features: {len(vulns['correlation_clusters'][:10]) + len(vulns['temporal_dependencies'][:10])}")

# Handle NaN and infinity
print(f"  Handling NaN/Inf values...")
print(f"    NaN before: {X_test.isna().sum().sum()}")
print(f"    Inf before: {np.isinf(X_test.values).sum()}")
X_test = X_test.fillna(0)
X_test = X_test.replace([np.inf, -np.inf], 0)
print(f"    NaN after: {X_test.isna().sum().sum()}")
print(f"    Inf after: {np.isinf(X_test.values).sum()}")

# Align features with training (critical for scaler/models)
print(f"\n🔧 Aligning features with training set...")
print(f"  Test features before: {len(X_test.columns)}")

# Add missing columns (fill with 0)
for col in exploited_feature_cols:
    if col not in X_test.columns:
        X_test[col] = 0

# Remove extra columns
X_test = X_test[exploited_feature_cols]

print(f"  Test features after: {len(X_test.columns)}")
print(f"  ✅ Features aligned with training")
print()

# =============================================================================
# 4. GENERATE PREDICTIONS
# =============================================================================

print("🎯 Generating predictions...")

# Scale for Ridge (healer)
X_test_scaled = scaler.transform(X_test)

# Get specialist predictions
healer_pred = healer.predict(X_test_scaled)
tank_pred = tank.predict(X_test)
dps_pred = dps.predict(X_test)

print(f"  Healer (Ridge) predictions: {len(healer_pred)}")
print(f"  Tank (Random Forest) predictions: {len(tank_pred)}")
print(f"  DPS (Gradient Boosting) predictions: {len(dps_pred)}")
print()

# =============================================================================
# 5. QUANTUM EXPLOITATION
# =============================================================================

print("⚛️  Applying quantum exploitation pipeline...")

predictions = []
allocations = []

# For test, we don't have historical returns, so use neutral basin
# In production, this would use rolling returns from train+test
for i in range(len(test_df)):
    specialist_preds = [healer_pred[i], tank_pred[i], dps_pred[i]]

    # For test, create dummy returns history (use mean from training)
    # In real deployment, this would be actual historical returns
    dummy_returns = pd.Series([0.0005] * 20)  # Neutral positive returns

    # Apply quantum exploitation
    allocation, report = exploiter.exploit(
        predictions=specialist_preds,
        features=X_test.iloc[[i]],
        returns_history=dummy_returns
    )

    allocations.append(allocation)
    predictions.append(report)

allocations = np.array(allocations)

print(f"  Mean allocation: {allocations.mean():.4f}")
print(f"  Min allocation: {allocations.min():.4f}")
print(f"  Max allocation: {allocations.max():.4f}")
print(f"  Std allocation: {allocations.std():.4f}")
print()

# =============================================================================
# 6. CREATE SUBMISSION
# =============================================================================

print("💾 Creating submission file...")

submission = pd.DataFrame({
    'date_id': test_df['date_id'],
    'prediction': allocations
})

# Ensure predictions are in valid range [0, 2]
submission['prediction'] = submission['prediction'].clip(0.0, 2.0)

print(f"  Submission shape: {submission.shape}")
print(f"\nFirst 10 predictions:")
print(submission.head(10))
print()

# Save as parquet
submission.to_parquet('submission_quantum.parquet', index=False)
print(f"✅ Submission saved to: submission_quantum.parquet")
print()

# =============================================================================
# 7. VALIDATE SUBMISSION
# =============================================================================

print("🔍 Validating submission...")

# Check format
assert list(submission.columns) == ['date_id', 'prediction'], "Column names must be ['date_id', 'prediction']"
assert len(submission) == len(test_df), f"Submission must have {len(test_df)} rows"
assert submission['prediction'].min() >= 0.0, "Predictions must be >= 0.0"
assert submission['prediction'].max() <= 2.0, "Predictions must be <= 2.0"
assert submission['date_id'].equals(test_df['date_id']), "date_id must match test set"

print("  ✅ Column names correct")
print("  ✅ Row count matches test set")
print("  ✅ Predictions in valid range [0, 2]")
print("  ✅ date_id matches test set")
print()

print("="*70)
print("🏆 SUBMISSION READY FOR KAGGLE")
print("="*70)
print(f"\nFile: submission_quantum.parquet")
print(f"Rows: {len(submission)}")
print(f"Mean allocation: {submission['prediction'].mean():.4f}")
print()
print("Next step: Upload to Kaggle competition")
print("="*70)
