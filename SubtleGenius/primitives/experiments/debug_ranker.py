"""Debug ranker weight updates"""

import numpy as np
import sys
sys.path.insert(0, '/home/user/HungryOrca/SubtleGenius/primitives')

from nspsa import PrimitiveRanker

# Simple rotation task
inp = np.array([[1, 2], [3, 4]])
out = np.rot90(inp, k=-1)

ranker = PrimitiveRanker()
prim = 'rotate_90_cw'
prim_idx = ranker.prim_to_idx[prim]

print("BEFORE UPDATE:")
print(f"  Weights: {ranker.weights[:, prim_idx]}")

# Extract features
features = ranker.extract_features(inp, out)
print(f"\nFeatures: {features}")
print(f"  Feature magnitudes: min={features.min():.4f}, max={features.max():.4f}")

# Compute prediction
logit = features @ ranker.weights[:, prim_idx]
p = 1.0 / (1.0 + np.exp(-logit))
print(f"\nLogit: {logit:.4f}")
print(f"Prediction (p): {p:.4f}")

# Error
reward = 1.0
error = reward - p
print(f"\nReward: {reward}")
print(f"Error: {error:.4f}")

# Gradient components
sigmoid_deriv = p * (1 - p)
print(f"\nSigmoid derivative (p*(1-p)): {sigmoid_deriv:.4f}")

grad_before_reg = error * sigmoid_deriv * features
print(f"Gradient (before reg): {grad_before_reg}")
print(f"  Gradient magnitude: {np.linalg.norm(grad_before_reg):.6f}")

# L2 regularization
l2_term = ranker.l2_lambda * ranker.weights[:, prim_idx]
print(f"\nL2 regularization term (lambda * weights): {l2_term}")
print(f"  L2 magnitude: {np.linalg.norm(l2_term):.6f}")

gradient = grad_before_reg - l2_term
print(f"\nGradient (after reg): {gradient}")
print(f"  Final gradient magnitude: {np.linalg.norm(gradient):.6f}")

# Momentum (starts at zero)
print(f"\nMomentum (before): {ranker.momentum[:, prim_idx]}")

# Update
ranker.update(inp, out, prim, reward=1.0)

print(f"\nAFTER UPDATE:")
print(f"  Weights: {ranker.weights[:, prim_idx]}")
print(f"  Weight change: {np.linalg.norm(ranker.weights[:, prim_idx] - ranker.weights[:, prim_idx]):.6f}")
print(f"  Momentum (after): {ranker.momentum[:, prim_idx]}")
print(f"  Learning rate: {ranker.learning_rate:.6f}")

# Try predicting again
logit_after = features @ ranker.weights[:, prim_idx]
p_after = 1.0 / (1.0 + np.exp(-logit_after))
print(f"\nPrediction after update: {p_after:.4f}")
print(f"Change in prediction: {p_after - p:.6f}")
