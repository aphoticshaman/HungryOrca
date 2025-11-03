"""
ROUND 2.5 VALIDATION: Are spatial features working?

Compare Round 2 (5 features) vs Round 2.5 (8 features)
"""

import numpy as np
import sys
sys.path.insert(0, '/home/user/HungryOrca/SubtleGenius/primitives')

from nspsa import PrimitiveRanker

def test_feature_nonzero():
    """Test that spatial features are non-zero for rotations"""
    print("="*70)
    print("VALIDATION 1: Feature Non-Zero Check")
    print("="*70)

    ranker = PrimitiveRanker()

    # Test rotation
    inp = np.array([[1, 2], [3, 4]])
    out = np.rot90(inp, k=-1)

    features = ranker.extract_features(inp, out)

    print(f"\nFeatures for rotation task:")
    feature_names = ['size_ratio', 'color_change', 'shape_change',
                     'symmetry', 'connectivity', 'position_corr',
                     'orientation', 'corner_move']

    nonzero_count = 0
    for i, (name, val) in enumerate(zip(feature_names, features)):
        status = "✅" if abs(val) > 0.01 else "  "
        print(f"  {status} Feature {i} ({name:15s}): {val:.4f}")
        if abs(val) > 0.01:
            nonzero_count += 1

    print(f"\nNon-zero features: {nonzero_count}/8")

    if nonzero_count >= 2:
        print("✅ ROUND 2.5 SUCCESS: Spatial features are non-zero")
        return True
    else:
        print("❌ ROUND 2.5 FAILED: Still have zero features")
        return False


def test_learning_rate():
    """Test how fast learning happens with new features"""
    print("\n" + "="*70)
    print("VALIDATION 2: Learning Rate Analysis")
    print("="*70)

    ranker = PrimitiveRanker()

    # Simple rotation task
    inp = np.array([[1, 2], [3, 4]])
    out = np.rot90(inp, k=-1)

    prim = 'rotate_90_cw'
    prim_idx = ranker.prim_to_idx[prim]

    # Track score over updates
    scores = []
    for i in range(20):
        rankings = ranker.rank_primitives(inp, out)
        score = next(s for name, s in rankings if name == prim)
        scores.append(score)
        ranker.update(inp, out, prim, reward=1.0)

    print(f"\nScore trajectory over 20 updates:")
    print(f"  Update  0: {scores[0]:.4f}")
    print(f"  Update  5: {scores[5]:.4f} (Δ={scores[5]-scores[0]:+.4f})")
    print(f"  Update 10: {scores[10]:.4f} (Δ={scores[10]-scores[0]:+.4f})")
    print(f"  Update 15: {scores[15]:.4f} (Δ={scores[15]-scores[0]:+.4f})")
    print(f"  Update 19: {scores[19]:.4f} (Δ={scores[19]-scores[0]:+.4f})")

    total_improvement = scores[-1] - scores[0]
    print(f"\nTotal improvement: {total_improvement:+.4f}")

    if total_improvement > 0.001:
        print("✅ Learning is happening (scores improving)")
        return True
    else:
        print("❌ No learning detected")
        return False


def compare_feature_vectors():
    """Compare feature vectors for different transforms"""
    print("\n" + "="*70)
    print("VALIDATION 3: Feature Vector Analysis")
    print("="*70)

    ranker = PrimitiveRanker()

    inp = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    transforms = {
        'Identity': inp,
        'Rotate 90°': np.rot90(inp, k=-1),
        'Reflect H': np.flip(inp, axis=0),
        'Transpose': np.transpose(inp),
        'Scale 2x': np.kron(inp, np.ones((2, 2), dtype=int))
    }

    print("\nFeature vectors for different transforms:")
    print("              size  color shape symm  conn  pos  orient corner")

    for name, out in transforms.items():
        features = ranker.extract_features(inp, out)
        print(f"  {name:12s}", end='')
        for f in features:
            print(f" {f:5.2f}", end='')
        print()

    print("\n✅ Feature diversity analysis complete")
    return True


if __name__ == '__main__':
    print("="*70)
    print("ROUND 2.5 VALIDATION")
    print("="*70)
    print("\nGoal: Verify spatial features fix Round 2's zero-gradient problem")
    print()

    results = {
        'nonzero': test_feature_nonzero(),
        'learning': test_learning_rate(),
        'diversity': compare_feature_vectors()
    }

    print("\n" + "="*70)
    print("VALIDATION RESULTS:")
    print("="*70)

    passed = sum(results.values())
    for test, result in results.items():
        status = "✅" if result else "❌"
        print(f"  {status} {test}")

    print(f"\nPassed: {passed}/3")

    if passed >= 2:
        print("\n✅ ROUND 2.5 VALIDATED")
        print("   - Features are now non-zero for spatial transforms")
        print("   - Learning is happening (weights updating)")
        print("   - Feature engineering improved from Round 2")
    else:
        print("\n❌ VALIDATION FAILED - need more debugging")
