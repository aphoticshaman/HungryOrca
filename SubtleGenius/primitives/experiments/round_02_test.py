"""
ROUND 2 TEST: Ranker Learning with Gradient Descent

Tests:
1. Ranker discovers all 20 primitives (not hardcoded 15)
2. Weights update properly after feedback
3. Learning converges (ranking improves over time)
4. Composition primitives get ranked appropriately
"""

import numpy as np
import sys
sys.path.insert(0, '/home/user/HungryOrca/SubtleGenius/primitives')

from nspsa import PrimitiveRanker
from symbolic_solver import SymbolicPrimitiveExecutor

def test_dynamic_discovery():
    """Test that ranker discovers all primitives dynamically"""
    print("="*70)
    print("TEST 1: Dynamic Primitive Discovery")
    print("="*70)

    ranker = PrimitiveRanker()

    print(f"\nDiscovered {ranker.num_primitives} primitives:")
    for i, name in enumerate(ranker.primitive_names):
        print(f"  {i:2d}. {name}")

    # Check for Round 1 compositions
    compositions = ['rotate_reflect_h', 'rotate_reflect_v', 'scale_rotate',
                   'tile_invert', 'reflect_transpose']

    found = sum(1 for comp in compositions if comp in ranker.primitive_names)

    print(f"\nComposition primitives found: {found}/{len(compositions)}")

    if ranker.num_primitives == 20 and found == 5:
        print("✅ Dynamic discovery working - found all 20 primitives")
        return True
    else:
        print(f"⚠️  Expected 20 primitives, found {ranker.num_primitives}")
        return False


def test_weight_updates():
    """Test that weights update properly with feedback"""
    print("\n" + "="*70)
    print("TEST 2: Weight Update Mechanics")
    print("="*70)

    ranker = PrimitiveRanker()

    # Task: rotation
    inp = np.array([[1, 2], [3, 4]])
    out = np.rot90(inp, k=-1)

    # Get initial ranking
    initial_rankings = ranker.rank_primitives(inp, out)
    initial_score = next(score for name, score in initial_rankings if name == 'rotate_90_cw')
    print(f"\nInitial score for rotate_90_cw: {initial_score:.4f}")

    # Capture initial weights
    prim_idx = ranker.prim_to_idx['rotate_90_cw']
    initial_weights = ranker.weights[:, prim_idx].copy()

    # Give positive feedback 10 times
    for i in range(10):
        ranker.update(inp, out, 'rotate_90_cw', reward=1.0)

    # Get final ranking
    final_rankings = ranker.rank_primitives(inp, out)
    final_score = next(score for name, score in final_rankings if name == 'rotate_90_cw')
    print(f"After 10 updates: {final_score:.4f}")

    # Check weights changed
    final_weights = ranker.weights[:, prim_idx]
    weight_change = np.linalg.norm(final_weights - initial_weights)

    print(f"\nWeight change (L2 norm): {weight_change:.4f}")
    print(f"Learning rate (after decay): {ranker.learning_rate:.6f}")

    if final_score > initial_score and weight_change > 0.01:
        print("✅ Weights updating properly")
        return True
    else:
        print("❌ Weights not updating or score not improving")
        return False


def test_learning_convergence():
    """Test that ranker learns over many updates"""
    print("\n" + "="*70)
    print("TEST 3: Learning Convergence")
    print("="*70)

    ranker = PrimitiveRanker()

    # Generate 50 rotation tasks
    tasks = []
    for _ in range(50):
        size = np.random.choice([2, 3, 4])
        inp = np.random.randint(0, 10, (size, size))
        out = np.rot90(inp, k=-1)
        tasks.append((inp, out, 'rotate_90_cw'))

    # Track score over time
    scores = []

    for i, (inp, out, prim) in enumerate(tasks):
        # Get current score
        rankings = ranker.rank_primitives(inp, out)
        score = next(s for name, s in rankings if name == prim)
        scores.append(score)

        # Update
        ranker.update(inp, out, prim, reward=1.0)

        if (i+1) % 10 == 0:
            print(f"  After {i+1} tasks: score={score:.4f}, lr={ranker.learning_rate:.6f}")

    # Analyze convergence
    early_avg = np.mean(scores[:10])
    late_avg = np.mean(scores[-10:])
    improvement = late_avg - early_avg

    print(f"\nScore trajectory:")
    print(f"  First 10 tasks: {early_avg:.4f}")
    print(f"  Last 10 tasks:  {late_avg:.4f}")
    print(f"  Improvement:    {improvement:+.4f}")

    if improvement > 0.05:
        print("✅ Learning converges - scores improving")
        return True
    else:
        print("⚠️  Weak convergence - scores barely changed")
        return False


def test_composition_ranking():
    """Test that composition primitives get ranked appropriately"""
    print("\n" + "="*70)
    print("TEST 4: Composition Primitive Ranking")
    print("="*70)

    ranker = PrimitiveRanker()

    # Task: rotate + reflect (composition)
    inp = np.array([[1, 2, 3], [4, 5, 6]])
    out = np.flip(np.rot90(inp, k=-1), axis=1)

    # Initial ranking
    print("\nInitial ranking for rotate+reflect task:")
    initial_rankings = ranker.rank_primitives(inp, out)
    for name, score in initial_rankings[:10]:
        marker = "**" if 'rotate' in name and 'reflect' in name else "  "
        print(f"  {marker} {name}: {score:.4f}")

    # Train on composition tasks
    for _ in range(20):
        ranker.update(inp, out, 'rotate_reflect_h', reward=1.0)

    # Final ranking
    print("\nAfter training on 20 composition tasks:")
    final_rankings = ranker.rank_primitives(inp, out)
    for name, score in final_rankings[:10]:
        marker = "**" if 'rotate' in name and 'reflect' in name else "  "
        print(f"  {marker} {name}: {score:.4f}")

    # Check if composition is in top 3
    top_3_names = [name for name, _ in final_rankings[:3]]

    if 'rotate_reflect_h' in top_3_names or 'rotate_reflect_v' in top_3_names:
        print("\n✅ Composition primitive ranked in top 3")
        return True
    else:
        print("\n⚠️  Composition primitive not in top 3")
        return False


if __name__ == '__main__':
    print("="*70)
    print("ROUND 2 TEST: Ranker Learning")
    print("="*70)

    results = {
        'discovery': test_dynamic_discovery(),
        'updates': test_weight_updates(),
        'convergence': test_learning_convergence(),
        'compositions': test_composition_ranking()
    }

    print("\n" + "="*70)
    print("ROUND 2 RESULTS:")
    print("="*70)

    passed = sum(results.values())
    total = len(results)

    for test_name, passed_test in results.items():
        status = "✅" if passed_test else "❌"
        print(f"  {status} {test_name}")

    print(f"\nPassed: {passed}/{total}")

    if passed == total:
        print("\n✅ ROUND 2 COMPLETE: All tests passed")
        print("   Ranker now:")
        print("   - Discovers 20 primitives dynamically")
        print("   - Updates with proper gradient descent")
        print("   - Learns to rank compositions correctly")
    else:
        print(f"\n⚠️  {total-passed} tests failed - need debugging")

    print("="*70)
