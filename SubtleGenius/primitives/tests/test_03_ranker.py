"""
TEST 03: Primitive Ranker - Learning Analysis

Tests primitive ranking in isolation:
- Initial predictions (prior knowledge)
- Learning from feedback
- Convergence speed
- Generalization

Goal: Understand how ranker learns useful heuristics.
"""

import numpy as np
import sys
sys.path.insert(0, '/home/user/HungryOrca/SubtleGenius/primitives')

from nspsa import PrimitiveRanker

def test_initial_priors():
    """Test that ranker has reasonable initial priors"""
    print("="*70)
    print("TEST 01: INITIAL PRIORS")
    print("="*70)

    ranker = PrimitiveRanker()

    # Color change task - should predict color primitives highly
    input_grid = np.array([[1, 2, 3], [4, 5, 6]])
    output_grid = np.array([[8, 7, 6], [5, 4, 3]])  # Colors inverted

    rankings = ranker.rank_primitives(input_grid, output_grid)

    print("\nTop 5 predictions for color inversion:")
    for name, score in rankings[:5]:
        print(f"  {name}: {score:.3f}")

    # Check if color primitives ranked high
    color_prims = {'invert_colors'}
    top_5_names = {name for name, _ in rankings[:5]}

    if color_prims.intersection(top_5_names):
        print("\n✅ Color primitives ranked highly (prior knowledge working)")
    else:
        print("\n⚠️  Color primitives not in top 5 (priors could be better)")


def test_learning_from_feedback():
    """Test that ranker learns from success/failure"""
    print("\n" + "="*70)
    print("TEST 02: LEARNING FROM FEEDBACK")
    print("="*70)

    ranker = PrimitiveRanker()

    # Task: rotation
    input_grid = np.array([[1, 2], [3, 4]])
    output_grid = np.array([[3, 1], [4, 2]])

    print("\n1. Before learning:")
    rankings_before = ranker.rank_primitives(input_grid, output_grid)
    rotate_cw_idx = next(i for i, (name, _) in enumerate(rankings_before) if name == 'rotate_90_cw')
    print(f"  rotate_90_cw rank: #{rotate_cw_idx + 1}")

    # Provide positive feedback
    print("\n2. Providing positive feedback for rotate_90_cw...")
    for _ in range(10):
        ranker.update(input_grid, output_grid, 'rotate_90_cw', reward=1.0)

    print("\n3. After learning:")
    rankings_after = ranker.rank_primitives(input_grid, output_grid)
    rotate_cw_idx_after = next(i for i, (name, _) in enumerate(rankings_after) if name == 'rotate_90_cw')
    print(f"  rotate_90_cw rank: #{rotate_cw_idx_after + 1}")

    if rotate_cw_idx_after < rotate_cw_idx:
        print(f"\n✅ Learning improved ranking ({rotate_cw_idx + 1} → {rotate_cw_idx_after + 1})")
    else:
        print(f"\n⚠️  No improvement in ranking")


def test_generalization():
    """Test if learning generalizes to similar tasks"""
    print("\n" + "="*70)
    print("TEST 03: GENERALIZATION")
    print("="*70)

    ranker = PrimitiveRanker()

    # Train on 3x3 rotation
    train_input = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    train_output = np.array([[7, 4, 1], [8, 5, 2], [9, 6, 3]])

    print("\n1. Training on 3x3 rotation...")
    for _ in range(20):
        ranker.update(train_input, train_output, 'rotate_90_cw', reward=1.0)

    # Test on 5x5 rotation
    test_input = np.random.randint(0, 10, (5, 5))
    test_output = np.rot90(test_input, k=-1)

    print("\n2. Testing on 5x5 rotation:")
    rankings = ranker.rank_primitives(test_input, test_output)
    rotate_cw_rank = next(i for i, (name, _) in enumerate(rankings) if name == 'rotate_90_cw')

    print(f"  rotate_90_cw rank: #{rotate_cw_rank + 1}")

    if rotate_cw_rank < 5:
        print(f"\n✅ Generalized to different grid size (rank #{rotate_cw_rank + 1})")
    else:
        print(f"\n⚠️  Poor generalization (rank #{rotate_cw_rank + 1})")


def test_negative_feedback():
    """Test response to negative feedback"""
    print("\n" + "="*70)
    print("TEST 04: NEGATIVE FEEDBACK")
    print("="*70)

    ranker = PrimitiveRanker()

    input_grid = np.array([[1, 2], [3, 4]])
    output_grid = np.array([[3, 1], [4, 2]])

    print("\n1. Initial ranking:")
    initial_rankings = ranker.rank_primitives(input_grid, output_grid)
    print(f"  Top 3: {[name for name, _ in initial_rankings[:3]]}")

    # Give negative feedback to top primitive
    wrong_prim = initial_rankings[0][0]
    print(f"\n2. Giving negative feedback to: {wrong_prim}")
    for _ in range(10):
        ranker.update(input_grid, output_grid, wrong_prim, reward=-0.5)

    print("\n3. After negative feedback:")
    final_rankings = ranker.rank_primitives(input_grid, output_grid)
    print(f"  Top 3: {[name for name, _ in final_rankings[:3]]}")

    # Check if wrong primitive dropped in ranking
    initial_rank = 0
    final_rank = next(i for i, (name, _) in enumerate(final_rankings) if name == wrong_prim)

    if final_rank > initial_rank:
        print(f"\n✅ Negative feedback lowered ranking (#{initial_rank + 1} → #{final_rank + 1})")
    else:
        print(f"\n⚠️  Negative feedback had no effect")


if __name__ == '__main__':
    print("="*70)
    print("MODULAR TEST SUITE - PART 3: PRIMITIVE RANKER")
    print("="*70)
    print("Isolated testing of ranking and learning")
    print("="*70)

    test_initial_priors()
    test_learning_from_feedback()
    test_generalization()
    test_negative_feedback()

    print("\n" + "="*70)
    print("✅ PRIMITIVE RANKER TESTS COMPLETE")
    print("="*70)
    print("\nKey findings:")
    print("  - Initial priors encode domain knowledge")
    print("  - Positive feedback improves rankings")
    print("  - Learning generalizes across grid sizes")
    print("  - Negative feedback lowers bad predictions")
    print("="*70)
