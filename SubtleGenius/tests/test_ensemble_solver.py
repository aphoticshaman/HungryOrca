"""
Test suite for Iteration 3: Ensemble Methods & Voting

Tests all new solvers and the voting mechanism.

Run with: python tests/test_ensemble_solver.py
"""

import sys
import os

# Add notebooks directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'notebooks'))

import numpy as np
from cell5_iteration3_ensemble import (
    detect_grid_arithmetic,
    apply_grid_arithmetic,
    detect_incomplete_symmetry,
    complete_symmetry,
    apply_symmetry_completion,
    detect_color_frequency_pattern,
    apply_color_frequency,
    vote_on_predictions,
    ensemble_solver,
    get_ensemble_statistics,
    SolverPrediction,
)


# ============================================================================
# TEST 1: GRID ARITHMETIC SOLVER
# ============================================================================

def test_grid_arithmetic_addition():
    """Test detection of addition pattern: output = input + constant"""
    print("Test 1: Grid arithmetic - addition")

    task_data = {
        "train": [
            {
                "input": [[1, 2], [3, 4]],
                "output": [[6, 7], [8, 9]]
            },
            {
                "input": [[0, 1], [2, 3]],
                "output": [[5, 6], [7, 8]]
            }
        ]
    }

    result = detect_grid_arithmetic(task_data)
    assert result is not None, "Should detect addition pattern"

    pattern_name, func, confidence = result
    assert pattern_name == "add_5", f"Expected 'add_5', got '{pattern_name}'"
    assert confidence > 0.9, f"Confidence should be high, got {confidence}"

    # Test on new input
    test_input = [[10, 20], [30, 40]]
    prediction, conf = apply_grid_arithmetic(test_input, task_data)
    expected = [[15, 25], [35, 45]]

    assert np.array_equal(prediction, expected), \
        f"Expected {expected}, got {prediction}"

    print("  ✓ Addition pattern detected and applied correctly")


def test_grid_arithmetic_modulo():
    """Test detection of modulo pattern: output = input % k"""
    print("Test 2: Grid arithmetic - modulo")

    task_data = {
        "train": [
            {
                "input": [[5, 7], [9, 11]],
                "output": [[2, 1], [0, 2]]
            },
            {
                "input": [[3, 6], [12, 15]],
                "output": [[0, 0], [0, 0]]
            }
        ]
    }

    result = detect_grid_arithmetic(task_data)
    assert result is not None, "Should detect modulo pattern"

    pattern_name, func, confidence = result
    assert pattern_name == "mod_3", f"Expected 'mod_3', got '{pattern_name}'"

    print("  ✓ Modulo pattern detected correctly")


# ============================================================================
# TEST 2: SYMMETRY COMPLETION SOLVER
# ============================================================================

def test_symmetry_detection():
    """Test detection of incomplete symmetry"""
    print("Test 3: Symmetry detection")

    # Horizontally incomplete symmetric grid
    grid = [
        [1, 2, 0, 0],
        [3, 4, 0, 0],
        [5, 6, 0, 0]
    ]

    result = detect_incomplete_symmetry(grid)
    assert result is not None, "Should detect incomplete horizontal symmetry"

    sym_type, score = result
    print(f"  Detected: {sym_type} with score {score:.2f}")
    assert 0.4 < score < 0.6, f"Score should be around 0.5, got {score}"

    print("  ✓ Incomplete symmetry detected")


def test_symmetry_completion():
    """Test completion of horizontal symmetry"""
    print("Test 4: Symmetry completion")

    grid = [
        [1, 2, 0, 0],
        [3, 4, 0, 0]
    ]

    result = complete_symmetry(grid, "horizontal")
    expected = [
        [1, 2, 2, 1],
        [3, 4, 4, 3]
    ]

    assert np.array_equal(result, expected), \
        f"Expected {expected}, got {result}"

    print("  ✓ Horizontal symmetry completed correctly")


def test_symmetry_vertical():
    """Test vertical symmetry completion"""
    print("Test 5: Vertical symmetry completion")

    grid = [
        [1, 2, 3],
        [0, 0, 0]
    ]

    result = complete_symmetry(grid, "vertical")
    expected = [
        [1, 2, 3],
        [1, 2, 3]
    ]

    assert np.array_equal(result, expected), \
        f"Expected {expected}, got {result}"

    print("  ✓ Vertical symmetry completed correctly")


# ============================================================================
# TEST 3: COLOR FREQUENCY SOLVER
# ============================================================================

def test_color_frequency_rare():
    """Test promote rare colors pattern"""
    print("Test 6: Color frequency - promote rare")

    task_data = {
        "train": [
            {
                "input": [[0, 0, 0, 0, 1, 2],
                         [0, 0, 0, 0, 1, 2]],
                "output": [[0, 0, 0, 0, 1, 2],
                          [0, 0, 0, 0, 1, 2]]
            }
        ]
    }

    result = detect_color_frequency_pattern(task_data)
    if result is not None:
        pattern_name, func, confidence = result
        print(f"  Detected pattern: {pattern_name} with confidence {confidence:.2f}")
        print("  ✓ Color frequency pattern detected")
    else:
        print("  ⚠ No pattern detected (expected for this simple case)")


# ============================================================================
# TEST 4: VOTING MECHANISM
# ============================================================================

def test_voting_agreement():
    """Test voting when multiple solvers agree"""
    print("Test 7: Voting - multiple solvers agree")

    test_input = [[1, 2], [3, 4]]

    predictions = [
        SolverPrediction([[5, 6], [7, 8]], 0.8, "solver_a"),
        SolverPrediction([[5, 6], [7, 8]], 0.7, "solver_b"),
        SolverPrediction([[9, 10], [11, 12]], 0.6, "solver_c"),
    ]

    result = vote_on_predictions(predictions, attempt=1, test_input=test_input)
    expected = [[5, 6], [7, 8]]

    assert np.array_equal(result, expected), \
        f"Expected {expected}, got {result}"

    print("  ✓ Voting correctly selected majority prediction")


def test_voting_confidence_weighted():
    """Test that voting is confidence-weighted, not just majority"""
    print("Test 8: Voting - confidence weighting")

    test_input = [[1, 2], [3, 4]]

    predictions = [
        SolverPrediction([[5, 6], [7, 8]], 0.95, "high_conf_solver"),
        SolverPrediction([[9, 10], [11, 12]], 0.4, "low_conf_solver_a"),
        SolverPrediction([[9, 10], [11, 12]], 0.4, "low_conf_solver_b"),
    ]

    result = vote_on_predictions(predictions, attempt=1, test_input=test_input)
    expected = [[5, 6], [7, 8]]

    assert np.array_equal(result, expected), \
        f"Expected high-confidence prediction, got {result}"

    print("  ✓ Voting correctly weighted by confidence (0.95 > 0.4+0.4)")


def test_voting_attempt_2():
    """Test that attempt 2 returns second-best prediction"""
    print("Test 9: Voting - attempt 2 returns alternative")

    test_input = [[1, 2], [3, 4]]

    predictions = [
        SolverPrediction([[5, 6], [7, 8]], 0.8, "solver_a"),
        SolverPrediction([[9, 10], [11, 12]], 0.7, "solver_b"),
    ]

    result = vote_on_predictions(predictions, attempt=2, test_input=test_input)
    expected = [[9, 10], [11, 12]]

    assert np.array_equal(result, expected), \
        f"Expected second-best prediction, got {result}"

    print("  ✓ Attempt 2 correctly returned second-best prediction")


# ============================================================================
# TEST 5: END-TO-END ENSEMBLE SOLVER
# ============================================================================

def test_ensemble_solver():
    """Test complete ensemble solver pipeline"""
    print("Test 10: End-to-end ensemble solver")

    task_data = {
        "train": [
            {
                "input": [[1, 2], [3, 4]],
                "output": [[6, 7], [8, 9]]
            },
            {
                "input": [[0, 1], [2, 3]],
                "output": [[5, 6], [7, 8]]
            }
        ]
    }

    test_input = [[10, 20], [30, 40]]

    # Should detect addition pattern
    result = ensemble_solver(test_input, task_data, attempt=1)
    expected = [[15, 25], [35, 45]]

    assert np.array_equal(result, expected), \
        f"Expected {expected}, got {result}"

    print("  ✓ Ensemble solver correctly solved task")


def test_ensemble_statistics():
    """Test ensemble statistics collection"""
    print("Test 11: Ensemble statistics")

    task_data = {
        "train": [
            {
                "input": [[1, 2], [3, 4]],
                "output": [[6, 7], [8, 9]]
            }
        ]
    }

    test_input = [[10, 20], [30, 40]]

    stats = get_ensemble_statistics(test_input, task_data)

    print(f"  Solvers triggered: {stats['num_solvers_triggered']}")
    print(f"  Solver names: {stats['solver_names']}")

    if 'vote_winner' in stats:
        print(f"  Vote winner: {stats['vote_winner']}")
        print(f"  Agreement level: {stats['agreement_level']}")

    assert stats['num_solvers_triggered'] >= 0, "Should report solver count"

    print("  ✓ Statistics collection working")


# ============================================================================
# TEST 6: FALLBACK BEHAVIOR
# ============================================================================

def test_ensemble_fallback():
    """Test that ensemble returns identity when no solvers match"""
    print("Test 12: Ensemble fallback to identity")

    task_data = {
        "train": [
            {
                "input": [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]],
                "output": [[99, 88, 77, 66, 55], [44, 33, 22, 11, 0]]
            }
        ]
    }

    test_input = [[1, 2], [3, 4]]

    # No solver should match this pattern
    result = ensemble_solver(test_input, task_data, attempt=1)

    # Should return identity (input unchanged)
    assert np.array_equal(result, test_input), \
        f"Expected identity fallback, got {result}"

    print("  ✓ Ensemble correctly falls back to identity")


# ============================================================================
# RUN ALL TESTS
# ============================================================================

def run_all_tests():
    """Run all test cases"""
    print("\n" + "=" * 70)
    print("ITERATION 3: ENSEMBLE METHODS - TEST SUITE")
    print("=" * 70 + "\n")

    tests = [
        test_grid_arithmetic_addition,
        test_grid_arithmetic_modulo,
        test_symmetry_detection,
        test_symmetry_completion,
        test_symmetry_vertical,
        test_color_frequency_rare,
        test_voting_agreement,
        test_voting_confidence_weighted,
        test_voting_attempt_2,
        test_ensemble_solver,
        test_ensemble_statistics,
        test_ensemble_fallback,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"  ✗ FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            failed += 1

    print("\n" + "=" * 70)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 70 + "\n")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
