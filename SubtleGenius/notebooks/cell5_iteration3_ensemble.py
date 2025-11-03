"""
Iteration 3: Ensemble Methods & Voting

This module implements ensemble voting across multiple diverse solvers.
Instead of cascading (pick first match), we generate predictions from ALL
solvers and use confidence-weighted voting to select the best answer.

Key Innovation: Wisdom of crowds - agreement between diverse solvers
indicates correctness, even if individual solvers aren't perfect.

Architecture:
    1. Object transformation solver (from Iteration 2)
    2. Pattern matching solver (from Iteration 1)
    3. Grid arithmetic solver (NEW)
    4. Symmetry completion solver (NEW)
    5. Color frequency solver (NEW)

Performance Target: 40-50% accuracy
Expected Improvement: +15-20% over Iteration 2

Author: SubtleGenius System
Date: 2025-11-02
"""

from typing import List, Dict, Optional, Tuple, Callable, Any
from dataclasses import dataclass
import numpy as np
from collections import Counter


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def grid_to_tuple(grid: List[List[int]]) -> Tuple:
    """Convert grid to hashable tuple for comparison."""
    return tuple(tuple(row) for row in grid)


def grids_equal(g1: List[List[int]], g2: List[List[int]]) -> bool:
    """Check if two grids are identical."""
    return np.array_equal(np.array(g1), np.array(g2))


# ============================================================================
# SOLVER 1: GRID ARITHMETIC (NEW)
# ============================================================================

def detect_grid_arithmetic(task_data: Dict) -> Optional[Tuple[str, Callable, float]]:
    """
    Detect if output = arithmetic_operation(input).

    Tests these operations:
        - Addition of constant: input + k
        - Multiplication by constant: input * k
        - Modulo operation: input % k
        - Subtraction: input - k
        - Min/max clipping: clip(input, min, max)

    Args:
        task_data: Dictionary with 'train' key containing training pairs

    Returns:
        (operation_name, operation_function, confidence) if pattern found
        None if no consistent arithmetic pattern
    """
    train_pairs = task_data.get('train', [])
    if len(train_pairs) < 1:
        return None

    # Test addition
    add_constants = []
    for pair in train_pairs:
        inp = np.array(pair['input'])
        out = np.array(pair['output'])

        # Must be same shape for arithmetic
        if inp.shape != out.shape:
            break

        diff = out - inp
        # Check if all differences are the same
        if len(np.unique(diff)) == 1:
            add_constants.append(diff.flat[0])
        else:
            break
    else:
        # All pairs have consistent addition
        if len(set(add_constants)) == 1:
            k = add_constants[0]
            if k != 0:  # Don't report identity as addition
                return (
                    f"add_{k}",
                    lambda grid: (np.array(grid) + k).tolist(),
                    0.95
                )

    # Test multiplication
    mult_constants = []
    for pair in train_pairs:
        inp = np.array(pair['input'])
        out = np.array(pair['output'])

        if inp.shape != out.shape:
            break

        # Avoid division by zero
        if np.any(inp == 0):
            # Check if output is also 0 where input is 0
            if not np.all(out[inp == 0] == 0):
                break

        non_zero_mask = inp != 0
        if np.any(non_zero_mask):
            ratios = out[non_zero_mask] / inp[non_zero_mask]
            if len(np.unique(ratios)) == 1:
                mult_constants.append(ratios[0])
            else:
                break
    else:
        if len(set(mult_constants)) == 1:
            k = mult_constants[0]
            if k != 1:  # Don't report identity as multiplication
                return (
                    f"multiply_{k}",
                    lambda grid: (np.array(grid) * k).astype(int).tolist(),
                    0.95
                )

    # Test modulo
    for mod_val in [2, 3, 4, 5, 10]:
        consistent = True
        for pair in train_pairs:
            inp = np.array(pair['input'])
            out = np.array(pair['output'])

            if inp.shape != out.shape:
                consistent = False
                break

            expected = inp % mod_val
            if not np.array_equal(out, expected):
                consistent = False
                break

        if consistent:
            return (
                f"mod_{mod_val}",
                lambda grid, m=mod_val: (np.array(grid) % m).tolist(),
                0.90
            )

    # Test min/max clipping
    for min_val in range(0, 5):
        for max_val in range(5, 10):
            consistent = True
            for pair in train_pairs:
                inp = np.array(pair['input'])
                out = np.array(pair['output'])

                if inp.shape != out.shape:
                    consistent = False
                    break

                expected = np.clip(inp, min_val, max_val)
                if not np.array_equal(out, expected):
                    consistent = False
                    break

            if consistent:
                return (
                    f"clip_{min_val}_{max_val}",
                    lambda grid, mn=min_val, mx=max_val: np.clip(np.array(grid), mn, mx).tolist(),
                    0.85
                )

    return None


def apply_grid_arithmetic(test_input: List[List[int]],
                          task_data: Dict) -> Optional[Tuple[List[List[int]], float]]:
    """
    Apply grid arithmetic transformation if detected.

    Returns:
        (prediction, confidence) if pattern found
        None if no arithmetic pattern
    """
    result = detect_grid_arithmetic(task_data)
    if result is None:
        return None

    operation_name, operation_func, confidence = result

    try:
        prediction = operation_func(test_input)
        return (prediction, confidence)
    except:
        return None


# ============================================================================
# SOLVER 2: SYMMETRY COMPLETION (NEW)
# ============================================================================

def measure_symmetry_score(grid: np.ndarray, symmetry_type: str) -> float:
    """
    Measure how symmetric a grid is (0.0 = not symmetric, 1.0 = perfect).

    Args:
        grid: Input grid as numpy array
        symmetry_type: "horizontal", "vertical", "diagonal", "rot180"

    Returns:
        Symmetry score 0.0-1.0
    """
    h, w = grid.shape

    if symmetry_type == "horizontal":
        # Compare left half with flipped right half
        left = grid[:, :w//2]
        right = np.fliplr(grid[:, w//2:])
        min_width = min(left.shape[1], right.shape[1])
        matches = np.sum(left[:, :min_width] == right[:, :min_width])
        total = h * min_width
        return matches / total if total > 0 else 0.0

    elif symmetry_type == "vertical":
        # Compare top half with flipped bottom half
        top = grid[:h//2, :]
        bottom = np.flipud(grid[h//2:, :])
        min_height = min(top.shape[0], bottom.shape[0])
        matches = np.sum(top[:min_height, :] == bottom[:min_height, :])
        total = min_height * w
        return matches / total if total > 0 else 0.0

    elif symmetry_type == "diagonal":
        # Compare with transpose
        if h != w:
            return 0.0  # Diagonal symmetry requires square grid
        matches = np.sum(grid == grid.T)
        total = h * w
        return matches / total if total > 0 else 0.0

    elif symmetry_type == "rot180":
        # Compare with 180° rotation
        rotated = np.rot90(grid, 2)
        matches = np.sum(grid == rotated)
        total = h * w
        return matches / total if total > 0 else 0.0

    return 0.0


def detect_incomplete_symmetry(grid: List[List[int]]) -> Optional[Tuple[str, float]]:
    """
    Detect if grid has partial symmetry that should be completed.

    Returns:
        (symmetry_type, score) if incomplete symmetry found (score 0.6-0.95)
        None if fully symmetric or no symmetry pattern
    """
    arr = np.array(grid)

    symmetry_types = ["horizontal", "vertical", "diagonal", "rot180"]
    scores = {}

    for sym_type in symmetry_types:
        score = measure_symmetry_score(arr, sym_type)
        scores[sym_type] = score

    # Find best symmetry type
    best_type = max(scores, key=scores.get)
    best_score = scores[best_type]

    # Incomplete symmetry: 60-95% symmetric
    if 0.6 <= best_score < 0.95:
        return (best_type, best_score)

    return None


def complete_symmetry(grid: List[List[int]], symmetry_type: str) -> List[List[int]]:
    """
    Complete the symmetry of a partially symmetric grid.

    Args:
        grid: Input grid
        symmetry_type: "horizontal", "vertical", "diagonal", "rot180"

    Returns:
        Grid with completed symmetry
    """
    arr = np.array(grid)
    h, w = arr.shape

    if symmetry_type == "horizontal":
        # Mirror left half to right half
        left = arr[:, :w//2]
        right_flipped = np.fliplr(left)

        if w % 2 == 0:
            return np.hstack([left, right_flipped]).tolist()
        else:
            # Keep middle column
            middle = arr[:, w//2:w//2+1]
            return np.hstack([left, middle, right_flipped]).tolist()

    elif symmetry_type == "vertical":
        # Mirror top half to bottom half
        top = arr[:h//2, :]
        bottom_flipped = np.flipud(top)

        if h % 2 == 0:
            return np.vstack([top, bottom_flipped]).tolist()
        else:
            # Keep middle row
            middle = arr[h//2:h//2+1, :]
            return np.vstack([top, middle, bottom_flipped]).tolist()

    elif symmetry_type == "diagonal":
        # Make symmetric across diagonal
        if h != w:
            return grid  # Can't complete non-square
        # Average with transpose to create symmetric matrix
        symmetric = (arr + arr.T) // 2
        return symmetric.tolist()

    elif symmetry_type == "rot180":
        # Make symmetric under 180° rotation
        rotated = np.rot90(arr, 2)
        # Average with rotated version
        symmetric = (arr + rotated) // 2
        return symmetric.tolist()

    return grid


def apply_symmetry_completion(test_input: List[List[int]],
                              task_data: Dict) -> Optional[Tuple[List[List[int]], float]]:
    """
    Apply symmetry completion if pattern detected in training data.

    Returns:
        (prediction, confidence) if symmetry pattern found
        None if no symmetry pattern
    """
    # Check if training pairs show symmetry completion pattern
    train_pairs = task_data.get('train', [])
    if len(train_pairs) < 1:
        return None

    symmetry_types_found = []

    for pair in train_pairs:
        inp = pair['input']
        out = pair['output']

        # Check if input has incomplete symmetry
        incomplete = detect_incomplete_symmetry(inp)
        if incomplete is None:
            continue

        sym_type, score = incomplete

        # Check if output completes this symmetry
        out_score = measure_symmetry_score(np.array(out), sym_type)

        # Output should be more symmetric than input
        if out_score > score + 0.1:
            symmetry_types_found.append(sym_type)

    if not symmetry_types_found:
        return None

    # Use most common symmetry type
    most_common = Counter(symmetry_types_found).most_common(1)[0][0]

    # Check if test input has this incomplete symmetry
    test_incomplete = detect_incomplete_symmetry(test_input)
    if test_incomplete is None:
        return None

    test_sym_type, test_score = test_incomplete

    # Apply symmetry completion
    prediction = complete_symmetry(test_input, test_sym_type)
    confidence = test_score * 0.8  # Reduce confidence for uncertainty

    return (prediction, confidence)


# ============================================================================
# SOLVER 3: COLOR FREQUENCY (NEW)
# ============================================================================

def detect_color_frequency_pattern(task_data: Dict) -> Optional[Tuple[str, Callable, float]]:
    """
    Detect if transformation is based on color frequency.

    Patterns:
        - "promote_rare": Keep only rare colors
        - "promote_common": Keep only common colors
        - "filter_by_threshold": Keep colors above/below frequency threshold

    Returns:
        (pattern_name, transform_function, confidence) if pattern found
        None if no frequency pattern
    """
    train_pairs = task_data.get('train', [])
    if len(train_pairs) < 1:
        return None

    # Test "promote rare" pattern
    promote_rare_consistent = True
    for pair in train_pairs:
        inp = np.array(pair['input'])
        out = np.array(pair['output'])

        # Count color frequencies in input
        unique, counts = np.unique(inp, return_counts=True)
        freq_dict = dict(zip(unique, counts))

        # Find rarest colors (bottom 20% by frequency)
        sorted_colors = sorted(freq_dict.items(), key=lambda x: x[1])
        threshold_idx = max(1, len(sorted_colors) // 5)
        rare_colors = set([c for c, _ in sorted_colors[:threshold_idx]])

        # Check if output contains only rare colors
        out_colors = set(np.unique(out))

        # Allow background color (0) even if not rare
        if not (out_colors <= (rare_colors | {0})):
            promote_rare_consistent = False
            break

    if promote_rare_consistent:
        def promote_rare_transform(grid):
            arr = np.array(grid)
            unique, counts = np.unique(arr, return_counts=True)
            freq_dict = dict(zip(unique, counts))

            sorted_colors = sorted(freq_dict.items(), key=lambda x: x[1])
            threshold_idx = max(1, len(sorted_colors) // 5)
            rare_colors = set([c for c, _ in sorted_colors[:threshold_idx]])

            # Keep only rare colors (and 0 as background)
            result = np.where(np.isin(arr, list(rare_colors)), arr, 0)
            return result.tolist()

        return ("promote_rare", promote_rare_transform, 0.75)

    # Test "promote common" pattern
    promote_common_consistent = True
    for pair in train_pairs:
        inp = np.array(pair['input'])
        out = np.array(pair['output'])

        # Find most common non-background color
        unique, counts = np.unique(inp, return_counts=True)
        freq_dict = dict(zip(unique, counts))

        # Remove background
        freq_dict.pop(0, None)

        if not freq_dict:
            continue

        most_common_color = max(freq_dict.items(), key=lambda x: x[1])[0]

        # Check if output is mostly this color
        out_most_common = Counter(out.flatten()).most_common(1)[0][0]

        if out_most_common != most_common_color:
            promote_common_consistent = False
            break

    if promote_common_consistent:
        def promote_common_transform(grid):
            arr = np.array(grid)
            unique, counts = np.unique(arr, return_counts=True)
            freq_dict = dict(zip(unique, counts))
            freq_dict.pop(0, None)

            if not freq_dict:
                return grid

            most_common_color = max(freq_dict.items(), key=lambda x: x[1])[0]

            # Keep only most common color (set others to 0)
            result = np.where(arr == most_common_color, arr, 0)
            return result.tolist()

        return ("promote_common", promote_common_transform, 0.70)

    return None


def apply_color_frequency(test_input: List[List[int]],
                          task_data: Dict) -> Optional[Tuple[List[List[int]], float]]:
    """
    Apply color frequency transformation if detected.

    Returns:
        (prediction, confidence) if pattern found
        None if no frequency pattern
    """
    result = detect_color_frequency_pattern(task_data)
    if result is None:
        return None

    pattern_name, transform_func, confidence = result

    try:
        prediction = transform_func(test_input)
        return (prediction, confidence)
    except:
        return None


# ============================================================================
# SOLVER 4 & 5: IMPORT FROM PREVIOUS ITERATIONS
# ============================================================================

def apply_pattern_matching(test_input: List[List[int]],
                          task_data: Dict) -> Optional[Tuple[List[List[int]], float]]:
    """
    Apply pattern matching from Iteration 1.

    Returns:
        (prediction, confidence) if pattern found
        None if no pattern
    """
    try:
        # Import from Iteration 1
        from cell5_iteration1_patterns import detect_combined_pattern

        result = detect_combined_pattern(task_data)
        if result is None:
            return None

        pattern_name, transform_func = result

        # Calculate confidence based on pattern type
        confidence_map = {
            "rotate_90_cw": 0.85,
            "rotate_180": 0.85,
            "rotate_270_cw": 0.85,
            "flip_horizontal": 0.85,
            "flip_vertical": 0.85,
            "color_mapping": 0.80,
        }

        confidence = confidence_map.get(pattern_name, 0.75)

        prediction = transform_func(test_input)
        return (prediction, confidence)

    except ImportError:
        return None


def apply_object_detection(test_input: List[List[int]],
                           task_data: Dict) -> Optional[Tuple[List[List[int]], float]]:
    """
    Apply object detection from Iteration 2.

    Returns:
        (prediction, confidence) if pattern found
        None if no pattern
    """
    try:
        # Import from Iteration 2
        from cell5_iteration2_objects import (
            detect_object_transformation_pattern,
            apply_object_transformation
        )

        pattern = detect_object_transformation_pattern(task_data)
        if pattern is None:
            return None

        prediction = apply_object_transformation(test_input, task_data, pattern)

        # Object detection is most specific, highest confidence
        confidence = 0.90

        return (prediction, confidence)

    except ImportError:
        return None


# ============================================================================
# ENSEMBLE VOTING SYSTEM
# ============================================================================

@dataclass
class SolverPrediction:
    """Container for a solver's prediction with metadata."""
    grid: List[List[int]]
    confidence: float
    solver_name: str


def collect_all_predictions(test_input: List[List[int]],
                            task_data: Dict) -> List[SolverPrediction]:
    """
    Collect predictions from all available solvers.

    Returns:
        List of SolverPrediction objects (may be empty if no solvers match)
    """
    predictions = []

    # Solver 1: Object detection (most specific)
    result = apply_object_detection(test_input, task_data)
    if result is not None:
        pred, conf = result
        predictions.append(SolverPrediction(pred, conf, "object_detection"))

    # Solver 2: Pattern matching
    result = apply_pattern_matching(test_input, task_data)
    if result is not None:
        pred, conf = result
        predictions.append(SolverPrediction(pred, conf, "pattern_matching"))

    # Solver 3: Grid arithmetic
    result = apply_grid_arithmetic(test_input, task_data)
    if result is not None:
        pred, conf = result
        predictions.append(SolverPrediction(pred, conf, "grid_arithmetic"))

    # Solver 4: Symmetry completion
    result = apply_symmetry_completion(test_input, task_data)
    if result is not None:
        pred, conf = result
        predictions.append(SolverPrediction(pred, conf, "symmetry_completion"))

    # Solver 5: Color frequency
    result = apply_color_frequency(test_input, task_data)
    if result is not None:
        pred, conf = result
        predictions.append(SolverPrediction(pred, conf, "color_frequency"))

    return predictions


def vote_on_predictions(predictions: List[SolverPrediction],
                       attempt: int,
                       test_input: List[List[int]]) -> List[List[int]]:
    """
    Vote on predictions using confidence-weighted voting.

    Algorithm:
        1. Group identical predictions
        2. Sum confidence scores for each unique prediction
        3. Sort by total confidence (descending)
        4. Return top for attempt 1, second for attempt 2

    Args:
        predictions: List of SolverPrediction objects
        attempt: 1 or 2 (which attempt we're generating)
        test_input: Original input (fallback if no predictions)

    Returns:
        Best prediction based on voting
    """
    if not predictions:
        # No predictions, return identity
        return test_input

    # Group by grid equality
    vote_groups = {}

    for pred in predictions:
        grid_key = grid_to_tuple(pred.grid)

        if grid_key not in vote_groups:
            vote_groups[grid_key] = {
                "grid": pred.grid,
                "total_confidence": 0.0,
                "solvers": []
            }

        vote_groups[grid_key]["total_confidence"] += pred.confidence
        vote_groups[grid_key]["solvers"].append(pred.solver_name)

    # Sort by total confidence (descending)
    ranked = sorted(vote_groups.values(),
                   key=lambda x: x["total_confidence"],
                   reverse=True)

    # Select based on attempt
    if attempt == 1:
        # Attempt 1: return highest confidence
        return ranked[0]["grid"]

    else:
        # Attempt 2: return second-best (or first if very confident)
        if ranked[0]["total_confidence"] > 0.95:
            # Very confident, use same for both attempts
            return ranked[0]["grid"]

        if len(ranked) > 1:
            return ranked[1]["grid"]
        else:
            # Only one prediction, return input as alternative
            return test_input


# ============================================================================
# MAIN ENSEMBLE SOLVER (PUBLIC API)
# ============================================================================

def ensemble_solver(test_input: List[List[int]],
                   task_data: Dict,
                   attempt: int = 1) -> List[List[int]]:
    """
    Ensemble solver using confidence-weighted voting.

    This is the main entry point for Iteration 3. It collects predictions
    from all available solvers and uses weighted voting to select the best.

    Args:
        test_input: Test input grid
        task_data: Task data including training pairs
        attempt: 1 or 2 (which attempt to generate)

    Returns:
        Prediction grid for this attempt
    """
    # Collect predictions from all solvers
    predictions = collect_all_predictions(test_input, task_data)

    # Vote on best prediction
    result = vote_on_predictions(predictions, attempt, test_input)

    return result


# ============================================================================
# STATISTICS & DEBUGGING
# ============================================================================

def get_ensemble_statistics(test_input: List[List[int]],
                           task_data: Dict) -> Dict[str, Any]:
    """
    Get detailed statistics about ensemble solver performance.

    Returns:
        Dictionary with:
            - num_solvers_triggered: How many solvers produced predictions
            - solver_names: List of solver names that triggered
            - confidence_scores: Dict mapping solver name to confidence
            - vote_winner: Which solver(s) won the vote
            - agreement_level: How many solvers agreed with winner
    """
    predictions = collect_all_predictions(test_input, task_data)

    stats = {
        "num_solvers_triggered": len(predictions),
        "solver_names": [p.solver_name for p in predictions],
        "confidence_scores": {p.solver_name: p.confidence for p in predictions},
    }

    if predictions:
        # Determine vote winner
        vote_groups = {}
        for pred in predictions:
            grid_key = grid_to_tuple(pred.grid)
            if grid_key not in vote_groups:
                vote_groups[grid_key] = {
                    "total_confidence": 0.0,
                    "solvers": []
                }
            vote_groups[grid_key]["total_confidence"] += pred.confidence
            vote_groups[grid_key]["solvers"].append(pred.solver_name)

        winner = max(vote_groups.values(), key=lambda x: x["total_confidence"])
        stats["vote_winner"] = winner["solvers"]
        stats["agreement_level"] = len(winner["solvers"])

    return stats
