#!/usr/bin/env python3
"""
TRADITIONAL ARC SOLVING APPROACHES (BOLT-ONS #9-#15)

Implements established techniques from ARC literature and competitions:
9. Symmetry-based solver
10. Histogram matching
11. Identity/copy solver
12. Majority vote ensemble
13. Nearest neighbor
14. Simple rule induction
15. Abstraction primitives

Author: HungryOrca BOLT-ON Framework
Date: 2025-11-02
"""

import numpy as np
from typing import List, Tuple, Optional
from collections import Counter


# ============================================================================
# BOLT-ON #9: Symmetry-Based Solver
# ============================================================================

class SymmetrySolver:
    """Detects and applies symmetry transformations."""

    def solve(self, train_pairs: List[Tuple[np.ndarray, np.ndarray]],
              test_input: np.ndarray) -> Optional[np.ndarray]:
        # Check if transformation is symmetry-based
        for inp, out in train_pairs:
            if inp.shape != out.shape:
                continue

            # Check if output is symmetric version of input
            if np.array_equal(out, np.fliplr(inp)):
                return np.fliplr(test_input)
            if np.array_equal(out, np.flipud(inp)):
                return np.flipud(test_input)
            if np.array_equal(out, np.rot90(inp)):
                return np.rot90(test_input)

        return None


# ============================================================================
# BOLT-ON #10: Histogram Matching
# ============================================================================

class HistogramSolver:
    """Matches color histograms between input and output."""

    def solve(self, train_pairs: List[Tuple[np.ndarray, np.ndarray]],
              test_input: np.ndarray) -> Optional[np.ndarray]:
        # Check if task maintains color distribution
        for inp, out in train_pairs:
            if inp.shape != out.shape:
                return None

            inp_hist = Counter(inp.flatten())
            out_hist = Counter(out.flatten())

            # If histograms match, it's a permutation
            if inp_hist == out_hist:
                # Task is spatial rearrangement, not color change
                # Try transposition
                if np.array_equal(out, inp.T):
                    return test_input.T
                # Try rotation
                for k in [1, 2, 3]:
                    if np.array_equal(out, np.rot90(inp, k)):
                        return np.rot90(test_input, k)

        return None


# ============================================================================
# BOLT-ON #11: Identity/Copy Solver
# ============================================================================

class IdentitySolver:
    """Returns input as output (handles no-op tasks)."""

    def solve(self, train_pairs: List[Tuple[np.ndarray, np.ndarray]],
              test_input: np.ndarray) -> Optional[np.ndarray]:
        # Check if any training example has output = input
        for inp, out in train_pairs:
            if np.array_equal(inp, out):
                return test_input.copy()

        return None


# ============================================================================
# BOLT-ON #12: Majority Vote Ensemble
# ============================================================================

class MajorityVoteSolver:
    """Combines predictions from multiple solvers via voting."""

    def __init__(self, base_solvers: List):
        self.base_solvers = base_solvers

    def solve(self, train_pairs: List[Tuple[np.ndarray, np.ndarray]],
              test_input: np.ndarray) -> Optional[np.ndarray]:
        predictions = []

        for solver in self.base_solvers:
            try:
                result = solver.solve(train_pairs, test_input)
                if result is not None:
                    predictions.append(result)
            except:
                continue

        if not predictions:
            return None

        # If all predictions agree, return it
        if len(predictions) > 0:
            first = predictions[0]
            if all(np.array_equal(p, first) for p in predictions):
                return first

            # Return most common shape
            shapes = [p.shape for p in predictions]
            most_common_shape = Counter(shapes).most_common(1)[0][0]
            candidates = [p for p in predictions if p.shape == most_common_shape]
            return candidates[0] if candidates else predictions[0]

        return None


# ============================================================================
# BOLT-ON #13: Nearest Neighbor
# ============================================================================

class NearestNeighborSolver:
    """Finds most similar training input, returns corresponding output."""

    def solve(self, train_pairs: List[Tuple[np.ndarray, np.ndarray]],
              test_input: np.ndarray) -> Optional[np.ndarray]:
        if not train_pairs:
            return None

        # Find training input most similar to test input
        best_match = None
        best_similarity = -1

        for inp, out in train_pairs:
            similarity = self._compute_similarity(test_input, inp)
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = out

        # Only return if similarity is high enough
        if best_similarity > 0.5:
            return best_match.copy()

        return None

    def _compute_similarity(self, grid1: np.ndarray, grid2: np.ndarray) -> float:
        """Compute similarity between two grids."""
        # Shape similarity
        if grid1.shape == grid2.shape:
            shape_sim = 1.0
        else:
            size1 = grid1.shape[0] * grid1.shape[1]
            size2 = grid2.shape[0] * grid2.shape[1]
            shape_sim = min(size1, size2) / max(size1, size2)

        # Color similarity
        colors1 = set(np.unique(grid1))
        colors2 = set(np.unique(grid2))
        color_sim = len(colors1 & colors2) / len(colors1 | colors2) if colors1 | colors2 else 0

        return (shape_sim + color_sim) / 2


# ============================================================================
# BOLT-ON #14: Simple Rule Induction
# ============================================================================

class RuleInductionSolver:
    """Induces simple rules from training examples."""

    def solve(self, train_pairs: List[Tuple[np.ndarray, np.ndarray]],
              test_input: np.ndarray) -> Optional[np.ndarray]:
        # Rule 1: If all outputs same size, use that size
        out_sizes = [out.shape for _, out in train_pairs]
        if len(set(out_sizes)) == 1:
            target_size = out_sizes[0]

            # If test input already this size, apply simple color map
            if test_input.shape == target_size:
                return self._apply_color_mapping(train_pairs, test_input)

            # If needs resizing, try uniform scaling
            if target_size[0] % test_input.shape[0] == 0:
                scale = target_size[0] // test_input.shape[0]
                return self._scale_uniform(test_input, scale)

        return None

    def _apply_color_mapping(self, train_pairs: List[Tuple[np.ndarray, np.ndarray]],
                            test_input: np.ndarray) -> Optional[np.ndarray]:
        """Apply consistent color mapping."""
        color_map = {}

        for inp, out in train_pairs:
            if inp.shape != out.shape:
                return None

            for r in range(inp.shape[0]):
                for c in range(inp.shape[1]):
                    in_c = int(inp[r, c])
                    out_c = int(out[r, c])
                    if in_c in color_map and color_map[in_c] != out_c:
                        return None
                    color_map[in_c] = out_c

        # Apply mapping
        output = test_input.copy()
        for r in range(output.shape[0]):
            for c in range(output.shape[1]):
                output[r, c] = color_map.get(int(test_input[r, c]), test_input[r, c])

        return output

    def _scale_uniform(self, grid: np.ndarray, scale: int) -> np.ndarray:
        """Uniformly scale grid by integer factor."""
        new_shape = (grid.shape[0] * scale, grid.shape[1] * scale)
        output = np.zeros(new_shape, dtype=grid.dtype)

        for r in range(new_shape[0]):
            for c in range(new_shape[1]):
                output[r, c] = grid[r // scale, c // scale]

        return output


# ============================================================================
# BOLT-ON #15: Abstraction Primitives
# ============================================================================

class AbstractionSolver:
    """Uses abstraction primitives (fill, move, copy, etc.)."""

    def solve(self, train_pairs: List[Tuple[np.ndarray, np.ndarray]],
              test_input: np.ndarray) -> Optional[np.ndarray]:
        # Check if task is "fill background"
        for inp, out in train_pairs:
            if inp.shape != out.shape:
                continue

            # Check if output fills 0s with most common non-zero color
            if np.all(inp != 0) == np.all(out != 0):
                # Same non-zero positions
                continue

            # Check fill operation
            non_zero_colors = [c for c in np.unique(out) if c != 0]
            if len(non_zero_colors) == 1:
                fill_color = non_zero_colors[0]
                # Test if output is input with 0s filled
                expected = inp.copy()
                expected[expected == 0] = fill_color

                if np.array_equal(expected, out):
                    # Apply fill to test
                    result = test_input.copy()
                    result[result == 0] = fill_color
                    return result

        return None
