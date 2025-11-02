#!/usr/bin/env python3
"""
BOLT-ON #6: Example-Based Solver

Direct pattern matching and analogy-based reasoning:
1. Find similar training examples
2. Apply analogous transformations
3. Use nearest-neighbor matching

Author: HungryOrca BOLT-ON Framework
Date: 2025-11-02
"""

import numpy as np
from typing import List, Tuple, Optional


class ExampleBasedSolver:
    """
    BOLT-ON #6: Solves by finding similar examples and applying analogies.

    Strategy: If test input looks like training input X,
    output should look like training output Y.
    """

    def solve(self, train_pairs: List[Tuple[np.ndarray, np.ndarray]],
              test_input: np.ndarray) -> Optional[np.ndarray]:
        """Solve using example-based reasoning."""
        if not train_pairs:
            return None

        # Strategy 1: Exact match (test input matches a training input)
        for inp, out in train_pairs:
            if np.array_equal(inp, test_input):
                return out.copy()

        # Strategy 2: Size-based analogy (find same-size training example)
        same_size_examples = [(inp, out) for inp, out in train_pairs
                              if inp.shape == test_input.shape]

        if same_size_examples:
            # Use first same-size example as template
            # Apply pixel-wise transformation pattern
            result = self._apply_pixel_transformation(
                same_size_examples[0][0],
                same_size_examples[0][1],
                test_input
            )
            if result is not None:
                return result

        # Strategy 3: Relative transformation (learn transformation, apply to test)
        transformation = self._learn_relative_transformation(train_pairs)
        if transformation:
            return self._apply_relative_transformation(test_input, transformation)

        return None

    def _apply_pixel_transformation(self, train_inp: np.ndarray,
                                   train_out: np.ndarray,
                                   test_inp: np.ndarray) -> Optional[np.ndarray]:
        """
        Apply pixel-wise transformation pattern.

        If train: A→B, then test: C→D where D follows same pattern as A→B.
        """
        if train_inp.shape != test_inp.shape:
            return None

        # Simple case: color swapping pattern
        # If train swaps colors, apply same swap to test
        color_map = {}
        for r in range(train_inp.shape[0]):
            for c in range(train_inp.shape[1]):
                in_c = int(train_inp[r, c])
                out_c = int(train_out[r, c])
                if in_c in color_map and color_map[in_c] != out_c:
                    # Position-dependent, not simple mapping
                    return None
                color_map[in_c] = out_c

        # Apply to test
        output = test_inp.copy()
        for r in range(output.shape[0]):
            for c in range(output.shape[1]):
                output[r, c] = color_map.get(int(test_inp[r, c]), test_inp[r, c])

        return output

    def _learn_relative_transformation(self, train_pairs: List[Tuple[np.ndarray, np.ndarray]]) -> Optional[dict]:
        """Learn transformation that's consistent across examples."""
        # Check if all have same size ratio
        size_ratios = []
        for inp, out in train_pairs:
            size_ratios.append((out.shape[0] / inp.shape[0], out.shape[1] / inp.shape[1]))

        if len(set(size_ratios)) == 1:
            ratio = size_ratios[0]
            return {'type': 'size_scale', 'ratio': ratio}

        return None

    def _apply_relative_transformation(self, test_inp: np.ndarray, transformation: dict) -> Optional[np.ndarray]:
        """Apply learned relative transformation."""
        if transformation['type'] == 'size_scale':
            ratio_r, ratio_c = transformation['ratio']
            target_shape = (int(test_inp.shape[0] * ratio_r), int(test_inp.shape[1] * ratio_c))

            # Simple scaling by nearest neighbor
            output = np.zeros(target_shape, dtype=test_inp.dtype)
            for r in range(target_shape[0]):
                for c in range(target_shape[1]):
                    src_r = int(r / ratio_r)
                    src_c = int(c / ratio_c)
                    output[r, c] = test_inp[min(src_r, test_inp.shape[0]-1),
                                          min(src_c, test_inp.shape[1]-1)]
            return output

        return None
