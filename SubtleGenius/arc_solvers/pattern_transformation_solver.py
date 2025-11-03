#!/usr/bin/env python3
"""
BOLT-ON #2: Pattern Transformation Solver

Handles pattern-based transformations that don't require object detection:
1. Tiling/repetition (simple and with variations)
2. Reflection/rotation patterns
3. In-place color mapping
4. Size changes (scaling, cropping)

Author: HungryOrca BOLT-ON Framework
Date: 2025-11-02
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class TransformationRule:
    """A learned transformation rule."""
    type: str  # 'tiling', 'reflection', 'color_map', etc.
    params: Dict  # Type-specific parameters
    confidence: float  # How confident we are (0-1)


class PatternTransformationSolver:
    """
    BOLT-ON #2: Detects and applies pattern-based transformations.

    Focuses on transformations that operate on the grid as a whole pattern,
    rather than individual objects.
    """

    def __init__(self):
        self.learned_rules = []

    def solve(self, train_pairs: List[Tuple[np.ndarray, np.ndarray]],
              test_input: np.ndarray) -> Optional[np.ndarray]:
        """
        Solve test input using pattern transformation learning.

        Args:
            train_pairs: List of (input, output) training examples
            test_input: Test input to transform

        Returns:
            Predicted output, or None if no transformation found
        """
        if not train_pairs:
            return None

        # Learn transformation rules from training examples
        rules = self._learn_transformations(train_pairs)

        if not rules:
            return None

        # Apply best rule to test input
        best_rule = max(rules, key=lambda r: r.confidence)
        return self._apply_transformation(test_input, best_rule)

    def _learn_transformations(self, train_pairs: List[Tuple[np.ndarray, np.ndarray]]) -> List[TransformationRule]:
        """Learn transformation rules from training examples."""
        rules = []

        # Try each transformation type
        tiling_rule = self._learn_tiling(train_pairs)
        if tiling_rule:
            rules.append(tiling_rule)

        reflection_rule = self._learn_reflection_pattern(train_pairs)
        if reflection_rule:
            rules.append(reflection_rule)

        color_map_rule = self._learn_color_mapping(train_pairs)
        if color_map_rule:
            rules.append(color_map_rule)

        return rules

    # ========================================================================
    # TILING TRANSFORMATIONS
    # ========================================================================

    def _learn_tiling(self, train_pairs: List[Tuple[np.ndarray, np.ndarray]]) -> Optional[TransformationRule]:
        """
        Learn tiling transformation (repeat input pattern N×M times).

        Handles:
        1. Simple tiling: output = input repeated N×M times
        2. Alternating tiling: alternate between original and transformed versions
        """
        # Check if all examples have same tiling ratio
        tile_ratios = []

        for inp, out in train_pairs:
            if out.shape[0] % inp.shape[0] != 0 or out.shape[1] % inp.shape[1] != 0:
                return None  # Not a tiling transformation

            tile_r = out.shape[0] // inp.shape[0]
            tile_c = out.shape[1] // inp.shape[1]
            tile_ratios.append((tile_r, tile_c))

        # Check if all ratios are the same
        if len(set(tile_ratios)) != 1:
            return None

        tile_r, tile_c = tile_ratios[0]

        # Check if it's simple tiling
        is_simple = self._check_simple_tiling(train_pairs[0][0], train_pairs[0][1], tile_r, tile_c)

        if is_simple:
            return TransformationRule(
                type='simple_tiling',
                params={'tile_rows': tile_r, 'tile_cols': tile_c},
                confidence=0.9
            )

        # Check if it's alternating tiling (e.g., checkerboard reflection)
        alternating_type = self._detect_alternating_pattern(train_pairs[0][0], train_pairs[0][1], tile_r, tile_c)

        if alternating_type:
            return TransformationRule(
                type='alternating_tiling',
                params={'tile_rows': tile_r, 'tile_cols': tile_c, 'alternation': alternating_type},
                confidence=0.8
            )

        return None

    def _check_simple_tiling(self, inp: np.ndarray, out: np.ndarray,
                            tile_r: int, tile_c: int) -> bool:
        """Check if output is simple repetition of input."""
        for r in range(out.shape[0]):
            for c in range(out.shape[1]):
                expected = inp[r % inp.shape[0], c % inp.shape[1]]
                if out[r, c] != expected:
                    return False
        return True

    def _detect_alternating_pattern(self, inp: np.ndarray, out: np.ndarray,
                                   tile_r: int, tile_c: int) -> Optional[str]:
        """
        Detect alternating tiling pattern.

        Returns:
            'row_flip': Alternate horizontal flips by row
            'col_flip': Alternate vertical flips by column
            'checkerboard_flip': Alternate flips in checkerboard pattern
            None: No alternating pattern detected
        """
        # Try row-wise alternating horizontal flip
        if self._check_alternating_row_flip(inp, out, tile_r, tile_c):
            return 'row_flip'

        # Try column-wise alternating vertical flip
        if self._check_alternating_col_flip(inp, out, tile_r, tile_c):
            return 'col_flip'

        # Try checkerboard alternating flip
        if self._check_checkerboard_flip(inp, out, tile_r, tile_c):
            return 'checkerboard_flip'

        return None

    def _check_alternating_row_flip(self, inp: np.ndarray, out: np.ndarray,
                                    tile_r: int, tile_c: int) -> bool:
        """Check if output alternates between original and horizontally flipped by row."""
        flipped = np.fliplr(inp)

        for r in range(out.shape[0]):
            tile_row_idx = r // inp.shape[0]
            use_flipped = (tile_row_idx % 2) == 1

            for c in range(out.shape[1]):
                src = flipped if use_flipped else inp
                expected = src[r % inp.shape[0], c % inp.shape[1]]
                if out[r, c] != expected:
                    return False
        return True

    def _check_alternating_col_flip(self, inp: np.ndarray, out: np.ndarray,
                                    tile_r: int, tile_c: int) -> bool:
        """Check if output alternates between original and vertically flipped by column."""
        flipped = np.flipud(inp)

        for r in range(out.shape[0]):
            for c in range(out.shape[1]):
                tile_col_idx = c // inp.shape[1]
                use_flipped = (tile_col_idx % 2) == 1

                src = flipped if use_flipped else inp
                expected = src[r % inp.shape[0], c % inp.shape[1]]
                if out[r, c] != expected:
                    return False
        return True

    def _check_checkerboard_flip(self, inp: np.ndarray, out: np.ndarray,
                                 tile_r: int, tile_c: int) -> bool:
        """Check if output alternates flips in checkerboard pattern."""
        flipped_h = np.fliplr(inp)
        flipped_v = np.flipud(inp)
        flipped_both = np.fliplr(np.flipud(inp))

        for r in range(out.shape[0]):
            tile_row_idx = r // inp.shape[0]
            for c in range(out.shape[1]):
                tile_col_idx = c // inp.shape[1]

                # Checkerboard pattern
                if (tile_row_idx % 2) == 0 and (tile_col_idx % 2) == 0:
                    src = inp  # Original
                elif (tile_row_idx % 2) == 0 and (tile_col_idx % 2) == 1:
                    src = flipped_h  # Flip horizontal
                elif (tile_row_idx % 2) == 1 and (tile_col_idx % 2) == 0:
                    src = flipped_v  # Flip vertical
                else:
                    src = flipped_both  # Flip both

                expected = src[r % inp.shape[0], c % inp.shape[1]]
                if out[r, c] != expected:
                    return False
        return True

    # ========================================================================
    # REFLECTION/ROTATION PATTERNS
    # ========================================================================

    def _learn_reflection_pattern(self, train_pairs: List[Tuple[np.ndarray, np.ndarray]]) -> Optional[TransformationRule]:
        """
        Learn reflection/rotation patterns (without tiling).

        E.g., output = input + reflected input (combined)
        """
        # Check if all examples have consistent reflection pattern
        # For now, skip - focus on tiling first
        return None

    # ========================================================================
    # COLOR MAPPING TRANSFORMATIONS
    # ========================================================================

    def _learn_color_mapping(self, train_pairs: List[Tuple[np.ndarray, np.ndarray]]) -> Optional[TransformationRule]:
        """
        Learn color mapping transformation for same-size grids.

        E.g., all 3s become 5s, all 7s become 2s, etc.
        """
        # Check if all examples have same shape
        for inp, out in train_pairs:
            if inp.shape != out.shape:
                return None  # Not a same-size transformation

        # Learn color mapping from first example
        inp, out = train_pairs[0]
        color_map = {}

        for r in range(inp.shape[0]):
            for c in range(inp.shape[1]):
                in_color = int(inp[r, c])
                out_color = int(out[r, c])

                if in_color in color_map:
                    if color_map[in_color] != out_color:
                        # Inconsistent mapping (color maps to different outputs)
                        # This is a position-dependent transformation
                        return None
                else:
                    color_map[in_color] = out_color

        # Validate mapping on all training examples
        for inp, out in train_pairs:
            if not self._validate_color_mapping(inp, out, color_map):
                return None

        return TransformationRule(
            type='color_mapping',
            params={'mapping': color_map},
            confidence=0.85
        )

    def _validate_color_mapping(self, inp: np.ndarray, out: np.ndarray,
                               color_map: Dict[int, int]) -> bool:
        """Validate that color mapping works for this example."""
        for r in range(inp.shape[0]):
            for c in range(inp.shape[1]):
                in_color = int(inp[r, c])
                expected_out = color_map.get(in_color, in_color)
                if out[r, c] != expected_out:
                    return False
        return True

    # ========================================================================
    # TRANSFORMATION APPLICATION
    # ========================================================================

    def _apply_transformation(self, test_input: np.ndarray,
                             rule: TransformationRule) -> Optional[np.ndarray]:
        """Apply learned transformation rule to test input."""
        if rule.type == 'simple_tiling':
            return self._apply_simple_tiling(test_input, rule.params)

        elif rule.type == 'alternating_tiling':
            return self._apply_alternating_tiling(test_input, rule.params)

        elif rule.type == 'color_mapping':
            return self._apply_color_mapping(test_input, rule.params)

        return None

    def _apply_simple_tiling(self, inp: np.ndarray, params: Dict) -> np.ndarray:
        """Apply simple tiling transformation."""
        tile_r = params['tile_rows']
        tile_c = params['tile_cols']

        out_shape = (inp.shape[0] * tile_r, inp.shape[1] * tile_c)
        output = np.zeros(out_shape, dtype=inp.dtype)

        for r in range(output.shape[0]):
            for c in range(output.shape[1]):
                output[r, c] = inp[r % inp.shape[0], c % inp.shape[1]]

        return output

    def _apply_alternating_tiling(self, inp: np.ndarray, params: Dict) -> np.ndarray:
        """Apply alternating tiling transformation."""
        tile_r = params['tile_rows']
        tile_c = params['tile_cols']
        alternation = params['alternation']

        out_shape = (inp.shape[0] * tile_r, inp.shape[1] * tile_c)
        output = np.zeros(out_shape, dtype=inp.dtype)

        if alternation == 'row_flip':
            flipped = np.fliplr(inp)
            for r in range(output.shape[0]):
                tile_row_idx = r // inp.shape[0]
                use_flipped = (tile_row_idx % 2) == 1
                src = flipped if use_flipped else inp

                for c in range(output.shape[1]):
                    output[r, c] = src[r % inp.shape[0], c % inp.shape[1]]

        elif alternation == 'col_flip':
            flipped = np.flipud(inp)
            for r in range(output.shape[0]):
                for c in range(output.shape[1]):
                    tile_col_idx = c // inp.shape[1]
                    use_flipped = (tile_col_idx % 2) == 1
                    src = flipped if use_flipped else inp
                    output[r, c] = src[r % inp.shape[0], c % inp.shape[1]]

        elif alternation == 'checkerboard_flip':
            flipped_h = np.fliplr(inp)
            flipped_v = np.flipud(inp)
            flipped_both = np.fliplr(np.flipud(inp))

            for r in range(output.shape[0]):
                tile_row_idx = r // inp.shape[0]
                for c in range(output.shape[1]):
                    tile_col_idx = c // inp.shape[1]

                    if (tile_row_idx % 2) == 0 and (tile_col_idx % 2) == 0:
                        src = inp
                    elif (tile_row_idx % 2) == 0 and (tile_col_idx % 2) == 1:
                        src = flipped_h
                    elif (tile_row_idx % 2) == 1 and (tile_col_idx % 2) == 0:
                        src = flipped_v
                    else:
                        src = flipped_both

                    output[r, c] = src[r % inp.shape[0], c % inp.shape[1]]

        return output

    def _apply_color_mapping(self, inp: np.ndarray, params: Dict) -> np.ndarray:
        """Apply color mapping transformation."""
        color_map = params['mapping']
        output = inp.copy()

        for r in range(output.shape[0]):
            for c in range(output.shape[1]):
                in_color = int(inp[r, c])
                output[r, c] = color_map.get(in_color, in_color)

        return output
