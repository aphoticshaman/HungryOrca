#!/usr/bin/env python3
"""
BOLT-ON #4: Size Transformation Solver

Handles dimension changes that aren't simple tiling:
1. Cropping (remove rows/columns by rule)
2. Extension (add rows/columns by pattern)
3. Selective scaling (scale specific regions)

Author: HungryOrca BOLT-ON Framework
Date: 2025-11-02
"""

import numpy as np
from typing import List, Tuple, Optional, Dict


class SizeTransformationSolver:
    """
    BOLT-ON #4: Learns and applies size transformation rules.

    Focuses on tasks where grid dimensions change in non-tiling ways.
    """

    def __init__(self):
        pass

    def solve(self, train_pairs: List[Tuple[np.ndarray, np.ndarray]],
              test_input: np.ndarray) -> Optional[np.ndarray]:
        """Solve using size transformation learning."""
        if not train_pairs:
            return None

        # Learn transformation rule
        rule = self._learn_size_transformation(train_pairs)

        if not rule:
            return None

        # Apply rule to test input
        return self._apply_transformation(test_input, rule)

    def _learn_size_transformation(self, train_pairs: List[Tuple[np.ndarray, np.ndarray]]) -> Optional[Dict]:
        """Learn size transformation rule from examples."""
        # Check all examples have consistent size ratio
        ratios = []
        for inp, out in train_pairs:
            row_ratio = out.shape[0] / inp.shape[0]
            col_ratio = out.shape[1] / inp.shape[1]
            ratios.append((row_ratio, col_ratio))

        # Check consistency
        if len(set(ratios)) != 1:
            return None  # Inconsistent size changes

        row_ratio, col_ratio = ratios[0]

        # Case 1: Extension (add rows/columns)
        if row_ratio > 1 or col_ratio > 1:
            return self._learn_extension(train_pairs, row_ratio, col_ratio)

        # Case 2: Cropping (remove rows/columns)
        if row_ratio < 1 or col_ratio < 1:
            return self._learn_cropping(train_pairs, row_ratio, col_ratio)

        return None

    def _learn_extension(self, train_pairs: List[Tuple[np.ndarray, np.ndarray]],
                        row_ratio: float, col_ratio: float) -> Optional[Dict]:
        """
        Learn row/column extension rule.

        Example: 6×3 → 9×3 means extend rows (add 3 rows)
        """
        inp, out = train_pairs[0]

        # Check if extension is by row repetition
        if col_ratio == 1.0 and row_ratio > 1:
            # Rows extended, columns unchanged
            # Check pattern: are original rows repeated?
            extension_type = self._detect_row_extension_pattern(inp, out)
            if extension_type:
                return {
                    'type': 'row_extension',
                    'pattern': extension_type,
                    'target_rows': out.shape[0]
                }

        # Check if extension is by column repetition
        if row_ratio == 1.0 and col_ratio > 1:
            extension_type = self._detect_col_extension_pattern(inp, out)
            if extension_type:
                return {
                    'type': 'col_extension',
                    'pattern': extension_type,
                    'target_cols': out.shape[1]
                }

        return None

    def _detect_row_extension_pattern(self, inp: np.ndarray, out: np.ndarray) -> Optional[str]:
        """
        Detect how rows are extended.

        Returns:
            'repeat': Rows are repeated (tile vertically)
            'append_pattern': Pattern from input is appended
        """
        # Check if output starts with input
        if np.array_equal(out[:inp.shape[0], :], inp):
            # Input is at the start, check what's added
            added_rows = out[inp.shape[0]:, :]

            # Check if added rows repeat the input
            if np.array_equal(added_rows[:inp.shape[0], :], inp):
                return 'repeat'

            return 'append_pattern'

        # Check if input rows are tiled
        is_tiled = True
        for r in range(out.shape[0]):
            if not np.array_equal(out[r, :], inp[r % inp.shape[0], :]):
                is_tiled = False
                break

        if is_tiled:
            return 'tile'

        return None

    def _detect_col_extension_pattern(self, inp: np.ndarray, out: np.ndarray) -> Optional[str]:
        """Detect how columns are extended."""
        # Similar to row extension
        if np.array_equal(out[:, :inp.shape[1]], inp):
            return 'append_pattern'

        # Check tiling
        is_tiled = True
        for c in range(out.shape[1]):
            if not np.array_equal(out[:, c], inp[:, c % inp.shape[1]]):
                is_tiled = False
                break

        if is_tiled:
            return 'tile'

        return None

    def _learn_cropping(self, train_pairs: List[Tuple[np.ndarray, np.ndarray]],
                       row_ratio: float, col_ratio: float) -> Optional[Dict]:
        """
        Learn cropping rule.

        Example: 3×7 → 3×3 means crop columns (keep first 3)
        """
        inp, out = train_pairs[0]

        # Check if it's simple cropping (take first N rows/columns)
        if row_ratio < 1.0 and col_ratio == 1.0:
            # Crop rows
            if np.array_equal(out, inp[:out.shape[0], :]):
                return {
                    'type': 'crop_rows',
                    'keep': 'first',
                    'target_rows': out.shape[0]
                }

        if col_ratio < 1.0 and row_ratio == 1.0:
            # Crop columns
            if np.array_equal(out, inp[:, :out.shape[1]]):
                return {
                    'type': 'crop_cols',
                    'keep': 'first',
                    'target_cols': out.shape[1]
                }

        # Check for selective cropping (keep non-zero columns, etc.)
        if col_ratio < 1.0:
            crop_rule = self._detect_selective_column_crop(inp, out)
            if crop_rule:
                return crop_rule

        return None

    def _detect_selective_column_crop(self, inp: np.ndarray, out: np.ndarray) -> Optional[Dict]:
        """
        Detect selective column cropping.

        Example: Keep only columns with non-zero values
        """
        # Check if output columns are a subset of input columns
        kept_cols = []
        for out_c in range(out.shape[1]):
            # Find which input column this matches
            for inp_c in range(inp.shape[1]):
                if np.array_equal(out[:, out_c], inp[:, inp_c]):
                    kept_cols.append(inp_c)
                    break

        if len(kept_cols) == out.shape[1]:
            # Output is a selection of input columns
            # Check pattern: are kept columns the first N non-zero columns?
            non_zero_cols = []
            for c in range(inp.shape[1]):
                if np.any(inp[:, c] != 0):
                    non_zero_cols.append(c)

            if kept_cols == non_zero_cols[:len(kept_cols)]:
                return {
                    'type': 'crop_cols_selective',
                    'rule': 'first_non_zero',
                    'target_cols': out.shape[1]
                }

        return None

    def _apply_transformation(self, inp: np.ndarray, rule: Dict) -> Optional[np.ndarray]:
        """Apply learned size transformation."""
        if rule['type'] == 'row_extension':
            return self._apply_row_extension(inp, rule)
        elif rule['type'] == 'col_extension':
            return self._apply_col_extension(inp, rule)
        elif rule['type'] == 'crop_rows':
            return self._apply_crop_rows(inp, rule)
        elif rule['type'] == 'crop_cols':
            return self._apply_crop_cols(inp, rule)
        elif rule['type'] == 'crop_cols_selective':
            return self._apply_selective_col_crop(inp, rule)

        return None

    def _apply_row_extension(self, inp: np.ndarray, rule: Dict) -> np.ndarray:
        """Apply row extension."""
        target_rows = rule['target_rows']
        pattern = rule['pattern']

        if pattern == 'tile' or pattern == 'repeat':
            # Tile rows to reach target
            output = np.zeros((target_rows, inp.shape[1]), dtype=inp.dtype)
            for r in range(target_rows):
                output[r, :] = inp[r % inp.shape[0], :]
            return output

        # Default: just repeat input
        tiles_needed = (target_rows + inp.shape[0] - 1) // inp.shape[0]
        tiled = np.tile(inp, (tiles_needed, 1))
        return tiled[:target_rows, :]

    def _apply_col_extension(self, inp: np.ndarray, rule: Dict) -> np.ndarray:
        """Apply column extension."""
        target_cols = rule['target_cols']
        pattern = rule['pattern']

        if pattern == 'tile':
            output = np.zeros((inp.shape[0], target_cols), dtype=inp.dtype)
            for c in range(target_cols):
                output[:, c] = inp[:, c % inp.shape[1]]
            return output

        tiles_needed = (target_cols + inp.shape[1] - 1) // inp.shape[1]
        tiled = np.tile(inp, (1, tiles_needed))
        return tiled[:, :target_cols]

    def _apply_crop_rows(self, inp: np.ndarray, rule: Dict) -> np.ndarray:
        """Apply row cropping."""
        target_rows = rule['target_rows']
        return inp[:target_rows, :]

    def _apply_crop_cols(self, inp: np.ndarray, rule: Dict) -> np.ndarray:
        """Apply column cropping."""
        target_cols = rule['target_cols']
        return inp[:, :target_cols]

    def _apply_selective_col_crop(self, inp: np.ndarray, rule: Dict) -> np.ndarray:
        """Apply selective column cropping."""
        if rule['rule'] == 'first_non_zero':
            # Keep first N non-zero columns
            target_cols = rule['target_cols']
            non_zero_cols = []

            for c in range(inp.shape[1]):
                if np.any(inp[:, c] != 0):
                    non_zero_cols.append(c)
                    if len(non_zero_cols) >= target_cols:
                        break

            # Extract these columns
            output = inp[:, non_zero_cols[:target_cols]]
            return output

        return None
