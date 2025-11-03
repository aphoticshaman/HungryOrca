#!/usr/bin/env python3
"""
BOLT-ON #3: Spatial Rule Solver

Learns and applies spatial transformation rules:
1. Shape detection (lines, rectangles, grids)
2. Rule-based transformations (move, color change, fill)
3. Pattern-based operations (extend, complete, filter)

Focuses on tasks that BOLT-ON #2 (pattern tiling) doesn't handle.

Author: HungryOrca BOLT-ON Framework
Date: 2025-11-02
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Set
from dataclasses import dataclass


@dataclass
class SpatialShape:
    """A detected spatial shape."""
    type: str  # 'line', 'rectangle', 'grid', 'region'
    color: int
    positions: Set[Tuple[int, int]]
    bounds: Tuple[int, int, int, int]  # (min_r, min_c, max_r, max_c)


@dataclass
class TransformationRule:
    """A spatial transformation rule."""
    type: str  # 'color_replace', 'region_clear', 'shape_fill', etc.
    source_condition: Dict  # What to match
    target_action: Dict  # What to do
    confidence: float


class SpatialRuleSolver:
    """
    BOLT-ON #3: Learns spatial transformation rules from examples.

    Handles:
    - Same-size transformations (color replacement, region operations)
    - Shape-based operations (fill, clear, modify specific shapes)
    - Selective transformations (apply rules to specific regions)
    """

    def __init__(self):
        pass

    def solve(self, train_pairs: List[Tuple[np.ndarray, np.ndarray]],
              test_input: np.ndarray) -> Optional[np.ndarray]:
        """
        Solve test input using spatial rule learning.

        Args:
            train_pairs: List of (input, output) training examples
            test_input: Test input to transform

        Returns:
            Predicted output, or None if no rule found
        """
        if not train_pairs:
            return None

        # Learn transformation rules
        rules = self._learn_rules(train_pairs)

        if not rules:
            return None

        # Apply best rule
        best_rule = max(rules, key=lambda r: r.confidence)
        return self._apply_rule(test_input, best_rule)

    def _learn_rules(self, train_pairs: List[Tuple[np.ndarray, np.ndarray]]) -> List[TransformationRule]:
        """Learn transformation rules from training examples."""
        rules = []

        # Try different rule types
        color_rule = self._learn_color_replacement(train_pairs)
        if color_rule:
            rules.append(color_rule)

        region_rule = self._learn_region_operation(train_pairs)
        if region_rule:
            rules.append(region_rule)

        shape_rule = self._learn_shape_operation(train_pairs)
        if shape_rule:
            rules.append(shape_rule)

        return rules

    # ========================================================================
    # COLOR REPLACEMENT RULES
    # ========================================================================

    def _learn_color_replacement(self, train_pairs: List[Tuple[np.ndarray, np.ndarray]]) -> Optional[TransformationRule]:
        """
        Learn global color replacement rule.

        Example: All 8s → 7s, all 1s → 0s (like task 009d5c81)
        """
        # Check if all examples have same shape
        for inp, out in train_pairs:
            if inp.shape != out.shape:
                return None

        # Learn color mapping from all training examples
        color_map = {}

        for inp, out in train_pairs:
            for r in range(inp.shape[0]):
                for c in range(inp.shape[1]):
                    in_color = int(inp[r, c])
                    out_color = int(out[r, c])

                    if in_color in color_map:
                        # Check consistency: same input color should always map to same output
                        if color_map[in_color] != out_color:
                            # Position-dependent - not a simple color map
                            return None
                    else:
                        color_map[in_color] = out_color

        # Validate mapping on all examples
        for inp, out in train_pairs:
            for r in range(inp.shape[0]):
                for c in range(inp.shape[1]):
                    in_color = int(inp[r, c])
                    expected = color_map[in_color]
                    if out[r, c] != expected:
                        return None

        return TransformationRule(
            type='color_replacement',
            source_condition={},
            target_action={'mapping': color_map},
            confidence=0.9
        )

    # ========================================================================
    # REGION OPERATION RULES
    # ========================================================================

    def _learn_region_operation(self, train_pairs: List[Tuple[np.ndarray, np.ndarray]]) -> Optional[TransformationRule]:
        """
        Learn region-based operations.

        Example: Clear all pixels of certain color in specific region
        """
        # Check same shape
        for inp, out in train_pairs:
            if inp.shape != out.shape:
                return None

        # Detect if operation is "keep some colors, remove others"
        kept_colors = set()
        removed_colors = set()

        for inp, out in train_pairs:
            input_colors = set(np.unique(inp))
            output_colors = set(np.unique(out))

            # Colors that appear in output are kept
            kept_colors.update(output_colors)

            # Colors that disappear (input has them, output doesn't)
            removed_colors.update(input_colors - output_colors)

        # If some colors are consistently removed, this is a filter operation
        if removed_colors:
            return TransformationRule(
                type='color_filter',
                source_condition={'remove_colors': list(removed_colors)},
                target_action={'keep_colors': list(kept_colors)},
                confidence=0.8
            )

        return None

    # ========================================================================
    # SHAPE OPERATION RULES
    # ========================================================================

    def _learn_shape_operation(self, train_pairs: List[Tuple[np.ndarray, np.ndarray]]) -> Optional[TransformationRule]:
        """
        Learn shape-based operations.

        Example: Find all rectangles of color X, fill them with color Y
        """
        # For now, defer shape operations to later refinement
        return None

    # ========================================================================
    # RULE APPLICATION
    # ========================================================================

    def _apply_rule(self, test_input: np.ndarray, rule: TransformationRule) -> Optional[np.ndarray]:
        """Apply learned rule to test input."""
        if rule.type == 'color_replacement':
            return self._apply_color_replacement(test_input, rule.target_action)

        elif rule.type == 'color_filter':
            return self._apply_color_filter(test_input, rule.source_condition, rule.target_action)

        return None

    def _apply_color_replacement(self, inp: np.ndarray, action: Dict) -> np.ndarray:
        """Apply color replacement rule."""
        color_map = action['mapping']
        output = inp.copy()

        for r in range(output.shape[0]):
            for c in range(output.shape[1]):
                in_color = int(inp[r, c])
                output[r, c] = color_map.get(in_color, in_color)

        return output

    def _apply_color_filter(self, inp: np.ndarray, condition: Dict, action: Dict) -> np.ndarray:
        """Apply color filter rule (remove certain colors)."""
        remove_colors = set(condition['remove_colors'])
        output = inp.copy()

        for r in range(output.shape[0]):
            for c in range(output.shape[1]):
                if int(inp[r, c]) in remove_colors:
                    output[r, c] = 0  # Replace with background

        return output
