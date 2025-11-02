#!/usr/bin/env python3
"""
PHASE 7 - WEEK 1: Full 8-Feature Extraction System

Implements sophisticated puzzle characterization from fuzzy controller research.

Features:
1. Symmetry strength - Reflection/rotation matching
2. Multi-scale complexity - Hierarchical pattern detection
3. Non-locality score - Global constraint detection
4. Criticality index - Percolation threshold proximity
5. Pattern entropy - Shannon entropy of colors
6. Grid size factor - Normalized size complexity
7. Color complexity - Unique color count
8. Transformation consistency - Size change variance

Author: HungryOrca Phase 7 Implementation
Date: 2025-11-02
"""

import numpy as np
from collections import Counter
from typing import List, Tuple, Dict
from dataclasses import dataclass


@dataclass
class PuzzleFeatures:
    """Container for all 8 extracted features."""
    symmetry_strength: float
    multi_scale_complexity: float
    non_locality_score: float
    criticality_index: float
    pattern_entropy: float
    grid_size_factor: float
    color_complexity: float
    transformation_consistency: float

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for easy access."""
        return {
            'symmetry_strength': self.symmetry_strength,
            'multi_scale_complexity': self.multi_scale_complexity,
            'non_locality_score': self.non_locality_score,
            'criticality_index': self.criticality_index,
            'pattern_entropy': self.pattern_entropy,
            'grid_size_factor': self.grid_size_factor,
            'color_complexity': self.color_complexity,
            'transformation_consistency': self.transformation_consistency
        }


class FullFeatureExtractor:
    """
    Production-ready 8-feature extraction system.

    Based on fuzzy_meta_controller_production.py research.
    """

    def extract(self, train_pairs: List[Tuple[np.ndarray, np.ndarray]],
                test_input: np.ndarray) -> PuzzleFeatures:
        """
        Extract all 8 features from puzzle.

        Args:
            train_pairs: List of (input_grid, output_grid) training examples
            test_input: Test input grid

        Returns:
            PuzzleFeatures object with all 8 features
        """
        # Average features across all training grids
        all_grids = []
        for inp, out in train_pairs:
            all_grids.append(inp)
            all_grids.append(out)
        all_grids.append(test_input)

        # Feature 1: Symmetry strength
        symmetry_scores = [self._compute_symmetry(grid) for grid in all_grids]
        symmetry_strength = np.mean(symmetry_scores)

        # Feature 2: Multi-scale complexity
        complexity_scores = [self._compute_multi_scale_complexity(grid) for grid in all_grids]
        multi_scale_complexity = np.mean(complexity_scores)

        # Feature 3: Non-locality score
        non_locality_scores = [self._compute_non_locality(grid) for grid in all_grids]
        non_locality_score = np.mean(non_locality_scores)

        # Feature 4: Criticality index
        criticality_scores = [self._compute_criticality(grid) for grid in all_grids]
        criticality_index = np.mean(criticality_scores)

        # Feature 5: Pattern entropy
        entropy_scores = [self._compute_entropy(grid) for grid in all_grids]
        pattern_entropy = np.mean(entropy_scores)

        # Feature 6: Grid size factor
        sizes = [grid.shape[0] * grid.shape[1] for grid in all_grids]
        grid_size_factor = np.mean(sizes) / (50 * 50)  # Normalize to max 50×50
        grid_size_factor = min(1.0, grid_size_factor)

        # Feature 7: Color complexity
        color_counts = [len(np.unique(grid)) for grid in all_grids]
        color_complexity = np.mean(color_counts) / 10.0  # Normalize to max 10 colors
        color_complexity = min(1.0, color_complexity)

        # Feature 8: Transformation consistency
        transformation_consistency = self._compute_transform_consistency(train_pairs)

        return PuzzleFeatures(
            symmetry_strength=float(symmetry_strength),
            multi_scale_complexity=float(multi_scale_complexity),
            non_locality_score=float(non_locality_score),
            criticality_index=float(criticality_index),
            pattern_entropy=float(pattern_entropy),
            grid_size_factor=float(grid_size_factor),
            color_complexity=float(color_complexity),
            transformation_consistency=float(transformation_consistency)
        )

    @staticmethod
    def _compute_symmetry(grid: np.ndarray) -> float:
        """
        Feature 1: Symmetry Strength

        Computes reflection and rotation symmetry matching.
        High score indicates symmetric patterns.

        Implementation from fuzzy_meta_controller_production.py:246-262
        """
        # Horizontal reflection
        h_match = np.mean(grid == np.flip(grid, axis=0))

        # Vertical reflection
        v_match = np.mean(grid == np.flip(grid, axis=1))

        # Rotational symmetry (if square)
        if grid.shape[0] == grid.shape[1]:
            r_match = np.mean(grid == np.rot90(grid))
        else:
            r_match = 0.0

        return max(h_match, v_match, r_match)

    @staticmethod
    def _compute_multi_scale_complexity(grid: np.ndarray) -> float:
        """
        Feature 2: Multi-Scale Complexity

        Measures variance preservation across scales.
        High score indicates hierarchical patterns.

        Implementation from fuzzy_meta_controller_production.py:265-281
        """
        data = grid.astype(float)

        # Variance at original scale
        original_var = np.var(data)

        # Downsample 2×
        if data.shape[0] >= 2 and data.shape[1] >= 2:
            downsampled = data[::2, ::2]
            downsampled_var = np.var(downsampled)

            # High complexity if variance preserved across scales
            complexity = abs(original_var - downsampled_var) / (original_var + 1)
            return min(1.0, complexity)

        return 0.5

    @staticmethod
    def _compute_non_locality(grid: np.ndarray) -> float:
        """
        Feature 3: Non-Locality Score

        Measures fragmentation of colored regions.
        High score indicates non-local constraints.

        Implementation from fuzzy_meta_controller_production.py:284-329
        """
        bg = Counter(grid.flatten()).most_common(1)[0][0]

        # Count connected components for each color
        total_components = 0
        total_pixels = 0

        for color in np.unique(grid):
            if color == bg:
                continue

            mask = (grid == color)
            pixel_count = np.sum(mask)

            if pixel_count > 0:
                # Simple component count (4-connectivity)
                labeled = np.zeros_like(grid, dtype=int)
                component_id = 1

                for i in range(grid.shape[0]):
                    for j in range(grid.shape[1]):
                        if mask[i, j] and labeled[i, j] == 0:
                            # Flood fill
                            stack = [(i, j)]
                            while stack:
                                ci, cj = stack.pop()
                                if labeled[ci, cj] == 0 and mask[ci, cj]:
                                    labeled[ci, cj] = component_id
                                    for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                                        ni, nj = ci + di, cj + dj
                                        if 0 <= ni < grid.shape[0] and 0 <= nj < grid.shape[1]:
                                            if mask[ni, nj] and labeled[ni, nj] == 0:
                                                stack.append((ni, nj))
                            component_id += 1

                total_components += (component_id - 1)
                total_pixels += pixel_count

        # High component count relative to pixels suggests non-local constraints
        if total_pixels > 0:
            fragmentation = total_components / total_pixels
            return min(1.0, fragmentation * 10)  # Normalize

        return 0.0

    @staticmethod
    def _compute_criticality(grid: np.ndarray) -> float:
        """
        Feature 4: Criticality Index

        Measures proximity to percolation threshold (p_c = 0.59).
        High score indicates phase transition patterns.

        Implementation from fuzzy_meta_controller_production.py:332-344
        """
        bg = Counter(grid.flatten()).most_common(1)[0][0]

        p_occupied = np.mean(grid != bg)
        p_critical = 0.59  # 2D square lattice percolation threshold

        # Close to critical → high score
        delta = abs(p_occupied - p_critical)
        criticality = max(0, 1 - delta / 0.3)  # Within 0.3 of critical

        return criticality

    @staticmethod
    def _compute_entropy(grid: np.ndarray) -> float:
        """
        Feature 5: Pattern Entropy

        Computes Shannon entropy of color distribution.
        High score indicates complex color patterns.

        Implementation from fuzzy_meta_controller_production.py:347-360
        """
        data = grid.flatten()
        color_counts = Counter(data)
        total = len(data)

        entropy = 0.0
        for count in color_counts.values():
            p = count / total
            if p > 0:
                entropy -= p * np.log2(p)

        # Normalize to 0-1 (max entropy for 10 colors is log2(10) ≈ 3.32)
        return min(1.0, entropy / 3.32)

    @staticmethod
    def _compute_transform_consistency(train_pairs: List[Tuple[np.ndarray, np.ndarray]]) -> float:
        """
        Feature 8: Transformation Consistency

        Measures variance in size change ratios across training pairs.
        High score indicates consistent transformations.

        Implementation from fuzzy_meta_controller_production.py:363-386
        """
        if len(train_pairs) < 2:
            return 1.0  # Single example assumed consistent

        # Check if output sizes are consistent
        input_sizes = [pair[0].shape for pair in train_pairs]
        output_sizes = [pair[1].shape for pair in train_pairs]

        # Size change patterns
        size_changes = [
            (out[0] / inp[0], out[1] / inp[1])
            for inp, out in zip(input_sizes, output_sizes)
        ]

        # Variance in size changes (low variance = consistent)
        if len(size_changes) > 1:
            ratios = np.array(size_changes)
            variance = np.mean(np.var(ratios, axis=0))
            consistency = max(0, 1 - variance)
        else:
            consistency = 1.0

        return consistency


# ============================================================================
# TESTING AND VALIDATION
# ============================================================================

def test_feature_extraction():
    """Quick validation test."""
    print("="*80)
    print("FULL FEATURE EXTRACTION - VALIDATION TEST")
    print("="*80)

    # Create simple test grids
    # Test 1: Symmetric grid
    symmetric_grid = np.array([
        [1, 2, 1],
        [2, 3, 2],
        [1, 2, 1]
    ])

    # Test 2: Random grid
    random_grid = np.random.randint(0, 5, (5, 5))

    train_pairs = [(symmetric_grid, random_grid)]
    test_input = symmetric_grid

    extractor = FullFeatureExtractor()
    features = extractor.extract(train_pairs, test_input)

    print("\nExtracted Features:")
    print(f"  1. Symmetry strength:          {features.symmetry_strength:.3f}")
    print(f"  2. Multi-scale complexity:     {features.multi_scale_complexity:.3f}")
    print(f"  3. Non-locality score:         {features.non_locality_score:.3f}")
    print(f"  4. Criticality index:          {features.criticality_index:.3f}")
    print(f"  5. Pattern entropy:            {features.pattern_entropy:.3f}")
    print(f"  6. Grid size factor:           {features.grid_size_factor:.3f}")
    print(f"  7. Color complexity:           {features.color_complexity:.3f}")
    print(f"  8. Transformation consistency: {features.transformation_consistency:.3f}")

    print("\n✅ All 8 features extracted successfully!")
    print("="*80)


if __name__ == '__main__':
    test_feature_extraction()
