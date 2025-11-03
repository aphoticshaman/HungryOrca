#!/usr/bin/env python3
"""
FUZZY-INTEGRATED ARC SOLVER

Combines:
1. Deep instrumentation insights (PROs of NO-GO components)
2. Fuzzy meta-controller (adaptive strategy weights)
3. BOLT-ON components (GO + refined NO-GO)

This is the integrated system for ARC Prize 2025.

Author: HungryOrca Fuzzy Integration
Date: 2025-11-02
"""

import numpy as np
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'SubtleGenius' / 'arc_solvers'))

# Import BOLT-ONs
from pattern_transformation_solver import PatternTransformationSolver
from example_based_solver import ExampleBasedSolver
from multiscale_object_solver import MultiScaleSolver
from traditional_approaches import NearestNeighborSolver, SymmetrySolver


# Simplified fuzzy system (from user's research)
class SimpleFuzzyController:
    """Lightweight fuzzy controller for strategy orchestration."""

    def compute_strategy_weights(self, train_pairs: List[Tuple[np.ndarray, np.ndarray]],
                                 test_input: np.ndarray) -> Dict[str, float]:
        """
        Compute strategy weights based on puzzle features.

        Returns weights for:
        - pattern_weight: Pattern/tiling transformations
        - example_weight: Example-based matching
        - object_weight: Object detection (multi-scale)
        - neighbor_weight: Nearest neighbor matching
        - symmetry_weight: Symmetry operations
        """

        # Extract features
        symmetry = self._compute_symmetry_strength(train_pairs)
        consistency = self._compute_consistency(train_pairs)
        size_ratio = self._compute_size_ratio(train_pairs)
        complexity = self._compute_complexity(train_pairs)

        weights = {}

        # Rule 1: High consistency + same size → example-based
        if consistency > 0.7 and size_ratio == 1.0:
            weights['example_weight'] = 0.8
            weights['pattern_weight'] = 0.4
        else:
            weights['example_weight'] = 0.5
            weights['pattern_weight'] = 0.6

        # Rule 2: High symmetry → emphasize symmetry + pattern
        if symmetry > 0.7:
            weights['symmetry_weight'] = 0.7
            weights['pattern_weight'] = 0.8
        else:
            weights['symmetry_weight'] = 0.3
            weights['pattern_weight'] = 0.6

        # Rule 3: Size changes → prioritize pattern solver
        if size_ratio != 1.0:
            weights['pattern_weight'] = 0.9
            weights['object_weight'] = 0.2
        else:
            weights['pattern_weight'] = 0.6
            weights['object_weight'] = 0.5

        # Rule 4: High complexity → use all strategies
        if complexity > 0.7:
            weights['neighbor_weight'] = 0.6
            weights['object_weight'] = 0.7
        else:
            weights['neighbor_weight'] = 0.4
            weights['object_weight'] = 0.3

        # Ensure all weights exist
        weights.setdefault('pattern_weight', 0.6)
        weights.setdefault('example_weight', 0.5)
        weights.setdefault('object_weight', 0.4)
        weights.setdefault('neighbor_weight', 0.4)
        weights.setdefault('symmetry_weight', 0.3)

        return weights

    def _compute_symmetry_strength(self, train_pairs: List) -> float:
        """Compute average symmetry across training inputs."""
        scores = []
        for inp, _ in train_pairs:
            # Horizontal symmetry
            h_sym = np.mean(inp == np.fliplr(inp))
            # Vertical symmetry
            v_sym = np.mean(inp == np.flipud(inp))
            scores.append(max(h_sym, v_sym))

        return np.mean(scores) if scores else 0.5

    def _compute_consistency(self, train_pairs: List) -> float:
        """Measure consistency of transformations."""
        if len(train_pairs) < 2:
            return 1.0

        size_ratios = []
        for inp, out in train_pairs:
            ratio = (out.shape[0] / inp.shape[0], out.shape[1] / inp.shape[1])
            size_ratios.append(ratio)

        # Check if all ratios are same
        if len(set(size_ratios)) == 1:
            return 0.9
        else:
            return 0.3

    def _compute_size_ratio(self, train_pairs: List) -> float:
        """Get size change ratio (average)."""
        if not train_pairs:
            return 1.0

        ratios = []
        for inp, out in train_pairs:
            ratio = (out.shape[0] * out.shape[1]) / (inp.shape[0] * inp.shape[1])
            ratios.append(ratio)

        return np.mean(ratios)

    def _compute_complexity(self, train_pairs: List) -> float:
        """Estimate complexity from color count and entropy."""
        complexities = []
        for inp, _ in train_pairs:
            num_colors = len(np.unique(inp))
            entropy = -np.sum([(np.sum(inp == c) / inp.size) * np.log2((np.sum(inp == c) / inp.size) + 1e-10)
                              for c in np.unique(inp)])
            complexity = min(1.0, (num_colors / 10 + entropy / 3.32) / 2)
            complexities.append(complexity)

        return np.mean(complexities) if complexities else 0.5


@dataclass
class SolverPrediction:
    """Prediction from a single solver."""
    solver_name: str
    prediction: Optional[np.ndarray]
    confidence: float


class FuzzyIntegratedARCSolver:
    """
    Main integrated solver using fuzzy controller + BOLT-ONs.

    Architecture:
    1. Extract puzzle features
    2. Fuzzy controller → strategy weights
    3. Run weighted solvers
    4. Aggregate predictions
    """

    def __init__(self):
        # Initialize BOLT-ON solvers
        self.pattern_solver = PatternTransformationSolver()
        self.example_solver = ExampleBasedSolver()
        self.object_solver = MultiScaleSolver()
        self.neighbor_solver = NearestNeighborSolver()
        self.symmetry_solver = SymmetrySolver()

        # Fuzzy controller
        self.fuzzy_controller = SimpleFuzzyController()

    def solve(self, train_pairs: List[Tuple[np.ndarray, np.ndarray]],
              test_input: np.ndarray) -> Optional[np.ndarray]:
        """
        Solve using fuzzy-integrated approach.

        Process:
        1. Compute strategy weights via fuzzy logic
        2. Run solvers with weights
        3. Select best prediction
        """

        # Get fuzzy strategy weights
        weights = self.fuzzy_controller.compute_strategy_weights(train_pairs, test_input)

        # Collect predictions from all solvers
        predictions = []

        # Pattern solver (GO component #1)
        if weights['pattern_weight'] > 0.3:
            try:
                pred = self.pattern_solver.solve(train_pairs, test_input)
                if pred is not None:
                    predictions.append(SolverPrediction(
                        solver_name='Pattern',
                        prediction=pred,
                        confidence=weights['pattern_weight']
                    ))
            except:
                pass

        # Example solver (GO component #2)
        if weights['example_weight'] > 0.3:
            try:
                pred = self.example_solver.solve(train_pairs, test_input)
                if pred is not None:
                    predictions.append(SolverPrediction(
                        solver_name='Example',
                        prediction=pred,
                        confidence=weights['example_weight']
                    ))
            except:
                pass

        # Object solver (NO-GO but high uplift potential)
        if weights['object_weight'] > 0.3:
            try:
                pred = self.object_solver.solve(train_pairs, test_input)
                if pred is not None:
                    predictions.append(SolverPrediction(
                        solver_name='Object',
                        prediction=pred,
                        confidence=weights['object_weight']
                    ))
            except:
                pass

        # Nearest neighbor solver (NO-GO but near-misses)
        if weights['neighbor_weight'] > 0.3:
            try:
                pred = self.neighbor_solver.solve(train_pairs, test_input)
                if pred is not None:
                    predictions.append(SolverPrediction(
                        solver_name='NearestNeighbor',
                        prediction=pred,
                        confidence=weights['neighbor_weight']
                    ))
            except:
                pass

        # Symmetry solver
        if weights['symmetry_weight'] > 0.3:
            try:
                pred = self.symmetry_solver.solve(train_pairs, test_input)
                if pred is not None:
                    predictions.append(SolverPrediction(
                        solver_name='Symmetry',
                        prediction=pred,
                        confidence=weights['symmetry_weight']
                    ))
            except:
                pass

        # No predictions
        if not predictions:
            return None

        # Select best prediction (highest confidence)
        best = max(predictions, key=lambda p: p.confidence)
        return best.prediction

    def solve_with_debug(self, train_pairs: List[Tuple[np.ndarray, np.ndarray]],
                        test_input: np.ndarray) -> Tuple[Optional[np.ndarray], Dict]:
        """Solve and return debug info (weights, predictions)."""
        weights = self.fuzzy_controller.compute_strategy_weights(train_pairs, test_input)

        predictions = []

        # Try all solvers
        for solver_name, solver, weight_key in [
            ('Pattern', self.pattern_solver, 'pattern_weight'),
            ('Example', self.example_solver, 'example_weight'),
            ('Object', self.object_solver, 'object_weight'),
            ('NearestNeighbor', self.neighbor_solver, 'neighbor_weight'),
            ('Symmetry', self.symmetry_solver, 'symmetry_weight'),
        ]:
            if weights[weight_key] > 0.3:
                try:
                    pred = solver.solve(train_pairs, test_input)
                    predictions.append({
                        'solver': solver_name,
                        'prediction': pred is not None,
                        'confidence': weights[weight_key]
                    })
                except:
                    predictions.append({
                        'solver': solver_name,
                        'prediction': False,
                        'confidence': weights[weight_key]
                    })

        debug_info = {
            'weights': weights,
            'predictions': predictions
        }

        # Get best prediction
        result = self.solve(train_pairs, test_input)

        return result, debug_info
