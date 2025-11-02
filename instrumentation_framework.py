#!/usr/bin/env python3
"""
Deep Instrumentation Framework for BOLT-ON Analysis

Provides sophisticated metrics to understand:
- Why components fail
- What partial successes occur
- Which features/patterns are learned
- Convergence behavior
- Uplift potential

Author: HungryOrca Deep Dive Framework
Date: 2025-11-02
"""

import numpy as np
import json
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from collections import defaultdict


@dataclass
class SolverMetrics:
    """Comprehensive metrics for solver performance."""
    solver_name: str
    task_id: str

    # Success metrics
    exact_match: bool
    partial_match_score: float  # 0-1, how close to correct

    # Prediction quality
    made_prediction: bool
    prediction_shape: Optional[Tuple[int, int]]
    expected_shape: Optional[Tuple[int, int]]
    shape_correct: bool

    # Color metrics
    color_accuracy: float  # Fraction of pixels with correct color
    color_distribution_error: float  # KL divergence of color histograms

    # Pattern recognition
    detected_pattern: Optional[str]
    pattern_confidence: float

    # Learning indicators
    learned_rule: bool
    rule_type: Optional[str]
    rule_confidence: float

    # Error analysis
    error_type: Optional[str]  # 'size', 'color', 'pattern', 'none', 'no_prediction'
    error_magnitude: float


class InstrumentedSolver:
    """Wrapper that instruments any solver with deep metrics."""

    def __init__(self, base_solver, solver_name: str):
        self.base_solver = base_solver
        self.solver_name = solver_name
        self.metrics_log = []

    def solve_with_metrics(self, train_pairs: List[Tuple[np.ndarray, np.ndarray]],
                           test_input: np.ndarray,
                           expected_output: np.ndarray,
                           task_id: str) -> SolverMetrics:
        """Solve and collect comprehensive metrics."""

        # Attempt to solve
        prediction = None
        detected_pattern = None
        pattern_confidence = 0.0
        learned_rule = False
        rule_type = None
        rule_confidence = 0.0

        try:
            prediction = self.base_solver.solve(train_pairs, test_input)

            # Try to extract internal state (pattern detection, rules learned)
            if hasattr(self.base_solver, '_learn_rules'):
                # Spatial/rule-based solver
                rules = self.base_solver._learn_rules(train_pairs)
                if rules:
                    learned_rule = True
                    rule_type = rules[0].type if hasattr(rules[0], 'type') else 'unknown'
                    rule_confidence = rules[0].confidence if hasattr(rules[0], 'confidence') else 0.0

            if hasattr(self.base_solver, '_learn_transformations'):
                # Pattern solver
                rules = self.base_solver._learn_transformations(train_pairs)
                if rules:
                    learned_rule = True
                    detected_pattern = rules[0].type if hasattr(rules[0], 'type') else 'unknown'
                    pattern_confidence = rules[0].confidence if hasattr(rules[0], 'confidence') else 0.0

        except Exception as e:
            # Solver crashed
            pass

        # Compute metrics
        made_prediction = prediction is not None

        if made_prediction:
            # Shape metrics
            pred_shape = prediction.shape
            exp_shape = expected_output.shape
            shape_correct = (pred_shape == exp_shape)

            # Exact match
            exact_match = np.array_equal(prediction, expected_output) if shape_correct else False

            # Partial match (if shapes match)
            if shape_correct:
                correct_pixels = np.sum(prediction == expected_output)
                total_pixels = prediction.size
                partial_match_score = correct_pixels / total_pixels
                color_accuracy = partial_match_score
            else:
                partial_match_score = 0.0
                color_accuracy = 0.0

            # Color distribution error
            if shape_correct:
                pred_hist = np.histogram(prediction, bins=10, range=(0, 9))[0]
                exp_hist = np.histogram(expected_output, bins=10, range=(0, 9))[0]
                pred_hist = pred_hist / (pred_hist.sum() + 1e-10)
                exp_hist = exp_hist / (exp_hist.sum() + 1e-10)

                # KL divergence
                kl_div = np.sum(exp_hist * np.log((exp_hist + 1e-10) / (pred_hist + 1e-10)))
                color_distribution_error = float(kl_div)
            else:
                color_distribution_error = 10.0  # High error

            # Error classification
            if exact_match:
                error_type = 'none'
                error_magnitude = 0.0
            elif not shape_correct:
                error_type = 'size'
                error_magnitude = abs(pred_shape[0] * pred_shape[1] - exp_shape[0] * exp_shape[1]) / (exp_shape[0] * exp_shape[1])
            elif color_accuracy < 0.5:
                error_type = 'color'
                error_magnitude = 1.0 - color_accuracy
            else:
                error_type = 'pattern'
                error_magnitude = 1.0 - partial_match_score

        else:
            # No prediction
            pred_shape = None
            exp_shape = expected_output.shape
            shape_correct = False
            exact_match = False
            partial_match_score = 0.0
            color_accuracy = 0.0
            color_distribution_error = 10.0
            error_type = 'no_prediction'
            error_magnitude = 1.0

        metrics = SolverMetrics(
            solver_name=self.solver_name,
            task_id=task_id,
            exact_match=exact_match,
            partial_match_score=partial_match_score,
            made_prediction=made_prediction,
            prediction_shape=pred_shape,
            expected_shape=exp_shape,
            shape_correct=shape_correct,
            color_accuracy=color_accuracy,
            color_distribution_error=color_distribution_error,
            detected_pattern=detected_pattern,
            pattern_confidence=pattern_confidence,
            learned_rule=learned_rule,
            rule_type=rule_type,
            rule_confidence=rule_confidence,
            error_type=error_type,
            error_magnitude=error_magnitude
        )

        self.metrics_log.append(metrics)
        return metrics


class MetricsAggregator:
    """Aggregates and analyzes metrics across all solvers and tasks."""

    def __init__(self):
        self.all_metrics: List[SolverMetrics] = []

    def add_metrics(self, metrics: List[SolverMetrics]):
        """Add metrics from instrumented run."""
        self.all_metrics.extend(metrics)

    def analyze_solver(self, solver_name: str) -> Dict[str, Any]:
        """Deep analysis of specific solver."""
        solver_metrics = [m for m in self.all_metrics if m.solver_name == solver_name]

        if not solver_metrics:
            return {}

        total = len(solver_metrics)

        analysis = {
            'solver_name': solver_name,
            'total_tasks': total,

            # Success rates
            'exact_matches': sum(m.exact_match for m in solver_metrics),
            'exact_match_rate': sum(m.exact_match for m in solver_metrics) / total,

            # Partial success (near misses)
            'made_predictions': sum(m.made_prediction for m in solver_metrics),
            'prediction_rate': sum(m.made_prediction for m in solver_metrics) / total,
            'avg_partial_match': np.mean([m.partial_match_score for m in solver_metrics]),
            'near_misses': sum(1 for m in solver_metrics if 0.7 <= m.partial_match_score < 1.0),

            # Shape correctness
            'shape_correct_count': sum(m.shape_correct for m in solver_metrics if m.made_prediction),
            'shape_correct_rate': sum(m.shape_correct for m in solver_metrics if m.made_prediction) / sum(m.made_prediction for m in solver_metrics) if any(m.made_prediction for m in solver_metrics) else 0,

            # Learning indicators
            'learned_rules': sum(m.learned_rule for m in solver_metrics),
            'learning_rate': sum(m.learned_rule for m in solver_metrics) / total,

            # Error breakdown
            'error_types': {
                'no_prediction': sum(1 for m in solver_metrics if m.error_type == 'no_prediction'),
                'size': sum(1 for m in solver_metrics if m.error_type == 'size'),
                'color': sum(1 for m in solver_metrics if m.error_type == 'color'),
                'pattern': sum(1 for m in solver_metrics if m.error_type == 'pattern'),
            },

            # Quality metrics
            'avg_color_accuracy': np.mean([m.color_accuracy for m in solver_metrics if m.made_prediction]) if any(m.made_prediction for m in solver_metrics) else 0,
        }

        return analysis

    def find_complementary_pairs(self) -> List[Tuple[str, str, float]]:
        """Find pairs of solvers that solve different tasks (complementarity)."""
        solver_names = list(set(m.solver_name for m in self.all_metrics))

        # Build solver → tasks solved mapping
        solver_tasks = defaultdict(set)
        for m in self.all_metrics:
            if m.exact_match:
                solver_tasks[m.solver_name].add(m.task_id)

        complementary = []
        for i, s1 in enumerate(solver_names):
            for s2 in solver_names[i+1:]:
                tasks1 = solver_tasks[s1]
                tasks2 = solver_tasks[s2]

                # Complementarity score (how many additional tasks does s2 solve?)
                additional = len(tasks2 - tasks1)
                total_solved = len(tasks1 | tasks2)

                if total_solved > 0:
                    complementarity = additional / total_solved
                    complementary.append((s1, s2, complementarity))

        return sorted(complementary, key=lambda x: x[2], reverse=True)

    def identify_uplift_opportunities(self) -> Dict[str, Any]:
        """Identify which NO-GO components have uplift potential."""
        opportunities = {}

        solver_names = list(set(m.solver_name for m in self.all_metrics))

        for solver_name in solver_names:
            analysis = self.analyze_solver(solver_name)

            # Uplift potential indicators
            uplift_score = 0.0
            reasons = []

            # 1. Near misses (high partial match but not exact)
            if analysis['near_misses'] > 0:
                uplift_score += 2.0 * analysis['near_misses']
                reasons.append(f"{analysis['near_misses']} near-miss tasks (>70% correct)")

            # 2. High learning rate but low success
            if analysis['learning_rate'] > 0.5 and analysis['exact_match_rate'] == 0:
                uplift_score += 1.5
                reasons.append(f"Learning rules ({analysis['learning_rate']*100:.0f}% of tasks) but failing")

            # 3. Correct shape but wrong colors
            if analysis['shape_correct_rate'] > 0.7 and analysis['exact_match_rate'] < 0.3:
                uplift_score += 1.0
                reasons.append(f"Shape correct ({analysis['shape_correct_rate']*100:.0f}%) but color errors")

            # 4. Makes predictions (not failing to act)
            if analysis['prediction_rate'] > 0.8 and analysis['exact_match_rate'] == 0:
                uplift_score += 0.5
                reasons.append("High prediction rate but wrong")

            if uplift_score > 0:
                opportunities[solver_name] = {
                    'uplift_score': uplift_score,
                    'reasons': reasons,
                    'analysis': analysis
                }

        return dict(sorted(opportunities.items(), key=lambda x: x[1]['uplift_score'], reverse=True))

    def export_report(self, filename: str):
        """Export comprehensive analysis report."""
        report = {
            'summary': {
                'total_metrics': len(self.all_metrics),
                'unique_solvers': len(set(m.solver_name for m in self.all_metrics)),
                'unique_tasks': len(set(m.task_id for m in self.all_metrics)),
            },
            'solver_analyses': {},
            'complementary_pairs': self.find_complementary_pairs(),
            'uplift_opportunities': self.identify_uplift_opportunities(),
        }

        solver_names = list(set(m.solver_name for m in self.all_metrics))
        for solver_name in solver_names:
            report['solver_analyses'][solver_name] = self.analyze_solver(solver_name)

        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        return report
