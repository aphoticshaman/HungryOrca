#!/usr/bin/env python3
"""
TESTS 1.3-1.8: Batch Feature Ablation Testing

Efficiently tests remaining 6 features while maintaining 5×3 protocol:
- Test 1.3: Non-locality score
- Test 1.4: Criticality index
- Test 1.5: Pattern entropy
- Test 1.6: Grid size factor
- Test 1.7: Color complexity
- Test 1.8: Transformation consistency

Each test: 5 runs × 3 conditions (A: baseline, B: feature alone, C: combined)

Author: HungryOrca Phase 7 Week 1
Date: 2025-11-02
"""

import json
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'SubtleGenius' / 'arc_solvers'))

from fuzzy_arc_integrated_solver import FuzzyIntegratedARCSolver, SimpleFuzzyController
from full_feature_extraction import FullFeatureExtractor
from pattern_transformation_solver import PatternTransformationSolver
from example_based_solver import ExampleBasedSolver

# Load test tasks (first 10)
with open('arc-agi_training_challenges.json', 'r') as f:
    challenges = json.load(f)

with open('arc-agi_training_solutions.json', 'r') as f:
    solutions = json.load(f)

task_ids = list(challenges.keys())[:10]


# ============================================================================
# BASELINE SOLVER (Condition A for all tests)
# ============================================================================

class BaselineSolver:
    """Current 4-feature fuzzy system (baseline for all tests)."""

    def __init__(self):
        self.solver = FuzzyIntegratedARCSolver()

    def solve(self, train_pairs, test_input):
        return self.solver.solve(train_pairs, test_input)


# ============================================================================
# FEATURE-SPECIFIC SOLVERS
# ============================================================================

class FeatureBasedSolver:
    """Generic solver that uses a single feature for decision-making."""

    def __init__(self, feature_name: str, threshold: float = 0.5):
        self.feature_extractor = FullFeatureExtractor()
        self.feature_name = feature_name
        self.threshold = threshold
        self.pattern_solver = PatternTransformationSolver()
        self.example_solver = ExampleBasedSolver()

    def solve(self, train_pairs, test_input):
        # Extract the specific feature
        features = self.feature_extractor.extract(train_pairs, test_input)
        feature_value = getattr(features, self.feature_name)

        # Simple rule: High feature value → pattern solver, else example solver
        if feature_value > self.threshold:
            try:
                return self.pattern_solver.solve(train_pairs, test_input)
            except:
                pass

        try:
            return self.example_solver.solve(train_pairs, test_input)
        except:
            return None


class CombinedFeatureSolver:
    """Baseline + one additional feature."""

    def __init__(self, feature_name: str, boost_factor: float = 1.5):
        self.simple_fuzzy = SimpleFuzzyController()
        self.feature_extractor = FullFeatureExtractor()
        self.feature_name = feature_name
        self.boost_factor = boost_factor
        self.pattern_solver = PatternTransformationSolver()
        self.example_solver = ExampleBasedSolver()

    def solve(self, train_pairs, test_input):
        # Get baseline fuzzy weights
        simple_weights = self.simple_fuzzy.compute_strategy_weights(train_pairs, test_input)

        # Get sophisticated feature
        features = self.feature_extractor.extract(train_pairs, test_input)
        feature_value = getattr(features, self.feature_name)

        # Boost pattern weight if feature is high
        pattern_weight = simple_weights['pattern_weight']
        example_weight = simple_weights['example_weight']

        if feature_value > 0.6:
            pattern_weight = min(1.0, pattern_weight * self.boost_factor)

        # Select solver
        predictions = []

        if pattern_weight > 0.3:
            try:
                pred = self.pattern_solver.solve(train_pairs, test_input)
                if pred is not None:
                    predictions.append((pred, pattern_weight))
            except:
                pass

        if example_weight > 0.3:
            try:
                pred = self.example_solver.solve(train_pairs, test_input)
                if pred is not None:
                    predictions.append((pred, example_weight))
            except:
                pass

        if predictions:
            best = max(predictions, key=lambda x: x[1])
            return best[0]

        return None


# ============================================================================
# TESTING HARNESS
# ============================================================================

def run_solver(solver, num_runs: int = 5):
    """Run solver multiple times on test set."""
    all_runs = []

    for run_num in range(1, num_runs + 1):
        solved_tasks = []

        for task_id in task_ids:
            task = challenges[task_id]
            train_pairs = [(np.array(ex['input']), np.array(ex['output']))
                          for ex in task['train']]

            for test_idx, test_ex in enumerate(task['test']):
                test_input = np.array(test_ex['input'])

                if task_id in solutions and test_idx < len(solutions[task_id]):
                    expected = np.array(solutions[task_id][test_idx])

                    try:
                        predicted = solver.solve(train_pairs, test_input)

                        if predicted is not None and np.array_equal(predicted, expected):
                            if task_id not in solved_tasks:
                                solved_tasks.append(task_id)
                    except:
                        pass

        accuracy = len(solved_tasks) / len(task_ids) * 100
        all_runs.append({
            'run_number': run_num,
            'tasks_solved': len(solved_tasks),
            'accuracy': accuracy,
            'solved_task_ids': solved_tasks
        })

    return all_runs


def test_feature(test_id: str, feature_name: str, hypothesis: str):
    """Test one feature following 5×3 protocol."""
    print(f"\n{'='*80}")
    print(f"TEST {test_id}: {feature_name.replace('_', ' ').title()} Feature")
    print(f"{'='*80}")
    print(f"Hypothesis: {hypothesis}")

    # Condition A: Baseline
    print(f"\nCondition A: Baseline (4 features) - 5 runs...")
    baseline_solver = BaselineSolver()
    baseline_results = run_solver(baseline_solver, num_runs=5)
    mean_a = np.mean([r['accuracy'] for r in baseline_results])

    # Condition B: Feature alone
    print(f"Condition B: {feature_name} alone - 5 runs...")
    feature_solver = FeatureBasedSolver(feature_name)
    feature_results = run_solver(feature_solver, num_runs=5)
    mean_b = np.mean([r['accuracy'] for r in feature_results])

    # Condition C: Combined
    print(f"Condition C: Baseline + {feature_name} - 5 runs...")
    combined_solver = CombinedFeatureSolver(feature_name)
    combined_results = run_solver(combined_solver, num_runs=5)
    mean_c = np.mean([r['accuracy'] for r in combined_results])

    # Analysis
    improvement = mean_c - mean_a

    print(f"\nResults:")
    print(f"  Baseline:  {mean_a:.1f}%")
    print(f"  Feature:   {mean_b:.1f}%")
    print(f"  Combined:  {mean_c:.1f}%")
    print(f"  Improvement: {improvement:+.1f} pp")

    # Decision
    if mean_c > mean_a and improvement >= 5.0:
        decision = "GO"
        symbol = "✅"
    elif mean_c > mean_a:
        decision = "MARGINAL"
        symbol = "⚠️"
    else:
        decision = "NO-GO"
        symbol = "❌"

    print(f"\n{symbol} Decision: {decision}")

    return {
        'test_id': test_id,
        'feature_name': feature_name,
        'hypothesis': hypothesis,
        'baseline_results': baseline_results,
        'feature_results': feature_results,
        'combined_results': combined_results,
        'mean_baseline': mean_a,
        'mean_feature': mean_b,
        'mean_combined': mean_c,
        'improvement': improvement,
        'decision': decision
    }


# ============================================================================
# MAIN BATCH TESTING
# ============================================================================

print("="*80)
print("BATCH FEATURE ABLATION: Tests 1.3-1.8")
print("="*80)
print("\nTesting 6 remaining features with 5×3 protocol each")
print("Total runs: 6 features × 3 conditions × 5 runs = 90 test runs")
print()

all_results = {}

# Test 1.3: Non-locality score
all_results['1.3'] = test_feature(
    '1.3',
    'non_locality_score',
    'Global constraint detection → better puzzle understanding'
)

# Test 1.4: Criticality index
all_results['1.4'] = test_feature(
    '1.4',
    'criticality_index',
    'Phase transition detection → better pattern recognition'
)

# Test 1.5: Pattern entropy
all_results['1.5'] = test_feature(
    '1.5',
    'pattern_entropy',
    'Color complexity → better strategy selection'
)

# Test 1.6: Grid size factor
all_results['1.6'] = test_feature(
    '1.6',
    'grid_size_factor',
    'Size-aware processing → better resource allocation'
)

# Test 1.7: Color complexity
all_results['1.7'] = test_feature(
    '1.7',
    'color_complexity',
    'Color diversity → better transformation detection'
)

# Test 1.8: Transformation consistency
all_results['1.8'] = test_feature(
    '1.8',
    'transformation_consistency',
    'Consistency detection → better generalization'
)

# ============================================================================
# WEEK 1 SUMMARY
# ============================================================================

print("\n" + "="*80)
print("WEEK 1 SUMMARY: All 8 Features Tested")
print("="*80)

# Include previous results
all_results['1.1'] = {
    'test_id': '1.1',
    'feature_name': 'symmetry_strength',
    'decision': 'NO-GO',
    'improvement': 0.0
}

all_results['1.2'] = {
    'test_id': '1.2',
    'feature_name': 'multi_scale_complexity',
    'decision': 'NO-GO',
    'improvement': 0.0
}

# Count decisions
go_count = sum(1 for r in all_results.values() if r['decision'] == 'GO')
marginal_count = sum(1 for r in all_results.values() if r['decision'] == 'MARGINAL')
no_go_count = sum(1 for r in all_results.values() if r['decision'] == 'NO-GO')

print(f"\nFeature Test Results:")
print(f"  ✅ GO:       {go_count}/8")
print(f"  ⚠️  MARGINAL: {marginal_count}/8")
print(f"  ❌ NO-GO:    {no_go_count}/8")

# Decision Gate 1
print("\n" + "="*80)
print("DECISION GATE 1: Week 1 Evaluation")
print("="*80)

gate_1_pass = go_count >= 6

print(f"\nCriteria:")
print(f"  Required: ≥6/8 features accepted")
print(f"  Actual:   {go_count}/8 features accepted")
print(f"  Status:   {'✅ PASS' if gate_1_pass else '❌ FAIL'}")

if not gate_1_pass:
    print(f"\nGate 1 Analysis:")
    print(f"  - {no_go_count} features showed no improvement")
    print(f"  - Current 4-feature system may already be near-optimal for test set")
    print(f"  - Sophisticated features don't add value individually")
    print(f"\nRecommended Actions:")
    print(f"  1. Test features in combination (synergy effects)")
    print(f"  2. Expand test set beyond 10 tasks")
    print(f"  3. Focus on Week 2 (fuzzy rules) instead")
    print(f"  4. Re-evaluate feature engineering approach")

# Export all results
with open('week_1_batch_results.json', 'w') as f:
    json.dump(all_results, f, indent=2, default=str)

print(f"\n✓ Results exported to: week_1_batch_results.json")
print("="*80)
