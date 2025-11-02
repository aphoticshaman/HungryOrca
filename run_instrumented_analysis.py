#!/usr/bin/env python3
"""
Run Instrumented Analysis on All BOLT-ONs

Collects deep metrics to identify:
- Near-miss tasks (high partial match)
- Learning successes (rules learned but wrong application)
- Shape correctness without color accuracy
- Complementary solver pairs
- Uplift opportunities

This reveals the PROs of NO-GO components.
"""

import json
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'SubtleGenius' / 'arc_solvers'))

from instrumentation_framework import InstrumentedSolver, MetricsAggregator
from multiscale_object_solver import MultiScaleSolver
from pattern_transformation_solver import PatternTransformationSolver
from spatial_rule_solver import SpatialRuleSolver
from size_transformation_solver import SizeTransformationSolver
from example_based_solver import ExampleBasedSolver
from grid_structure_solver import GridStructureSolver
from traditional_approaches import (
    SymmetrySolver, HistogramSolver, IdentitySolver,
    NearestNeighborSolver, RuleInductionSolver, AbstractionSolver
)

# Load 10 ARC tasks
with open('arc-agi_training_challenges.json', 'r') as f:
    challenges = json.load(f)

with open('arc-agi_training_solutions.json', 'r') as f:
    solutions = json.load(f)

task_ids = list(challenges.keys())[:10]

print("="*80)
print("INSTRUMENTED DEEP DIVE ANALYSIS")
print("="*80)
print("\nCollecting comprehensive metrics on all BOLT-ONs...")
print("Goal: Identify uplift opportunities in NO-GO components\n")

# Create instrumented versions
instrumented_solvers = [
    InstrumentedSolver(MultiScaleSolver(), 'BOLTON-01-Object'),
    InstrumentedSolver(PatternTransformationSolver(), 'BOLTON-02-Pattern'),
    InstrumentedSolver(SpatialRuleSolver(), 'BOLTON-03-Rules'),
    InstrumentedSolver(SizeTransformationSolver(), 'BOLTON-04-Size'),
    InstrumentedSolver(ExampleBasedSolver(), 'BOLTON-06-Example'),
    InstrumentedSolver(GridStructureSolver(), 'BOLTON-07-Grid'),
    InstrumentedSolver(SymmetrySolver(), 'BOLTON-09-Symmetry'),
    InstrumentedSolver(HistogramSolver(), 'BOLTON-10-Histogram'),
    InstrumentedSolver(IdentitySolver(), 'BOLTON-11-Identity'),
    InstrumentedSolver(NearestNeighborSolver(), 'BOLTON-13-NearestNeighbor'),
    InstrumentedSolver(RuleInductionSolver(), 'BOLTON-14-RuleInduction'),
    InstrumentedSolver(AbstractionSolver(), 'BOLTON-15-Abstraction'),
]

aggregator = MetricsAggregator()

# Run instrumented tests
for task_id in task_ids:
    print(f"Analyzing task {task_id}...")

    task = challenges[task_id]
    train_pairs = [(np.array(ex['input']), np.array(ex['output']))
                   for ex in task['train']]

    for test_idx, test_ex in enumerate(task['test']):
        test_input = np.array(test_ex['input'])

        if task_id in solutions and test_idx < len(solutions[task_id]):
            expected = np.array(solutions[task_id][test_idx])

            # Test each instrumented solver
            for instrumented in instrumented_solvers:
                metrics = instrumented.solve_with_metrics(
                    train_pairs, test_input, expected, task_id
                )
                aggregator.add_metrics([metrics])

print("\n" + "="*80)
print("ANALYSIS RESULTS")
print("="*80)

# Export full report
report = aggregator.export_report('instrumented_analysis_report.json')

# Print key findings
print("\n### UPLIFT OPPORTUNITIES (PROs of NO-GO Components) ###\n")

opportunities = report['uplift_opportunities']

if opportunities:
    for solver_name, data in opportunities.items():
        print(f"{solver_name}:")
        print(f"  Uplift Score: {data['uplift_score']:.1f}")
        print(f"  Reasons:")
        for reason in data['reasons']:
            print(f"    - {reason}")

        analysis = data['analysis']
        print(f"  Metrics:")
        print(f"    - Exact matches: {analysis['exact_matches']}/{analysis['total_tasks']}")
        print(f"    - Predictions made: {analysis['made_predictions']}/{analysis['total_tasks']}")
        print(f"    - Avg partial match: {analysis['avg_partial_match']:.2f}")
        print(f"    - Near misses: {analysis['near_misses']}")
        print()
else:
    print("No uplift opportunities identified (all NO-GO components failed completely)\n")

print("\n### COMPLEMENTARY SOLVER PAIRS ###\n")

complementary = report['complementary_pairs'][:5]  # Top 5

if complementary:
    for s1, s2, score in complementary:
        print(f"{s1} + {s2}: Complementarity = {score:.2f}")
else:
    print("No complementary pairs found\n")

print("\n### SOLVER SUMMARIES ###\n")

for solver_name, analysis in report['solver_analyses'].items():
    status = "GO" if analysis['exact_match_rate'] > 0 else "NO-GO"

    print(f"{solver_name} [{status}]:")
    print(f"  Success: {analysis['exact_matches']}/10")
    print(f"  Predictions: {analysis['made_predictions']}/10")
    print(f"  Avg partial match: {analysis['avg_partial_match']:.2f}")
    print(f"  Near misses (>70%): {analysis['near_misses']}")
    print(f"  Rules learned: {analysis['learned_rules']}/10")

    if analysis['made_predictions'] > 0:
        print(f"  Shape correct rate: {analysis['shape_correct_rate']:.2%}")
        print(f"  Avg color accuracy: {analysis['avg_color_accuracy']:.2%}")

    print()

print("="*80)
print(f"Full report saved to: instrumented_analysis_report.json")
print("="*80)
