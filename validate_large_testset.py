#!/usr/bin/env python3
"""
PHASE 5: Validation on Expanded Test Set (50+ tasks)

Tests fuzzy-integrated system on larger task set to validate scalability.

Author: HungryOrca Validation Framework
Date: 2025-11-02
"""

import json
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'SubtleGenius' / 'arc_solvers'))

from fuzzy_arc_integrated_solver import FuzzyIntegratedARCSolver
from pattern_transformation_solver import PatternTransformationSolver

# Load ALL training tasks
with open('arc-agi_training_challenges.json', 'r') as f:
    challenges = json.load(f)

with open('arc-agi_training_solutions.json', 'r') as f:
    solutions = json.load(f)

# Test on first 50 tasks
task_ids = list(challenges.keys())[:50]

print("="*80)
print("LARGE-SCALE VALIDATION: 50 ARC Tasks")
print("="*80)
print(f"\nTesting on {len(task_ids)} tasks")
print("Baseline: Pattern solver (1/10 = 10%)")
print("Expected: Fuzzy-integrated (2/10 = 20%)")
print("\nGoal: Validate consistent improvement on larger set\n")

# Initialize solvers
baseline_solver = PatternTransformationSolver()
fuzzy_solver = FuzzyIntegratedARCSolver()

# Test baseline
print("Testing BASELINE (Pattern Solver)...")
baseline_solved = 0
baseline_total = 0
baseline_task_ids = []

for i, task_id in enumerate(task_ids):
    if (i + 1) % 10 == 0:
        print(f"  Progress: {i+1}/{len(task_ids)} tasks...")

    task = challenges[task_id]
    train_pairs = [(np.array(ex['input']), np.array(ex['output']))
                   for ex in task['train']]

    for test_idx, test_ex in enumerate(task['test']):
        test_input = np.array(test_ex['input'])

        if task_id in solutions and test_idx < len(solutions[task_id]):
            expected = np.array(solutions[task_id][test_idx])

            try:
                predicted = baseline_solver.solve(train_pairs, test_input)

                if predicted is not None and np.array_equal(predicted, expected):
                    baseline_solved += 1
                    if task_id not in baseline_task_ids:
                        baseline_task_ids.append(task_id)
            except:
                pass

            baseline_total += 1

baseline_accuracy = baseline_solved / baseline_total if baseline_total > 0 else 0

print(f"\nBaseline Results:")
print(f"  Tasks solved: {len(baseline_task_ids)}/{len(task_ids)}")
print(f"  Test examples solved: {baseline_solved}/{baseline_total}")
print(f"  Accuracy: {baseline_accuracy:.2%}")

# Test fuzzy-integrated
print("\n" + "="*80)
print("Testing FUZZY-INTEGRATED...")
fuzzy_solved = 0
fuzzy_total = 0
fuzzy_task_ids = []

for i, task_id in enumerate(task_ids):
    if (i + 1) % 10 == 0:
        print(f"  Progress: {i+1}/{len(task_ids)} tasks...")

    task = challenges[task_id]
    train_pairs = [(np.array(ex['input']), np.array(ex['output']))
                   for ex in task['train']]

    for test_idx, test_ex in enumerate(task['test']):
        test_input = np.array(test_ex['input'])

        if task_id in solutions and test_idx < len(solutions[task_id]):
            expected = np.array(solutions[task_id][test_idx])

            try:
                predicted = fuzzy_solver.solve(train_pairs, test_input)

                if predicted is not None and np.array_equal(predicted, expected):
                    fuzzy_solved += 1
                    if task_id not in fuzzy_task_ids:
                        fuzzy_task_ids.append(task_id)
            except:
                pass

            fuzzy_total += 1

fuzzy_accuracy = fuzzy_solved / fuzzy_total if fuzzy_total > 0 else 0

print(f"\nFuzzy-Integrated Results:")
print(f"  Tasks solved: {len(fuzzy_task_ids)}/{len(task_ids)}")
print(f"  Test examples solved: {fuzzy_solved}/{fuzzy_total}")
print(f"  Accuracy: {fuzzy_accuracy:.2%}")

# Comparison
print("\n" + "="*80)
print("COMPARISON")
print("="*80)

improvement_tasks = len(fuzzy_task_ids) - len(baseline_task_ids)
improvement_pct = (fuzzy_accuracy - baseline_accuracy) * 100

print(f"\nTask-level improvement: +{improvement_tasks} tasks")
print(f"  Baseline: {len(baseline_task_ids)}/{len(task_ids)} = {len(baseline_task_ids)/len(task_ids)*100:.1f}%")
print(f"  Fuzzy: {len(fuzzy_task_ids)}/{len(task_ids)} = {len(fuzzy_task_ids)/len(task_ids)*100:.1f}%")

print(f"\nAccuracy improvement: +{improvement_pct:.1f} percentage points")
print(f"  Baseline: {baseline_accuracy:.2%}")
print(f"  Fuzzy: {fuzzy_accuracy:.2%}")

if len(fuzzy_task_ids) > len(baseline_task_ids):
    print("\n✅ VALIDATION SUCCESS: Fuzzy-integrated consistently outperforms baseline")
    print(f"   Relative improvement: {(len(fuzzy_task_ids) / len(baseline_task_ids) - 1) * 100:.1f}%")
else:
    print("\n⚠️  No improvement on larger set")

# Additional tasks solved by fuzzy
additional_tasks = set(fuzzy_task_ids) - set(baseline_task_ids)
if additional_tasks:
    print(f"\nAdditional tasks solved by fuzzy (not by baseline):")
    for task_id in additional_tasks:
        print(f"  - {task_id}")

# Export results
results = {
    'test_set_size': len(task_ids),
    'baseline': {
        'tasks_solved': len(baseline_task_ids),
        'accuracy': baseline_accuracy,
        'solved_task_ids': baseline_task_ids
    },
    'fuzzy_integrated': {
        'tasks_solved': len(fuzzy_task_ids),
        'accuracy': fuzzy_accuracy,
        'solved_task_ids': fuzzy_task_ids
    },
    'improvement': {
        'tasks': improvement_tasks,
        'accuracy_points': improvement_pct
    }
}

with open('large_testset_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n✓ Results exported to: large_testset_results.json")
print("="*80)
