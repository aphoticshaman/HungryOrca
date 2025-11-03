#!/usr/bin/env python3
"""
TEST 2.1: First 10 Fuzzy Rules Ablation

Protocol (x5):
- Condition A: Transformations only (0 rules) - 5 runs
- Condition B: 10 fuzzy rules alone - 5 runs
- Condition C: Transformations + 10 fuzzy rules - 5 runs

Hypothesis: Fuzzy rule orchestration improves transformation selection

Author: HungryOrca Phase 7 Week 2
Date: 2025-11-02
"""

import json
import numpy as np
from fuzzy_transformation_solver import FuzzyTransformationSolver

# Load test tasks
with open('arc-agi_training_challenges.json', 'r') as f:
    challenges = json.load(f)

with open('arc-agi_training_solutions.json', 'r') as f:
    solutions = json.load(f)

task_ids = list(challenges.keys())[:10]


def run_solver(num_rules: int, num_runs: int = 5):
    """Run solver multiple times."""
    all_runs = []

    for run_num in range(1, num_runs + 1):
        solver = FuzzyTransformationSolver(num_rules=num_rules)
        solved_tasks = []

        for task_id in task_ids:
            task = challenges[task_id]
            train_pairs = [(np.array(ex['input']), np.array(ex['output']))
                          for ex in task['train']]

            test_input = np.array(task['test'][0]['input'])
            expected = np.array(solutions[task_id][0]) if task_id in solutions else None

            try:
                predicted = solver.solve(train_pairs, test_input)

                if predicted is not None and expected is not None and np.array_equal(predicted, expected):
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


print("="*80)
print("TEST 2.1: FIRST 10 FUZZY RULES ABLATION")
print("="*80)

# Condition A: Baseline (0 rules)
print("\n--- CONDITION A: Transformations Only (0 rules) - 5 runs ---")
baseline_results = run_solver(num_rules=0, num_runs=5)
for r in baseline_results:
    print(f"  Run {r['run_number']}: {r['tasks_solved']}/10 ({r['accuracy']:.1f}%)")

# Condition C: Combined (10 rules)
print("\n--- CONDITION C: Transformations + 10 Fuzzy Rules - 5 runs ---")
combined_results = run_solver(num_rules=10, num_runs=5)
for r in combined_results:
    print(f"  Run {r['run_number']}: {r['tasks_solved']}/10 ({r['accuracy']:.1f}%)")

# Analysis
mean_a = np.mean([r['accuracy'] for r in baseline_results])
mean_c = np.mean([r['accuracy'] for r in combined_results])
improvement = mean_c - mean_a

print("\n" + "="*80)
print("STATISTICAL ANALYSIS")
print("="*80)
print(f"\nCondition A (Baseline): {mean_a:.1f}%")
print(f"Condition C (10 Rules): {mean_c:.1f}%")
print(f"Improvement: {improvement:+.1f} pp")

# Decision
if mean_c > mean_a and improvement >= 5.0:
    decision = "✅ GO"
    print(f"\n{decision} - Accept 10 fuzzy rules (improvement ≥ 5%)")
elif mean_c > mean_a:
    decision = "⚠️ MARGINAL"
    print(f"\n{decision} - Improvement < 5% threshold")
else:
    decision = "❌ NO-GO"
    print(f"\n{decision} - No improvement")

# Export
results = {
    'test_id': '2.1',
    'baseline_results': baseline_results,
    'combined_results': combined_results,
    'mean_baseline': mean_a,
    'mean_combined': mean_c,
    'improvement': improvement,
    'decision': decision
}

with open('test_2_1_fuzzy_rules_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n✓ Results exported to: test_2_1_fuzzy_rules_results.json")
print("="*80)
