#!/usr/bin/env python3
"""
TEST BOLT-ON #2 + #3 COMBINED: Multi-Solver (Pattern + Rules)

Comparison:
- Baseline (NSPSA): 0/10 (0%)
- BOLT-ON #2 alone: 1/10 (10%)
- BOLT-ON #2 + #3: ? (testing now)

Components:
- BOLT-ON #2: Pattern transformations (tiling, reflection, color mapping)
- BOLT-ON #3: Spatial rule learning (color replacement, region operations)

Note: BOLT-ON #1 (object detection) deferred for later testing per user guidance.

Success criteria: Solve ≥ 2 tasks (improvement over BOLT-ON #2 alone)
"""

import json
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'SubtleGenius' / 'arc_solvers'))

from pattern_transformation_solver import PatternTransformationSolver
from spatial_rule_solver import SpatialRuleSolver

# Load first 10 ARC tasks
with open('arc-agi_training_challenges.json', 'r') as f:
    challenges = json.load(f)

with open('arc-agi_training_solutions.json', 'r') as f:
    solutions = json.load(f)

task_ids = list(challenges.keys())[:10]

print("="*80)
print("BOLT-ON #2 + #3 COMBINED TEST: Pattern + Rules Solver")
print("="*80)
print(f"\nTesting {len(task_ids)} tasks")
print("Baseline: 0/10 (0%)")
print("BOLT-ON #2 alone: 1/10 (10%)")
print("Goal: Solve ≥ 2 tasks\n")

pattern_solver = PatternTransformationSolver()
rule_solver = SpatialRuleSolver()

results = []
for i, task_id in enumerate(task_ids):
    print(f"[{i+1}/10] Task {task_id}")

    task = challenges[task_id]

    train_pairs = [(np.array(ex['input']), np.array(ex['output']))
                   for ex in task['train']]

    # Solve each test input
    correct = 0
    total = 0

    for test_idx, test_ex in enumerate(task['test']):
        test_input = np.array(test_ex['input'])

        # Get expected output
        if task_id in solutions and test_idx < len(solutions[task_id]):
            expected_output = solutions[task_id][test_idx]
        else:
            expected_output = None

        if expected_output is not None:
            expected = np.array(expected_output)

            # Try solvers in order of confidence
            # 1. Pattern solver (tiling, reflection)
            predicted = pattern_solver.solve(train_pairs, test_input)
            solver_used = "Pattern"

            # 2. If pattern solver failed, try rule solver
            if predicted is None:
                predicted = rule_solver.solve(train_pairs, test_input)
                solver_used = "Rule" if predicted is not None else None

            if predicted is not None and np.array_equal(predicted, expected):
                correct += 1
                print(f"  ✅ Test {test_idx}: SOLVED (using {solver_used} solver)")
            else:
                print(f"  ❌ Test {test_idx}: Wrong/No solution")

            total += 1

    accuracy = correct / total if total > 0 else 0
    results.append({
        'task_id': task_id,
        'solved': accuracy > 0,
        'accuracy': accuracy
    })

    if accuracy > 0:
        print(f"  Task accuracy: {correct}/{total} = {100*accuracy:.0f}%")

print("\n" + "="*80)
print("BOLT-ON #2 + #3 COMBINED RESULTS")
print("="*80)

solved = sum(1 for r in results if r['solved'])
total_tasks = len(results)

print(f"Tasks solved: {solved}/{total_tasks} = {100*solved/total_tasks:.0f}%")
print(f"Avg accuracy: {100*np.mean([r['accuracy'] for r in results]):.1f}%")

improvement_over_baseline = solved - 0
improvement_over_bolton2 = solved - 1

print(f"\nImprovement over baseline: +{improvement_over_baseline} tasks")
print(f"Improvement over BOLT-ON #2 alone: +{improvement_over_bolton2} tasks")

if solved >= 2:
    print("\n✅ BOLT-ON #3 VALIDATED")
    print("   Spatial rule solver ADDS VALUE")
    print("   → KEEP both BOLT-ON #2 and #3")
    print("   → Proceed to test BOLT-ON #4")
elif solved == 1:
    print("\n⚠️  BOLT-ON #3 NO IMPROVEMENT")
    print("   Same performance as BOLT-ON #2 alone")
    print("   → Try BOLT-ON #4 (may need #3 later)")
else:
    print("\n❌ REGRESSION!")
    print("   Performance worse than BOLT-ON #2 alone")
    print("   → Debug BOLT-ON #3 or skip it")
