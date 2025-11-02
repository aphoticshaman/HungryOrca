#!/usr/bin/env python3
"""
TEST BOLT-ON #1 + #2 COMBINED: Does Multi-Solver Improve Over Baseline?

Comparison:
- Baseline (0%): NSPSA with pixel-level primitives
- BOLT-ON #1 (0%): Multi-scale object detection
- BOLT-ON #2: Pattern transformations (tiling, color mapping)
- COMBINED: BOLT-ON #1 + BOLT-ON #2 working together

Success criteria: Solve ≥ 1 task (any improvement over 0%)
"""

import json
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'SubtleGenius' / 'arc_solvers'))

from multiscale_object_solver import MultiScaleSolver
from pattern_transformation_solver import PatternTransformationSolver

# Load first 10 ARC tasks (same as baseline test)
with open('arc-agi_training_challenges.json', 'r') as f:
    challenges = json.load(f)

with open('arc-agi_training_solutions.json', 'r') as f:
    solutions = json.load(f)

task_ids = list(challenges.keys())[:10]

print("="*80)
print("BOLT-ON #1 + #2 COMBINED TEST: Multi-Solver on 10 ARC Tasks")
print("="*80)
print(f"\nTesting {len(task_ids)} tasks")
print("Baseline performance: 0/10 (0%)")
print("BOLT-ON #1 alone: 0/10 (0%)")
print("Goal: Solve ≥ 1 task\n")

object_solver = MultiScaleSolver()
pattern_solver = PatternTransformationSolver()

results = []
for i, task_id in enumerate(task_ids):
    print(f"[{i+1}/10] Task {task_id}")

    task = challenges[task_id]
    solution = solutions.get(task_id, {})

    train_pairs = [(np.array(ex['input']), np.array(ex['output']))
                   for ex in task['train']]

    # Solve each test input
    correct = 0
    total = 0

    for test_idx, test_ex in enumerate(task['test']):
        test_input = np.array(test_ex['input'])

        # Get expected output from solution
        if task_id in solutions and test_idx < len(solutions[task_id]):
            expected_output = solutions[task_id][test_idx]  # Already the grid!
        else:
            expected_output = None

        if expected_output is not None:
            expected = np.array(expected_output)

            # Try BOLT-ON #2 first (pattern transformations)
            predicted = pattern_solver.solve(train_pairs, test_input)

            # If pattern solver failed, try BOLT-ON #1 (object detection)
            if predicted is None:
                predicted = object_solver.solve(train_pairs, test_input)

            solver_used = None
            if predicted is not None:
                # Check which solver was used
                pattern_result = pattern_solver.solve(train_pairs, test_input)
                if pattern_result is not None and np.array_equal(predicted, pattern_result):
                    solver_used = "Pattern"
                else:
                    solver_used = "Object"

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
print("BOLT-ON #1 + #2 COMBINED RESULTS")
print("="*80)

solved = sum(1 for r in results if r['solved'])
total_tasks = len(results)

print(f"Tasks solved: {solved}/{total_tasks} = {100*solved/total_tasks:.0f}%")
print(f"Avg accuracy: {100*np.mean([r['accuracy'] for r in results]):.1f}%")

improvement = solved - 0  # Baseline was 0
print(f"\nImprovement over baseline: +{improvement} tasks")

if solved >= 1:
    print("\n✅ BOLT-ON #1 + #2 VALIDATED")
    print("   Combined multi-solver IMPROVES over baseline")
    print("   → KEEP both components")
    print("   → Proceed to BOLT-ON #3")
else:
    print("\n⚠️  COMBINED SYSTEM NEEDS MORE COMPONENTS")
    print("   No improvement yet with #1 + #2")
    print("   → Add BOLT-ON #3 (more transformation types)")
    print("   → May need specialized solvers for these task types")
