#!/usr/bin/env python3
"""
TEST BOLT-ON #5: Meta-Solver with Multiple Base Solvers

Components:
- BOLT-ON #2: Pattern transformations (GO - 1 task solved)
- BOLT-ON #5: Meta-solver orchestration (NEW)

Also includes deferred solvers in meta-solver pipeline:
- BOLT-ON #1: Object detection (NO-GO alone, but may help in ensemble)
- BOLT-ON #3: Spatial rules (NO-GO alone, but may help in ensemble)
- BOLT-ON #4: Size transforms (NO-GO alone, but may help in ensemble)

Hypothesis: Meta-solver can find synergies between NO-GO components.

Success criteria: Solve ≥ 2 tasks
"""

import json
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'SubtleGenius' / 'arc_solvers'))

from pattern_transformation_solver import PatternTransformationSolver
from multiscale_object_solver import MultiScaleSolver
from spatial_rule_solver import SpatialRuleSolver
from size_transformation_solver import SizeTransformationSolver
from meta_solver import MetaSolver

# Load first 10 ARC tasks
with open('arc-agi_training_challenges.json', 'r') as f:
    challenges = json.load(f)

with open('arc-agi_training_solutions.json', 'r') as f:
    solutions = json.load(f)

task_ids = list(challenges.keys())[:10]

print("="*80)
print("BOLT-ON #5 TEST: Meta-Solver with All Components")
print("="*80)
print(f"\nTesting {len(task_ids)} tasks")
print("Previous best (BOLT-ON #2): 1/10 (10%)")
print("Goal: Solve ≥ 2 tasks\n")

# Initialize all solvers
pattern_solver = PatternTransformationSolver()
object_solver = MultiScaleSolver()
rule_solver = SpatialRuleSolver()
size_solver = SizeTransformationSolver()

# Create meta-solver with all base solvers
# Order by expected success rate (pattern solver first, since it's proven)
meta_solver = MetaSolver(base_solvers=[
    pattern_solver,
    size_solver,
    rule_solver,
    object_solver,
])

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

            # Use meta-solver
            predicted = meta_solver.solve(train_pairs, test_input)

            if predicted is not None and np.array_equal(predicted, expected):
                correct += 1
                print(f"  ✅ Test {test_idx}: SOLVED (meta-solver)")
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
print("BOLT-ON #5 META-SOLVER RESULTS")
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
    print("\n✅ BOLT-ON #5 VALIDATED")
    print("   Meta-solver orchestration ADDS VALUE")
    print("   → KEEP meta-solver approach")
    print("   → Some NO-GO components may have synergistic value")
elif solved == 1:
    print("\n⚠️  BOLT-ON #5 NO IMPROVEMENT")
    print("   Same performance as BOLT-ON #2 alone")
    print("   → Meta-solver doesn't find synergies yet")
    print("   → Need deep dive on why NO-GO components fail")
else:
    print("\n❌ REGRESSION!")
    print("   Performance worse than BOLT-ON #2 alone")
    print("   → Meta-solver has bugs or heuristics too restrictive")

print("\n" + "="*80)
print("CURRENT GO/NO-GO STATUS")
print("="*80)
print("GO:")
print("  - BOLT-ON #2: Pattern Transformations (1 task)")
print("\nNO-GO (pending deep dive):")
print("  - BOLT-ON #1: Object Detection")
print("  - BOLT-ON #3: Spatial Rules")
print("  - BOLT-ON #4: Size Transformations")
print("  - BOLT-ON #5: Meta-Solver (if no improvement)")
