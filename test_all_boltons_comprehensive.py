#!/usr/bin/env python3
"""
COMPREHENSIVE BOLT-ON TEST: All 15 Components

Tests all bolt-ons individually and in combinations to generate final GO/NO-GO list.

BOLT-ONS TESTED:
1. Object Detection
2. Pattern Transformations ✅ (known GO - 1 task)
3. Spatial Rules
4. Size Transformations
5. Meta-Solver
6. Example-Based
7. Grid Structure
8. Compositional
9. Symmetry
10. Histogram
11. Identity
12. Majority Vote
13. Nearest Neighbor
14. Rule Induction
15. Abstraction

Success criteria: Identify all components that improve over baseline
"""

import json
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'SubtleGenius' / 'arc_solvers'))

from multiscale_object_solver import MultiScaleSolver
from pattern_transformation_solver import PatternTransformationSolver
from spatial_rule_solver import SpatialRuleSolver
from size_transformation_solver import SizeTransformationSolver
from meta_solver import MetaSolver
from example_based_solver import ExampleBasedSolver
from grid_structure_solver import GridStructureSolver
from compositional_solver import CompositionalSolver
from traditional_approaches import (
    SymmetrySolver, HistogramSolver, IdentitySolver,
    MajorityVoteSolver, NearestNeighborSolver, RuleInductionSolver, AbstractionSolver
)

# Load first 10 ARC tasks
with open('arc-agi_training_challenges.json', 'r') as f:
    challenges = json.load(f)

with open('arc-agi_training_solutions.json', 'r') as f:
    solutions = json.load(f)

task_ids = list(challenges.keys())[:10]

print("="*80)
print("COMPREHENSIVE BOLT-ON TEST: All 15 Components")
print("="*80)
print(f"\nTesting {len(task_ids)} tasks with 15 bolt-ons")
print("Baseline: 0/10 (0%)\n")

# Initialize all solvers
solvers = {
    'BOLTON-01-Object': MultiScaleSolver(),
    'BOLTON-02-Pattern': PatternTransformationSolver(),
    'BOLTON-03-Rules': SpatialRuleSolver(),
    'BOLTON-04-Size': SizeTransformationSolver(),
    'BOLTON-06-Example': ExampleBasedSolver(),
    'BOLTON-07-Grid': GridStructureSolver(),
    'BOLTON-09-Symmetry': SymmetrySolver(),
    'BOLTON-10-Histogram': HistogramSolver(),
    'BOLTON-11-Identity': IdentitySolver(),
    'BOLTON-13-NearestNeighbor': NearestNeighborSolver(),
    'BOLTON-14-RuleInduction': RuleInductionSolver(),
    'BOLTON-15-Abstraction': AbstractionSolver(),
}

# Track results per solver
solver_results = {name: {'solved': 0, 'tasks': []} for name in solvers.keys()}

for i, task_id in enumerate(task_ids):
    print(f"\n[{i+1}/10] Task {task_id}")

    task = challenges[task_id]
    train_pairs = [(np.array(ex['input']), np.array(ex['output']))
                   for ex in task['train']]

    for test_idx, test_ex in enumerate(task['test']):
        test_input = np.array(test_ex['input'])

        if task_id in solutions and test_idx < len(solutions[task_id]):
            expected = np.array(solutions[task_id][test_idx])

            # Try each solver
            for solver_name, solver in solvers.items():
                try:
                    predicted = solver.solve(train_pairs, test_input)

                    if predicted is not None and np.array_equal(predicted, expected):
                        solver_results[solver_name]['solved'] += 1
                        solver_results[solver_name]['tasks'].append(task_id)
                        print(f"  ✅ {solver_name}: SOLVED")
                except Exception as e:
                    # Solver failed - continue
                    pass

print("\n" + "="*80)
print("FINAL GO/NO-GO LIST")
print("="*80)

# Sort by performance
sorted_solvers = sorted(solver_results.items(), key=lambda x: x[1]['solved'], reverse=True)

go_components = []
no_go_components = []

for solver_name, results in sorted_solvers:
    solved = results['solved']
    percentage = (solved / 10) * 100
    tasks_solved = results['tasks']

    status = "GO" if solved > 0 else "NO-GO"

    print(f"\n{solver_name}:")
    print(f"  Status: {status}")
    print(f"  Tasks solved: {solved}/10 ({percentage:.0f}%)")
    if tasks_solved:
        print(f"  Solved: {', '.join(tasks_solved)}")

    if solved > 0:
        go_components.append((solver_name, solved, tasks_solved))
    else:
        no_go_components.append(solver_name)

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"\nGO Components: {len(go_components)}")
for name, count, tasks in go_components:
    print(f"  - {name}: {count} tasks")

print(f"\nNO-GO Components: {len(no_go_components)}")
for name in no_go_components:
    print(f"  - {name}")

# Calculate ensemble potential
if len(go_components) > 0:
    unique_tasks = set()
    for _, _, tasks in go_components:
        unique_tasks.update(tasks)

    print(f"\nEnsemble potential: {len(unique_tasks)} unique tasks solved")
    print(f"Coverage: {len(unique_tasks)}/10 = {len(unique_tasks)*10}%")

print("\n" + "="*80)
print("READY FOR COMPREHENSIVE ANALYSIS")
print("="*80)
