#!/usr/bin/env python3
"""
Quick check: What's our PARTIAL match rate?

The other branch got 60% partial match. Let's see our baseline.
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


def partial_match_score(predicted, expected):
    """Compute partial match (0-1)."""
    if predicted is None or expected is None:
        return 0.0

    if predicted.shape != expected.shape:
        return 0.0

    matches = np.sum(predicted == expected)
    total = predicted.size
    return matches / total if total > 0 else 0.0


print("="*60)
print("BASELINE TRANSFORMATION LIBRARY - PARTIAL MATCH TEST")
print("="*60)

solver = FuzzyTransformationSolver(num_rules=0)  # No rules

exact_matches = 0
partial_scores = []

for task_id in task_ids:
    task = challenges[task_id]
    train_pairs = [(np.array(ex['input']), np.array(ex['output']))
                  for ex in task['train']]

    test_input = np.array(task['test'][0]['input'])
    expected = np.array(solutions[task_id][0]) if task_id in solutions else None

    try:
        predicted = solver.solve(train_pairs, test_input)

        if predicted is not None and expected is not None:
            partial_score = partial_match_score(predicted, expected)
            partial_scores.append(partial_score)

            if np.array_equal(predicted, expected):
                exact_matches += 1
                print(f"✓ {task_id}: EXACT (100%)")
            elif partial_score > 0.5:
                print(f"≈ {task_id}: {partial_score*100:.1f}% match")
            else:
                print(f"✗ {task_id}: {partial_score*100:.1f}% match")
        else:
            print(f"✗ {task_id}: No prediction")
            partial_scores.append(0.0)
    except Exception as e:
        print(f"✗ {task_id}: Error - {str(e)[:50]}")
        partial_scores.append(0.0)

print("\n" + "="*60)
print(f"Exact matches: {exact_matches}/10 ({exact_matches*10:.1f}%)")
print(f"Avg partial match: {np.mean(partial_scores)*100:.1f}%")
print(f"High similarity (>50%): {sum(1 for s in partial_scores if s > 0.5)}/10")
print("="*60)
