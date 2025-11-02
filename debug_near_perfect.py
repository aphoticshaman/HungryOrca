#!/usr/bin/env python3
"""
Debug: Why are we getting 90%+ but not 100%?

Check the specific differences on high-scoring tasks.
"""

import json
import numpy as np
from collaborative_multi_agent_solver import CollaborativeSolver

with open('arc-agi_training_challenges.json', 'r') as f:
    challenges = json.load(f)

with open('arc-agi_training_solutions.json', 'r') as f:
    solutions = json.load(f)

# Focus on high-scoring tasks
high_score_tasks = ['00d62c1b', '045e512c', '025d127b', '009d5c81']

solver = CollaborativeSolver(verbose=True)

for task_id in high_score_tasks:
    print("\n" + "="*80)
    print(f"TASK: {task_id}")
    print("="*80)

    task = challenges[task_id]
    train_pairs = [(np.array(ex['input']), np.array(ex['output']))
                  for ex in task['train']]

    test_input = np.array(task['test'][0]['input'])
    expected = np.array(solutions[task_id][0]) if task_id in solutions else None

    predicted = solver.solve(train_pairs, test_input)

    if predicted is not None and expected is not None and predicted.shape == expected.shape:
        score = np.sum(predicted == expected) / predicted.size

        print(f"\nOverall match: {score*100:.1f}%")

        # Find differences
        diff_mask = predicted != expected
        num_diff = np.sum(diff_mask)

        print(f"Differing cells: {num_diff}/{predicted.size}")

        if num_diff > 0 and num_diff < 20:
            print("\nDifferences:")
            for i in range(predicted.shape[0]):
                for j in range(predicted.shape[1]):
                    if diff_mask[i, j]:
                        print(f"  [{i},{j}]: predicted={predicted[i,j]}, expected={expected[i,j]}")

        # Show grids for first task
        if task_id == high_score_tasks[0]:
            print("\nTest Input:")
            print(test_input)
            print("\nPredicted:")
            print(predicted)
            print("\nExpected:")
            print(expected)

    print("\n")
