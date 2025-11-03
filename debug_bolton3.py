#!/usr/bin/env python3
"""Debug why BOLT-ON #3 isn't solving task 009d5c81."""

import json
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'SubtleGenius' / 'arc_solvers'))

from spatial_rule_solver import SpatialRuleSolver

# Load task 009d5c81
with open('arc-agi_training_challenges.json', 'r') as f:
    challenges = json.load(f)

with open('arc-agi_training_solutions.json', 'r') as f:
    solutions = json.load(f)

task_id = '009d5c81'
task = challenges[task_id]

train_pairs = [(np.array(ex['input']), np.array(ex['output']))
               for ex in task['train']]

solver = SpatialRuleSolver()

# Learn rules
rules = solver._learn_rules(train_pairs)

print(f'Task {task_id}')
print(f'Training examples: {len(train_pairs)}')
print(f'Rules learned: {len(rules)}')
for rule in rules:
    print(f'  - {rule.type}: {rule.target_action} (confidence: {rule.confidence})')

# Analyze training examples manually
print('\n=== Manual Analysis ===')
for i, (inp, out) in enumerate(train_pairs[:2]):  # First 2 examples
    print(f'\nExample {i}:')
    print(f'  Input colors: {set(np.unique(inp))}')
    print(f'  Output colors: {set(np.unique(out))}')

    # Check color mapping
    color_map = {}
    for r in range(inp.shape[0]):
        for c in range(inp.shape[1]):
            in_c = int(inp[r, c])
            out_c = int(out[r, c])
            if in_c in color_map:
                if color_map[in_c] != out_c:
                    print(f'  Inconsistent mapping: {in_c} → {color_map[in_c]} AND {out_c}')
            else:
                color_map[in_c] = out_c

    print(f'  Color mapping: {color_map}')

# Try solving test input
test_input = np.array(task['test'][0]['input'])
predicted = solver.solve(train_pairs, test_input)

if predicted is not None:
    print('\nPrediction made!')
    expected = np.array(solutions[task_id][0])
    if np.array_equal(predicted, expected):
        print('✅ CORRECT!')
    else:
        print('❌ Wrong prediction')
else:
    print('\n❌ No prediction made (no rules learned)')
