#!/usr/bin/env python3
"""Examine ARC tasks to understand transformation patterns"""

import json

# Load data
with open('arc-agi_training_challenges.json', 'r') as f:
    challenges = json.load(f)

with open('arc-agi_training_solutions.json', 'r') as f:
    solutions = json.load(f)

# Examine first 3 tasks
for i, (task_id, task) in enumerate(list(challenges.items())[:3]):
    print("="*60)
    print(f"Task #{i+1}: {task_id}")
    print("="*60)

    # Show first training example
    train_ex = task['train'][0]
    inp = train_ex['input']
    out = train_ex['output']

    print(f"\nTraining example 1:")
    print(f"Input  ({len(inp)}x{len(inp[0]) if inp else 0}):")
    for row in inp:
        print(f"  {row}")

    print(f"\nOutput ({len(out)}x{len(out[0]) if out else 0}):")
    for row in out:
        print(f"  {row}")

    # Analyze transformation
    print(f"\nAnalysis:")
    print(f"  - Size change: {len(inp)}x{len(inp[0]) if inp else 0} -> {len(out)}x{len(out[0]) if out else 0}")

    inp_colors = set()
    for row in inp:
        inp_colors.update(row)
    out_colors = set()
    for row in out:
        out_colors.update(row)

    print(f"  - Input colors: {sorted(inp_colors)}")
    print(f"  - Output colors: {sorted(out_colors)}")
    print(f"  - Total training examples: {len(task['train'])}")
    print()
