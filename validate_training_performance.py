#!/usr/bin/env python3
"""Validate submission against training set to find patterns."""

import json
import numpy as np

# Load training data
with open('arc-agi_training_challenges.json') as f:
    train_challenges = json.load(f)

with open('arc-agi_training_solutions.json') as f:
    train_solutions = json.load(f)

print("="*80)
print("TRAINING SET PERFORMANCE ANALYSIS")
print("="*80)
print(f"Training tasks: {len(train_challenges)}")

# We need to generate predictions for training tasks
# For now, let's just analyze what task types exist

task_characteristics = []

for task_id, task in train_challenges.items():
    if task_id not in train_solutions:
        continue
    
    # Analyze training examples
    train_pairs = task['train']
    test_pairs = task['test']
    solutions = train_solutions[task_id]
    
    # Get dimensions
    input_shapes = [np.array(pair['input']).shape for pair in train_pairs]
    output_shapes = [np.array(pair['output']).shape for pair in train_pairs]
    
    # Check if shape-preserving
    shape_preserving = all(
        i == o for i, o in zip(input_shapes, output_shapes)
    )
    
    # Count colors
    all_colors = set()
    for pair in train_pairs:
        all_colors.update(np.unique(np.array(pair['input'])))
        all_colors.update(np.unique(np.array(pair['output'])))
    
    task_characteristics.append({
        'task_id': task_id,
        'num_train': len(train_pairs),
        'num_test': len(test_pairs),
        'shape_preserving': shape_preserving,
        'num_colors': len(all_colors),
        'input_shapes': input_shapes,
        'output_shapes': output_shapes,
    })

# Categorize tasks
shape_preserving_tasks = [t for t in task_characteristics if t['shape_preserving']]
simple_tasks = [t for t in task_characteristics if t['num_colors'] <= 4]

print(f"\n📊 Task Categories:")
print(f"   Shape-preserving: {len(shape_preserving_tasks)} ({len(shape_preserving_tasks)/len(task_characteristics)*100:.1f}%)")
print(f"   Simple (≤4 colors): {len(simple_tasks)} ({len(simple_tasks)/len(task_characteristics)*100:.1f}%)")

# Show a few example tasks for compositional testing
print(f"\n🎯 Sample Tasks for Compositional Testing:")
for i, char in enumerate(task_characteristics[:5]):
    print(f"\n{i+1}. Task {char['task_id']}:")
    print(f"   Training examples: {char['num_train']}")
    print(f"   Shape-preserving: {char['shape_preserving']}")
    print(f"   Colors: {char['num_colors']}")
    print(f"   Input shapes: {char['input_shapes'][:2]}")
    print(f"   Output shapes: {char['output_shapes'][:2]}")

print(f"\n💡 Next Step: Run REAL_TIME_BUDGET_SOLVER.py on these tasks!")
print(f"   Expected: Compositional chaining should improve shape-preserving tasks")
