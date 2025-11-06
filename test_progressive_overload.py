#!/usr/bin/env python3
"""
🧪 Test Progressive Overload Strategy on Diverse ARC Puzzles

Tests across multiple dimensions:
- Grid size: small (<10x10), medium (10-20), large (>20)
- Color complexity: simple (2-3 colors), medium (4-6), complex (7+)
- Primitives: shapes, objects, patterns
- Transforms: rotation, scaling, tiling, pattern_completion, etc.

Author: Ryan Cardwell & Claude
Date: November 2025
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict
import time

from lucidorca_quantum import LucidOrcaQuantum


def analyze_task_characteristics(task):
    """Analyze task characteristics for classification"""

    # Get all grids (train inputs/outputs + test inputs)
    grids = []
    for example in task.get('train', []):
        grids.append(np.array(example['input']))
        grids.append(np.array(example['output']))
    for example in task.get('test', []):
        grids.append(np.array(example['input']))

    # Grid size (max dimensions)
    max_h = max(g.shape[0] for g in grids)
    max_w = max(g.shape[1] for g in grids)
    max_dim = max(max_h, max_w)

    if max_dim < 10:
        grid_size = 'small'
    elif max_dim < 20:
        grid_size = 'medium'
    else:
        grid_size = 'large'

    # Color complexity (unique colors across all grids)
    all_colors = set()
    for g in grids:
        all_colors.update(np.unique(g).tolist())

    num_colors = len(all_colors)
    if num_colors <= 3:
        color_complexity = 'simple'
    elif num_colors <= 6:
        color_complexity = 'medium'
    else:
        color_complexity = 'complex'

    # Primitive detection (rough heuristics)
    # Check for rectangular objects, patterns, etc.
    has_objects = False
    has_patterns = False

    for g in grids:
        # Objects: distinct connected components
        if len(np.unique(g)) > 2:  # More than background
            has_objects = True

        # Patterns: repeating structures
        if g.shape[0] >= 3 and g.shape[1] >= 3:
            # Check for repeating rows/cols
            if len(set([tuple(row) for row in g])) < g.shape[0] * 0.7:
                has_patterns = True

    if has_objects and has_patterns:
        primitive = 'complex'
    elif has_objects:
        primitive = 'objects'
    elif has_patterns:
        primitive = 'patterns'
    else:
        primitive = 'shapes'

    return {
        'grid_size': grid_size,
        'color_complexity': color_complexity,
        'primitive': primitive,
        'max_dim': max_dim,
        'num_colors': num_colors
    }


def compare_grids(pred, truth):
    """Compare predicted grid to ground truth"""
    try:
        pred_arr = np.array(pred)
        truth_arr = np.array(truth)

        if pred_arr.shape != truth_arr.shape:
            return False

        return np.array_equal(pred_arr, truth_arr)
    except:
        return False


def evaluate_solutions(solutions, ground_truth, test_tasks):
    """
    Evaluate solutions against ground truth

    Returns:
        - Overall accuracy
        - Breakdown by characteristics
        - Per-task results
    """

    results = {
        'correct': 0,
        'total': 0,
        'by_grid_size': defaultdict(lambda: {'correct': 0, 'total': 0}),
        'by_color': defaultdict(lambda: {'correct': 0, 'total': 0}),
        'by_primitive': defaultdict(lambda: {'correct': 0, 'total': 0}),
        'tasks': {}
    }

    for task_id in solutions:
        if task_id not in ground_truth:
            continue

        # Get characteristics
        chars = analyze_task_characteristics(test_tasks[task_id])

        # Check each test output
        pred = solutions[task_id]
        truth = ground_truth[task_id]

        task_correct = False

        for i, truth_output in enumerate(truth):
            if i >= len(pred):
                break

            pred_attempts = pred[i]

            # Check if EITHER attempt matches (that's how ARC scoring works)
            attempt1_match = compare_grids(pred_attempts['attempt_1'], truth_output)
            attempt2_match = compare_grids(pred_attempts['attempt_2'], truth_output)

            if attempt1_match or attempt2_match:
                task_correct = True
                break

        # Record results
        results['total'] += 1
        if task_correct:
            results['correct'] += 1
            results['by_grid_size'][chars['grid_size']]['correct'] += 1
            results['by_color'][chars['color_complexity']]['correct'] += 1
            results['by_primitive'][chars['primitive']]['correct'] += 1

        results['by_grid_size'][chars['grid_size']]['total'] += 1
        results['by_color'][chars['color_complexity']]['total'] += 1
        results['by_primitive'][chars['primitive']]['total'] += 1

        results['tasks'][task_id] = {
            'correct': task_correct,
            'characteristics': chars
        }

    return results


def main():
    """Run comprehensive progressive overload test"""

    print("\n" + "="*70)
    print("🧪 PROGRESSIVE OVERLOAD COMPREHENSIVE TEST")
    print("="*70)
    print("Testing across:")
    print("  - Grid sizes: small/medium/large")
    print("  - Color complexity: simple/medium/complex")
    print("  - Primitives: shapes/objects/patterns/complex")
    print("  - Time budget: 45 minutes (2700s)")
    print("="*70)

    # Load evaluation data (has ground truth)
    data_dir = Path(".")

    eval_challenges_path = data_dir / "arc-agi_evaluation_challenges.json"
    eval_solutions_path = data_dir / "arc-agi_evaluation_solutions.json"

    if not eval_challenges_path.exists():
        print(f"❌ Not found: {eval_challenges_path}")
        print("   Please ensure ARC evaluation data is in current directory")
        return

    print(f"\n📂 Loading evaluation data...")
    with open(eval_challenges_path, 'r') as f:
        eval_tasks = json.load(f)

    with open(eval_solutions_path, 'r') as f:
        ground_truth = json.load(f)

    print(f"   Tasks: {len(eval_tasks)}")

    # Analyze characteristics distribution
    print(f"\n📊 Analyzing task characteristics...")
    char_stats = {
        'grid_size': defaultdict(int),
        'color': defaultdict(int),
        'primitive': defaultdict(int)
    }

    for task_id, task in eval_tasks.items():
        chars = analyze_task_characteristics(task)
        char_stats['grid_size'][chars['grid_size']] += 1
        char_stats['color'][chars['color_complexity']] += 1
        char_stats['primitive'][chars['primitive']] += 1

    print(f"\n  Grid Sizes:")
    for size, count in sorted(char_stats['grid_size'].items()):
        print(f"    {size:8s}: {count:3d} ({count/len(eval_tasks)*100:.1f}%)")

    print(f"\n  Color Complexity:")
    for comp, count in sorted(char_stats['color'].items()):
        print(f"    {comp:8s}: {count:3d} ({count/len(eval_tasks)*100:.1f}%)")

    print(f"\n  Primitives:")
    for prim, count in sorted(char_stats['primitive'].items()):
        print(f"    {prim:8s}: {count:3d} ({count/len(eval_tasks)*100:.1f}%)")

    # Initialize solver
    print(f"\n🌊⚛️ Initializing LucidOrca Quantum...")
    solver = LucidOrcaQuantum()

    # Run progressive overload (45 minutes)
    print(f"\n🏋️ Running 45-minute progressive overload test...")
    solutions = solver.solve_test_set(eval_tasks, time_budget=2700)  # 45 min

    # Evaluate against ground truth
    print(f"\n📊 Evaluating solutions...")
    results = evaluate_solutions(solutions, ground_truth, eval_tasks)

    # Print results
    print("\n" + "="*70)
    print("🏆 PROGRESSIVE OVERLOAD TEST RESULTS")
    print("="*70)

    accuracy = results['correct'] / results['total'] * 100 if results['total'] > 0 else 0
    print(f"\n  Overall Accuracy: {results['correct']}/{results['total']} = {accuracy:.1f}%")

    print(f"\n  By Grid Size:")
    for size in ['small', 'medium', 'large']:
        stats = results['by_grid_size'][size]
        if stats['total'] > 0:
            acc = stats['correct'] / stats['total'] * 100
            print(f"    {size.capitalize():8s}: {stats['correct']:3d}/{stats['total']:3d} = {acc:5.1f}%")

    print(f"\n  By Color Complexity:")
    for comp in ['simple', 'medium', 'complex']:
        stats = results['by_color'][comp]
        if stats['total'] > 0:
            acc = stats['correct'] / stats['total'] * 100
            print(f"    {comp.capitalize():8s}: {stats['correct']:3d}/{stats['total']:3d} = {acc:5.1f}%")

    print(f"\n  By Primitive Type:")
    for prim in ['shapes', 'objects', 'patterns', 'complex']:
        stats = results['by_primitive'][prim]
        if stats['total'] > 0:
            acc = stats['correct'] / stats['total'] * 100
            print(f"    {prim.capitalize():8s}: {stats['correct']:3d}/{stats['total']:3d} = {acc:5.1f}%")

    print("\n" + "="*70)

    # Target analysis
    print(f"\n🎯 TARGET ANALYSIS:")
    print(f"   Current:  {accuracy:.1f}%")
    print(f"   Target:   85.0% (Grand Prize)")
    print(f"   Gap:      {85.0 - accuracy:+.1f}%")

    if accuracy >= 85.0:
        print(f"\n   ✅ GRAND PRIZE TARGET ACHIEVED!")
    elif accuracy >= 70.0:
        print(f"\n   🥈 Close! Need {85.0 - accuracy:.1f}% more")
    else:
        print(f"\n   📈 Need significant improvement")

    print("\n" + "="*70)

    # Save detailed results
    output_path = Path("test_results.json")
    with open(output_path, 'w') as f:
        json.dump({
            'overall': {
                'correct': results['correct'],
                'total': results['total'],
                'accuracy': accuracy
            },
            'by_characteristic': {
                'grid_size': dict(results['by_grid_size']),
                'color': dict(results['by_color']),
                'primitive': dict(results['by_primitive'])
            },
            'per_task': results['tasks']
        }, f, indent=2)

    print(f"\n💾 Detailed results saved: {output_path}")
    print("="*70)


if __name__ == "__main__":
    main()
