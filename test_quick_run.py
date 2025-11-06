#!/usr/bin/env python3
"""
🏃 Quick Progressive Overload Test - Sub 2-minute runs

Fast validation test with 6 decimal precision metrics
Tests 20 tasks with 2-minute timeout each to validate fixes

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
    grids = []
    for example in task.get('train', []):
        grids.append(np.array(example['input']))
        grids.append(np.array(example['output']))
    for example in task.get('test', []):
        grids.append(np.array(example['input']))

    max_h = max(g.shape[0] for g in grids)
    max_w = max(g.shape[1] for g in grids)
    max_dim = max(max_h, max_w)

    if max_dim < 10:
        grid_size = 'small'
    elif max_dim < 20:
        grid_size = 'medium'
    else:
        grid_size = 'large'

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

    return {
        'grid_size': grid_size,
        'color_complexity': color_complexity,
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


def main():
    """Run quick validation test - 20 tasks x 2min = 40min total"""

    print("\n" + "="*70)
    print("🏃 QUICK PROGRESSIVE OVERLOAD TEST (Sub 2-Minute Runs)")
    print("="*70)
    print("Purpose: Validate fixes and tune for fast runs")
    print("Config: 20 tasks x 120s = 40 minutes total")
    print("Precision: 6 decimal places for fine-tuning")
    print("="*70)

    # Load evaluation data
    data_dir = Path(".")
    eval_challenges_path = data_dir / "arc-agi_evaluation_challenges.json"
    eval_solutions_path = data_dir / "arc-agi_evaluation_solutions.json"

    if not eval_challenges_path.exists():
        print(f"❌ Not found: {eval_challenges_path}")
        return

    print(f"\n📂 Loading evaluation data...")
    with open(eval_challenges_path, 'r') as f:
        all_tasks = json.load(f)

    with open(eval_solutions_path, 'r') as f:
        all_solutions = json.load(f)

    # Select first 20 tasks for quick test
    test_tasks = dict(list(all_tasks.items())[:20])
    print(f"   Selected: {len(test_tasks)} tasks for quick validation")

    # Analyze distribution
    char_stats = defaultdict(int)
    for task_id, task in test_tasks.items():
        chars = analyze_task_characteristics(task)
        char_stats[f"grid_{chars['grid_size']}"] += 1
        char_stats[f"color_{chars['color_complexity']}"] += 1

    print(f"\n  Quick Test Distribution:")
    for key, count in sorted(char_stats.items()):
        print(f"    {key:20s}: {count:3d} ({count/len(test_tasks)*100:.6f}%)")

    # Initialize solver
    print(f"\n🌊⚛️ Initializing LucidOrca Quantum...")
    solver = LucidOrcaQuantum()

    # Run progressive overload (120s per task = 2min)
    print(f"\n🏋️ Running test (120s per task)...")
    start_time = time.time()
    solutions = solver.solve_test_set(test_tasks, time_budget=120 * len(test_tasks))
    elapsed = time.time() - start_time

    # Evaluate
    print(f"\n📊 Evaluating solutions...")
    correct = 0
    total = 0

    results = {
        'by_grid_size': defaultdict(lambda: {'correct': 0, 'total': 0}),
        'by_color': defaultdict(lambda: {'correct': 0, 'total': 0}),
    }

    for task_id in solutions:
        if task_id not in all_solutions:
            continue

        chars = analyze_task_characteristics(test_tasks[task_id])
        pred = solutions[task_id]
        truth = all_solutions[task_id]

        task_correct = False

        for i, truth_output in enumerate(truth):
            if i >= len(pred):
                break

            pred_attempts = pred[i]

            # Check if EITHER attempt matches
            attempt1_match = compare_grids(pred_attempts['attempt_1'], truth_output)
            attempt2_match = compare_grids(pred_attempts['attempt_2'], truth_output)

            if attempt1_match or attempt2_match:
                task_correct = True
                break

        total += 1
        if task_correct:
            correct += 1
            results['by_grid_size'][chars['grid_size']]['correct'] += 1
            results['by_color'][chars['color_complexity']]['correct'] += 1

        results['by_grid_size'][chars['grid_size']]['total'] += 1
        results['by_color'][chars['color_complexity']]['total'] += 1

    # Print results with 6 decimal precision
    print("\n" + "="*70)
    print("🏆 QUICK TEST RESULTS (6 DECIMAL PRECISION)")
    print("="*70)

    accuracy = (correct / total * 100) if total > 0 else 0
    print(f"\n  Overall: {correct}/{total} = {accuracy:.6f}%")

    print(f"\n  By Grid Size:")
    for size in ['small', 'medium', 'large']:
        stats = results['by_grid_size'][size]
        if stats['total'] > 0:
            acc = stats['correct'] / stats['total'] * 100
            print(f"    {size.capitalize():8s}: {stats['correct']:3d}/{stats['total']:3d} = {acc:8.6f}%")

    print(f"\n  By Color Complexity:")
    for comp in ['simple', 'medium', 'complex']:
        stats = results['by_color'][comp]
        if stats['total'] > 0:
            acc = stats['correct'] / stats['total'] * 100
            print(f"    {comp.capitalize():8s}: {stats['correct']:3d}/{stats['total']:3d} = {acc:8.6f}%")

    print(f"\n  Performance:")
    print(f"    Total time:    {elapsed:.6f}s ({elapsed/60:.6f} min)")
    print(f"    Avg per task:  {elapsed/total:.6f}s")
    print(f"    Tasks/minute:  {total/(elapsed/60):.6f}")

    print("\n" + "="*70)
    print(f"🎯 TARGET ANALYSIS:")
    print(f"   Current:  {accuracy:.6f}%")
    print(f"   Target:   85.000000%")
    print(f"   Gap:      {85.0 - accuracy:+.6f}%")

    if accuracy >= 85.0:
        print(f"\n   ✅ GRAND PRIZE TARGET ACHIEVED!")
    elif accuracy >= 50.0:
        print(f"\n   🥈 Good progress! Need {85.0 - accuracy:.6f}% more")
    elif accuracy >= 10.0:
        print(f"\n   📈 Solver is working! Need optimization")
    else:
        print(f"\n   ⚠️  Need significant improvement")

    print("="*70)


if __name__ == "__main__":
    main()
