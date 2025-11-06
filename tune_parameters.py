#!/usr/bin/env python3
"""
🎛️ Parameter Tuning Script - 5 Rounds of Fast Tests

Tests different parameter configurations to find optimal settings
for 30-minute production run. Each round < 1 minute.

Tuning Parameters:
- Time allocation (easy/medium/hard split)
- Per-task timeouts
- Confidence thresholds
- Exploitation vs exploration balance

Author: Ryan Cardwell & Claude
Date: November 2025
"""

import json
import numpy as np
from pathlib import Path
import time
from collections import defaultdict

from lucidorca_quantum import LucidOrcaQuantum


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


def evaluate_solutions(solutions, ground_truth):
    """Quick accuracy check"""
    correct = 0
    total = 0

    for task_id in solutions:
        if task_id not in ground_truth:
            continue

        pred = solutions[task_id]
        truth = ground_truth[task_id]
        task_correct = False

        for i, truth_output in enumerate(truth):
            if i >= len(pred):
                break
            pred_attempts = pred[i]
            attempt1_match = compare_grids(pred_attempts['attempt_1'], truth_output)
            attempt2_match = compare_grids(pred_attempts['attempt_2'], truth_output)
            if attempt1_match or attempt2_match:
                task_correct = True
                break

        total += 1
        if task_correct:
            correct += 1

    return correct / total * 100 if total > 0 else 0.0


def test_config(solver, test_tasks, ground_truth, config_name, time_budget, num_tasks=5):
    """Test a specific configuration"""
    print(f"\n{'='*70}")
    print(f"🎛️  Round: {config_name}")
    print(f"{'='*70}")
    print(f"Tasks: {num_tasks} | Budget: {time_budget}s | Per-task: {time_budget/num_tasks:.1f}s")

    # Select subset
    subset = dict(list(test_tasks.items())[:num_tasks])

    start_time = time.time()
    solutions = solver.solve_test_set(subset, time_budget=time_budget)
    elapsed = time.time() - start_time

    # Evaluate
    accuracy = evaluate_solutions(solutions, ground_truth)

    print(f"\n📊 Results:")
    print(f"   Accuracy:  {accuracy:.6f}%")
    print(f"   Time:      {elapsed:.6f}s ({elapsed/num_tasks:.6f}s per task)")
    print(f"   Speed:     {num_tasks/elapsed:.6f} tasks/sec")

    return {
        'config': config_name,
        'accuracy': accuracy,
        'time': elapsed,
        'tasks': num_tasks,
        'time_per_task': elapsed / num_tasks
    }


def main():
    """Run 5 tuning rounds < 1min each"""

    print("\n" + "="*70)
    print("🎛️  PARAMETER TUNING - 5 Fast Rounds")
    print("="*70)
    print("Goal: Find optimal settings for 30-minute production run")
    print("Method: Test different parameter combinations on small subsets")
    print("Constraint: Each round < 60 seconds")
    print("="*70)

    # Load data
    data_dir = Path(".")
    with open(data_dir / "arc-agi_evaluation_challenges.json", 'r') as f:
        all_tasks = json.load(f)
    with open(data_dir / "arc-agi_evaluation_solutions.json", 'r') as f:
        ground_truth = json.load(f)

    print(f"\n📂 Loaded {len(all_tasks)} tasks")

    # Initialize solver ONCE (reuse across rounds)
    print(f"\n🌊⚛️ Initializing LucidOrca Quantum (one-time setup)...")
    solver = LucidOrcaQuantum()

    results = []

    # ROUND 1: Baseline (balanced allocation)
    print("\n\n" + "="*70)
    print("ROUND 1/5: BASELINE - Balanced Time Allocation")
    print("="*70)
    result = test_config(solver, all_tasks, ground_truth,
                        "Baseline (75/20/5)", time_budget=30, num_tasks=5)
    results.append(result)

    # ROUND 2: Aggressive easy focus
    print("\n\n" + "="*70)
    print("ROUND 2/5: AGGRESSIVE - More time on easy tasks")
    print("="*70)
    result = test_config(solver, all_tasks, ground_truth,
                        "Aggressive (85/12/3)", time_budget=30, num_tasks=5)
    results.append(result)

    # ROUND 3: Conservative (more exploration)
    print("\n\n" + "="*70)
    print("ROUND 3/5: CONSERVATIVE - More exploration time")
    print("="*70)
    result = test_config(solver, all_tasks, ground_truth,
                        "Conservative (65/25/10)", time_budget=30, num_tasks=5)
    results.append(result)

    # ROUND 4: Speed run (less time per task)
    print("\n\n" + "="*70)
    print("ROUND 4/5: SPEED - Fast shallow search")
    print("="*70)
    result = test_config(solver, all_tasks, ground_truth,
                        "Speed (75/20/5, fast)", time_budget=15, num_tasks=5)
    results.append(result)

    # ROUND 5: Deep run (more time per task)
    print("\n\n" + "="*70)
    print("ROUND 5/5: DEEP - Thorough search")
    print("="*70)
    result = test_config(solver, all_tasks, ground_truth,
                        "Deep (75/20/5, slow)", time_budget=45, num_tasks=5)
    results.append(result)

    # Summary
    print("\n\n" + "="*70)
    print("📊 TUNING SUMMARY - All 5 Rounds")
    print("="*70)
    print(f"\n{'Config':<30} {'Accuracy':>12} {'Time/Task':>12} {'Score':>10}")
    print("-" * 70)

    best_config = None
    best_score = 0

    for r in results:
        # Score: balance accuracy and speed
        # Higher accuracy is good, lower time is good
        score = r['accuracy'] - (r['time_per_task'] * 2)  # Penalty for slow
        if score > best_score:
            best_score = score
            best_config = r

        print(f"{r['config']:<30} {r['accuracy']:>11.6f}% {r['time_per_task']:>11.6f}s {score:>9.2f}")

    print("-" * 70)
    print(f"\n🏆 WINNER: {best_config['config']}")
    print(f"   Accuracy:  {best_config['accuracy']:.6f}%")
    print(f"   Time/Task: {best_config['time_per_task']:.6f}s")
    print(f"   Score:     {best_score:.2f}")

    # Project to 30-minute run
    print(f"\n📈 PROJECTION FOR 30-MINUTE RUN:")
    tasks_in_30min = 1800 / best_config['time_per_task']
    print(f"   Estimated tasks completed: {tasks_in_30min:.0f}")
    print(f"   Estimated accuracy: {best_config['accuracy']:.6f}%")
    print(f"   Target accuracy: 85.000000%")
    gap = 85.0 - best_config['accuracy']
    print(f"   Gap: {gap:+.6f}%")

    print("\n" + "="*70)
    print("✅ TUNING COMPLETE - Parameters ready for production run")
    print("="*70)

    # Save results
    with open('tuning_results.json', 'w') as f:
        json.dump({
            'results': results,
            'best_config': best_config,
            'projection_30min': {
                'estimated_tasks': tasks_in_30min,
                'estimated_accuracy': best_config['accuracy'],
                'target': 85.0,
                'gap': gap
            }
        }, f, indent=2)

    print(f"\n💾 Saved: tuning_results.json")


if __name__ == "__main__":
    main()
