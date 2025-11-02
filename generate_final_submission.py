#!/usr/bin/env python3
"""
FINAL SUBMISSION GENERATOR - ARC Prize 2025

Uses best-performing solver: Evolving Specialists (51.9% avg, 1 exact)

Time management: 150s total / 240 tasks = 0.625s per task
"""

import json
import numpy as np
from evolving_specialist_system import EvolvingSolver
import time


def generate_submission():
    """Generate submission.json for all 240 test tasks."""

    print(f"{'='*80}")
    print(f"🚀 FINAL SUBMISSION GENERATOR - ARC Prize 2025")
    print(f"{'='*80}\n")

    # Load test challenges
    with open('arc-agi_test_challenges.json') as f:
        test_tasks = json.load(f)

    print(f"Loaded {len(test_tasks)} test tasks")

    # Time budget
    total_budget = 150.0  # 2.5 minutes
    time_per_task = total_budget / len(test_tasks)

    print(f"Time budget: {total_budget:.1f}s total, {time_per_task:.3f}s per task")
    print(f"{'='*80}\n")

    submission = {}
    start_total = time.time()

    for idx, (task_id, task) in enumerate(test_tasks.items()):
        if (idx + 1) % 50 == 0:
            elapsed = time.time() - start_total
            print(f"Progress: {idx+1}/{len(test_tasks)} tasks ({elapsed:.1f}s elapsed)")

        # Extract training pairs
        train_pairs = [(np.array(p['input']), np.array(p['output']))
                      for p in task['train']]

        # Each test task has 1-2 test inputs
        attempts = []

        for test_pair in task['test']:
            test_input = np.array(test_pair['input'])

            # Solve with time budget
            solver = EvolvingSolver(time_limit=time_per_task, verbose=False)

            try:
                result = solver.solve(train_pairs, test_input)

                if result is not None:
                    attempts.append(result.tolist())
                else:
                    # Fallback: return input
                    attempts.append(test_input.tolist())

            except Exception:
                # Fallback: return input
                attempts.append(test_input.tolist())

        # Each task needs 2 attempts (even if only 1 test input)
        while len(attempts) < 2:
            attempts.append(attempts[0] if attempts else [[0]])

        submission[task_id] = attempts

    total_time = time.time() - start_total

    print(f"\n{'='*80}")
    print(f"✅ SUBMISSION COMPLETE")
    print(f"{'='*80}")
    print(f"Tasks: {len(submission)}")
    print(f"Time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Avg per task: {total_time/len(submission):.2f}s")

    # Save
    with open('submission.json', 'w') as f:
        json.dump(submission, f)

    print(f"\n💾 Saved: submission.json")

    # Check file size
    import os
    size = os.path.getsize('submission.json')
    print(f"📦 Size: {size:,} bytes ({size/1024:.1f} KB)")

    print(f"\n🎯 Ready for Kaggle upload!")


if __name__ == '__main__':
    generate_submission()
