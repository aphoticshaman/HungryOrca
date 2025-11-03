#!/usr/bin/env python3
"""
Test fixed fill algorithm on ALL near-miss tasks.
"""

import json
import numpy as np
from fixed_fill_specialist import solve_with_fixed_fill


def test_all_near_miss():
    """Test fixed algorithm on all near-miss tasks."""

    print(f"{'='*80}")
    print(f"🧪 TESTING FIXED FILL ON ALL NEAR-MISS TASKS")
    print(f"{'='*80}\n")

    # Load data
    with open('arc-agi_training_challenges.json') as f:
        challenges = json.load(f)

    with open('arc-agi_training_solutions.json') as f:
        solutions = json.load(f)

    # Near-miss tasks from our analysis
    near_miss_tasks = [
        ('00d62c1b', 91.8),  # Fill task
        ('045e512c', 90.2),  # Unknown
        ('025d127b', 88.0),  # Unknown
        ('00dbd492', 78.0),  # Unknown
        ('03560426', 71.0),  # Unknown
    ]

    results = []

    for task_id, old_score in near_miss_tasks:
        print(f"\n{'='*80}")
        print(f"Task: {task_id} (old score: {old_score}%)")
        print(f"{'='*80}")

        task = challenges[task_id]
        train_pairs = [(np.array(p['input']), np.array(p['output'])) for p in task['train']]
        test_input = np.array(task['test'][0]['input'])
        test_output = np.array(solutions[task_id][0])

        try:
            # Test fixed fill
            predicted = solve_with_fixed_fill(train_pairs, test_input)

            # Score
            exact = np.array_equal(predicted, test_output)
            score = np.sum(predicted == test_output) / test_output.size if predicted.shape == test_output.shape else 0.0

            print(f"\n📊 Result: {score*100:.1f}% {'✅ EXACT' if exact else ''}")

            improvement = score * 100 - old_score
            if improvement > 0:
                print(f"📈 Improvement: +{improvement:.1f}%")
            elif improvement < 0:
                print(f"📉 Regression: {improvement:.1f}%")

            results.append({
                'task_id': task_id,
                'old_score': old_score,
                'new_score': score * 100,
                'improvement': improvement,
                'exact': exact,
            })

        except Exception as e:
            print(f"❌ Error: {str(e)[:80]}")
            results.append({
                'task_id': task_id,
                'old_score': old_score,
                'new_score': 0.0,
                'improvement': -old_score,
                'exact': False,
            })

    # Summary
    print(f"\n\n{'='*80}")
    print(f"📊 SUMMARY: FIXED ALGORITHM RESULTS")
    print(f"{'='*80}\n")

    print(f"{'Task':<12} {'Old':<8} {'New':<8} {'Change':<10} {'Status'}")
    print(f"{'-'*60}")

    for r in results:
        status = '✅ EXACT' if r['exact'] else ''
        change = f"+{r['improvement']:.1f}%" if r['improvement'] >= 0 else f"{r['improvement']:.1f}%"
        print(f"{r['task_id']:<12} {r['old_score']:>6.1f}% {r['new_score']:>6.1f}% {change:>9} {status}")

    # Overall stats
    exact_count = sum(r['exact'] for r in results)
    avg_improvement = np.mean([r['improvement'] for r in results])

    print(f"\n{'-'*60}")
    print(f"Exact matches: {exact_count}/{len(results)}")
    print(f"Avg improvement: {avg_improvement:+.1f}%")

    return results


if __name__ == '__main__':
    results = test_all_near_miss()
