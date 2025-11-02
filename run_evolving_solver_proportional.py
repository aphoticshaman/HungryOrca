#!/usr/bin/env python3
"""
Run evolving solver with PROPORTIONAL time management.

Time distributed across ALL tasks, not spent on just first few.

Example:
- Total budget: 150s (2.5 minutes)
- 10 tasks = 15s per task
- Attempt ALL tasks with fraction of time each
"""

import json
import numpy as np
import time
from evolving_specialist_system import EvolvingSolver


def load_evaluation_tasks(path='arc-agi_training_challenges.json'):
    """Load training tasks for testing."""
    with open(path) as f:
        challenges = json.load(f)

    # Load solutions too
    with open('arc-agi_training_solutions.json') as f:
        solutions = json.load(f)

    # Merge challenges with solutions
    for task_id in challenges:
        if task_id in solutions:
            challenges[task_id]['test'][0]['output'] = solutions[task_id][0]

    return challenges


def partial_match_score(pred, true):
    """Calculate partial match percentage."""
    if pred is None or true is None:
        return 0.0

    # Resize if needed
    if pred.shape != true.shape:
        return 0.0

    matches = np.sum(pred == true)
    total = true.size
    return matches / total


def run_proportional_time_test(total_time_budget=150.0, max_tasks=10):
    """
    Run with proportional time management.

    Key insight: Distribute time across ALL tasks, not all time on first task!
    """

    print(f"{'='*80}")
    print(f"🚀 EVOLVING SOLVER - PROPORTIONAL TIME MANAGEMENT TEST")
    print(f"{'='*80}")
    print(f"Total budget: {total_time_budget:.1f}s")
    print(f"Tasks to attempt: {max_tasks}")
    print(f"Time per task: {total_time_budget/max_tasks:.1f}s")
    print(f"{'='*80}\n")

    # Load tasks
    tasks = load_evaluation_tasks()
    task_ids = list(tasks.keys())[:max_tasks]

    # Calculate time per task
    time_per_task = total_time_budget / len(task_ids)

    results = []
    start_total = time.time()

    for idx, task_id in enumerate(task_ids):
        task = tasks[task_id]

        print(f"\n{'='*80}")
        print(f"Task {idx+1}/{len(task_ids)}: {task_id}")
        print(f"Time budget: {time_per_task:.1f}s")
        print(f"{'='*80}")

        # Extract train/test
        train_pairs = [(np.array(pair['input']), np.array(pair['output']))
                       for pair in task['train']]
        test_input = np.array(task['test'][0]['input'])
        test_output = np.array(task['test'][0]['output'])

        # Create solver with time budget for THIS task
        solver = EvolvingSolver(time_limit=time_per_task, verbose=True)

        # Solve
        task_start = time.time()
        try:
            result = solver.solve(train_pairs, test_input)
            task_time = time.time() - task_start

            # Score
            score = partial_match_score(result, test_output)
            exact = score == 1.0

            results.append({
                'task_id': task_id,
                'score': float(score),
                'exact': bool(exact),
                'time': float(task_time),
            })

            print(f"\n📊 Result: {score*100:.1f}% match {'✅ EXACT' if exact else ''}")
            print(f"⏱️ Time used: {task_time:.1f}s / {time_per_task:.1f}s budgeted")

        except Exception as e:
            print(f"❌ Error: {str(e)[:60]}")
            results.append({
                'task_id': task_id,
                'score': 0.0,
                'exact': False,
                'time': float(time.time() - task_start),
            })

        # Check if we're over total budget
        elapsed_total = time.time() - start_total
        if elapsed_total > total_time_budget:
            print(f"\n⏱️ Total time budget exceeded, stopping")
            break

    # Summary
    print(f"\n\n{'='*80}")
    print(f"📊 FINAL RESULTS - PROPORTIONAL TIME MANAGEMENT")
    print(f"{'='*80}")

    total_time = time.time() - start_total
    avg_score = np.mean([r['score'] for r in results])
    exact_count = sum(r['exact'] for r in results)

    print(f"\nTasks attempted: {len(results)}/{max_tasks}")
    print(f"Average partial match: {avg_score*100:.1f}%")
    print(f"Exact matches: {exact_count}/{len(results)}")
    print(f"Total time: {total_time:.1f}s / {total_time_budget:.1f}s budgeted")
    print(f"Time per task avg: {total_time/len(results):.1f}s")

    print(f"\n{'='*80}")
    print(f"INDIVIDUAL RESULTS:")
    print(f"{'='*80}")
    for r in results:
        status = "✅ EXACT" if r['exact'] else f"{r['score']*100:.1f}%"
        print(f"{r['task_id']}: {status} ({r['time']:.1f}s)")

    # Save results
    output = {
        'config': {
            'total_time_budget': total_time_budget,
            'max_tasks': max_tasks,
            'time_per_task': time_per_task,
        },
        'summary': {
            'tasks_attempted': len(results),
            'avg_partial_match': float(avg_score),
            'exact_matches': exact_count,
            'total_time': total_time,
        },
        'results': results,
    }

    with open('evolving_solver_proportional_results.json', 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n💾 Results saved to: evolving_solver_proportional_results.json")

    return avg_score, exact_count


if __name__ == '__main__':
    avg_score, exact_count = run_proportional_time_test(
        total_time_budget=150.0,  # 2.5 minutes
        max_tasks=10
    )

    print(f"\n\n🎯 BASELINE COMPARISON:")
    print(f"   Previous (Collaborative): 51.8% partial, 0 exact")
    print(f"   Current (Evolving):       {avg_score*100:.1f}% partial, {exact_count} exact")

    improvement = (avg_score - 0.518) * 100
    if improvement > 0:
        print(f"   📈 Improvement: +{improvement:.1f}%")
    elif improvement < 0:
        print(f"   📉 Regression: {improvement:.1f}%")
    else:
        print(f"   ➖ No change")
