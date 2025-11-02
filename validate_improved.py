"""Quick validation of improved solver."""
import json
import numpy as np
from arc_solver_improved import ImprovedARCSolver

solver = ImprovedARCSolver()

# Load training data
with open('arc-agi_training_challenges.json', 'r') as f:
    challenges = json.load(f)
with open('arc-agi_training_solutions.json', 'r') as f:
    solutions = json.load(f)

# Test on first 20 tasks
perfect = 0
partial = 0
total = 0

task_ids = list(challenges.keys())[:20]

for task_id in task_ids:
    task = challenges[task_id]
    expected = solutions[task_id]

    try:
        # Test each test example
        for test_idx, test_pair in enumerate(task['test']):
            total += 1
            test_task = {
                'train': task['train'],
                'test': [{'input': test_pair['input']}]
            }

            predictions = solver.solve_task(test_task)
            expected_output = expected[test_idx]

            # Check match
            for pred in predictions:
                if np.array_equal(np.array(pred), np.array(expected_output)):
                    perfect += 1
                    print(f"✓ {task_id} test {test_idx}: PERFECT")
                    break
            else:
                # Check partial
                best_sim = 0
                for pred in predictions:
                    arr1, arr2 = np.array(pred), np.array(expected_output)
                    if arr1.shape == arr2.shape:
                        sim = np.sum(arr1 == arr2) / arr1.size
                        best_sim = max(best_sim, sim)

                if best_sim > 0.7:
                    partial += 1
                    print(f"≈ {task_id} test {test_idx}: PARTIAL ({best_sim:.1%})")
                else:
                    print(f"✗ {task_id} test {test_idx}: FAILED ({best_sim:.1%})")
    except Exception as e:
        print(f"✗ {task_id}: ERROR - {str(e)[:50]}")
        total += 1

print(f"\n{'='*60}")
print(f"Results on 20 tasks:")
print(f"Perfect: {perfect}/{total} ({perfect/total*100:.1f}%)")
print(f"Partial: {partial}/{total} ({partial/total*100:.1f}%)")
print(f"{'='*60}")
