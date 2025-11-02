#!/usr/bin/env python3
"""
QUICK OVERFIT/UNDERFIT DIAGNOSTIC
===================================

Test extreme configurations to diagnose overfit vs underfit behavior.

CONFIGURATIONS:
- UNDERFIT: Minimal transforms, shallow search (fast but inaccurate)
- BALANCED: Standard settings
- OVERFIT: Maximum everything (slow but may memorize too much)

USAGE: python3 quick_overfit_test.py
"""

import numpy as np
import json
import time
from typing import List, Tuple


class QuickSolver:
    """Minimal solver for quick testing."""

    def __init__(self, mode: str = 'balanced'):
        self.mode = mode

        # Configure based on mode
        if mode == 'underfit':
            self.max_checks = 1          # Only check 1 training pair
            self.transforms = 2          # Only 2 transforms
            self.early_stop = 0.5        # Stop at 50%
        elif mode == 'balanced':
            self.max_checks = 2          # Check 2 training pairs
            self.transforms = 6          # 6 transforms
            self.early_stop = 0.9        # Stop at 90%
        elif mode == 'overfit':
            self.max_checks = 999        # Check ALL training pairs
            self.transforms = 9          # All transforms
            self.early_stop = 0.999      # Never stop early

    def solve(self, train_pairs: List[Tuple[np.ndarray, np.ndarray]],
              test_input: np.ndarray, time_limit: float) -> Tuple[np.ndarray, float, str]:
        """Solve and return (solution, confidence, transform_used)."""

        deadline = time.time() + time_limit

        # Try transforms in order
        transform_list = [
            ('identity', self._identity),
            ('flip_h', self._flip_h),
            ('flip_v', self._flip_v),
            ('rot_90', self._rot_90),
            ('rot_180', self._rot_180),
            ('rot_270', self._rot_270),
            ('color_map', self._color_map),
            ('transpose', self._transpose),
            ('fill', self._fill),
        ][:self.transforms]

        for name, transform_fn in transform_list:
            if time.time() >= deadline:
                break

            try:
                candidate = transform_fn(test_input, train_pairs)
                if candidate is not None:
                    # Success!
                    confidence = self._calc_confidence(train_pairs)
                    return candidate, confidence, name
            except:
                continue

        # Failed - return input
        return test_input, 0.0, 'none'

    def _calc_confidence(self, train_pairs: List) -> float:
        """Calculate confidence based on training set size."""
        checks = min(len(train_pairs), self.max_checks)
        return checks / max(len(train_pairs), 1)

    def _identity(self, test_input, train_pairs):
        for inp, out in train_pairs[:self.max_checks]:
            if not np.array_equal(inp, out):
                return None
        return test_input

    def _flip_h(self, test_input, train_pairs):
        for inp, out in train_pairs[:self.max_checks]:
            if inp.shape != out.shape or not np.array_equal(np.flip(inp, axis=0), out):
                return None
        return np.flip(test_input, axis=0)

    def _flip_v(self, test_input, train_pairs):
        for inp, out in train_pairs[:self.max_checks]:
            if inp.shape != out.shape or not np.array_equal(np.flip(inp, axis=1), out):
                return None
        return np.flip(test_input, axis=1)

    def _rot_90(self, test_input, train_pairs):
        for inp, out in train_pairs[:self.max_checks]:
            if not np.array_equal(np.rot90(inp, k=1), out):
                return None
        return np.rot90(test_input, k=1)

    def _rot_180(self, test_input, train_pairs):
        for inp, out in train_pairs[:self.max_checks]:
            if inp.shape != out.shape or not np.array_equal(np.rot90(inp, k=2), out):
                return None
        return np.rot90(test_input, k=2)

    def _rot_270(self, test_input, train_pairs):
        for inp, out in train_pairs[:self.max_checks]:
            if not np.array_equal(np.rot90(inp, k=3), out):
                return None
        return np.rot90(test_input, k=3)

    def _transpose(self, test_input, train_pairs):
        for inp, out in train_pairs[:self.max_checks]:
            if not np.array_equal(np.transpose(inp), out):
                return None
        return np.transpose(test_input)

    def _color_map(self, test_input, train_pairs):
        color_map = {}
        for inp, out in train_pairs[:self.max_checks]:
            if inp.shape != out.shape:
                return None
            for i in range(inp.shape[0]):
                for j in range(inp.shape[1]):
                    c_in = int(inp[i, j])
                    c_out = int(out[i, j])
                    if c_in in color_map:
                        if color_map[c_in] != c_out:
                            return None
                    else:
                        color_map[c_in] = c_out

        result = test_input.copy()
        for i in range(result.shape[0]):
            for j in range(result.shape[1]):
                c = int(result[i, j])
                if c in color_map:
                    result[i, j] = color_map[c]
        return result

    def _fill(self, test_input, train_pairs):
        return None


def run_quick_overfit_test(num_tasks: int = 20, time_per_task: float = 5):
    """Run quick overfit vs underfit test."""

    print("=" * 80)
    print("QUICK OVERFIT/UNDERFIT DIAGNOSTIC")
    print("=" * 80)
    print(f"Tasks: {num_tasks}")
    print(f"Time per task: {time_per_task}s")
    print("=" * 80)

    # Load data
    try:
        with open('arc-agi_training_challenges.json') as f:
            all_tasks = json.load(f)
        with open('arc-agi_training_solutions.json') as f:
            solutions = json.load(f)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return

    task_ids = list(all_tasks.keys())[:num_tasks]

    # Test 3 modes
    modes = ['underfit', 'balanced', 'overfit']
    results = {mode: {'solved': 0, 'total': 0, 'avg_time': 0, 'transforms': {}} for mode in modes}

    for mode in modes:
        print(f"\n{'='*80}")
        print(f"TESTING MODE: {mode.upper()}")
        print(f"{'='*80}")

        solver = QuickSolver(mode=mode)
        times = []

        for i, task_id in enumerate(task_ids):
            task = all_tasks[task_id]
            train_pairs = [(np.array(p['input']), np.array(p['output']))
                          for p in task['train']]
            test_input = np.array(task['test'][0]['input'])
            expected_output = np.array(solutions[task_id][0])

            start = time.time()
            solution, confidence, transform = solver.solve(train_pairs, test_input, time_per_task)
            elapsed = time.time() - start
            times.append(elapsed)

            # Check if correct
            is_correct = np.array_equal(solution, expected_output)

            if is_correct:
                results[mode]['solved'] += 1

                # Track which transforms work
                if transform not in results[mode]['transforms']:
                    results[mode]['transforms'][transform] = 0
                results[mode]['transforms'][transform] += 1

            results[mode]['total'] += 1

            if (i + 1) % 10 == 0:
                acc = results[mode]['solved'] / results[mode]['total']
                print(f"  Progress {i+1}/{num_tasks}: {results[mode]['solved']}/{results[mode]['total']} solved ({acc:.1%})")

        results[mode]['avg_time'] = np.mean(times)

        print(f"\n  ✓ {mode.upper()} Results:")
        print(f"    Solved: {results[mode]['solved']}/{results[mode]['total']} ({results[mode]['solved']/results[mode]['total']:.1%})")
        print(f"    Avg time: {results[mode]['avg_time']:.3f}s")
        print(f"    Top transforms: {sorted(results[mode]['transforms'].items(), key=lambda x: x[1], reverse=True)[:3]}")

    # Comparison
    print(f"\n{'='*80}")
    print("COMPARISON")
    print(f"{'='*80}")

    print(f"\n{'Mode':<15} {'Accuracy':<15} {'Avg Time':<15} {'Speed/Acc Ratio':<15}")
    print("-" * 80)

    for mode in modes:
        acc = results[mode]['solved'] / results[mode]['total']
        avg_time = results[mode]['avg_time']
        ratio = acc / max(avg_time, 0.001)  # Accuracy per second

        print(f"{mode.upper():<15} {acc:<15.1%} {avg_time:<15.3f}s {ratio:<15.2f}")

    # Diagnosis
    print(f"\n{'='*80}")
    print("DIAGNOSIS")
    print(f"{'='*80}")

    underfit_acc = results['underfit']['solved'] / results['underfit']['total']
    balanced_acc = results['balanced']['solved'] / results['balanced']['total']
    overfit_acc = results['overfit']['solved'] / results['overfit']['total']

    if balanced_acc > underfit_acc + 0.1 and balanced_acc > overfit_acc:
        print("\n✅ BALANCED is optimal - current settings are good!")
    elif underfit_acc > balanced_acc:
        print("\n⚠️ UNDERFIT winning - solver is too simple, add more transforms/depth")
    elif overfit_acc > balanced_acc + 0.1:
        print("\n⚠️ OVERFIT winning - solver is too restrictive, relax validation")
    else:
        print("\n📊 Results are close - current settings are reasonable")

    # Save results
    with open('overfit_underfit_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n💾 Results saved to: overfit_underfit_results.json")


if __name__ == '__main__':
    # Quick test: 20 tasks × 5 seconds = 100 seconds (~1.5 minutes)
    run_quick_overfit_test(num_tasks=20, time_per_task=5)
