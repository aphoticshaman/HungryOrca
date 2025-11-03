#!/usr/bin/env python3
"""
ABLATION TEST: Spiking Neural Network Approach
================================================

Test whether SPIKING (work/rest/sleep cycles) improves performance
compared to continuous processing.

THREE CONFIGURATIONS:
1. UNDERFIT - Minimal spiking (small bursts, long rests)
   - Batch: 5 tasks
   - Rest: 20s
   - Sleep: Every 10 tasks (30s)
   - May not push hardware enough

2. OPTIMAL - Balanced spiking (medium bursts, medium rests)
   - Batch: 20 tasks
   - Rest: 10s
   - Sleep: Every 50 tasks (15s)
   - Balanced stress/recovery

3. OVERFIT - Aggressive spiking (large bursts, short rests)
   - Batch: 50 tasks
   - Rest: 5s
   - Sleep: Every 100 tasks (10s)
   - May burn out, too much overhead

CONTROL: No spiking - continuous processing

RUN 3 ITERATIONS MINIMUM for statistical significance.

USAGE: python3 ablation_test_spiking.py
"""

import numpy as np
import json
import time
import gc
import psutil
from typing import List, Tuple, Dict
from collections import defaultdict


class SpikingSolver:
    """Solver with configurable spiking parameters."""

    def __init__(self, batch_size: int, rest_duration: float,
                 sleep_interval: int, sleep_duration: float, mode: str):
        self.batch_size = batch_size
        self.rest_duration = rest_duration
        self.sleep_interval = sleep_interval
        self.sleep_duration = sleep_duration
        self.mode = mode

        self.tasks_completed = 0
        self.spikes_fired = 0
        self.sleeps_taken = 0

    def solve_task(self, train_pairs: List, test_input: np.ndarray,
                   time_limit: float) -> Tuple[np.ndarray, float]:
        """Solve single task with basic transforms."""

        deadline = time.time() + time_limit
        best_solution = test_input.copy()
        best_score = 0.0

        transforms = [
            self._identity,
            self._flip_h,
            self._flip_v,
            self._rot_90,
            self._rot_180,
            self._color_map,
        ]

        for transform in transforms:
            if time.time() >= deadline:
                break

            try:
                candidate = transform(test_input, train_pairs)
                if candidate is not None:
                    score = self._validate(candidate, train_pairs)
                    if score > best_score:
                        best_solution = candidate
                        best_score = score
                        if score >= 0.999:
                            break
            except:
                continue

        return best_solution, best_score

    def solve_batch_with_spiking(self, tasks: List, solutions: Dict,
                                 time_per_task: float) -> Dict[str, any]:
        """Solve tasks with SPIKING rhythm."""

        print(f"\n{'='*60}")
        print(f"MODE: {self.mode}")
        print(f"Batch: {self.batch_size}, Rest: {self.rest_duration}s, "
              f"Sleep: every {self.sleep_interval} ({self.sleep_duration}s)")
        print(f"{'='*60}")

        start_time = time.time()
        results = []
        cpu_usage = []
        ram_usage = []

        for i, task_id in enumerate(tasks):
            task = tasks[task_id]
            train_pairs = [(np.array(p['input']), np.array(p['output']))
                          for p in task['train']]
            test_input = np.array(task['test'][0]['input'])
            ground_truth = np.array(solutions[task_id][0])

            # SPIKE: Solve task
            solution, score = self.solve_task(train_pairs, test_input, time_per_task)
            similarity = self._calc_similarity(solution, ground_truth)

            results.append({
                'task_id': task_id,
                'similarity': similarity,
                'perfect': similarity >= 0.999
            })

            self.tasks_completed += 1

            # Monitor resources
            cpu_usage.append(psutil.cpu_percent(interval=0.01))
            ram_usage.append(psutil.virtual_memory().percent)

            # SPIKE BOUNDARY: Rest after batch
            if (self.tasks_completed % self.batch_size == 0 and
                self.tasks_completed < len(tasks)):

                # REST PHASE
                self.spikes_fired += 1
                print(f"  ⚡ SPIKE #{self.spikes_fired} complete "
                      f"({self.batch_size} tasks)")
                print(f"  😮‍💨 REFRACTORY: Resting {self.rest_duration}s...")

                gc.collect()  # Garbage collect
                time.sleep(self.rest_duration)

                cpu_after = psutil.cpu_percent(interval=0.1)
                ram_after = psutil.virtual_memory().percent
                print(f"     CPU: {cpu_after:.0f}%, RAM: {ram_after:.0f}%")

            # SLEEP PHASE: Deep consolidation
            if (self.tasks_completed % self.sleep_interval == 0 and
                self.tasks_completed < len(tasks)):

                self.sleeps_taken += 1
                print(f"  😴 SLEEP #{self.sleeps_taken}: Deep consolidation "
                      f"{self.sleep_duration}s...")

                gc.collect()
                time.sleep(self.sleep_duration)

                cpu_after = psutil.cpu_percent(interval=0.1)
                ram_after = psutil.virtual_memory().percent
                print(f"     CPU: {cpu_after:.0f}%, RAM: {ram_after:.0f}%")

            # Progress
            if (i + 1) % 10 == 0:
                avg_sim = np.mean([r['similarity'] for r in results])
                perfect = sum([r['perfect'] for r in results])
                print(f"  Progress {i+1}/{len(tasks)}: "
                      f"Avg={avg_sim:.1%}, Perfect={perfect}")

        elapsed = time.time() - start_time

        return {
            'results': results,
            'elapsed': elapsed,
            'spikes': self.spikes_fired,
            'sleeps': self.sleeps_taken,
            'avg_cpu': np.mean(cpu_usage),
            'avg_ram': np.mean(ram_usage),
            'mode': self.mode
        }

    def _calc_similarity(self, pred: np.ndarray, truth: np.ndarray) -> float:
        """Calculate similarity."""
        if pred.shape != truth.shape:
            return 0.0
        return np.sum(pred == truth) / truth.size

    def _validate(self, candidate, train_pairs):
        """Quick validation."""
        if len(train_pairs) == 0:
            return 0.5
        scores = []
        for inp, out in train_pairs[:2]:
            if candidate.shape == out.shape:
                scores.append(0.8)
            else:
                scores.append(0.2)
        return np.mean(scores)

    # Basic transforms
    def _identity(self, test_input, train_pairs):
        for inp, out in train_pairs[:1]:
            if not np.array_equal(inp, out):
                return None
        return test_input

    def _flip_h(self, test_input, train_pairs):
        for inp, out in train_pairs[:1]:
            if inp.shape != out.shape or not np.array_equal(np.flip(inp, axis=0), out):
                return None
        return np.flip(test_input, axis=0)

    def _flip_v(self, test_input, train_pairs):
        for inp, out in train_pairs[:1]:
            if inp.shape != out.shape or not np.array_equal(np.flip(inp, axis=1), out):
                return None
        return np.flip(test_input, axis=1)

    def _rot_90(self, test_input, train_pairs):
        for inp, out in train_pairs[:1]:
            if not np.array_equal(np.rot90(inp, k=1), out):
                return None
        return np.rot90(test_input, k=1)

    def _rot_180(self, test_input, train_pairs):
        for inp, out in train_pairs[:1]:
            if inp.shape != out.shape or not np.array_equal(np.rot90(inp, k=2), out):
                return None
        return np.rot90(test_input, k=2)

    def _color_map(self, test_input, train_pairs):
        color_map = {}
        for inp, out in train_pairs[:1]:
            if inp.shape != out.shape:
                return None
            for i in range(inp.shape[0]):
                for j in range(inp.shape[1]):
                    c_in, c_out = int(inp[i, j]), int(out[i, j])
                    if c_in in color_map and color_map[c_in] != c_out:
                        return None
                    color_map[c_in] = c_out

        result = test_input.copy()
        for i in range(result.shape[0]):
            for j in range(result.shape[1]):
                if result[i, j] in color_map:
                    result[i, j] = color_map[result[i, j]]
        return result


class ContinuousSolver(SpikingSolver):
    """Control: No spiking - continuous processing."""

    def solve_batch_continuous(self, tasks: List, solutions: Dict,
                              time_per_task: float) -> Dict[str, any]:
        """Solve tasks WITHOUT spiking (control)."""

        print(f"\n{'='*60}")
        print(f"MODE: CONTROL (Continuous - No Spiking)")
        print(f"{'='*60}")

        start_time = time.time()
        results = []
        cpu_usage = []
        ram_usage = []

        for i, task_id in enumerate(tasks):
            task = tasks[task_id]
            train_pairs = [(np.array(p['input']), np.array(p['output']))
                          for p in task['train']]
            test_input = np.array(task['test'][0]['input'])
            ground_truth = np.array(solutions[task_id][0])

            # Continuous solving (no breaks)
            solution, score = self.solve_task(train_pairs, test_input, time_per_task)
            similarity = self._calc_similarity(solution, ground_truth)

            results.append({
                'task_id': task_id,
                'similarity': similarity,
                'perfect': similarity >= 0.999
            })

            cpu_usage.append(psutil.cpu_percent(interval=0.01))
            ram_usage.append(psutil.virtual_memory().percent)

            if (i + 1) % 10 == 0:
                avg_sim = np.mean([r['similarity'] for r in results])
                perfect = sum([r['perfect'] for r in results])
                print(f"  Progress {i+1}/{len(tasks)}: "
                      f"Avg={avg_sim:.1%}, Perfect={perfect}")

        elapsed = time.time() - start_time

        return {
            'results': results,
            'elapsed': elapsed,
            'spikes': 0,
            'sleeps': 0,
            'avg_cpu': np.mean(cpu_usage),
            'avg_ram': np.mean(ram_usage),
            'mode': 'CONTROL'
        }


def run_spiking_ablation(num_tasks: int = 30, iterations: int = 3):
    """
    Run ablation test for spiking approach.

    Args:
        num_tasks: Number of tasks per test (default 30)
        iterations: Number of test iterations (default 3)
    """

    print("=" * 80)
    print("ABLATION TEST: Spiking Neural Network Approach")
    print("=" * 80)
    print(f"Tasks per run: {num_tasks}")
    print(f"Iterations: {iterations}")
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

    # Configurations
    configs = [
        {
            'name': 'CONTROL (No Spiking)',
            'batch_size': 999,  # No batching
            'rest_duration': 0,
            'sleep_interval': 999,
            'sleep_duration': 0,
            'continuous': True
        },
        {
            'name': 'UNDERFIT (Minimal Spiking)',
            'batch_size': 5,
            'rest_duration': 20,
            'sleep_interval': 10,
            'sleep_duration': 30,
            'continuous': False
        },
        {
            'name': 'OPTIMAL (Balanced Spiking)',
            'batch_size': 20,
            'rest_duration': 10,
            'sleep_interval': 50,
            'sleep_duration': 15,
            'continuous': False
        },
        {
            'name': 'OVERFIT (Aggressive Spiking)',
            'batch_size': 50,
            'rest_duration': 5,
            'sleep_interval': 100,
            'sleep_duration': 10,
            'continuous': False
        },
    ]

    all_results = defaultdict(list)

    # Run iterations
    for iteration in range(iterations):
        print(f"\n{'#'*80}")
        print(f"ITERATION {iteration + 1}/{iterations}")
        print(f"{'#'*80}")

        # Select tasks for this iteration
        task_ids = list(all_tasks.keys())[iteration*num_tasks:(iteration+1)*num_tasks]
        tasks = {tid: all_tasks[tid] for tid in task_ids}

        print(f"Testing on {len(tasks)} tasks (offset: {iteration*num_tasks})")

        # Test each configuration
        for config in configs:
            mode = config['name']

            if config['continuous']:
                # CONTROL: Continuous
                solver = ContinuousSolver(
                    batch_size=0,
                    rest_duration=0,
                    sleep_interval=0,
                    sleep_duration=0,
                    mode=mode
                )
                result = solver.solve_batch_continuous(tasks, solutions, time_per_task=10)
            else:
                # TREATMENT: Spiking
                solver = SpikingSolver(
                    batch_size=config['batch_size'],
                    rest_duration=config['rest_duration'],
                    sleep_interval=config['sleep_interval'],
                    sleep_duration=config['sleep_duration'],
                    mode=mode
                )
                result = solver.solve_batch_with_spiking(tasks, solutions, time_per_task=10)

            # Store results
            all_results[mode].append(result)

            # Print summary
            similarities = [r['similarity'] for r in result['results']]
            perfect = sum([r['perfect'] for r in result['results']])

            print(f"\n  ✓ {mode}:")
            print(f"    Perfect: {perfect}/{len(tasks)} ({perfect/len(tasks):.1%})")
            print(f"    Avg Similarity: {np.mean(similarities):.1%}")
            print(f"    Time: {result['elapsed']:.1f}s")
            print(f"    Spikes: {result['spikes']}, Sleeps: {result['sleeps']}")
            print(f"    CPU: {result['avg_cpu']:.0f}%, RAM: {result['avg_ram']:.0f}%")

    # Aggregate results
    print(f"\n{'='*80}")
    print(f"AGGREGATED RESULTS ({iterations} iterations)")
    print(f"{'='*80}")

    comparison = []

    for mode, results_list in all_results.items():
        all_similarities = []
        all_perfect = []
        all_times = []

        for result in results_list:
            similarities = [r['similarity'] for r in result['results']]
            perfect = sum([r['perfect'] for r in result['results']])

            all_similarities.extend(similarities)
            all_perfect.append(perfect)
            all_times.append(result['elapsed'])

        avg_similarity = np.mean(all_similarities)
        avg_perfect = np.mean(all_perfect)
        avg_time = np.mean(all_times)
        total_tasks = len(all_similarities)

        comparison.append({
            'mode': mode,
            'avg_similarity': avg_similarity,
            'avg_perfect': avg_perfect,
            'total_perfect': sum(all_perfect),
            'total_tasks': total_tasks,
            'avg_time': avg_time
        })

        print(f"\n{mode}:")
        print(f"  Avg Similarity: {avg_similarity:.1%}")
        print(f"  Avg Perfect: {avg_perfect:.1f}/{num_tasks} ({avg_perfect/num_tasks:.1%})")
        print(f"  Total Perfect: {sum(all_perfect)}/{total_tasks} ({sum(all_perfect)/total_tasks:.1%})")
        print(f"  Avg Time: {avg_time:.1f}s")

    # Determine winner
    print(f"\n{'='*80}")
    print("WINNER DETERMINATION")
    print(f"{'='*80}")

    sorted_by_perfect = sorted(comparison, key=lambda x: x['avg_similarity'], reverse=True)

    print(f"\n{'Mode':<30} {'Avg Similarity':<15} {'Perfect Rate':<15} {'Time (s)':<10}")
    print("-" * 80)

    for rank, result in enumerate(sorted_by_perfect, 1):
        perfect_rate = result['total_perfect'] / result['total_tasks']
        print(f"{rank}. {result['mode']:<27} {result['avg_similarity']:<15.1%} "
              f"{perfect_rate:<15.1%} {result['avg_time']:<10.1f}")

    winner = sorted_by_perfect[0]
    control = next(r for r in comparison if 'CONTROL' in r['mode'])

    improvement = winner['avg_similarity'] - control['avg_similarity']

    print(f"\n{'='*80}")
    print(f"🏆 WINNER: {winner['mode']}")
    print(f"{'='*80}")
    print(f"Avg Similarity: {winner['avg_similarity']:.1%}")
    print(f"Perfect Rate: {winner['total_perfect']}/{winner['total_tasks']} "
          f"({winner['total_perfect']/winner['total_tasks']:.1%})")
    print(f"Improvement over CONTROL: {improvement:+.1%}")

    if improvement > 0.05:
        print(f"\n✅ RESULT: PASS - Spiking shows +{improvement:.1%} improvement")
        print("   Recommendation: USE spiking approach")
    elif improvement > 0.02:
        print(f"\n⚠️ RESULT: MARGINAL - Spiking shows +{improvement:.1%} improvement")
        print("   Recommendation: Consider using spiking with refinements")
    else:
        print(f"\n❌ RESULT: FAIL - Spiking shows only +{improvement:.1%} improvement")
        print("   Recommendation: Do NOT use spiking (overhead not worth it)")

    # Save results
    with open('ablation_spiking_results.json', 'w') as f:
        json.dump({
            'iterations': iterations,
            'tasks_per_iteration': num_tasks,
            'comparison': comparison,
            'winner': winner['mode'],
            'improvement': float(improvement)
        }, f, indent=2)

    print(f"\n💾 Results saved to: ablation_spiking_results.json")


if __name__ == '__main__':
    # Run 3 iterations minimum (30 tasks × 3 = 90 tasks total)
    run_spiking_ablation(num_tasks=30, iterations=3)
