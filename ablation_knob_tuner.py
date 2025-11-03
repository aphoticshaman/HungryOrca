#!/usr/bin/env python3
"""
ABLATION KNOB TUNER - Quick Diagnostic A/B Testing
====================================================

Tune hyperparameters with quick ablation tests to find optimal settings.

TUNABLE KNOBS:
1. Search depth (1-step vs 2-step transforms)
2. Transform selection (minimal vs comprehensive)
3. Validation strictness (how many training pairs to check)
4. Time allocation (greedy vs thorough)
5. Overfit vs underfit balance

USAGE: python3 ablation_knob_tuner.py
"""

import numpy as np
import json
import time
from typing import List, Tuple, Dict
from collections import defaultdict


# =============================================================================
# KNOB CONFIGURATIONS
# =============================================================================

KNOBS = {
    'search_depth': {
        'underfit': 1,      # Only single transforms (fast, may miss complex patterns)
        'balanced': 2,       # Up to 2-step compositions (good balance)
        'overfit': 3,        # Up to 3-step compositions (slow, may overfit)
    },

    'transform_set': {
        'minimal': ['identity', 'flip_h', 'flip_v', 'color_map'],  # Fast, 4 transforms
        'standard': ['identity', 'flip_h', 'flip_v', 'rot_90', 'rot_180', 'color_map'],  # Balanced, 6 transforms
        'comprehensive': ['identity', 'flip_h', 'flip_v', 'rot_90', 'rot_180', 'rot_270',
                         'color_map', 'transpose', 'fill'],  # Slow, 9 transforms
    },

    'validation_strictness': {
        'lenient': 1,       # Check only 1 training pair (fast, may be wrong)
        'moderate': 2,      # Check 2 training pairs (balanced)
        'strict': 999,      # Check ALL training pairs (slow, more reliable)
    },

    'time_allocation': {
        'greedy': 0.3,      # Stop at first 80%+ solution (fast, lower accuracy)
        'balanced': 0.6,    # Stop at first 90%+ solution (balanced)
        'thorough': 0.99,   # Keep searching until 99%+ (slow, higher accuracy)
    },
}


# =============================================================================
# SOLVER WITH TUNABLE KNOBS
# =============================================================================

class TunableSolver:
    """ARC solver with tunable hyperparameters."""

    def __init__(self, config: Dict):
        self.search_depth = config['search_depth']
        self.transform_set = config['transform_set']
        self.validation_strictness = config['validation_strictness']
        self.time_allocation = config['time_allocation']

    def solve(self, train_pairs: List[Tuple[np.ndarray, np.ndarray]],
              test_input: np.ndarray, time_limit: float) -> Tuple[np.ndarray, float]:
        """Solve with current knob settings."""

        deadline = time.time() + time_limit
        best_solution = test_input.copy()
        best_score = 0.0

        # Phase 1: Single transforms
        for transform_name in self.transform_set:
            if time.time() >= deadline:
                break

            try:
                transform_fn = getattr(self, f'_transform_{transform_name}')
                candidate = transform_fn(test_input, train_pairs)

                if candidate is not None:
                    score = self._score(candidate, train_pairs)
                    if score > best_score:
                        best_solution = candidate
                        best_score = score

                        # Early stopping based on time_allocation knob
                        if score >= self.time_allocation:
                            return best_solution, best_score

            except Exception as e:
                continue

        # Phase 2: Compositional transforms (if search_depth > 1)
        if self.search_depth >= 2 and time.time() < deadline:
            for t1_name in self.transform_set[:3]:  # Limit combinations
                if time.time() >= deadline:
                    break

                try:
                    t1_fn = getattr(self, f'_transform_{t1_name}')
                    intermediate = t1_fn(test_input, train_pairs)

                    if intermediate is None:
                        continue

                    for t2_name in self.transform_set[:3]:
                        if time.time() >= deadline:
                            break

                        try:
                            t2_fn = getattr(self, f'_transform_{t2_name}')
                            candidate = t2_fn(intermediate, train_pairs)

                            if candidate is not None:
                                score = self._score(candidate, train_pairs)
                                if score > best_score:
                                    best_solution = candidate
                                    best_score = score

                                    if score >= self.time_allocation:
                                        return best_solution, best_score

                        except:
                            continue

                except:
                    continue

        # Phase 3: 3-step compositions (if search_depth >= 3)
        if self.search_depth >= 3 and time.time() < deadline:
            # Try a few 3-step combinations (very expensive)
            for t1_name in self.transform_set[:2]:
                if time.time() >= deadline:
                    break
                try:
                    t1_fn = getattr(self, f'_transform_{t1_name}')
                    int1 = t1_fn(test_input, train_pairs)
                    if int1 is None:
                        continue

                    for t2_name in self.transform_set[:2]:
                        if time.time() >= deadline:
                            break
                        try:
                            t2_fn = getattr(self, f'_transform_{t2_name}')
                            int2 = t2_fn(int1, train_pairs)
                            if int2 is None:
                                continue

                            # Only try color_map as final step
                            if 'color_map' in self.transform_set:
                                candidate = self._transform_color_map(int2, train_pairs)
                                if candidate is not None:
                                    score = self._score(candidate, train_pairs)
                                    if score > best_score:
                                        best_solution = candidate
                                        best_score = score
                        except:
                            continue
                except:
                    continue

        return best_solution, best_score

    def _score(self, candidate: np.ndarray, train_pairs: List) -> float:
        """Score by checking shape consistency with training outputs."""
        if len(train_pairs) == 0:
            return 0.0

        # Check against training outputs
        matches = 0
        checks = min(len(train_pairs), self.validation_strictness)

        for inp, out in train_pairs[:checks]:
            if candidate.shape == out.shape:
                matches += 1

        return matches / checks

    # Transform implementations
    def _transform_identity(self, test_input, train_pairs):
        """Check if input == output."""
        checks = min(len(train_pairs), self.validation_strictness)
        for inp, out in train_pairs[:checks]:
            if not np.array_equal(inp, out):
                return None
        return test_input

    def _transform_flip_h(self, test_input, train_pairs):
        """Horizontal flip."""
        checks = min(len(train_pairs), self.validation_strictness)
        for inp, out in train_pairs[:checks]:
            if inp.shape != out.shape or not np.array_equal(np.flip(inp, axis=0), out):
                return None
        return np.flip(test_input, axis=0)

    def _transform_flip_v(self, test_input, train_pairs):
        """Vertical flip."""
        checks = min(len(train_pairs), self.validation_strictness)
        for inp, out in train_pairs[:checks]:
            if inp.shape != out.shape or not np.array_equal(np.flip(inp, axis=1), out):
                return None
        return np.flip(test_input, axis=1)

    def _transform_rot_90(self, test_input, train_pairs):
        """90° rotation."""
        checks = min(len(train_pairs), self.validation_strictness)
        for inp, out in train_pairs[:checks]:
            if not np.array_equal(np.rot90(inp, k=1), out):
                return None
        return np.rot90(test_input, k=1)

    def _transform_rot_180(self, test_input, train_pairs):
        """180° rotation."""
        checks = min(len(train_pairs), self.validation_strictness)
        for inp, out in train_pairs[:checks]:
            if inp.shape != out.shape or not np.array_equal(np.rot90(inp, k=2), out):
                return None
        return np.rot90(test_input, k=2)

    def _transform_rot_270(self, test_input, train_pairs):
        """270° rotation."""
        checks = min(len(train_pairs), self.validation_strictness)
        for inp, out in train_pairs[:checks]:
            if not np.array_equal(np.rot90(inp, k=3), out):
                return None
        return np.rot90(test_input, k=3)

    def _transform_transpose(self, test_input, train_pairs):
        """Transpose."""
        checks = min(len(train_pairs), self.validation_strictness)
        for inp, out in train_pairs[:checks]:
            if not np.array_equal(np.transpose(inp), out):
                return None
        return np.transpose(test_input)

    def _transform_color_map(self, test_input, train_pairs):
        """Color mapping."""
        color_map = {}
        checks = min(len(train_pairs), self.validation_strictness)

        for inp, out in train_pairs[:checks]:
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

    def _transform_fill(self, test_input, train_pairs):
        """Fill (simplified)."""
        return None


# =============================================================================
# ABLATION TEST RUNNER
# =============================================================================

def run_knob_ablation(num_tasks: int = 10, time_per_task: float = 10):
    """
    Test different knob configurations.

    Args:
        num_tasks: Number of tasks to test (default 10)
        time_per_task: Time per task in seconds (default 10)
    """

    print("=" * 80)
    print("ABLATION KNOB TUNER - Quick Diagnostic Testing")
    print("=" * 80)
    print(f"Tasks: {num_tasks}")
    print(f"Time per task: {time_per_task}s")
    print("=" * 80)

    # Load test data
    try:
        with open('arc-agi_training_challenges.json') as f:
            all_tasks = json.load(f)
    except FileNotFoundError:
        print("ERROR: arc-agi_training_challenges.json not found")
        return

    # Select tasks
    task_ids = list(all_tasks.keys())[:num_tasks]

    # Test configurations
    configs = [
        # BASELINE
        {
            'name': 'BASELINE (fast, underfit)',
            'search_depth': 1,
            'transform_set': KNOBS['transform_set']['minimal'],
            'validation_strictness': 1,
            'time_allocation': 0.6,
        },

        # STANDARD
        {
            'name': 'STANDARD (balanced)',
            'search_depth': 2,
            'transform_set': KNOBS['transform_set']['standard'],
            'validation_strictness': 2,
            'time_allocation': 0.6,
        },

        # AGGRESSIVE (high accuracy, slower)
        {
            'name': 'AGGRESSIVE (thorough, may overfit)',
            'search_depth': 2,
            'transform_set': KNOBS['transform_set']['comprehensive'],
            'validation_strictness': 999,
            'time_allocation': 0.99,
        },

        # DEPTH TEST
        {
            'name': 'DEPTH-3 (compositional)',
            'search_depth': 3,
            'transform_set': KNOBS['transform_set']['standard'],
            'validation_strictness': 2,
            'time_allocation': 0.6,
        },
    ]

    results = {}

    # Run each configuration
    for config in configs:
        config_name = config['name']
        print(f"\n{'='*80}")
        print(f"TESTING: {config_name}")
        print(f"{'='*80}")
        print(f"  search_depth: {config['search_depth']}")
        print(f"  transforms: {len(config['transform_set'])}")
        print(f"  validation_strictness: {config['validation_strictness']}")
        print(f"  time_allocation: {config['time_allocation']}")
        print()

        solver = TunableSolver(config)

        scores = []
        times = []
        perfect = 0

        for i, task_id in enumerate(task_ids):
            task = all_tasks[task_id]
            train_pairs = [(np.array(p['input']), np.array(p['output']))
                          for p in task['train']]
            test_input = np.array(task['test'][0]['input'])

            start = time.time()
            solution, score = solver.solve(train_pairs, test_input, time_per_task)
            elapsed = time.time() - start

            scores.append(score)
            times.append(elapsed)

            if score >= 0.999:
                perfect += 1

            if (i + 1) % 5 == 0:
                avg_score = np.mean(scores)
                avg_time = np.mean(times)
                print(f"  Progress {i+1}/{num_tasks}: avg_score={avg_score:.1%}, avg_time={avg_time:.2f}s, perfect={perfect}")

        # Final stats
        avg_score = np.mean(scores)
        avg_time = np.mean(times)

        results[config_name] = {
            'avg_score': avg_score,
            'avg_time': avg_time,
            'perfect': perfect,
            'config': config
        }

        print(f"\n  ✓ {config_name}:")
        print(f"    Avg score: {avg_score:.1%}")
        print(f"    Avg time: {avg_time:.2f}s")
        print(f"    Perfect: {perfect}/{num_tasks}")

    # Compare results
    print(f"\n{'='*80}")
    print("COMPARISON")
    print(f"{'='*80}")

    sorted_results = sorted(results.items(), key=lambda x: x[1]['avg_score'], reverse=True)

    print(f"\n{'Configuration':<40} {'Score':<10} {'Time':<10} {'Perfect':<10}")
    print("-" * 80)
    for name, result in sorted_results:
        print(f"{name:<40} {result['avg_score']:<10.1%} {result['avg_time']:<10.2f}s {result['perfect']}/{num_tasks}")

    # Recommendation
    winner = sorted_results[0]
    print(f"\n{'='*80}")
    print(f"🏆 WINNER: {winner[0]}")
    print(f"{'='*80}")
    print(f"Score: {winner[1]['avg_score']:.1%}")
    print(f"Time: {winner[1]['avg_time']:.2f}s per task")
    print(f"Perfect: {winner[1]['perfect']}/{num_tasks}")
    print(f"\n✅ RECOMMENDATION: Use these knob settings for TurboOrcav2")

    # Save results
    save_data = {
        'num_tasks': num_tasks,
        'time_per_task': time_per_task,
        'results': {k: {**v, 'config': {**v['config'], 'transform_set': list(v['config']['transform_set'])}}
                   for k, v in results.items()}
    }

    with open('ablation_knob_results.json', 'w') as f:
        json.dump(save_data, f, indent=2)

    print(f"\n💾 Results saved to: ablation_knob_results.json")


if __name__ == '__main__':
    # Quick diagnostic: 10 tasks × 10 seconds = 100 seconds total (~2 minutes)
    run_knob_ablation(num_tasks=10, time_per_task=10)
