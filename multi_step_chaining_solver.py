#!/usr/bin/env python3
"""
MULTI-STEP TRANSFORMATION CHAINING

Top priority from meta-analysis: Try composition of transforms
(e.g., rotate THEN color map, flip THEN fill)

TUNABLE KNOBS:
- max_chain_length: How many transforms to chain (1-3)
- min_confidence: Threshold for keeping result (0.5-0.9)
- num_attempts: How many chains to try (10-100)

Test 3 configurations:
1. UNDERFIT: Conservative (short chains, high confidence)
2. OVERFIT: Aggressive (long chains, low confidence, many attempts)
3. HYBRID: Balanced
"""

import numpy as np
import json
from typing import List, Tuple, Optional, Callable
from itertools import combinations


class MultiStepChainingSolver:
    """
    Solver that tries multi-step transformation chains.

    Knobs:
    - max_chain_length: 1-3 (1=single transform, 3=triple composition)
    - min_confidence: 0.5-0.9 (threshold to accept result)
    - num_attempts: 10-100 (how many chains to try)
    """

    def __init__(self,
                 max_chain_length: int = 2,
                 min_confidence: float = 0.7,
                 num_attempts: int = 30,
                 verbose: bool = False):
        self.max_chain_length = max_chain_length
        self.min_confidence = min_confidence
        self.num_attempts = num_attempts
        self.verbose = verbose

        # Atomic transforms to chain
        self.transforms = {
            'identity': self._identity,
            'flip_h': self._flip_h,
            'flip_v': self._flip_v,
            'rotate_90': self._rotate_90,
            'rotate_180': self._rotate_180,
            'rotate_270': self._rotate_270,
            'transpose': self._transpose,
            'color_map': self._color_map,
            'fill_interior': self._fill_interior,
            'remove_grid': self._remove_grid,
            'tile_2x2': self._tile_2x2,
            'crop_to_content': self._crop_to_content,
        }

    def solve(self, train_pairs: List[Tuple[np.ndarray, np.ndarray]],
              test_input: np.ndarray) -> Optional[np.ndarray]:
        """Solve by trying multi-step transformation chains."""

        if self.verbose:
            print(f"\n🔗 Multi-Step Chaining Solver")
            print(f"   Config: max_chain={self.max_chain_length}, "
                  f"min_conf={self.min_confidence}, attempts={self.num_attempts}")

        best_result = None
        best_score = 0.0
        best_chain = []

        attempts = 0

        # Try chains of increasing length
        for chain_length in range(1, self.max_chain_length + 1):
            if attempts >= self.num_attempts:
                break

            # Try all combinations of transforms
            transform_names = list(self.transforms.keys())

            for chain in self._generate_chains(transform_names, chain_length):
                if attempts >= self.num_attempts:
                    break

                attempts += 1

                # Apply chain to training examples
                score = self._evaluate_chain(chain, train_pairs)

                if score >= self.min_confidence:
                    # Apply to test input
                    result = self._apply_chain(chain, test_input, train_pairs)

                    if result is not None and score > best_score:
                        best_result = result
                        best_score = score
                        best_chain = chain

                        if self.verbose:
                            print(f"   ✓ Found chain: {' → '.join(chain)} (score: {score:.2f})")

        if self.verbose and best_result is not None:
            print(f"   🎯 Best: {' → '.join(best_chain)} ({best_score:.2f})")

        return best_result

    def _generate_chains(self, transform_names: List[str], length: int):
        """Generate chains of given length."""
        if length == 1:
            for name in transform_names:
                yield [name]
        else:
            # Generate combinations with replacement
            import itertools
            for chain in itertools.product(transform_names, repeat=length):
                yield list(chain)

    def _evaluate_chain(self, chain: List[str], train_pairs: List[Tuple]) -> float:
        """Evaluate how well chain works on training examples."""

        scores = []

        for inp, out in train_pairs[:2]:  # Check first 2 for speed
            result = self._apply_chain(chain, inp, train_pairs)

            if result is None:
                scores.append(0.0)
            elif result.shape == out.shape:
                matches = np.sum(result == out)
                score = matches / out.size
                scores.append(score)
            else:
                scores.append(0.0)

        return np.mean(scores) if scores else 0.0

    def _apply_chain(self, chain: List[str], grid: np.ndarray,
                     train_pairs: List[Tuple]) -> Optional[np.ndarray]:
        """Apply sequence of transforms."""

        result = grid.copy()

        for transform_name in chain:
            transform_fn = self.transforms[transform_name]

            try:
                # Some transforms need training data (color_map, fill_interior)
                if transform_name in ['color_map', 'fill_interior']:
                    result = transform_fn(result, train_pairs)
                else:
                    result = transform_fn(result)

                if result is None:
                    return None

            except Exception:
                return None

        return result

    # ========================================================================
    # ATOMIC TRANSFORMS
    # ========================================================================

    def _identity(self, grid: np.ndarray) -> np.ndarray:
        return grid

    def _flip_h(self, grid: np.ndarray) -> np.ndarray:
        return np.flip(grid, axis=0)

    def _flip_v(self, grid: np.ndarray) -> np.ndarray:
        return np.flip(grid, axis=1)

    def _rotate_90(self, grid: np.ndarray) -> np.ndarray:
        return np.rot90(grid, k=1)

    def _rotate_180(self, grid: np.ndarray) -> np.ndarray:
        return np.rot90(grid, k=2)

    def _rotate_270(self, grid: np.ndarray) -> np.ndarray:
        return np.rot90(grid, k=3)

    def _transpose(self, grid: np.ndarray) -> np.ndarray:
        if grid.shape[0] == grid.shape[1]:
            return grid.T
        return grid

    def _color_map(self, grid: np.ndarray, train_pairs: List[Tuple]) -> np.ndarray:
        """Learn and apply color mapping from training."""

        # Learn color mapping
        color_map = {}

        for inp, out in train_pairs:
            if inp.shape == out.shape:
                for i in range(inp.shape[0]):
                    for j in range(inp.shape[1]):
                        c_in = inp[i, j]
                        c_out = out[i, j]

                        if c_in not in color_map:
                            color_map[c_in] = c_out
                        elif color_map[c_in] != c_out:
                            return grid  # Inconsistent mapping

        # Apply mapping
        result = grid.copy()
        for i in range(result.shape[0]):
            for j in range(result.shape[1]):
                if result[i, j] in color_map:
                    result[i, j] = color_map[result[i, j]]

        return result

    def _fill_interior(self, grid: np.ndarray, train_pairs: List[Tuple]) -> np.ndarray:
        """Fill interior regions."""

        # Find what color to fill with from training
        fill_color = None

        for inp, out in train_pairs:
            if inp.shape == out.shape:
                diff = out != inp
                if np.any(diff):
                    fill_color = out[diff][0]
                    break

        if fill_color is None:
            return grid

        # Find interior cells
        exterior = set()
        h, w = grid.shape
        stack = []

        # Edge cells
        for i in range(h):
            stack.extend([(i, 0), (i, w-1)])
        for j in range(w):
            stack.extend([(0, j), (h-1, j)])

        # Flood fill from edges
        while stack:
            i, j = stack.pop()
            if (i, j) in exterior or not (0 <= i < h and 0 <= j < w):
                continue

            exterior.add((i, j))

            for di, dj in [(0,1), (1,0), (0,-1), (-1,0)]:
                stack.append((i+di, j+dj))

        # Fill interior
        result = grid.copy()
        for i in range(h):
            for j in range(w):
                if (i, j) not in exterior:
                    result[i, j] = fill_color

        return result

    def _remove_grid(self, grid: np.ndarray) -> np.ndarray:
        """Remove most common color (grid lines)."""

        from collections import Counter
        colors = Counter(grid.flatten())

        if len(colors) > 1:
            grid_color = colors.most_common(1)[0][0]
            result = grid.copy()
            result[result == grid_color] = 0
            return result

        return grid

    def _tile_2x2(self, grid: np.ndarray) -> np.ndarray:
        """Tile input 2x2."""
        return np.tile(grid, (2, 2))

    def _crop_to_content(self, grid: np.ndarray) -> np.ndarray:
        """Crop to bounding box of non-zero content."""

        nonzero = np.argwhere(grid != 0)

        if len(nonzero) == 0:
            return grid

        min_row = nonzero[:, 0].min()
        max_row = nonzero[:, 0].max()
        min_col = nonzero[:, 1].min()
        max_col = nonzero[:, 1].max()

        return grid[min_row:max_row+1, min_col:max_col+1]


def test_configurations():
    """Test 3 knob configurations: underfit, overfit, hybrid."""

    print(f"{'='*80}")
    print(f"🎛️ MULTI-STEP CHAINING: PARAMETER TUNING TEST")
    print(f"{'='*80}\n")

    # Load data
    with open('arc-agi_training_challenges.json') as f:
        challenges = json.load(f)

    with open('arc-agi_training_solutions.json') as f:
        solutions = json.load(f)

    task_ids = list(challenges.keys())[:10]

    # 3 Configurations
    configs = {
        'UNDERFIT': {
            'max_chain_length': 1,  # Single transforms only
            'min_confidence': 0.9,  # Very high bar
            'num_attempts': 10,     # Few attempts
        },
        'HYBRID': {
            'max_chain_length': 2,  # 2-step chains
            'min_confidence': 0.7,  # Balanced
            'num_attempts': 30,     # Moderate
        },
        'OVERFIT': {
            'max_chain_length': 3,  # 3-step chains
            'min_confidence': 0.5,  # Low bar (try more)
            'num_attempts': 50,     # Many attempts
        },
    }

    results_by_config = {}

    for config_name, params in configs.items():
        print(f"\n{'='*80}")
        print(f"🎛️ Configuration: {config_name}")
        print(f"{'='*80}")
        print(f"Parameters:")
        for k, v in params.items():
            print(f"  {k}: {v}")
        print()

        solver = MultiStepChainingSolver(**params, verbose=False)

        results = []

        for task_id in task_ids:
            task = challenges[task_id]

            train_pairs = [(np.array(p['input']), np.array(p['output']))
                          for p in task['train']]
            test_input = np.array(task['test'][0]['input'])
            test_output = np.array(solutions[task_id][0])

            # Solve
            result = solver.solve(train_pairs, test_input)

            # Score
            if result is None:
                score = 0.0
            elif result.shape != test_output.shape:
                score = 0.0
            else:
                matches = np.sum(result == test_output)
                score = matches / test_output.size

            exact = (score == 1.0)

            results.append({
                'task_id': task_id,
                'score': float(score),
                'exact': bool(exact),
            })

            status = "✅" if exact else f"{score*100:.1f}%"
            print(f"  {task_id}: {status}")

        # Summary
        avg_score = np.mean([r['score'] for r in results])
        exact_count = sum(r['exact'] for r in results)

        print(f"\n📊 {config_name} Results:")
        print(f"   Average: {avg_score*100:.1f}%")
        print(f"   Exact: {exact_count}/{len(results)}")

        results_by_config[config_name] = {
            'params': params,
            'avg_score': float(avg_score),
            'exact_count': exact_count,
            'results': results,
        }

    # Compare
    print(f"\n\n{'='*80}")
    print(f"📊 CONFIGURATION COMPARISON")
    print(f"{'='*80}\n")

    for name, data in results_by_config.items():
        print(f"{name:12s}: {data['avg_score']*100:5.1f}% avg, {data['exact_count']} exact")

    # Determine winner
    best_config = max(results_by_config.items(), key=lambda x: x[1]['avg_score'])

    print(f"\n🏆 Winner: {best_config[0]} ({best_config[1]['avg_score']*100:.1f}%)")

    # Save
    with open('multi_step_chaining_results.json', 'w') as f:
        json.dump(results_by_config, f, indent=2)

    print(f"\n💾 Results saved to: multi_step_chaining_results.json")

    return results_by_config


if __name__ == '__main__':
    test_configurations()
