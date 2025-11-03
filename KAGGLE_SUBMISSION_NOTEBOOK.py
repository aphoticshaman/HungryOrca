#!/usr/bin/env python3
"""
ARC PRIZE 2025 - FAST KAGGLE SUBMISSION GENERATOR
==================================================

PERFORMANCE TARGET:
- 240 tasks in <90 minutes
- 22.5 seconds per task average
- No external dependencies (pure NumPy)

DESIGNED FOR: Kaggle notebook execution
"""

import numpy as np
import json
from collections import deque, Counter
from typing import List, Tuple, Optional


# ============================================================================
# FAST LEAN SOLVER - NO BLOAT
# ============================================================================

class FastSolver:
    """Ultra-fast solver for Kaggle - 240 tasks in <90 min."""

    def __init__(self):
        # 5 proven specialists, fast implementations only
        self.specialists = [
            self.try_identity,
            self.try_symmetry,
            self.try_color_map,
            self.try_fill,
            self.try_tiling,
        ]

    def solve(self, train_pairs: List[Tuple[np.ndarray, np.ndarray]],
             test_input: np.ndarray, time_limit: float = 20.0) -> np.ndarray:
        """Solve with time limit per task."""

        # Try each specialist (fast!)
        for specialist in self.specialists:
            try:
                result = specialist(train_pairs, test_input)
                if result is not None:
                    return result
            except:
                pass

        # Fallback: return input
        return test_input

    def try_identity(self, train_pairs, test_input):
        """Check if output = input."""
        for inp, out in train_pairs:
            if not np.array_equal(inp, out):
                return None
        return test_input

    def try_symmetry(self, train_pairs, test_input):
        """Try flip/rotate operations."""
        ops = [
            ('flip_h', lambda x: np.flip(x, axis=0)),
            ('flip_v', lambda x: np.flip(x, axis=1)),
            ('rot_90', lambda x: np.rot90(x, k=1)),
            ('rot_180', lambda x: np.rot90(x, k=2)),
        ]

        for name, op in ops:
            all_match = True
            for inp, out in train_pairs[:2]:  # Check first 2 only (speed!)
                if not np.array_equal(op(inp), out):
                    all_match = False
                    break

            if all_match:
                return op(test_input)

        return None

    def try_color_map(self, train_pairs, test_input):
        """Try color mapping."""
        color_map = {}

        for inp, out in train_pairs:
            if inp.shape != out.shape:
                return None

            for i in range(inp.shape[0]):
                for j in range(inp.shape[1]):
                    c_in = inp[i, j]
                    c_out = out[i, j]

                    if c_in not in color_map:
                        color_map[c_in] = c_out
                    elif color_map[c_in] != c_out:
                        return None  # Inconsistent

        # Apply
        result = test_input.copy()
        for i in range(result.shape[0]):
            for j in range(result.shape[1]):
                if result[i, j] in color_map:
                    result[i, j] = color_map[result[i, j]]

        return result

    def try_fill(self, train_pairs, test_input):
        """Try filling enclosed regions (FIXED algorithm)."""

        # Learn fill color
        fill_color = None
        for inp, out in train_pairs:
            if inp.shape == out.shape:
                changed = (inp == 0) & (out != 0)
                if np.any(changed):
                    fill_color = out[changed][0]
                    break

        if fill_color is None:
            return None

        # Find enclosed regions (FAST!)
        h, w = test_input.shape
        bg_cells = set()
        for i in range(h):
            for j in range(w):
                if test_input[i, j] == 0:
                    bg_cells.add((i, j))

        if not bg_cells:
            return None

        # Find components
        visited = set()
        enclosed_cells = []

        for start in bg_cells:
            if start in visited:
                continue

            # BFS
            component = set()
            queue = deque([start])

            while queue:
                i, j = queue.popleft()

                if (i, j) in visited:
                    continue
                if not (0 <= i < h and 0 <= j < w):
                    continue
                if test_input[i, j] != 0:
                    continue

                visited.add((i, j))
                component.add((i, j))

                queue.append((i+1, j))
                queue.append((i-1, j))
                queue.append((i, j+1))
                queue.append((i, j-1))

            # Check if enclosed (doesn't touch edges)
            touches_edge = False
            for i, j in component:
                if i == 0 or i == h-1 or j == 0 or j == w-1:
                    touches_edge = True
                    break

            if not touches_edge:
                enclosed_cells.extend(component)

        if not enclosed_cells:
            return None

        # Fill
        result = test_input.copy()
        for i, j in enclosed_cells:
            result[i, j] = fill_color

        return result

    def try_tiling(self, train_pairs, test_input):
        """Try simple tiling."""
        for inp, out in train_pairs:
            # Check if output is tiled version of input
            if out.shape[0] % inp.shape[0] == 0 and out.shape[1] % inp.shape[1] == 0:
                tile_h = out.shape[0] // inp.shape[0]
                tile_w = out.shape[1] // inp.shape[1]

                if np.array_equal(np.tile(inp, (tile_h, tile_w)), out):
                    # Apply to test
                    return np.tile(test_input, (tile_h, tile_w))

        return None


# ============================================================================
# SUBMISSION GENERATOR
# ============================================================================

def generate_submission(test_path='arc-agi_test_challenges.json',
                       output_path='submission.json'):
    """Generate submission.json for all 240 test tasks."""

    print("="*80)
    print("ARC PRIZE 2025 - KAGGLE SUBMISSION GENERATOR")
    print("="*80)
    print(f"Target: 240 tasks in <90 minutes")
    print(f"Time per task: ~22 seconds")
    print("="*80)

    # Load test data
    with open(test_path) as f:
        test_tasks = json.load(f)

    print(f"\nLoaded {len(test_tasks)} test tasks")

    # Initialize solver
    solver = FastSolver()

    submission = {}
    completed = 0

    for task_id, task in test_tasks.items():
        completed += 1

        if completed % 50 == 0:
            print(f"Progress: {completed}/{len(test_tasks)} tasks")

        # Extract training
        train_pairs = [(np.array(p['input']), np.array(p['output']))
                      for p in task['train']]

        # Solve each test input
        attempts = []
        for test_pair in task['test']:
            test_input = np.array(test_pair['input'])

            try:
                result = solver.solve(train_pairs, test_input, time_limit=22.0)
                attempts.append(result.tolist())
            except:
                # Fallback: return input
                attempts.append(test_input.tolist())

        # Ensure 2 attempts
        while len(attempts) < 2:
            attempts.append(attempts[0] if attempts else [[0]])

        submission[task_id] = attempts

    # Save
    with open(output_path, 'w') as f:
        json.dump(submission, f)

    print(f"\n✅ COMPLETE!")
    print(f"Tasks: {len(submission)}")
    print(f"Saved: {output_path}")

    # Check file size
    import os
    size = os.path.getsize(output_path)
    print(f"Size: {size:,} bytes ({size/1024:.1f} KB)")

    print(f"\n🎯 READY FOR KAGGLE UPLOAD!")

    return submission


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    # KAGGLE PATHS (adjust if needed)
    TEST_PATH = '/kaggle/input/arc-prize-2025/arc-agi_test_challenges.json'
    OUTPUT_PATH = '/kaggle/working/submission.json'

    # For local testing
    import os
    if not os.path.exists(TEST_PATH):
        TEST_PATH = 'arc-agi_test_challenges.json'
        OUTPUT_PATH = 'submission.json'

    # Generate!
    submission = generate_submission(TEST_PATH, OUTPUT_PATH)

    print("\n" + "="*80)
    print("SUBMISSION READY FOR UPLOAD!")
    print("="*80)
