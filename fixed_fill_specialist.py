#!/usr/bin/env python3
"""
FIXED FILL SPECIALIST

Root cause identified:
- Current flood-fill FROM EDGES misses enclosed regions
- Testing shows SAME cells missed regardless of fill color
- Validation shows only 94-99% match even on TRAINING data

NEW APPROACH:
Instead of flood-fill from edges to find exterior,
directly identify enclosed regions by checking connectivity.
"""

import numpy as np
from collections import deque
from typing import List, Tuple, Set


def find_enclosed_regions_improved(grid: np.ndarray, boundary_color: int = None) -> List[Set[Tuple[int, int]]]:
    """
    Find enclosed regions using improved algorithm.

    Instead of flood-filling from edges, identify regions by:
    1. Find all connected components of background (0)
    2. Check which components are enclosed by boundaries
    3. Return list of enclosed region cells
    """

    h, w = grid.shape

    # Find all background cells
    bg_cells = set()
    for i in range(h):
        for j in range(w):
            if grid[i, j] == 0:
                bg_cells.add((i, j))

    if not bg_cells:
        return []

    # Find connected components of background
    components = []
    visited = set()

    for start_cell in bg_cells:
        if start_cell in visited:
            continue

        # BFS to find component
        component = set()
        queue = deque([start_cell])

        while queue:
            i, j = queue.popleft()

            if (i, j) in visited:
                continue

            if not (0 <= i < h and 0 <= j < w):
                continue

            if grid[i, j] != 0:
                continue

            visited.add((i, j))
            component.add((i, j))

            # 4-connected neighbors
            for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                queue.append((i + di, j + dj))

        components.append(component)

    # Determine which components are enclosed
    # A component is enclosed if NO cell touches the edge
    enclosed = []

    for component in components:
        touches_edge = False

        for i, j in component:
            if i == 0 or i == h - 1 or j == 0 or j == w - 1:
                touches_edge = True
                break

        if not touches_edge:
            # This component is enclosed!
            enclosed.append(component)

    return enclosed


def fill_enclosed_regions(grid: np.ndarray, fill_color: int) -> np.ndarray:
    """Fill all enclosed regions with given color."""

    result = grid.copy()

    enclosed_regions = find_enclosed_regions_improved(grid)

    for region in enclosed_regions:
        for i, j in region:
            result[i, j] = fill_color

    return result


def learn_fill_color_from_training(train_pairs: List[Tuple[np.ndarray, np.ndarray]]) -> int:
    """Learn what color to fill with from training examples."""

    # Look at all training examples
    fill_colors = []

    for inp, out in train_pairs:
        if inp.shape != out.shape:
            continue

        # Find cells that changed from 0 to non-zero
        changed = (inp == 0) & (out != 0)

        if np.any(changed):
            # What color were they filled with?
            filled_colors = out[changed]
            if len(filled_colors) > 0:
                # Most common fill color
                from collections import Counter
                color = Counter(filled_colors).most_common(1)[0][0]
                fill_colors.append(int(color))

    # Return most common across all examples
    if fill_colors:
        from collections import Counter
        return Counter(fill_colors).most_common(1)[0][0]

    # Default
    return 1


def solve_with_fixed_fill(train_pairs: List[Tuple[np.ndarray, np.ndarray]],
                          test_input: np.ndarray) -> np.ndarray:
    """Solve using fixed fill algorithm."""

    # Learn fill color from ALL training examples
    fill_color = learn_fill_color_from_training(train_pairs)

    print(f"Learned fill color: {fill_color}")

    # Validate on training examples
    print(f"\nValidating on training examples:")
    for idx, (inp, out) in enumerate(train_pairs):
        if inp.shape != out.shape:
            continue

        predicted = fill_enclosed_regions(inp, fill_color)

        match = np.array_equal(predicted, out)
        if not match:
            score = np.sum(predicted == out) / out.size
            print(f"  Training {idx+1}: {score*100:.1f}% {'✅ EXACT' if match else '❌'}")
        else:
            print(f"  Training {idx+1}: 100.0% ✅ EXACT")

    # Apply to test
    result = fill_enclosed_regions(test_input, fill_color)

    return result


def test_fixed_fill():
    """Test fixed fill on problematic task."""

    import json

    print(f"{'='*80}")
    print(f"🔧 TESTING FIXED FILL ALGORITHM")
    print(f"{'='*80}\n")

    # Load task 00d62c1b (the 91.8% task)
    with open('arc-agi_training_challenges.json') as f:
        challenges = json.load(f)

    with open('arc-agi_training_solutions.json') as f:
        solutions = json.load(f)

    task_id = '00d62c1b'
    task = challenges[task_id]

    train_pairs = [(np.array(p['input']), np.array(p['output'])) for p in task['train']]
    test_input = np.array(task['test'][0]['input'])
    test_output = np.array(solutions[task_id][0])

    print(f"Task: {task_id}")
    print(f"Training examples: {len(train_pairs)}")
    print(f"Test input shape: {test_input.shape}")

    # Solve with fixed algorithm
    predicted = solve_with_fixed_fill(train_pairs, test_input)

    # Check result
    exact_match = np.array_equal(predicted, test_output)
    if not exact_match:
        score = np.sum(predicted == test_output) / test_output.size
        print(f"\n📊 Test Result: {score*100:.1f}% {'✅ EXACT!' if exact_match else '❌'}")
    else:
        print(f"\n📊 Test Result: 100.0% ✅ EXACT MATCH!")

    if not exact_match:
        # Show differences
        diff_mask = (predicted != test_output)
        num_diff = np.sum(diff_mask)
        print(f"\nDifferences: {num_diff} cells")

        if num_diff > 0:
            diff_positions = np.argwhere(diff_mask)
            print(f"First 10 differences:")
            for pos in diff_positions[:10]:
                i, j = pos
                print(f"  ({i},{j}): predicted={predicted[i,j]}, actual={test_output[i,j]}")

    return exact_match


if __name__ == '__main__':
    success = test_fixed_fill()

    if success:
        print(f"\n{'='*80}")
        print(f"✅ FIX SUCCESSFUL - EXACT MATCH ACHIEVED!")
        print(f"{'='*80}")
    else:
        print(f"\n{'='*80}")
        print(f"⚠️ Still not exact - need further investigation")
        print(f"{'='*80}")
