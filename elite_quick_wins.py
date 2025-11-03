#!/usr/bin/env python3
"""
ELITE MODE INSIGHTS - PHASE 1 QUICK WINS

Implement 3 highest-ROI, lowest-complexity insights:
1. Structure Preservation (Insight #7): +5-7% - Filter transforms that break structure
2. Iterative Dynamics (Insight #4): +2-4% - Test CA and iterated functions
3. Weighted Ensemble (Insight #6): +4-6% - Superposition resolution

Expected total: +11-17%
Current: 53.9% → Target: 60-70%+
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from collections import Counter
import json


# ============================================================================
# INSIGHT #7: STRUCTURE PRESERVATION
# ============================================================================

def label_components(mask: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    Label connected components in binary mask.

    Returns: (labeled_array, num_components)
    """
    from collections import deque

    h, w = mask.shape
    labeled = np.zeros_like(mask, dtype=int)
    component_id = 0

    for i in range(h):
        for j in range(w):
            if mask[i, j] and labeled[i, j] == 0:
                # Start new component
                component_id += 1
                queue = deque([(i, j)])

                while queue:
                    ci, cj = queue.popleft()

                    if not (0 <= ci < h and 0 <= cj < w):
                        continue

                    if not mask[ci, cj] or labeled[ci, cj] > 0:
                        continue

                    labeled[ci, cj] = component_id

                    # 4-connected neighbors
                    queue.append((ci+1, cj))
                    queue.append((ci-1, cj))
                    queue.append((ci, cj+1))
                    queue.append((ci, cj-1))

    return labeled, component_id


def compute_structural_properties(grid: np.ndarray) -> Dict:
    """
    Extract structural properties that transformations should preserve.

    Properties:
    - num_components: Connected components
    - num_colors: Unique colors
    - max_component_size: Largest connected component
    - has_holes: Whether topology has holes
    - symmetry: Horizontal/vertical symmetry
    """

    h, w = grid.shape

    # Connected components (non-zero) - manual implementation
    labeled, num_components = label_components(grid != 0)

    # Component sizes
    component_sizes = [np.sum(labeled == i) for i in range(1, num_components + 1)]
    max_component_size = max(component_sizes) if component_sizes else 0

    # Colors
    num_colors = len(np.unique(grid))

    # Symmetry
    h_symmetric = np.array_equal(grid, np.flip(grid, axis=0))
    v_symmetric = np.array_equal(grid, np.flip(grid, axis=1))

    # Holes (Euler characteristic approximation)
    # β₀ - β₁ + β₂ = χ
    # For 2D grids: χ = V - E + F ≈ components - holes
    # Simple approximation: enclosed background regions
    has_holes = detect_holes_simple(grid)

    return {
        'num_components': num_components,
        'num_colors': num_colors,
        'max_component_size': max_component_size,
        'has_holes': has_holes,
        'h_symmetric': h_symmetric,
        'v_symmetric': v_symmetric,
        'shape': grid.shape,
    }


def detect_holes_simple(grid: np.ndarray) -> bool:
    """Simple hole detection: enclosed background regions."""

    # Label background components
    bg_mask = (grid == 0)
    labeled_bg, num_bg = label_components(bg_mask)

    if num_bg <= 1:
        return False

    # Check if any component doesn't touch edges
    h, w = grid.shape
    for comp_id in range(1, num_bg + 1):
        comp_mask = (labeled_bg == comp_id)

        # Check edges
        touches_edge = (
            np.any(comp_mask[0, :]) or
            np.any(comp_mask[-1, :]) or
            np.any(comp_mask[:, 0]) or
            np.any(comp_mask[:, -1])
        )

        if not touches_edge:
            return True  # Found enclosed region (hole)

    return False


def preserves_structure(input_props: Dict, output_props: Dict,
                       tolerance: float = 0.2) -> bool:
    """
    Check if transformation preserves key structural properties.

    Returns True if output respects input structure.
    """

    # Must preserve component count (or within tolerance)
    if output_props['num_components'] > input_props['num_components'] * (1 + tolerance):
        return False

    # Color count shouldn't explode
    if output_props['num_colors'] > input_props['num_colors'] * 2:
        return False

    # If input has holes, output should too (topology preservation)
    if input_props['has_holes'] and not output_props['has_holes']:
        return False

    # If input is symmetric, consider preserving it (soft constraint)
    # Don't enforce, but prefer

    return True


# ============================================================================
# INSIGHT #4: ITERATIVE DYNAMICS (CELLULAR AUTOMATA)
# ============================================================================

def manual_convolve(grid: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Manual 2D convolution (simple implementation)."""
    h, w = grid.shape
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2

    # Pad with zeros
    padded = np.pad(grid, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
    result = np.zeros((h, w), dtype=grid.dtype)

    for i in range(h):
        for j in range(w):
            result[i, j] = np.sum(padded[i:i+kh, j:j+kw] * kernel)

    return result


def apply_ca_game_of_life(grid: np.ndarray) -> np.ndarray:
    """Apply one step of Conway's Game of Life."""

    # Count neighbors (3x3 kernel, exclude center)
    kernel = np.ones((3, 3), dtype=int)
    kernel[1, 1] = 0

    # Only consider non-zero as "alive"
    alive = (grid != 0).astype(int)
    neighbor_count = manual_convolve(alive, kernel)

    # Rules: alive with 2-3 neighbors stays alive, dead with 3 neighbors becomes alive
    new_alive = ((alive == 1) & ((neighbor_count == 2) | (neighbor_count == 3))) | \
                ((alive == 0) & (neighbor_count == 3))

    # Preserve colors from original
    result = np.where(new_alive, grid, 0)

    return result


def apply_ca_rule_90(grid: np.ndarray) -> np.ndarray:
    """Elementary CA Rule 90 (XOR with neighbors)."""
    result = grid.copy()

    h, w = grid.shape

    # Apply row by row
    for i in range(h):
        new_row = np.zeros(w, dtype=grid.dtype)
        for j in range(w):
            left = grid[i, j-1] if j > 0 else 0
            right = grid[i, j+1] if j < w-1 else 0
            new_row[j] = left ^ right  # XOR
        result[i] = new_row

    return result


def test_cellular_automata(input_grid: np.ndarray, output_grid: np.ndarray,
                          max_steps: int = 10) -> Optional[Tuple[str, int]]:
    """
    Test if output = CA^n(input) for some CA rule and n steps.

    Returns: (ca_name, n_steps) if found, None otherwise
    """

    ca_rules = {
        'game_of_life': apply_ca_game_of_life,
        'rule_90': apply_ca_rule_90,
    }

    for ca_name, ca_func in ca_rules.items():
        current = input_grid.copy()

        for n in range(1, max_steps + 1):
            current = ca_func(current)

            if np.array_equal(current, output_grid):
                return (ca_name, n)

    return None


def apply_iterated_function(grid: np.ndarray, func_name: str,
                            n_steps: int) -> np.ndarray:
    """Apply a function n times."""

    funcs = {
        'game_of_life': apply_ca_game_of_life,
        'rule_90': apply_ca_rule_90,
    }

    if func_name not in funcs:
        return grid

    result = grid.copy()
    for _ in range(n_steps):
        result = funcs[func_name](result)

    return result


# ============================================================================
# INSIGHT #6: WEIGHTED ENSEMBLE (SUPERPOSITION RESOLUTION)
# ============================================================================

def score_on_training(result_func, train_pairs: List[Tuple[np.ndarray, np.ndarray]]) -> float:
    """Score how well a transformation function works on training examples."""

    scores = []

    for inp, out in train_pairs:
        try:
            predicted = result_func(inp)

            if predicted is None or predicted.shape != out.shape:
                scores.append(0.0)
            else:
                match = np.sum(predicted == out) / out.size
                scores.append(match)
        except:
            scores.append(0.0)

    return np.mean(scores) if scores else 0.0


def weighted_ensemble_vote(results: List[np.ndarray], weights: np.ndarray) -> np.ndarray:
    """
    Weighted voting across multiple results.

    Each cell: vote for color with highest weighted support.
    """

    if not results or all(r is None for r in results):
        return None

    # Filter out None results
    valid_results = [(r, w) for r, w in zip(results, weights) if r is not None]

    if not valid_results:
        return None

    results, weights = zip(*valid_results)

    # Assume same shape (take first)
    shape = results[0].shape
    output = np.zeros(shape, dtype=results[0].dtype)

    # Vote cell by cell
    for i in range(shape[0]):
        for j in range(shape[1]):
            # Collect weighted votes for each color
            color_votes = {}

            for result, weight in zip(results, weights):
                if result.shape == shape:
                    color = result[i, j]
                    color_votes[color] = color_votes.get(color, 0) + weight

            # Winner takes cell
            if color_votes:
                winner = max(color_votes.items(), key=lambda x: x[1])[0]
                output[i, j] = winner

    return output


# ============================================================================
# ELITE-ENHANCED SOLVER
# ============================================================================

class EliteEnhancedSolver:
    """
    Solver enhanced with Elite Mode Phase 1 insights.

    Combines:
    - Existing evolving specialists
    - Structure preservation filtering
    - CA/iterated dynamics detection
    - Weighted ensemble voting
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def solve(self, train_pairs: List[Tuple[np.ndarray, np.ndarray]],
             test_input: np.ndarray) -> Optional[np.ndarray]:
        """Enhanced solving with Elite insights."""

        if self.verbose:
            print(f"\n🔬 ELITE-ENHANCED SOLVER")

        # Compute input structure
        input_props = compute_structural_properties(test_input)

        if self.verbose:
            print(f"   Input structure: {input_props['num_components']} components, "
                  f"{input_props['num_colors']} colors")

        # Generate multiple candidate solutions
        candidates = []

        # 1. Check for CA/iterative dynamics (Insight #4)
        if len(train_pairs) > 0:
            ca_result = test_cellular_automata(train_pairs[0][0], train_pairs[0][1])

            if ca_result:
                ca_name, n_steps = ca_result
                result = apply_iterated_function(test_input, ca_name, n_steps)
                candidates.append(('ca_' + ca_name, result, 0.95))  # High confidence for CA

                if self.verbose:
                    print(f"   ✓ CA detected: {ca_name} × {n_steps} steps")

        # 2. Try evolving specialists (existing system)
        from evolving_specialist_system import EvolvingSolver

        ev_solver = EvolvingSolver(time_limit=10.0, verbose=False)
        ev_result = ev_solver.solve(train_pairs, test_input)

        if ev_result is not None:
            # Score on training
            score = score_on_training(lambda x: ev_result if np.array_equal(x, test_input) else None,
                                     train_pairs)
            candidates.append(('evolving', ev_result, score))

            if self.verbose:
                print(f"   ✓ Evolving specialists: {score*100:.1f}% confidence")

        # 3. Structure-preserving transforms (Insight #7)
        # Filter candidates that break structure
        valid_candidates = []

        for name, result, score in candidates:
            if result is None:
                continue

            output_props = compute_structural_properties(result)

            if preserves_structure(input_props, output_props):
                valid_candidates.append((name, result, score))

                if self.verbose:
                    print(f"   ✓ {name}: preserves structure")
            else:
                if self.verbose:
                    print(f"   ✗ {name}: breaks structure (filtered)")

        if not valid_candidates:
            # If all filtered, use original candidates
            valid_candidates = [(n, r, s) for n, r, s in candidates if r is not None]

        if not valid_candidates:
            return None

        # 4. Weighted ensemble (Insight #6)
        if len(valid_candidates) == 1:
            return valid_candidates[0][1]

        names, results, scores = zip(*valid_candidates)
        weights = np.array(scores)

        if weights.sum() == 0:
            weights = np.ones(len(weights))

        weights /= weights.sum()

        if self.verbose:
            print(f"   🎯 Ensemble: {len(valid_candidates)} candidates")
            for name, weight in zip(names, weights):
                print(f"      - {name}: {weight*100:.1f}%")

        final_result = weighted_ensemble_vote(list(results), weights)

        return final_result


def test_elite_solver():
    """Test Elite-enhanced solver on training tasks."""

    print(f"{'='*80}")
    print(f"🔬 TESTING ELITE-ENHANCED SOLVER (Phase 1 Quick Wins)")
    print(f"{'='*80}\n")

    # Load data
    with open('arc-agi_training_challenges.json') as f:
        challenges = json.load(f)

    with open('arc-agi_training_solutions.json') as f:
        solutions = json.load(f)

    task_ids = list(challenges.keys())[:10]

    solver = EliteEnhancedSolver(verbose=True)

    results = []

    for task_id in task_ids:
        print(f"\n{'='*80}")
        print(f"Task: {task_id}")
        print(f"{'='*80}")

        task = challenges[task_id]
        train_pairs = [(np.array(p['input']), np.array(p['output'])) for p in task['train']]
        test_input = np.array(task['test'][0]['input'])
        test_output = np.array(solutions[task_id][0])

        # Solve
        predicted = solver.solve(train_pairs, test_input)

        # Score
        if predicted is None:
            score = 0.0
            exact = False
        elif predicted.shape != test_output.shape:
            score = 0.0
            exact = False
        else:
            score = np.sum(predicted == test_output) / test_output.size
            exact = (score == 1.0)

        print(f"\n📊 Result: {score*100:.1f}% {'✅ EXACT' if exact else ''}")

        results.append({
            'task_id': task_id,
            'score': float(score),
            'exact': bool(exact),
        })

    # Summary
    print(f"\n\n{'='*80}")
    print(f"📊 ELITE-ENHANCED SOLVER RESULTS")
    print(f"{'='*80}\n")

    avg_score = np.mean([r['score'] for r in results])
    exact_count = sum(r['exact'] for r in results)

    print(f"Average: {avg_score*100:.1f}%")
    print(f"Exact matches: {exact_count}/{len(results)}")

    print(f"\nComparison:")
    print(f"  Baseline (before fix):  51.9% avg, 1 exact")
    print(f"  After forensic fix:     53.9% avg, 2 exact")
    print(f"  Elite-enhanced:         {avg_score*100:.1f}% avg, {exact_count} exact")

    improvement = (avg_score * 100) - 53.9
    print(f"\n{'📈' if improvement > 0 else '📉'} Change from forensic: {improvement:+.1f}%")

    return results


if __name__ == '__main__':
    results = test_elite_solver()
