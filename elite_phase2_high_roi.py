#!/usr/bin/env python3
"""
ELITE MODE PHASE 2 - HIGHEST ROI INSIGHTS

From elite_mode_puzzles.py analysis:
1. Insight #1: Algebraic Colors (+5-8%) - Modular arithmetic detection
2. Insight #10: Fractal Compression (+8-12%) - Box-counting dimension

Combined expected: +13-20% improvement!
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
import json


# ============================================================================
# INSIGHT #1: ALGEBRAIC COLORS (Field/Group Operations)
# ============================================================================

def detect_modular_arithmetic(train_pairs: List[Tuple[np.ndarray, np.ndarray]],
                              max_modulus: int = 20) -> Optional[Dict]:
    """
    Detect if transformation is modular arithmetic on colors.

    Tests: output[i,j] = (a * input[i,j] + b) mod p

    Returns: {'type': 'affine', 'a': mult, 'b': add, 'p': modulus}
            or None if no pattern found
    """

    # Only works if shapes match
    if not all(inp.shape == out.shape for inp, out in train_pairs):
        return None

    # Try different moduli
    for p in range(3, max_modulus + 1):
        # Try affine transformations: output = (a * input + b) mod p
        for a in range(1, p):
            for b in range(p):
                # Test on all training pairs
                all_match = True

                for inp, out in train_pairs:
                    predicted = (inp * a + b) % p

                    if not np.array_equal(predicted, out):
                        all_match = False
                        break

                if all_match:
                    return {
                        'type': 'affine_mod',
                        'a': a,
                        'b': b,
                        'p': p,
                        'formula': f'(color × {a} + {b}) mod {p}'
                    }

    # Try additive: output = (input + b) mod p
    for p in range(3, max_modulus + 1):
        for b in range(p):
            all_match = True

            for inp, out in train_pairs:
                predicted = (inp + b) % p

                if not np.array_equal(predicted, out):
                    all_match = False
                    break

            if all_match:
                return {
                    'type': 'additive_mod',
                    'b': b,
                    'p': p,
                    'formula': f'(color + {b}) mod {p}'
                }

    # Try multiplicative: output = (input * a) mod p
    for p in range(3, max_modulus + 1):
        for a in range(1, p):
            all_match = True

            for inp, out in train_pairs:
                predicted = (inp * a) % p

                if not np.array_equal(predicted, out):
                    all_match = False
                    break

            if all_match:
                return {
                    'type': 'multiplicative_mod',
                    'a': a,
                    'p': p,
                    'formula': f'(color × {a}) mod {p}'
                }

    return None


def apply_algebraic_transform(grid: np.ndarray, transform: Dict) -> np.ndarray:
    """Apply detected algebraic transformation."""

    if transform['type'] == 'affine_mod':
        return (grid * transform['a'] + transform['b']) % transform['p']
    elif transform['type'] == 'additive_mod':
        return (grid + transform['b']) % transform['p']
    elif transform['type'] == 'multiplicative_mod':
        return (grid * transform['a']) % transform['p']
    else:
        return grid


# ============================================================================
# INSIGHT #10: FRACTAL COMPRESSION (Self-Similarity)
# ============================================================================

def compute_fractal_dimension(grid: np.ndarray, scales: List[int] = None) -> float:
    """
    Compute box-counting fractal dimension.

    D = log(N) / log(1/r)

    where N = number of boxes containing pattern at scale r
    """

    if scales is None:
        scales = [2, 4, 8, 16]

    # Convert to binary (foreground/background)
    binary = (grid != 0).astype(int)

    counts = []

    for scale in scales:
        # Partition into boxes of size scale × scale
        h, w = binary.shape
        boxes_h = h // scale
        boxes_w = w // scale

        occupied_boxes = 0

        for i in range(boxes_h):
            for j in range(boxes_w):
                box = binary[i*scale:(i+1)*scale, j*scale:(j+1)*scale]

                # Box is occupied if contains any foreground
                if np.any(box):
                    occupied_boxes += 1

        counts.append(occupied_boxes)

    # Linear fit on log-log plot
    # D = -slope of log(N) vs log(r)
    if len(scales) < 2 or all(c == 0 for c in counts):
        return 0.0

    log_scales = np.log(scales)
    log_counts = np.log([max(c, 1) for c in counts])  # Avoid log(0)

    # Polyfit: log(N) = -D * log(r) + constant
    coeffs = np.polyfit(log_scales, log_counts, 1)
    dimension = -coeffs[0]

    return max(0.0, dimension)  # Dimension should be non-negative


def detect_self_similarity(grid: np.ndarray, generator_sizes: List[int] = None) -> Optional[Dict]:
    """
    Detect if grid is self-similar (fractal).

    Find smallest repeating unit (generator) that produces grid recursively.
    """

    if generator_sizes is None:
        generator_sizes = [2, 3, 4, 5]

    h, w = grid.shape

    for size in generator_sizes:
        if h % size != 0 or w % size != 0:
            continue

        # Extract generator (top-left block)
        generator = grid[:size, :size]

        # Check if grid is tiled repetition of generator
        tiles_h = h // size
        tiles_w = w // size

        is_tiled = True

        for i in range(tiles_h):
            for j in range(tiles_w):
                tile = grid[i*size:(i+1)*size, j*size:(j+1)*size]

                if not np.array_equal(tile, generator):
                    is_tiled = False
                    break

            if not is_tiled:
                break

        if is_tiled:
            return {
                'type': 'simple_tiling',
                'generator_size': size,
                'generator': generator.tolist(),
                'repetitions': (tiles_h, tiles_w)
            }

    # Check for scaled self-similarity (Sierpinski-like)
    # Grid at scale 2 should look like generator pattern
    if h >= 4 and w >= 4:
        # Downsample by factor of 2
        downsampled = grid[::2, ::2]

        # Check if pattern similar to original
        if downsampled.shape[0] >= 2 and downsampled.shape[1] >= 2:
            # Compute similarity
            # For fractal, small part looks like whole
            corner = grid[:downsampled.shape[0], :downsampled.shape[1]]

            # Count matching cells
            matches = np.sum((downsampled != 0) == (corner != 0))
            total = downsampled.size

            similarity = matches / total

            if similarity > 0.8:
                return {
                    'type': 'scaled_self_similar',
                    'scale_factor': 2,
                    'similarity': float(similarity)
                }

    return None


def apply_fractal_pattern(generator: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """
    Generate large grid from small generator using tiling.
    """

    gen_h, gen_w = generator.shape
    target_h, target_w = target_size

    tiles_h = target_h // gen_h
    tiles_w = target_w // gen_w

    # Tile the generator
    result = np.tile(generator, (tiles_h, tiles_w))

    # Crop to exact target size
    return result[:target_h, :target_w]


# ============================================================================
# ELITE PHASE 2 SOLVER
# ============================================================================

class ElitePhase2Solver:
    """
    Solver with highest-ROI Elite insights:
    - Algebraic color operations (+5-8%)
    - Fractal compression (+8-12%)
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def solve(self, train_pairs: List[Tuple[np.ndarray, np.ndarray]],
             test_input: np.ndarray) -> Optional[np.ndarray]:
        """Solve with Elite Phase 2 insights."""

        if self.verbose:
            print(f"\n🔬 ELITE PHASE 2 SOLVER")

        # Try Insight #1: Algebraic Colors
        algebraic = detect_modular_arithmetic(train_pairs)

        if algebraic:
            result = apply_algebraic_transform(test_input, algebraic)

            if self.verbose:
                print(f"   ✓ Algebraic pattern detected: {algebraic['formula']}")

            return result

        # Try Insight #10: Fractal Compression
        # Check if output is fractal expansion of input
        for inp, out in train_pairs[:2]:
            fractal_info = detect_self_similarity(out)

            if fractal_info and fractal_info['type'] == 'simple_tiling':
                # Check if output is tiled version of input
                gen_size = fractal_info['generator_size']

                if np.array_equal(inp, np.array(fractal_info['generator'])):
                    # Output is tiling of input!
                    result = apply_fractal_pattern(test_input, out.shape)

                    if self.verbose:
                        print(f"   ✓ Fractal tiling detected: {gen_size}×{gen_size} generator")

                    return result

        # Check if input has self-similar structure
        input_fractal = detect_self_similarity(test_input)

        if input_fractal and input_fractal['type'] == 'simple_tiling':
            # Try extracting generator and applying transformation
            gen_size = input_fractal['generator_size']
            generator = test_input[:gen_size, :gen_size]

            if self.verbose:
                print(f"   ✓ Input is self-similar: {gen_size}×{gen_size} tiles")

            # See if transformation applied to generator produces output generator
            # (More complex logic needed here)

        # Compute fractal dimensions for analysis
        if self.verbose:
            input_dim = compute_fractal_dimension(test_input)
            print(f"   Input fractal dimension: {input_dim:.3f}")

        # Fallback to existing solver
        from evolving_specialist_system import EvolvingSolver

        ev_solver = EvolvingSolver(time_limit=10.0, verbose=False)
        result = ev_solver.solve(train_pairs, test_input)

        return result


# ============================================================================
# TESTING
# ============================================================================

def test_elite_phase2():
    """Test Elite Phase 2 solver."""

    print(f"{'='*80}")
    print(f"🔬 TESTING ELITE PHASE 2 - HIGHEST ROI INSIGHTS")
    print(f"{'='*80}\n")

    print("Insights:")
    print("  #1: Algebraic Colors (+5-8%)")
    print("  #10: Fractal Compression (+8-12%)")
    print("  Expected combined: +13-20%\n")

    # Load data
    with open('arc-agi_training_challenges.json') as f:
        challenges = json.load(f)

    with open('arc-agi_training_solutions.json') as f:
        solutions = json.load(f)

    task_ids = list(challenges.keys())[:10]

    solver = ElitePhase2Solver(verbose=True)

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
    print(f"📊 ELITE PHASE 2 RESULTS")
    print(f"{'='*80}\n")

    avg_score = np.mean([r['score'] for r in results])
    exact_count = sum(r['exact'] for r in results)

    print(f"Average: {avg_score*100:.1f}%")
    print(f"Exact matches: {exact_count}/{len(results)}")

    print(f"\nComparison:")
    print(f"  Baseline (before fix):  51.9% avg, 1 exact")
    print(f"  After forensic fix:     53.9% avg, 2 exact")
    print(f"  Elite Phase 1:          53.9% avg, 2 exact")
    print(f"  Elite Phase 2:          {avg_score*100:.1f}% avg, {exact_count} exact")

    improvement = (avg_score * 100) - 53.9
    print(f"\n{'📈' if improvement > 0 else '📉'} Change from Phase 1: {improvement:+.1f}%")

    # Save
    output = {
        'avg_score': float(avg_score),
        'exact_count': exact_count,
        'results': results,
    }

    with open('elite_phase2_results.json', 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n💾 Results saved: elite_phase2_results.json")

    return results


if __name__ == '__main__':
    test_elite_phase2()
