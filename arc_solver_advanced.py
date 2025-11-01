#!/usr/bin/env python3
"""
ARC Advanced Solver - NSM + SDP Implementation
===============================================

Implements Improvement Vectors 1-3:
1. Explicit Symbolic Representation (not neural weights)
2. Beam Search with MDL Pruning (systematic discovery)
3. Rich Primitive Library (50+ operations)

Target: 50-60% accuracy (Phase 1)
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Callable, Optional, Set
from dataclasses import dataclass
from abc import ABC, abstractmethod
from collections import defaultdict
import itertools
import json
from scipy.ndimage import label
import heapq


# =============================================================================
# VECTOR 1: EXPLICIT SYMBOLIC REPRESENTATION
# =============================================================================

@dataclass
class Grid:
    """Grid wrapper with metadata"""
    data: np.ndarray

    def __post_init__(self):
        if not isinstance(self.data, np.ndarray):
            self.data = np.array(self.data, dtype=np.int32)

    @property
    def shape(self):
        return self.data.shape

    def __eq__(self, other):
        if not isinstance(other, Grid):
            return False
        return np.array_equal(self.data, other.data)

    def __hash__(self):
        return hash(self.data.tobytes())


class GridOperation(ABC):
    """Base class for explicit grid transformations"""

    @abstractmethod
    def apply(self, grid: Grid) -> Grid:
        """Apply transformation to grid"""
        pass

    @abstractmethod
    def description_length(self) -> int:
        """Kolmogorov complexity proxy (for MDL)"""
        pass

    def inverse(self) -> Optional['GridOperation']:
        """Return inverse operation if exists"""
        return None

    def __repr__(self):
        return self.__class__.__name__


# =============================================================================
# VECTOR 3: RICH PRIMITIVE LIBRARY (50+ OPERATIONS)
# =============================================================================

# -------------------------
# Spatial Operations
# -------------------------

class Rotate90(GridOperation):
    def __init__(self, times: int = 1):
        self.times = times % 4

    def apply(self, grid: Grid) -> Grid:
        return Grid(np.rot90(grid.data, k=self.times))

    def description_length(self) -> int:
        return 2

    def inverse(self):
        return Rotate90(times=4 - self.times)


class FlipHorizontal(GridOperation):
    def apply(self, grid: Grid) -> Grid:
        return Grid(np.fliplr(grid.data))

    def description_length(self) -> int:
        return 2

    def inverse(self):
        return FlipHorizontal()


class FlipVertical(GridOperation):
    def apply(self, grid: Grid) -> Grid:
        return Grid(np.flipud(grid.data))

    def description_length(self) -> int:
        return 2

    def inverse(self):
        return FlipVertical()


class Transpose(GridOperation):
    def apply(self, grid: Grid) -> Grid:
        return Grid(grid.data.T)

    def description_length(self) -> int:
        return 2

    def inverse(self):
        return Transpose()


class Translate(GridOperation):
    def __init__(self, dx: int, dy: int, fill: int = 0):
        self.dx = dx
        self.dy = dy
        self.fill = fill

    def apply(self, grid: Grid) -> Grid:
        h, w = grid.shape
        result = np.full((h, w), self.fill, dtype=np.int32)

        # Source coordinates
        src_y_start = max(0, -self.dy)
        src_y_end = min(h, h - self.dy)
        src_x_start = max(0, -self.dx)
        src_x_end = min(w, w - self.dx)

        # Destination coordinates
        dst_y_start = max(0, self.dy)
        dst_y_end = min(h, h + self.dy)
        dst_x_start = max(0, self.dx)
        dst_x_end = min(w, w + self.dx)

        result[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = \
            grid.data[src_y_start:src_y_end, src_x_start:src_x_end]

        return Grid(result)

    def description_length(self) -> int:
        return 3


# -------------------------
# Color Operations
# -------------------------

class RecolorMap(GridOperation):
    def __init__(self, mapping: Dict[int, int]):
        self.mapping = mapping

    def apply(self, grid: Grid) -> Grid:
        result = grid.data.copy()
        for from_color, to_color in self.mapping.items():
            result[grid.data == from_color] = to_color
        return Grid(result)

    def description_length(self) -> int:
        return 1 + len(self.mapping)


class SwapColors(GridOperation):
    def __init__(self, color1: int, color2: int):
        self.color1 = color1
        self.color2 = color2

    def apply(self, grid: Grid) -> Grid:
        result = grid.data.copy()
        mask1 = grid.data == self.color1
        mask2 = grid.data == self.color2
        result[mask1] = self.color2
        result[mask2] = self.color1
        return Grid(result)

    def description_length(self) -> int:
        return 3


class ReplaceColor(GridOperation):
    def __init__(self, from_color: int, to_color: int):
        self.from_color = from_color
        self.to_color = to_color

    def apply(self, grid: Grid) -> Grid:
        result = grid.data.copy()
        result[grid.data == self.from_color] = self.to_color
        return Grid(result)

    def description_length(self) -> int:
        return 3


# -------------------------
# Pattern Operations
# -------------------------

class TilePattern(GridOperation):
    def __init__(self, n_x: int, n_y: int):
        self.n_x = n_x
        self.n_y = n_y

    def apply(self, grid: Grid) -> Grid:
        return Grid(np.tile(grid.data, (self.n_y, self.n_x)))

    def description_length(self) -> int:
        return 3


class ExtractRegion(GridOperation):
    def __init__(self, y_start: int, y_end: int, x_start: int, x_end: int):
        self.y_start = y_start
        self.y_end = y_end
        self.x_start = x_start
        self.x_end = x_end

    def apply(self, grid: Grid) -> Grid:
        h, w = grid.shape
        y_start = max(0, min(h, self.y_start))
        y_end = max(0, min(h, self.y_end))
        x_start = max(0, min(w, self.x_start))
        x_end = max(0, min(w, self.x_end))

        return Grid(grid.data[y_start:y_end, x_start:x_end])

    def description_length(self) -> int:
        return 5


class ScaleUp(GridOperation):
    def __init__(self, factor: int):
        self.factor = factor

    def apply(self, grid: Grid) -> Grid:
        return Grid(np.repeat(np.repeat(grid.data, self.factor, axis=0),
                             self.factor, axis=1))

    def description_length(self) -> int:
        return 2


class ScaleDown(GridOperation):
    def __init__(self, factor: int):
        self.factor = factor

    def apply(self, grid: Grid) -> Grid:
        h, w = grid.shape
        new_h = h // self.factor
        new_w = w // self.factor

        result = np.zeros((new_h, new_w), dtype=np.int32)
        for i in range(new_h):
            for j in range(new_w):
                # Take most common value in block
                block = grid.data[i*self.factor:(i+1)*self.factor,
                                 j*self.factor:(j+1)*self.factor]
                values, counts = np.unique(block, return_counts=True)
                result[i, j] = values[np.argmax(counts)]

        return Grid(result)

    def description_length(self) -> int:
        return 2


# -------------------------
# Object-Based Operations
# -------------------------

@dataclass
class GridObject:
    """Represents a connected component"""
    color: int
    mask: np.ndarray
    bbox: Tuple[int, int, int, int]  # (y_min, y_max, x_min, x_max)
    size: int
    centroid: Tuple[float, float]


def extract_objects(grid: Grid, connectivity: int = 4) -> List[GridObject]:
    """Extract connected components as objects"""
    objects = []

    for color in range(1, 10):  # Skip background (0)
        if color not in grid.data:
            continue

        mask = (grid.data == color)
        labeled, n_components = label(mask)

        for i in range(1, n_components + 1):
            obj_mask = (labeled == i)

            # Bounding box
            rows = np.where(obj_mask.any(axis=1))[0]
            cols = np.where(obj_mask.any(axis=0))[0]

            if len(rows) == 0 or len(cols) == 0:
                continue

            bbox = (rows[0], rows[-1] + 1, cols[0], cols[-1] + 1)

            # Centroid
            y_coords, x_coords = np.where(obj_mask)
            centroid = (np.mean(y_coords), np.mean(x_coords))

            objects.append(GridObject(
                color=color,
                mask=obj_mask,
                bbox=bbox,
                size=np.sum(obj_mask),
                centroid=centroid
            ))

    return objects


class FilterByColor(GridOperation):
    def __init__(self, keep_color: int, background: int = 0):
        self.keep_color = keep_color
        self.background = background

    def apply(self, grid: Grid) -> Grid:
        result = np.full_like(grid.data, self.background)
        result[grid.data == self.keep_color] = self.keep_color
        return Grid(result)

    def description_length(self) -> int:
        return 3


# -------------------------
# Composite Operations
# -------------------------

class CompositeOperation(GridOperation):
    """Sequence of operations applied in order"""

    def __init__(self, ops: List[GridOperation]):
        self.ops = ops

    def apply(self, grid: Grid) -> Grid:
        result = grid
        for op in self.ops:
            result = op.apply(result)
        return result

    def description_length(self) -> int:
        return sum(op.description_length() for op in self.ops)

    def __repr__(self):
        return " → ".join(repr(op) for op in self.ops)


class Identity(GridOperation):
    """Identity operation (no change)"""

    def apply(self, grid: Grid) -> Grid:
        return Grid(grid.data.copy())

    def description_length(self) -> int:
        return 1


# =============================================================================
# PRIMITIVE LIBRARY
# =============================================================================

def get_all_primitives() -> List[GridOperation]:
    """Get all primitive operations"""

    primitives = []

    # Spatial
    primitives.extend([
        Rotate90(1), Rotate90(2), Rotate90(3),
        FlipHorizontal(), FlipVertical(), Transpose(),
    ])

    # Translations (small offsets)
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx != 0 or dy != 0:
                primitives.append(Translate(dx, dy))

    # Color operations
    # Color replacements (common pairs)
    common_color_pairs = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)]
    for c1, c2 in common_color_pairs:
        primitives.append(ReplaceColor(c1, c2))
        primitives.append(ReplaceColor(c2, c1))
        primitives.append(SwapColors(c1, c2))

    # Pattern operations
    for n in [2, 3]:
        primitives.append(TilePattern(n, 1))
        primitives.append(TilePattern(1, n))
        primitives.append(TilePattern(n, n))

    # Scaling
    for factor in [2, 3]:
        primitives.append(ScaleUp(factor))
        primitives.append(ScaleDown(factor))

    # Object filtering
    for color in range(1, 10):
        primitives.append(FilterByColor(color))

    # Identity
    primitives.append(Identity())

    return primitives


# =============================================================================
# VECTOR 2: BEAM SEARCH WITH MDL PRUNING
# =============================================================================

class RuleSearcher:
    """Beam search for transformation rules with MDL pruning"""

    def __init__(self, primitives: List[GridOperation], beam_width: int = 100):
        self.primitives = primitives
        self.beam_width = beam_width
        self.verified_cache = {}

    def search(self, examples: List[Tuple[Grid, Grid]],
               max_depth: int = 3) -> List[CompositeOperation]:
        """Search for rules that explain all examples"""

        print(f"  Searching rules (max depth {max_depth}, beam width {self.beam_width})...")

        # Convert examples to Grid objects
        examples = [(Grid(inp), Grid(out)) for inp, out in examples]

        # Initialize beam with single-operation rules
        beam = []

        for op in self.primitives:
            if self._verifies_on_all(op, examples):
                complexity = op.description_length()
                fit_score = self._measure_fit(op, examples)
                mdl_score = fit_score - 0.5 * complexity

                heapq.heappush(beam, (-mdl_score, complexity, CompositeOperation([op])))

        # Keep only top beam_width
        beam = heapq.nlargest(self.beam_width, beam)

        print(f"    Depth 1: {len(beam)} valid rules")

        # Iteratively extend rules
        for depth in range(2, max_depth + 1):
            candidates = []

            for neg_score, _, rule in beam:
                # Extend with one more operation
                for op in self.primitives:
                    new_ops = rule.ops + [op]
                    new_rule = CompositeOperation(new_ops)

                    # Check if still valid
                    if self._verifies_on_all(new_rule, examples):
                        complexity = new_rule.description_length()
                        fit_score = self._measure_fit(new_rule, examples)
                        mdl_score = fit_score - 0.5 * complexity

                        heapq.heappush(candidates, (-mdl_score, complexity, new_rule))

            # Prune to beam width
            if candidates:
                beam = heapq.nlargest(self.beam_width, candidates)
                print(f"    Depth {depth}: {len(beam)} valid rules")
            else:
                print(f"    Depth {depth}: No new valid rules, stopping")
                break

            # Early stopping if perfect simple rule found
            if beam and -beam[0][0] > 9.5:  # Very high score
                print(f"    Found excellent rule, stopping early")
                break

        # Extract and sort final rules
        final_rules = [rule for _, _, rule in beam]
        final_rules.sort(key=lambda r: r.description_length())

        print(f"    Found {len(final_rules)} total valid rules")

        return final_rules

    def _verifies_on_all(self, rule: GridOperation, examples: List[Tuple[Grid, Grid]]) -> bool:
        """Check if rule produces correct output on all examples"""

        # Check cache
        cache_key = (id(rule), tuple(id(e) for e in examples))
        if cache_key in self.verified_cache:
            return self.verified_cache[cache_key]

        try:
            for inp, out in examples:
                predicted = rule.apply(inp)
                if not predicted == out:
                    self.verified_cache[cache_key] = False
                    return False

            self.verified_cache[cache_key] = True
            return True

        except Exception:
            self.verified_cache[cache_key] = False
            return False

    def _measure_fit(self, rule: GridOperation, examples: List[Tuple[Grid, Grid]]) -> float:
        """Measure how well rule fits examples"""

        score = 0.0
        for inp, out in examples:
            try:
                predicted = rule.apply(inp)

                # Exact match = high score
                if predicted == out:
                    score += 10.0
                else:
                    # Partial credit
                    if predicted.shape == out.shape:
                        score += np.mean(predicted.data == out.data) * 5.0

            except Exception:
                pass

        return score / len(examples)


# =============================================================================
# MAIN SOLVER
# =============================================================================

def solve_arc_task(task: Dict[str, Any],
                   primitives: List[GridOperation] = None,
                   max_depth: int = 3,
                   beam_width: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve one ARC task using advanced solver

    Returns:
        (attempt_1, attempt_2) - TWO DIFFERENT attempts
    """

    if primitives is None:
        primitives = get_all_primitives()

    # Extract examples
    train_examples = [(ex['input'], ex['output']) for ex in task['train']]
    test_input = task['test'][0]['input']

    print(f"\n  Solving task with {len(train_examples)} examples...")
    print(f"  Primitives: {len(primitives)}")

    # Search for rules
    searcher = RuleSearcher(primitives, beam_width=beam_width)
    valid_rules = searcher.search(train_examples, max_depth=max_depth)

    if not valid_rules:
        print("  ⚠️ No valid rules found! Using fallback...")
        # Fallback
        attempt_1 = test_input
        attempt_2 = np.rot90(test_input)
        return attempt_1, attempt_2

    # Use top 2 rules for diversity
    print(f"  ✓ Best rule: {valid_rules[0]}")
    attempt_1 = valid_rules[0].apply(Grid(test_input)).data

    if len(valid_rules) >= 2:
        print(f"  ✓ 2nd rule: {valid_rules[1]}")
        attempt_2 = valid_rules[1].apply(Grid(test_input)).data
    else:
        print(f"  ⚠️ Only one rule found")
        attempt_2 = attempt_1

    # Check if identical
    if np.array_equal(attempt_1, attempt_2):
        print(f"  ⚠️ Attempts are IDENTICAL")
    else:
        print(f"  ✓ Attempts are DIFFERENT")

    return attempt_1, attempt_2


# =============================================================================
# DATASET SOLVER
# =============================================================================

def solve_dataset(input_path: str, output_path: str):
    """Solve full dataset"""

    print(f"Loading: {input_path}")
    with open(input_path, 'r') as f:
        tasks = json.load(f)

    print(f"Loaded {len(tasks)} tasks")
    print(f"Primitives: {len(get_all_primitives())}")

    submission = {}
    identical_count = 0

    for i, (task_id, task_data) in enumerate(tasks.items(), 1):
        print(f"\n{'='*80}")
        print(f"Task {i}/{len(tasks)}: {task_id}")
        print(f"{'='*80}")

        try:
            attempt_1, attempt_2 = solve_arc_task(task_data, max_depth=2, beam_width=30)

            if np.array_equal(attempt_1, attempt_2):
                identical_count += 1

            submission[task_id] = [{
                "attempt_1": attempt_1.tolist(),
                "attempt_2": attempt_2.tolist()
            }]

        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            test_input = task_data['test'][0]['input']
            submission[task_id] = [{
                "attempt_1": test_input,
                "attempt_2": np.rot90(test_input).tolist()
            }]

    print(f"\n{'='*80}")
    print(f"RESULTS")
    print(f"{'='*80}")
    print(f"Total tasks: {len(submission)}")
    print(f"Identical attempts: {identical_count}/{len(submission)} ({100*identical_count/len(submission):.1f}%)")
    print(f"Different attempts: {len(submission)-identical_count}/{len(submission)} ({100*(len(submission)-identical_count)/len(submission):.1f}%)")

    print(f"\nSaving to: {output_path}")
    with open(output_path, 'w') as f:
        json.dump(submission, f, indent=2)

    print(f"✓ Done!")


if __name__ == '__main__':
    import sys

    if len(sys.argv) != 3:
        print("Usage: python arc_solver_advanced.py <input.json> <output.json>")
        sys.exit(1)

    solve_dataset(sys.argv[1], sys.argv[2])
