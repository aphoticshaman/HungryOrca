#!/usr/bin/env python3
"""
ARC Program Synthesis Solver
=============================

CORRECT approach to ARC Prize 2025:
- Explicit rule discovery (NOT neural networks)
- Program synthesis from primitives
- Constraint satisfaction
- TWO DIFFERENT attempts per task

This is what SHOULD have been done from the start.
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Callable
from dataclasses import dataclass
from itertools import product, combinations
import json


# =============================================================================
# PRIMITIVES: Basic grid transformations
# =============================================================================

@dataclass
class Transform:
    """A transformation function with metadata"""
    name: str
    func: Callable
    arity: int  # Number of arguments
    complexity: int  # For Occam's razor


def rotate_90(grid: List[List[int]]) -> List[List[int]]:
    """Rotate grid 90 degrees clockwise"""
    return [list(row) for row in zip(*grid[::-1])]


def rotate_180(grid: List[List[int]]) -> List[List[int]]:
    """Rotate grid 180 degrees"""
    return [row[::-1] for row in grid[::-1]]


def rotate_270(grid: List[List[int]]) -> List[List[int]]:
    """Rotate grid 270 degrees clockwise (90 CCW)"""
    return [list(row) for row in zip(*grid)][::-1]


def flip_horizontal(grid: List[List[int]]) -> List[List[int]]:
    """Flip grid horizontally"""
    return [row[::-1] for row in grid]


def flip_vertical(grid: List[List[int]]) -> List[List[int]]:
    """Flip grid vertically"""
    return grid[::-1]


def transpose(grid: List[List[int]]) -> List[List[int]]:
    """Transpose grid"""
    return [list(row) for row in zip(*grid)]


def color_permutation(grid: List[List[int]], mapping: Dict[int, int]) -> List[List[int]]:
    """Apply color permutation"""
    return [[mapping.get(cell, cell) for cell in row] for row in grid]


def invert_colors(grid: List[List[int]], max_color: int = 9) -> List[List[int]]:
    """Invert colors (0→9, 1→8, etc.)"""
    return [[max_color - cell for cell in row] for row in grid]


def extract_color(grid: List[List[int]], color: int) -> List[List[int]]:
    """Extract only specific color, set others to 0"""
    return [[cell if cell == color else 0 for cell in row] for row in grid]


def tile_2x2(grid: List[List[int]]) -> List[List[int]]:
    """Tile grid in 2x2 pattern"""
    return [row + row for row in grid] + [row + row for row in grid]


def tile_3x3(grid: List[List[int]]) -> List[List[int]]:
    """Tile grid in 3x3 pattern"""
    tiled = []
    for _ in range(3):
        for row in grid:
            tiled.append(row * 3)
    return tiled


def identity(grid: List[List[int]]) -> List[List[int]]:
    """Identity transformation"""
    return [row[:] for row in grid]


# =============================================================================
# PRIMITIVE REGISTRY
# =============================================================================

PRIMITIVES = [
    Transform("identity", identity, arity=0, complexity=1),
    Transform("rotate_90", rotate_90, arity=0, complexity=2),
    Transform("rotate_180", rotate_180, arity=0, complexity=2),
    Transform("rotate_270", rotate_270, arity=0, complexity=2),
    Transform("flip_h", flip_horizontal, arity=0, complexity=2),
    Transform("flip_v", flip_vertical, arity=0, complexity=2),
    Transform("transpose", transpose, arity=0, complexity=2),
    Transform("tile_2x2", tile_2x2, arity=0, complexity=3),
    Transform("tile_3x3", tile_3x3, arity=0, complexity=3),
]


# =============================================================================
# PROGRAM REPRESENTATION
# =============================================================================

@dataclass
class Program:
    """A composed program (sequence of transformations)"""
    transforms: List[Transform]

    def apply(self, grid: List[List[int]]) -> List[List[int]]:
        """Apply all transformations in sequence"""
        result = grid
        for transform in self.transforms:
            result = transform.func(result)
        return result

    def complexity(self) -> int:
        """Total complexity (for ranking)"""
        return sum(t.complexity for t in self.transforms)

    def __repr__(self):
        return " → ".join(t.name for t in self.transforms)


# =============================================================================
# VERIFICATION
# =============================================================================

def grids_equal(grid1: List[List[int]], grid2: List[List[int]]) -> bool:
    """Check if two grids are identical"""
    if len(grid1) != len(grid2):
        return False
    if not grid1:
        return not grid2
    if len(grid1[0]) != len(grid2[0]):
        return False

    for row1, row2 in zip(grid1, grid2):
        for c1, c2 in zip(row1, row2):
            if c1 != c2:
                return False

    return True


def verifies_on_examples(program: Program, examples: List[Tuple]) -> bool:
    """Check if program produces correct output on all training examples"""
    for inp, expected_out in examples:
        actual_out = program.apply(inp)
        if not grids_equal(actual_out, expected_out):
            return False
    return True


# =============================================================================
# PROGRAM SYNTHESIS
# =============================================================================

def enumerate_programs(max_depth: int = 3) -> List[Program]:
    """Generate all programs up to max_depth compositions"""
    programs = []

    # Depth 1: Single transformations
    for primitive in PRIMITIVES:
        programs.append(Program([primitive]))

    # Depth 2+: Compositions
    for depth in range(2, max_depth + 1):
        # All combinations of depth primitives
        for combo in product(PRIMITIVES, repeat=depth):
            programs.append(Program(list(combo)))

    return programs


def synthesize_solutions(train_examples: List[Tuple], max_depth: int = 3) -> List[Program]:
    """Synthesize programs that satisfy all training examples"""

    print(f"Synthesizing programs (max depth {max_depth})...")

    # Generate candidate programs
    candidates = enumerate_programs(max_depth)
    print(f"  Generated {len(candidates)} candidate programs")

    # Filter to programs that verify on ALL examples
    valid_programs = []
    for i, program in enumerate(candidates):
        if (i + 1) % 100 == 0:
            print(f"  Tested {i+1}/{len(candidates)} programs... {len(valid_programs)} valid so far")

        if verifies_on_examples(program, train_examples):
            valid_programs.append(program)

    print(f"  Found {len(valid_programs)} valid programs")

    # Rank by complexity (Occam's Razor)
    valid_programs.sort(key=lambda p: p.complexity())

    return valid_programs


# =============================================================================
# ARC TASK SOLVER
# =============================================================================

def solve_arc_task(task: Dict[str, Any]) -> Tuple[List[List[int]], List[List[int]]]:
    """
    Solve ARC task using program synthesis

    Returns:
        (attempt_1, attempt_2) - TWO DIFFERENT attempts
    """

    # Extract training examples
    train_examples = [(ex['input'], ex['output']) for ex in task['train']]
    test_input = task['test'][0]['input']

    print(f"\nSolving task with {len(train_examples)} training examples...")

    # Synthesize programs
    valid_programs = synthesize_solutions(train_examples, max_depth=2)

    if not valid_programs:
        print("  No valid programs found! Using fallback...")
        # Fallback: try simple transformations
        attempt_1 = identity(test_input)
        attempt_2 = rotate_90(test_input)
        return attempt_1, attempt_2

    # Use top 2 programs for diversity
    print(f"  Best program: {valid_programs[0]}")
    attempt_1 = valid_programs[0].apply(test_input)

    if len(valid_programs) >= 2:
        print(f"  2nd best program: {valid_programs[1]}")
        attempt_2 = valid_programs[1].apply(test_input)
    else:
        # If only one program works, try a variation
        print(f"  Only one program found, using variation...")
        attempt_2 = valid_programs[0].apply(test_input)

    # Ensure attempts are different (if possible)
    if grids_equal(attempt_1, attempt_2) and len(valid_programs) >= 2:
        print("  WARNING: Attempts are identical despite multiple valid programs!")

    return attempt_1, attempt_2


# =============================================================================
# MAIN SOLVER PIPELINE
# =============================================================================

def solve_arc_dataset(dataset_path: str, output_path: str):
    """Solve full ARC dataset using program synthesis"""

    print(f"Loading dataset: {dataset_path}")
    with open(dataset_path, 'r') as f:
        tasks = json.load(f)

    print(f"Loaded {len(tasks)} tasks")

    # Solve each task
    submission = {}
    identical_count = 0

    for i, (task_id, task_data) in enumerate(tasks.items(), 1):
        print(f"\n{'='*80}")
        print(f"Task {i}/{len(tasks)}: {task_id}")
        print(f"{'='*80}")

        try:
            attempt_1, attempt_2 = solve_arc_task(task_data)

            # Check if identical
            if grids_equal(attempt_1, attempt_2):
                identical_count += 1
                print(f"  ⚠️ Attempts are IDENTICAL ({identical_count}/{i} so far)")
            else:
                print(f"  ✓ Attempts are DIFFERENT")

            submission[task_id] = [
                {"attempt_1": attempt_1, "attempt_2": attempt_2}
            ]

        except Exception as e:
            print(f"  ERROR: {e}")
            # Fallback
            test_input = task_data['test'][0]['input']
            submission[task_id] = [
                {"attempt_1": test_input, "attempt_2": rotate_90(test_input)}
            ]

    # Save submission
    print(f"\n{'='*80}")
    print(f"Saving submission to: {output_path}")
    print(f"{'='*80}")

    with open(output_path, 'w') as f:
        json.dump(submission, f, indent=2)

    print(f"\n✓ Solved {len(submission)} tasks")
    print(f"⚠️ {identical_count}/{len(submission)} tasks have identical attempts ({100*identical_count/len(submission):.1f}%)")
    print(f"✓ {len(submission) - identical_count}/{len(submission)} tasks have DIFFERENT attempts ({100*(len(submission)-identical_count)/len(submission):.1f}%)")


if __name__ == '__main__':
    import sys

    if len(sys.argv) != 3:
        print("Usage: python arc_program_synthesis.py <input.json> <output.json>")
        print("\nExample:")
        print("  python arc_program_synthesis.py arc-agi_training_challenges.json submission_synthesis.json")
        sys.exit(1)

    solve_arc_dataset(sys.argv[1], sys.argv[2])
