"""
ARC PRIZE 2025 IMPROVED SOLVER
================================

Enhanced solver with pattern learning from training examples.
Implements core transformations based on observed input->output patterns.
"""

import numpy as np
import json
from typing import List, Tuple, Dict, Optional, Any, Callable
from collections import Counter
from dataclasses import dataclass
import copy


# ============================================================================
# CORE DATA STRUCTURES
# ============================================================================

@dataclass
class ARCGrid:
    """Represents an ARC grid."""
    data: np.ndarray

    @classmethod
    def from_list(cls, grid_list: List[List[int]]):
        return cls(np.array(grid_list, dtype=int))

    def to_list(self) -> List[List[int]]:
        return self.data.tolist()

    @property
    def height(self) -> int:
        return self.data.shape[0]

    @property
    def width(self) -> int:
        return self.data.shape[1]

    def get_colors(self) -> List[int]:
        return list(np.unique(self.data))

    def copy(self):
        return ARCGrid(self.data.copy())


# ============================================================================
# CORE TRANSFORMATION PRIMITIVES
# ============================================================================

class TransformationLibrary:
    """Library of atomic transformations."""

    @staticmethod
    def identity(grid: np.ndarray) -> np.ndarray:
        return grid.copy()

    @staticmethod
    def rotate_90(grid: np.ndarray) -> np.ndarray:
        return np.rot90(grid, k=1)

    @staticmethod
    def rotate_180(grid: np.ndarray) -> np.ndarray:
        return np.rot90(grid, k=2)

    @staticmethod
    def rotate_270(grid: np.ndarray) -> np.ndarray:
        return np.rot90(grid, k=3)

    @staticmethod
    def flip_horizontal(grid: np.ndarray) -> np.ndarray:
        return np.flip(grid, axis=0)

    @staticmethod
    def flip_vertical(grid: np.ndarray) -> np.ndarray:
        return np.flip(grid, axis=1)

    @staticmethod
    def transpose(grid: np.ndarray) -> np.ndarray:
        return grid.T

    @staticmethod
    def extract_color(grid: np.ndarray, color: int, bg: int = 0) -> np.ndarray:
        """Extract only cells of specific color."""
        result = np.full_like(grid, bg)
        result[grid == color] = color
        return result

    @staticmethod
    def replace_color(grid: np.ndarray, old_color: int, new_color: int) -> np.ndarray:
        """Replace one color with another."""
        result = grid.copy()
        result[grid == old_color] = new_color
        return result

    @staticmethod
    def scale_up_2x(grid: np.ndarray) -> np.ndarray:
        """Scale grid up by repeating each cell 2x2."""
        return np.repeat(np.repeat(grid, 2, axis=0), 2, axis=1)

    @staticmethod
    def scale_down_2x(grid: np.ndarray) -> np.ndarray:
        """Scale grid down by sampling every other cell."""
        return grid[::2, ::2]

    @staticmethod
    def fill_background(grid: np.ndarray, new_bg: int = 0) -> np.ndarray:
        """Fill most common color with new background."""
        bg = Counter(grid.flatten()).most_common(1)[0][0]
        result = grid.copy()
        result[grid == bg] = new_bg
        return result

    @staticmethod
    def crop_to_content(grid: np.ndarray, bg: int = 0) -> np.ndarray:
        """Crop to bounding box of non-background."""
        mask = grid != bg
        if not mask.any():
            return grid

        rows = np.where(mask.any(axis=1))[0]
        cols = np.where(mask.any(axis=0))[0]

        if len(rows) == 0 or len(cols) == 0:
            return grid

        return grid[rows[0]:rows[-1]+1, cols[0]:cols[-1]+1]

    @staticmethod
    def tile_pattern(grid: np.ndarray, rows: int, cols: int) -> np.ndarray:
        """Tile pattern to fill larger grid."""
        return np.tile(grid, (rows, cols))

    @staticmethod
    def overlay(grid1: np.ndarray, grid2: np.ndarray, bg: int = 0) -> np.ndarray:
        """Overlay grid2 onto grid1 (non-bg pixels from grid2 replace grid1)."""
        result = grid1.copy()
        mask = grid2 != bg
        if result.shape == grid2.shape:
            result[mask] = grid2[mask]
        return result


# ============================================================================
# PATTERN MATCHER
# ============================================================================

class PatternMatcher:
    """Match patterns between input and output to learn transformations."""

    def __init__(self):
        self.transforms = TransformationLibrary()

    def find_best_transform(self, input_grid: np.ndarray,
                           output_grid: np.ndarray) -> List[Tuple[Callable, float]]:
        """
        Find transformations that convert input to output.

        Returns:
            List of (transform_function, score) tuples sorted by score
        """
        candidates = []

        # Try all basic transformations
        transform_list = [
            ('identity', self.transforms.identity),
            ('rotate_90', self.transforms.rotate_90),
            ('rotate_180', self.transforms.rotate_180),
            ('rotate_270', self.transforms.rotate_270),
            ('flip_h', self.transforms.flip_horizontal),
            ('flip_v', self.transforms.flip_vertical),
            ('transpose', self.transforms.transpose),
        ]

        for name, transform in transform_list:
            try:
                result = transform(input_grid)
                score = self._similarity(result, output_grid)
                if score > 0:
                    candidates.append((transform, score, name))
            except:
                pass

        # Try color transformations
        input_colors = np.unique(input_grid)
        output_colors = np.unique(output_grid)

        # Color mapping
        if len(input_colors) == len(output_colors):
            # Try mapping colors
            for i_col in input_colors:
                for o_col in output_colors:
                    try:
                        result = self.transforms.replace_color(input_grid, i_col, o_col)
                        score = self._similarity(result, output_grid)
                        if score > 0:
                            candidates.append((
                                lambda g, ic=i_col, oc=o_col: self.transforms.replace_color(g, ic, oc),
                                score,
                                f'color_{i_col}_to_{o_col}'
                            ))
                    except:
                        pass

        # Try scaling
        if output_grid.shape[0] == input_grid.shape[0] * 2:
            try:
                result = self.transforms.scale_up_2x(input_grid)
                score = self._similarity(result, output_grid)
                if score > 0:
                    candidates.append((self.transforms.scale_up_2x, score, 'scale_up_2x'))
            except:
                pass

        if output_grid.shape[0] * 2 == input_grid.shape[0]:
            try:
                result = self.transforms.scale_down_2x(input_grid)
                score = self._similarity(result, output_grid)
                if score > 0:
                    candidates.append((self.transforms.scale_down_2x, score, 'scale_down_2x'))
            except:
                pass

        # Try cropping
        try:
            result = self.transforms.crop_to_content(input_grid)
            score = self._similarity(result, output_grid)
            if score > 0:
                candidates.append((self.transforms.crop_to_content, score, 'crop'))
        except:
            pass

        # Sort by score
        candidates.sort(key=lambda x: x[1], reverse=True)

        return candidates

    def _similarity(self, grid1: np.ndarray, grid2: np.ndarray) -> float:
        """Compute similarity between grids."""
        if grid1.shape != grid2.shape:
            return 0.0

        matches = np.sum(grid1 == grid2)
        total = grid1.size

        return matches / total if total > 0 else 0.0


# ============================================================================
# IMPROVED ARC SOLVER
# ============================================================================

class ImprovedARCSolver:
    """
    Improved solver that learns transformations from training examples.
    """

    def __init__(self):
        self.pattern_matcher = PatternMatcher()
        self.transforms = TransformationLibrary()

    def solve_task(self, task: Dict) -> List[List[List[int]]]:
        """
        Solve an ARC task by learning from training examples.

        Args:
            task: Dict with 'train' and 'test' keys

        Returns:
            List of 2 solution attempts
        """
        train_pairs = task['train']
        test_input = ARCGrid.from_list(task['test'][0]['input'])

        # Learn transformations from training pairs
        learned_transforms = self._learn_from_training(train_pairs)

        # Apply learned transforms to test input
        candidates = []

        for transform, score, name in learned_transforms[:10]:  # Top 10 transforms
            try:
                result = transform(test_input.data)
                candidates.append((result, score, name))
            except:
                pass

        # Add fallback strategies
        candidates.extend(self._fallback_strategies(test_input))

        # Sort by score and deduplicate
        candidates.sort(key=lambda x: x[1], reverse=True)
        unique_solutions = []
        seen = set()

        for result, score, name in candidates:
            result_tuple = tuple(map(tuple, result.tolist()))
            if result_tuple not in seen:
                unique_solutions.append(result.tolist())
                seen.add(result_tuple)

            if len(unique_solutions) >= 2:
                break

        # Ensure we have 2 solutions
        while len(unique_solutions) < 2:
            unique_solutions.append(test_input.to_list())

        return unique_solutions[:2]

    def _learn_from_training(self, train_pairs: List[Dict]) -> List[Tuple]:
        """Learn transformations from training examples."""
        all_transforms = []

        for pair in train_pairs:
            input_grid = np.array(pair['input'])
            output_grid = np.array(pair['output'])

            # Find transforms that work for this pair
            transforms = self.pattern_matcher.find_best_transform(input_grid, output_grid)
            all_transforms.extend(transforms)

        # Aggregate by transform name and average scores
        transform_scores = {}
        transform_funcs = {}

        for func, score, name in all_transforms:
            if name not in transform_scores:
                transform_scores[name] = []
                transform_funcs[name] = func

            transform_scores[name].append(score)

        # Compute average scores
        result = []
        for name, scores in transform_scores.items():
            avg_score = np.mean(scores)
            result.append((transform_funcs[name], avg_score, name))

        result.sort(key=lambda x: x[1], reverse=True)
        return result

    def _fallback_strategies(self, test_grid: ARCGrid) -> List[Tuple]:
        """Fallback strategies when no learned transform works."""
        fallbacks = []

        # Identity (return input as-is)
        fallbacks.append((test_grid.data.copy(), 0.01, 'identity'))

        # All basic transformations
        try:
            fallbacks.append((self.transforms.rotate_90(test_grid.data), 0.005, 'rotate_90_fallback'))
        except:
            pass

        try:
            fallbacks.append((self.transforms.flip_horizontal(test_grid.data), 0.005, 'flip_h_fallback'))
        except:
            pass

        try:
            fallbacks.append((self.transforms.crop_to_content(test_grid.data), 0.005, 'crop_fallback'))
        except:
            pass

        return fallbacks


# ============================================================================
# SUBMISSION GENERATOR
# ============================================================================

class ImprovedSubmissionGenerator:
    """Generate submission using improved solver."""

    def __init__(self, test_challenges_path: str = 'arc-agi_test_challenges.json'):
        self.test_challenges_path = test_challenges_path
        self.solver = ImprovedARCSolver()

    def generate_submission(self, output_path: str = 'submission.json', verbose: bool = True):
        """Generate complete submission file."""
        # Load test challenges
        with open(self.test_challenges_path, 'r') as f:
            test_challenges = json.load(f)

        submission = {}
        total_tasks = len(test_challenges)

        if verbose:
            print(f"🚀 ARC Prize 2025 IMPROVED Submission Generator")
            print(f"=" * 60)
            print(f"Total tasks: {total_tasks}")
            print(f"=" * 60)

        for idx, (task_id, task) in enumerate(test_challenges.items(), 1):
            try:
                # Solve task
                solutions = self.solver.solve_task(task)

                # Format for submission
                submission[task_id] = [
                    {
                        'attempt_1': solutions[0],
                        'attempt_2': solutions[1]
                    }
                ]

                if verbose and idx % 10 == 0:
                    print(f"✓ Solved {idx}/{total_tasks} tasks ({idx*100//total_tasks}%)")

            except Exception as e:
                # Fallback: use test input
                test_input = task['test'][0]['input']
                submission[task_id] = [
                    {
                        'attempt_1': test_input,
                        'attempt_2': test_input
                    }
                ]

                if verbose:
                    print(f"⚠ Task {task_id} failed: {str(e)[:50]}")

        # Save submission
        with open(output_path, 'w') as f:
            json.dump(submission, f)

        if verbose:
            print(f"=" * 60)
            print(f"✅ Submission saved to: {output_path}")
            print(f"🎮 WAKA WAKA! Ready for ARC Prize 2025!")
            print(f"=" * 60)

        return submission


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Generate improved submission."""
    generator = ImprovedSubmissionGenerator()
    submission = generator.generate_submission(
        output_path='submission.json',
        verbose=True
    )

    print(f"\n📊 Submission Statistics:")
    print(f"   Total tasks: {len(submission)}")
    print(f"   Uses pattern learning from training examples")
    print(f"   Ready for upload to Kaggle!")
    print(f"\n🚀 Next steps:")
    print(f"   1. Validate with: python3 validate_solver.py")
    print(f"   2. Upload submission.json to ARC Prize 2025")
    print(f"   3. WAKA WAKA! 🎮🧠⚡")


if __name__ == "__main__":
    main()
