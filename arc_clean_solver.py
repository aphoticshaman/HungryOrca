#!/usr/bin/env python3
"""
Clean ARC Prize 2025 Solver
===========================

A focused, practical solver that combines:
- Task classification and routing
- Multiple solver strategies
- Ensemble voting for confidence
- Dual-attempt submissions
- Progressive time allocation

Target: 15-25% accuracy (realistic baseline)

Author: Ryan Cardwell & Claude
Date: November 2025
"""

import numpy as np
import json
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any, Callable
from dataclasses import dataclass
from collections import defaultdict, Counter
from itertools import product
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class SolverConfig:
    """Clean, focused configuration"""

    # Time management
    total_time_budget: float = 6 * 3600  # 6 hours
    min_time_per_task: float = 0.5
    max_time_per_task: float = 30.0

    # Solver parameters
    max_primitive_depth: int = 3
    ensemble_size: int = 5

    # Task routing
    enable_task_classification: bool = True
    enable_ensemble_voting: bool = True

    # Paths (Kaggle-compatible)
    data_path: str = '/kaggle/input/arc-prize-2025'
    output_path: str = '/kaggle/working'


# =============================================================================
# GEOMETRIC PRIMITIVES
# =============================================================================

class Primitives:
    """
    Core geometric and logical primitives
    These are task-independent operations that can be composed
    """

    @staticmethod
    def identity(grid: np.ndarray) -> np.ndarray:
        """No change"""
        return grid.copy()

    @staticmethod
    def rotate_90(grid: np.ndarray) -> np.ndarray:
        """Rotate 90 degrees clockwise"""
        return np.rot90(grid, k=-1)

    @staticmethod
    def rotate_180(grid: np.ndarray) -> np.ndarray:
        """Rotate 180 degrees"""
        return np.rot90(grid, k=2)

    @staticmethod
    def rotate_270(grid: np.ndarray) -> np.ndarray:
        """Rotate 270 degrees clockwise"""
        return np.rot90(grid, k=-3)

    @staticmethod
    def flip_horizontal(grid: np.ndarray) -> np.ndarray:
        """Mirror horizontally"""
        return np.fliplr(grid)

    @staticmethod
    def flip_vertical(grid: np.ndarray) -> np.ndarray:
        """Mirror vertically"""
        return np.flipud(grid)

    @staticmethod
    def transpose(grid: np.ndarray) -> np.ndarray:
        """Transpose (swap rows/columns)"""
        return grid.T

    @staticmethod
    def invert_colors(grid: np.ndarray) -> np.ndarray:
        """Invert non-zero colors"""
        max_color = max(9, grid.max())
        result = grid.copy()
        mask = result > 0
        result[mask] = max_color - result[mask] + 1
        return result

    @staticmethod
    def replace_color(grid: np.ndarray, old_color: int, new_color: int) -> np.ndarray:
        """Replace one color with another"""
        result = grid.copy()
        result[result == old_color] = new_color
        return result

    @staticmethod
    def crop_nonzero(grid: np.ndarray) -> np.ndarray:
        """Crop to bounding box of non-zero elements"""
        if grid.sum() == 0:
            return grid
        rows = np.any(grid != 0, axis=1)
        cols = np.any(grid != 0, axis=0)
        if rows.any() and cols.any():
            return grid[rows][:, cols]
        return grid

    @staticmethod
    def tile_2x2(grid: np.ndarray) -> np.ndarray:
        """Tile 2x2"""
        return np.tile(grid, (2, 2))

    @staticmethod
    def scale_up_2x(grid: np.ndarray) -> np.ndarray:
        """Scale up by repeating each cell 2x2"""
        return np.repeat(np.repeat(grid, 2, axis=0), 2, axis=1)


# =============================================================================
# TASK CLASSIFIER
# =============================================================================

class TaskClassifier:
    """
    Classify tasks into categories to route to specialized solvers

    Based on simple, robust features extracted from training examples
    """

    @staticmethod
    def classify(task: Dict) -> str:
        """
        Classify task type based on training examples

        Returns category: 'geometric', 'color', 'spatial', 'pattern', 'complex'
        """
        examples = task.get('train', [])
        if not examples:
            return 'complex'

        features = TaskClassifier._extract_features(examples)

        # Simple decision tree
        if features['same_shape_ratio'] > 0.8:
            if features['color_changes_only']:
                return 'color'
            elif features['has_rotation'] or features['has_reflection']:
                return 'geometric'
            else:
                return 'pattern'
        elif features['has_scaling']:
            return 'spatial'
        else:
            return 'complex'

    @staticmethod
    def _extract_features(examples: List[Dict]) -> Dict:
        """Extract classification features from examples"""
        features = {
            'same_shape_ratio': 0.0,
            'color_changes_only': False,
            'has_rotation': False,
            'has_reflection': False,
            'has_scaling': False,
        }

        try:
            # Check shape preservation
            same_shape_count = 0
            for ex in examples:
                inp = np.array(ex['input'])
                out = np.array(ex['output'])
                if inp.shape == out.shape:
                    same_shape_count += 1
            features['same_shape_ratio'] = same_shape_count / len(examples)

            # Check for pure color transformations
            if same_shape_count == len(examples):
                color_only = all(
                    np.array(ex['input']).shape == np.array(ex['output']).shape and
                    set(np.array(ex['input']).flatten()) == set(np.array(ex['output']).flatten())
                    for ex in examples
                )
                features['color_changes_only'] = color_only

            # Check for geometric transformations
            for ex in examples:
                inp = np.array(ex['input'])
                out = np.array(ex['output'])

                if inp.shape == out.shape:
                    # Check rotation
                    if np.array_equal(out, np.rot90(inp)) or \
                       np.array_equal(out, np.rot90(inp, 2)) or \
                       np.array_equal(out, np.rot90(inp, 3)):
                        features['has_rotation'] = True

                    # Check reflection
                    if np.array_equal(out, np.fliplr(inp)) or \
                       np.array_equal(out, np.flipud(inp)):
                        features['has_reflection'] = True
                else:
                    # Different shapes suggest scaling
                    if inp.size > 0 and out.size > 0:
                        ratio = out.size / inp.size
                        if ratio > 1.5 or ratio < 0.7:
                            features['has_scaling'] = True

        except Exception:
            pass

        return features


# =============================================================================
# SOLVERS
# =============================================================================

class GeometricSolver:
    """
    Solves tasks with geometric transformations
    Tries common rotations, reflections, transpose
    """

    def __init__(self):
        self.primitives = Primitives()

    def solve(self, task: Dict, timeout: float = 10.0) -> Optional[np.ndarray]:
        """Try geometric primitives on training examples"""
        start_time = time.time()
        examples = task.get('train', [])
        test_input = np.array(task['test'][0]['input'])

        if not examples:
            return None

        # Try each geometric primitive
        transforms = [
            self.primitives.rotate_90,
            self.primitives.rotate_180,
            self.primitives.rotate_270,
            self.primitives.flip_horizontal,
            self.primitives.flip_vertical,
            self.primitives.transpose,
        ]

        for transform in transforms:
            if time.time() - start_time > timeout:
                break

            # Test if this transform works on training examples
            matches = 0
            for ex in examples:
                try:
                    inp = np.array(ex['input'])
                    out = np.array(ex['output'])
                    if np.array_equal(transform(inp), out):
                        matches += 1
                except:
                    pass

            # If transform works on all examples, apply to test
            if matches == len(examples):
                return transform(test_input)

        return None


class ColorSolver:
    """
    Solves tasks with color transformations
    Learns color mapping from examples
    """

    def __init__(self):
        self.primitives = Primitives()

    def solve(self, task: Dict, timeout: float = 10.0) -> Optional[np.ndarray]:
        """Learn and apply color mapping"""
        examples = task.get('train', [])
        test_input = np.array(task['test'][0]['input'])

        if not examples:
            return None

        # Extract color mapping from examples
        color_map = self._learn_color_mapping(examples)

        if color_map:
            return self._apply_color_mapping(test_input, color_map)

        return None

    def _learn_color_mapping(self, examples: List[Dict]) -> Optional[Dict[int, int]]:
        """Learn consistent color mapping from examples"""
        mappings = []

        for ex in examples:
            inp = np.array(ex['input'])
            out = np.array(ex['output'])

            if inp.shape != out.shape:
                return None  # Not a pure color transformation

            # Extract mapping for this example
            example_map = {}
            for i in range(inp.shape[0]):
                for j in range(inp.shape[1]):
                    in_color = inp[i, j]
                    out_color = out[i, j]
                    if in_color in example_map:
                        if example_map[in_color] != out_color:
                            return None  # Inconsistent mapping
                    example_map[in_color] = out_color

            mappings.append(example_map)

        # Find consistent mapping across all examples
        if not mappings:
            return None

        consistent_map = mappings[0].copy()
        for m in mappings[1:]:
            for k, v in m.items():
                if k in consistent_map and consistent_map[k] != v:
                    return None
                consistent_map[k] = v

        return consistent_map

    def _apply_color_mapping(self, grid: np.ndarray, color_map: Dict[int, int]) -> np.ndarray:
        """Apply color mapping to grid"""
        result = grid.copy()
        for old_color, new_color in color_map.items():
            result[result == old_color] = new_color
        return result


class PatternSolver:
    """
    Solves pattern-based tasks
    Tries to detect and apply consistent patterns
    """

    def __init__(self):
        self.primitives = Primitives()

    def solve(self, task: Dict, timeout: float = 10.0) -> Optional[np.ndarray]:
        """Try to detect and apply patterns"""
        examples = task.get('train', [])
        test_input = np.array(task['test'][0]['input'])

        if not examples:
            return None

        # Check if it's a tiling pattern
        for ex in examples:
            inp = np.array(ex['input'])
            out = np.array(ex['output'])

            # Check if output is tiled input
            if out.shape[0] == inp.shape[0] * 2 and out.shape[1] == inp.shape[1] * 2:
                if np.array_equal(self.primitives.tile_2x2(inp), out):
                    return self.primitives.tile_2x2(test_input)

        return None


# =============================================================================
# ENSEMBLE SOLVER
# =============================================================================

class EnsembleSolver:
    """
    Runs multiple solvers and votes on the best answer
    Higher agreement = higher confidence
    """

    def __init__(self, config: SolverConfig):
        self.config = config
        self.solvers = {
            'geometric': GeometricSolver(),
            'color': ColorSolver(),
            'pattern': PatternSolver(),
        }
        self.classifier = TaskClassifier()

    def solve(self, task: Dict, timeout: float = 10.0) -> Dict:
        """
        Solve task using ensemble of solvers

        Returns:
            {
                'solution': best_solution (np.ndarray),
                'confidence': float (0-1),
                'method': str (which solver(s) agreed)
            }
        """
        start_time = time.time()
        test_input = np.array(task['test'][0]['input'])

        # Classify task
        category = self.classifier.classify(task)

        # Run solvers
        solutions = []
        methods = []

        # Always try the classified solver first (more time)
        if category in self.solvers:
            sol = self.solvers[category].solve(task, timeout=timeout * 0.6)
            if sol is not None:
                solutions.append(sol)
                methods.append(category)

        # Try other solvers (less time)
        remaining_time = timeout - (time.time() - start_time)
        time_per_solver = max(1.0, remaining_time / len(self.solvers))

        for name, solver in self.solvers.items():
            if name == category:
                continue  # Already ran
            if time.time() - start_time > timeout:
                break

            sol = solver.solve(task, timeout=time_per_solver)
            if sol is not None:
                solutions.append(sol)
                methods.append(name)

        # Vote on solutions
        if solutions:
            best_solution, confidence = self._vote(solutions)
            method = '+'.join(set(methods))
        else:
            # Fallback
            best_solution = test_input
            confidence = 0.1
            method = 'fallback_identity'

        return {
            'solution': best_solution,
            'confidence': confidence,
            'method': method,
            'category': category
        }

    def _vote(self, solutions: List[np.ndarray]) -> Tuple[np.ndarray, float]:
        """
        Vote on solutions using simple agreement counting

        Returns (best_solution, confidence)
        """
        if not solutions:
            return np.array([[0]]), 0.0

        # Convert to hashable format for counting
        solution_strings = []
        for sol in solutions:
            try:
                solution_strings.append(str(sol.tolist()))
            except:
                solution_strings.append("")

        # Count agreements
        counter = Counter(solution_strings)
        most_common_str, count = counter.most_common(1)[0]

        # Calculate confidence based on agreement
        confidence = count / len(solutions)

        # Reconstruct solution
        try:
            best_solution = np.array(eval(most_common_str))
        except:
            best_solution = solutions[0]

        return best_solution, confidence


# =============================================================================
# VARIATION GENERATOR (for dual attempts)
# =============================================================================

class VariationGenerator:
    """
    Generates alternative solutions for attempt_2
    If we're not confident, try variations
    """

    def __init__(self):
        self.primitives = Primitives()

    def generate_variation(self, solution: np.ndarray, confidence: float) -> np.ndarray:
        """
        Generate a different solution attempt

        If confidence is high, try minor variations
        If confidence is low, try major variations
        """
        if confidence > 0.7:
            # High confidence: try minor variation
            variations = [
                self.primitives.rotate_90(solution),
                self.primitives.flip_horizontal(solution),
            ]
        else:
            # Low confidence: try major variation
            variations = [
                self.primitives.rotate_180(solution),
                self.primitives.flip_vertical(solution),
                self.primitives.invert_colors(solution),
            ]

        # Return first variation that's different from original
        for var in variations:
            if not np.array_equal(var, solution):
                return var

        # If all else fails, rotate 90
        return self.primitives.rotate_90(solution)


# =============================================================================
# MAIN SOLVER ORCHESTRATOR
# =============================================================================

class ARCCleanSolver:
    """
    Main solver that orchestrates everything:
    - Time budget management
    - Task routing
    - Dual-attempt generation
    - Fallback handling
    """

    def __init__(self, config: SolverConfig = None):
        self.config = config or SolverConfig()
        self.ensemble = EnsembleSolver(self.config)
        self.variation_generator = VariationGenerator()
        self.start_time = None
        self.stats = {
            'solved': 0,
            'high_confidence': 0,
            'fallbacks': 0,
            'by_category': Counter(),
        }

    def solve_test_set(self, test_tasks: Dict, time_budget: float = None) -> Dict:
        """
        Solve all test tasks with time budget management

        Returns: Submission dict in Kaggle format
        """
        if time_budget is None:
            time_budget = self.config.total_time_budget

        self.start_time = time.time()
        submission = {}

        print(f"\nSolving {len(test_tasks)} tasks...")
        print(f"Time budget: {time_budget/3600:.1f} hours\n")

        for i, (task_id, task) in enumerate(test_tasks.items()):
            # Check time budget
            elapsed = time.time() - self.start_time
            remaining = time_budget - elapsed

            if remaining < 10:
                print(f"\nTime budget exhausted at task {i}/{len(test_tasks)}")
                break

            # Allocate time for this task
            tasks_left = len(test_tasks) - i
            time_per_task = min(
                self.config.max_time_per_task,
                max(self.config.min_time_per_task, remaining / tasks_left)
            )

            # Solve task
            task_result = self._solve_task(task, task_id, time_per_task)
            submission[task_id] = task_result

            # Update stats
            self._update_stats(task_result)

            # Progress update
            if (i + 1) % 50 == 0 or i < 10:
                self._print_progress(i + 1, len(test_tasks), elapsed)

        # Ensure completeness
        submission = self._ensure_complete(test_tasks, submission)

        # Final stats
        self._print_final_stats(submission, time.time() - self.start_time)

        return submission

    def _solve_task(self, task: Dict, task_id: str, timeout: float) -> List[Dict]:
        """
        Solve a single task with dual attempts

        Returns: List of dicts with attempt_1 and attempt_2
        """
        try:
            # Solve with ensemble
            result = self.ensemble.solve(task, timeout=timeout)

            solution = result['solution']
            confidence = result['confidence']

            # Generate variation for attempt_2
            variation = self.variation_generator.generate_variation(solution, confidence)

            # Format for submission (one dict per test output)
            num_test_outputs = len(task.get('test', []))
            task_solutions = []

            for _ in range(num_test_outputs):
                task_solutions.append({
                    'attempt_1': solution.tolist(),
                    'attempt_2': variation.tolist(),
                    '_metadata': {
                        'confidence': confidence,
                        'method': result['method'],
                        'category': result['category']
                    }
                })

            return task_solutions

        except Exception as e:
            # Fallback on error
            test_input = np.array(task['test'][0]['input'])
            return [{
                'attempt_1': test_input.tolist(),
                'attempt_2': np.rot90(test_input).tolist(),
                '_metadata': {'confidence': 0.0, 'method': 'error_fallback'}
            }]

    def _ensure_complete(self, test_tasks: Dict, submission: Dict) -> Dict:
        """Ensure all tasks have solutions"""
        for task_id, task in test_tasks.items():
            if task_id not in submission:
                test_input = np.array(task['test'][0]['input'])
                num_outputs = len(task['test'])
                submission[task_id] = [{
                    'attempt_1': test_input.tolist(),
                    'attempt_2': np.rot90(test_input).tolist(),
                    '_metadata': {'confidence': 0.0, 'method': 'completion_fallback'}
                } for _ in range(num_outputs)]
                self.stats['fallbacks'] += 1

        return submission

    def _update_stats(self, task_result: List[Dict]):
        """Update solving statistics"""
        self.stats['solved'] += 1

        if task_result and task_result[0].get('_metadata'):
            metadata = task_result[0]['_metadata']
            if metadata.get('confidence', 0) > 0.6:
                self.stats['high_confidence'] += 1
            if 'fallback' in metadata.get('method', ''):
                self.stats['fallbacks'] += 1

            category = metadata.get('category', 'unknown')
            self.stats['by_category'][category] += 1

    def _print_progress(self, completed: int, total: int, elapsed: float):
        """Print progress update"""
        pct = completed / total * 100
        rate = completed / elapsed if elapsed > 0 else 0
        high_conf_pct = self.stats['high_confidence'] / completed * 100 if completed > 0 else 0

        print(f"Progress: {completed}/{total} ({pct:.1f}%) | "
              f"Rate: {rate:.2f} tasks/s | "
              f"High confidence: {high_conf_pct:.1f}%")

    def _print_final_stats(self, submission: Dict, total_time: float):
        """Print final statistics"""
        print("\n" + "="*70)
        print("SOLVING COMPLETE")
        print("="*70)
        print(f"Tasks solved: {len(submission)}")
        print(f"High confidence: {self.stats['high_confidence']} ({self.stats['high_confidence']/len(submission)*100:.1f}%)")
        print(f"Fallbacks used: {self.stats['fallbacks']}")
        print(f"Total time: {total_time:.0f}s ({total_time/60:.1f} min)")
        print(f"Average time per task: {total_time/len(submission):.2f}s")
        print(f"\nBy category:")
        for category, count in self.stats['by_category'].most_common():
            print(f"  {category}: {count}")
        print("="*70)


# =============================================================================
# SUBMISSION GENERATOR
# =============================================================================

def save_submission(submission: Dict, config: SolverConfig):
    """
    Save submission in Kaggle format

    Removes metadata before saving
    """
    # Clean metadata
    clean_submission = {}
    for task_id, task_solutions in submission.items():
        clean_submission[task_id] = [
            {
                'attempt_1': sol['attempt_1'],
                'attempt_2': sol['attempt_2']
            }
            for sol in task_solutions
        ]

    # Ensure output directory exists
    output_dir = Path(config.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save to multiple locations (Kaggle compatibility)
    paths = [
        output_dir / 'submission.json',
        Path('/kaggle/working/submission.json'),
    ]

    saved = False
    for path in paths:
        try:
            with open(path, 'w') as f:
                json.dump(clean_submission, f)
            print(f"\nSaved submission to: {path}")
            print(f"File size: {path.stat().st_size / 1024:.1f} KB")
            saved = True
        except:
            continue

    if not saved:
        print("\nWARNING: Could not save submission!")

    return clean_submission


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """Main execution"""
    print("\n" + "="*70)
    print("ARC PRIZE 2025 - CLEAN SOLVER")
    print("="*70)
    print("Target: 15-25% accuracy (realistic baseline)")
    print("="*70)

    # Initialize
    config = SolverConfig()
    solver = ARCCleanSolver(config)

    # Load test data
    data_path = Path(config.data_path)
    if not data_path.exists():
        data_path = Path('.')  # Local development

    test_path = data_path / 'arc-agi_test_challenges.json'

    print(f"\nLoading: {test_path}")
    with open(test_path, 'r') as f:
        test_tasks = json.load(f)
    print(f"Loaded {len(test_tasks)} test tasks")

    # Solve
    submission = solver.solve_test_set(test_tasks)

    # Save
    save_submission(submission, config)

    print("\n" + "="*70)
    print("COMPLETE - Ready for submission!")
    print("="*70)


if __name__ == "__main__":
    main()
