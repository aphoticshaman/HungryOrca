"""
Validation Harness - Test Solvers on Training Data with Ground Truth

This is THE critical infrastructure piece that prevents:
- Building in a vacuum
- Discovering bugs in Kaggle
- Guessing accuracy instead of measuring it

LAELD Rule #1: VALIDATE ON TRAINING DATA BEFORE KAGGLE

Usage:
    from validation_harness import ValidationHarness

    harness = ValidationHarness('data/arc-agi_training_challenges.json')
    results = harness.validate(ensemble_solver)
    harness.print_report(results)
"""

import json
import numpy as np
from typing import Dict, List, Callable, Any, Tuple
from collections import defaultdict
from pathlib import Path
import time


class ValidationHarness:
    """Test solvers on training data with ground truth"""

    def __init__(self, training_data_path: str):
        """Initialize with training data path"""
        self.training_data_path = training_data_path
        self.training_data = self._load_training_data()

    def _load_training_data(self) -> Dict:
        """Load training challenges"""
        path = Path(self.training_data_path)
        if not path.exists():
            raise FileNotFoundError(f"Training data not found: {self.training_data_path}")

        with open(path, 'r') as f:
            return json.load(f)

    def validate(self,
                solver_func: Callable,
                max_tasks: int = None,
                verbose: bool = True) -> Dict:
        """
        Validate solver on training data

        Args:
            solver_func: Function(test_input, task_data, attempt) -> prediction
            max_tasks: Limit to first N tasks (for quick testing)
            verbose: Print progress

        Returns:
            Dict with comprehensive validation results
        """
        results = {
            'total_tasks': 0,
            'total_test_cases': 0,
            'correct': 0,
            'incorrect': 0,
            'errors': 0,
            'solver_triggers': defaultdict(int),
            'solver_correct': defaultdict(int),
            'solver_incorrect': defaultdict(int),
            'task_results': {},
            'error_log': []
        }

        tasks = list(self.training_data.items())
        if max_tasks:
            tasks = tasks[:max_tasks]

        start_time = time.time()

        for task_idx, (task_id, task_data) in enumerate(tasks, 1):
            results['total_tasks'] += 1

            if verbose and task_idx % 50 == 0:
                elapsed = time.time() - start_time
                rate = task_idx / elapsed
                eta = (len(tasks) - task_idx) / rate
                print(f"  Progress: {task_idx}/{len(tasks)} "
                      f"({task_idx/len(tasks)*100:.1f}%) | "
                      f"Rate: {rate:.1f} tasks/s | ETA: {eta:.0f}s")

            task_result = self._validate_task(task_id, task_data, solver_func)
            results['task_results'][task_id] = task_result

            # Aggregate stats
            results['total_test_cases'] += task_result['test_cases']
            results['correct'] += task_result['correct']
            results['incorrect'] += task_result['incorrect']
            results['errors'] += task_result['errors']

            # Track solver performance
            if task_result['solver_name']:
                solver = task_result['solver_name']
                results['solver_triggers'][solver] += 1
                if task_result['correct'] > 0:
                    results['solver_correct'][solver] += task_result['correct']
                if task_result['incorrect'] > 0:
                    results['solver_incorrect'][solver] += task_result['incorrect']

        elapsed = time.time() - start_time
        results['elapsed_seconds'] = elapsed
        results['tasks_per_second'] = results['total_tasks'] / elapsed

        # Calculate derived metrics
        results['overall_accuracy'] = (
            results['correct'] / max(1, results['total_test_cases'])
        )

        # Per-solver metrics
        results['solver_stats'] = {}
        for solver in results['solver_triggers'].keys():
            triggers = results['solver_triggers'][solver]
            correct = results['solver_correct'][solver]
            incorrect = results['solver_incorrect'][solver]

            coverage = triggers / results['total_tasks']
            accuracy = correct / max(1, correct + incorrect)
            contribution = coverage * accuracy

            results['solver_stats'][solver] = {
                'triggers': triggers,
                'coverage': coverage,
                'correct': correct,
                'incorrect': incorrect,
                'accuracy': accuracy,
                'contribution': contribution
            }

        return results

    def _validate_task(self,
                      task_id: str,
                      task_data: Dict,
                      solver_func: Callable) -> Dict:
        """Validate solver on a single task"""
        task_result = {
            'task_id': task_id,
            'test_cases': 0,
            'correct': 0,
            'incorrect': 0,
            'errors': 0,
            'solver_name': None
        }

        # Test on each test case in the task
        for test_idx, test_pair in enumerate(task_data['test']):
            task_result['test_cases'] += 1

            test_input = test_pair['input']
            expected_output = test_pair['output']

            try:
                # Run solver (attempt 1)
                prediction = solver_func(test_input, task_data, attempt=1, task_id=task_id)

                # Extract solver name if possible (from logs or return value)
                # For now, assume solver_func returns just the grid

                # Compare prediction to ground truth
                if self._grids_equal(prediction, expected_output):
                    task_result['correct'] += 1
                else:
                    task_result['incorrect'] += 1

            except Exception as e:
                task_result['errors'] += 1
                # Log error but don't crash

        return task_result

    def _grids_equal(self, grid1: Any, grid2: Any) -> bool:
        """Check if two grids are equal"""
        try:
            arr1 = np.array(grid1)
            arr2 = np.array(grid2)
            return np.array_equal(arr1, arr2)
        except:
            return False

    def print_report(self, results: Dict, verbose: bool = True):
        """Print comprehensive validation report"""
        print("\n" + "="*70)
        print("VALIDATION HARNESS REPORT")
        print("="*70)

        # Overall metrics
        print(f"\n📊 OVERALL PERFORMANCE")
        print(f"  Tasks tested: {results['total_tasks']}")
        print(f"  Test cases: {results['total_test_cases']}")
        print(f"  Correct: {results['correct']} ({results['correct']/results['total_test_cases']*100:.1f}%)")
        print(f"  Incorrect: {results['incorrect']} ({results['incorrect']/results['total_test_cases']*100:.1f}%)")
        print(f"  Errors: {results['errors']}")
        print(f"  Overall accuracy: {results['overall_accuracy']*100:.1f}%")

        # Performance
        print(f"\n⏱️  PERFORMANCE")
        print(f"  Time: {results['elapsed_seconds']:.1f}s")
        print(f"  Rate: {results['tasks_per_second']:.1f} tasks/s")

        # Per-solver stats
        if results['solver_stats']:
            print(f"\n🔧 SOLVER BREAKDOWN")
            print(f"  {'Solver':<20} {'Triggers':>8} {'Coverage':>10} {'Accuracy':>10} {'Contribution':>13}")
            print(f"  {'-'*20} {'-'*8} {'-'*10} {'-'*10} {'-'*13}")

            for solver, stats in sorted(results['solver_stats'].items(),
                                       key=lambda x: x[1]['contribution'],
                                       reverse=True):
                print(f"  {solver:<20} "
                      f"{stats['triggers']:>8} "
                      f"{stats['coverage']*100:>9.1f}% "
                      f"{stats['accuracy']*100:>9.1f}% "
                      f"{stats['contribution']*100:>12.1f}%")

        print("\n" + "="*70)
        print("✅ VALIDATION COMPLETE")
        print("="*70 + "\n")

    def quick_validate(self, solver_func: Callable, n_tasks: int = 10) -> Dict:
        """Quick validation on first N tasks"""
        print(f"\n🚀 QUICK VALIDATION ({n_tasks} tasks)")
        return self.validate(solver_func, max_tasks=n_tasks, verbose=False)


# Example usage
if __name__ == "__main__":
    print("Validation Harness - Infrastructure Test")
    print("="*70)
    print("\nThis harness tests solvers on training data with ground truth.")
    print("\nUsage:")
    print("  harness = ValidationHarness('data/arc-agi_training_challenges.json')")
    print("  results = harness.validate(my_solver_function)")
    print("  harness.print_report(results)")
    print("\n" + "="*70)
