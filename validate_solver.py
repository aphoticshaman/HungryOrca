"""
ARC SOLVER VALIDATION & ACCURACY TESTING
=========================================

Validate the solver against training and evaluation datasets where solutions are known.
"""

import json
import numpy as np
from arc_solver_production import ARCSolver, ARCGrid
from typing import Dict, List, Tuple


class SolverValidator:
    """Validate solver performance on datasets with known solutions."""

    def __init__(self):
        self.solver = ARCSolver()

    def validate_on_dataset(self, challenges_path: str, solutions_path: str,
                          dataset_name: str = "Dataset", max_tasks: int = None) -> Dict:
        """
        Validate solver on a dataset with known solutions.

        Args:
            challenges_path: Path to challenges JSON
            solutions_path: Path to solutions JSON
            dataset_name: Name for reporting
            max_tasks: Max tasks to test (None = all)

        Returns:
            Dict with validation results
        """
        print(f"\n{'='*60}")
        print(f"VALIDATING ON {dataset_name}")
        print(f"{'='*60}\n")

        # Load data
        with open(challenges_path, 'r') as f:
            challenges = json.load(f)

        with open(solutions_path, 'r') as f:
            solutions = json.load(f)

        results = {
            'total_tasks': 0,
            'perfect_matches': 0,
            'partial_matches': 0,
            'failures': 0,
            'task_results': {}
        }

        task_ids = list(challenges.keys())
        if max_tasks:
            task_ids = task_ids[:max_tasks]

        for idx, task_id in enumerate(task_ids, 1):
            task = challenges[task_id]
            expected_solutions = solutions[task_id]

            results['total_tasks'] += 1

            try:
                # Test each test example in the task
                task_perfect = True
                task_partial = False

                for test_idx, test_pair in enumerate(task['test']):
                    # Create modified task with this test example
                    test_task = {
                        'train': task['train'],
                        'test': [{'input': test_pair['input']}]
                    }

                    # Get solver predictions
                    predictions = self.solver.solve_task(test_task)

                    # Get expected output
                    expected = expected_solutions[test_idx]

                    # Check if either prediction matches
                    match_found = False
                    for pred in predictions:
                        if self._grids_equal(pred, expected):
                            match_found = True
                            break
                        elif self._compute_similarity(pred, expected) > 0.8:
                            task_partial = True

                    if not match_found:
                        task_perfect = False

                # Score the task
                if task_perfect:
                    results['perfect_matches'] += 1
                    results['task_results'][task_id] = 'perfect'
                elif task_partial:
                    results['partial_matches'] += 1
                    results['task_results'][task_id] = 'partial'
                else:
                    results['failures'] += 1
                    results['task_results'][task_id] = 'failed'

            except Exception as e:
                results['failures'] += 1
                results['task_results'][task_id] = f'error: {str(e)[:50]}'

            # Progress update
            if idx % 10 == 0 or idx == len(task_ids):
                perfect_pct = (results['perfect_matches'] / results['total_tasks']) * 100
                print(f"Progress: {idx}/{len(task_ids)} | "
                      f"Perfect: {results['perfect_matches']} ({perfect_pct:.1f}%) | "
                      f"Partial: {results['partial_matches']} | "
                      f"Failed: {results['failures']}")

        # Final stats
        perfect_pct = (results['perfect_matches'] / results['total_tasks']) * 100
        partial_pct = (results['partial_matches'] / results['total_tasks']) * 100

        print(f"\n{'='*60}")
        print(f"RESULTS for {dataset_name}:")
        print(f"{'='*60}")
        print(f"Total Tasks:      {results['total_tasks']}")
        print(f"Perfect Matches:  {results['perfect_matches']} ({perfect_pct:.1f}%)")
        print(f"Partial Matches:  {results['partial_matches']} ({partial_pct:.1f}%)")
        print(f"Failures:         {results['failures']}")
        print(f"{'='*60}\n")

        return results

    def _grids_equal(self, grid1: List[List[int]], grid2: List[List[int]]) -> bool:
        """Check if two grids are exactly equal."""
        arr1 = np.array(grid1)
        arr2 = np.array(grid2)

        if arr1.shape != arr2.shape:
            return False

        return np.array_equal(arr1, arr2)

    def _compute_similarity(self, grid1: List[List[int]], grid2: List[List[int]]) -> float:
        """Compute similarity between two grids."""
        arr1 = np.array(grid1)
        arr2 = np.array(grid2)

        if arr1.shape != arr2.shape:
            return 0.0

        matches = np.sum(arr1 == arr2)
        total = arr1.size

        return matches / total if total > 0 else 0.0

    def validate_submission_format(self, submission_path: str,
                                   sample_path: str) -> bool:
        """
        Validate that submission matches the expected format.

        Args:
            submission_path: Path to generated submission.json
            sample_path: Path to sample_submission.json

        Returns:
            True if format is valid
        """
        print(f"\n{'='*60}")
        print(f"VALIDATING SUBMISSION FORMAT")
        print(f"{'='*60}\n")

        with open(submission_path, 'r') as f:
            submission = json.load(f)

        with open(sample_path, 'r') as f:
            sample = json.load(f)

        issues = []

        # Check number of tasks
        if len(submission) != len(sample):
            issues.append(f"Task count mismatch: {len(submission)} vs {len(sample)} expected")

        # Check task IDs match
        submission_ids = set(submission.keys())
        sample_ids = set(sample.keys())

        missing_ids = sample_ids - submission_ids
        if missing_ids:
            issues.append(f"Missing task IDs: {list(missing_ids)[:5]}...")

        extra_ids = submission_ids - sample_ids
        if extra_ids:
            issues.append(f"Extra task IDs: {list(extra_ids)[:5]}...")

        # Check format of each task
        for task_id in list(submission.keys())[:5]:  # Sample first 5
            task_data = submission[task_id]

            if not isinstance(task_data, list):
                issues.append(f"Task {task_id}: Should be a list, got {type(task_data)}")
                continue

            if len(task_data) != 1:
                issues.append(f"Task {task_id}: Should have 1 element, got {len(task_data)}")
                continue

            attempts = task_data[0]
            if not isinstance(attempts, dict):
                issues.append(f"Task {task_id}: Attempts should be dict, got {type(attempts)}")
                continue

            if 'attempt_1' not in attempts or 'attempt_2' not in attempts:
                issues.append(f"Task {task_id}: Missing attempt_1 or attempt_2")
                continue

            # Check each attempt is a valid grid
            for attempt_name in ['attempt_1', 'attempt_2']:
                attempt = attempts[attempt_name]
                if not isinstance(attempt, list):
                    issues.append(f"Task {task_id} {attempt_name}: Should be list")
                    continue

                if not all(isinstance(row, list) for row in attempt):
                    issues.append(f"Task {task_id} {attempt_name}: All rows should be lists")

        if issues:
            print("❌ FORMAT VALIDATION FAILED:\n")
            for issue in issues:
                print(f"  - {issue}")
            print()
            return False
        else:
            print("✅ FORMAT VALIDATION PASSED")
            print(f"  - {len(submission)} tasks")
            print(f"  - All task IDs match sample")
            print(f"  - All formats correct (attempt_1, attempt_2)")
            print()
            return True


def main():
    """Run full validation suite."""
    validator = SolverValidator()

    # Test on smaller samples first
    print("="*60)
    print("ARC SOLVER VALIDATION SUITE")
    print("="*60)

    # 1. Validate on training data (sample)
    train_results = validator.validate_on_dataset(
        'arc-agi_training_challenges.json',
        'arc-agi_training_solutions.json',
        dataset_name="TRAINING SET (50 tasks)",
        max_tasks=50  # Test on first 50 for speed
    )

    # 2. Validate on evaluation data (sample)
    eval_results = validator.validate_on_dataset(
        'arc-agi_evaluation_challenges.json',
        'arc-agi_evaluation_solutions.json',
        dataset_name="EVALUATION SET (50 tasks)",
        max_tasks=50
    )

    # 3. Validate submission format
    format_valid = validator.validate_submission_format(
        'submission.json',
        'sample_submission.json'
    )

    # Summary
    print("\n" + "="*60)
    print("OVERALL SUMMARY")
    print("="*60)

    train_accuracy = (train_results['perfect_matches'] / train_results['total_tasks']) * 100
    eval_accuracy = (eval_results['perfect_matches'] / eval_results['total_tasks']) * 100

    print(f"\nTraining Accuracy:   {train_accuracy:.1f}% ({train_results['perfect_matches']}/{train_results['total_tasks']})")
    print(f"Evaluation Accuracy: {eval_accuracy:.1f}% ({eval_results['perfect_matches']}/{eval_results['total_tasks']})")
    print(f"Format Valid:        {'✅ Yes' if format_valid else '❌ No'}")

    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)

    if train_accuracy < 10:
        print("⚠️  Low accuracy detected. Consider:")
        print("   - Implementing more sophisticated pattern matching")
        print("   - Adding DSL (Domain Specific Language) synthesis")
        print("   - Using the fuzzy meta-controller framework")
        print("   - Training on specific pattern types")
    elif train_accuracy < 30:
        print("📊 Moderate accuracy. Improvements possible:")
        print("   - Tune strategy weights")
        print("   - Add more transformation strategies")
        print("   - Implement the 5 physics-inspired insights fully")
    else:
        print("✅ Good baseline performance!")
        print("   - Continue refining strategies")
        print("   - Test on full evaluation set")

    print("\n🎮 Validation complete! WAKA WAKA! 🧠⚡\n")


if __name__ == "__main__":
    main()
