#!/usr/bin/env python3
"""
ARC Prize 2025 Submission Analysis Script
Analyzes submission.json file against ARC Prize scoring criteria
"""

import json
import sys
from collections import defaultdict

def load_json_file(filepath):
    """Load a JSON file and return the data."""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        sys.exit(1)

def analyze_submission(submission_path, evaluation_solutions_path=None):
    """
    Analyze the submission file for ARC Prize 2025.

    ARC Prize 2025 Scoring:
    - Each task can have 1-3 test outputs
    - For each test output, you get 2 attempts
    - Score = (correct answers) / (total test outputs)
    - Maximum score: 100% (all test outputs correct)
    """

    print("="*80)
    print("ARC PRIZE 2025 SUBMISSION ANALYSIS")
    print("="*80)
    print()

    # Load submission
    submission = load_json_file(submission_path)

    # Load evaluation solutions if provided
    solutions = None
    if evaluation_solutions_path:
        solutions = load_json_file(evaluation_solutions_path)

    # Analysis metrics
    total_tasks = len(submission)
    total_test_outputs = 0
    total_attempts = 0
    tasks_with_multiple_tests = 0
    empty_attempts = 0
    non_empty_attempts = 0

    # Grid size analysis
    grid_sizes = defaultdict(int)

    # Task details
    task_details = []

    print(f"Total Tasks in Submission: {total_tasks}")
    print()

    for task_id, attempts in submission.items():
        # In ARC format, submission is {task_id: [attempt1_grid, attempt2_grid]}
        # or {task_id: [{attempt_1: grid, attempt_2: grid}]} for test format

        # Detect format
        if isinstance(attempts, list) and len(attempts) > 0:
            # Check if it's the dict format (with attempt_1/attempt_2 keys)
            if isinstance(attempts[0], dict) and 'attempt_1' in attempts[0]:
                # Old format - keep existing logic
                num_tests = len(attempts)
                total_test_outputs += num_tests

                if num_tests > 1:
                    tasks_with_multiple_tests += 1

                task_info = {
                    'task_id': task_id,
                    'num_test_outputs': num_tests,
                    'attempts': []
                }

                for test_idx, test_dict in enumerate(attempts):
                    for attempt_key in ['attempt_1', 'attempt_2']:
                        total_attempts += 1

                        if attempt_key in test_dict:
                            grid = test_dict[attempt_key]

                            # Check if grid is empty (all zeros or trivial)
                            is_empty = True
                            if grid and len(grid) > 0:
                                for row in grid:
                                    if any(cell != 0 for cell in row):
                                        is_empty = False
                                        break

                            if is_empty:
                                empty_attempts += 1
                            else:
                                non_empty_attempts += 1

                            # Analyze grid size
                            grid_size = 'empty'
                            if grid:
                                grid_size = f"{len(grid)}x{len(grid[0]) if grid[0] else 0}"
                                grid_sizes[grid_size] += 1

                            task_info['attempts'].append({
                                'test_idx': test_idx,
                                'attempt': attempt_key,
                                'grid_size': grid_size,
                                'is_empty': is_empty
                            })
            else:
                # New format - attempts is just a list of grids [grid1, grid2]
                # Each task has 2 attempts for a single test output
                num_tests = 1  # One test output per task
                total_test_outputs += 1

                task_info = {
                    'task_id': task_id,
                    'num_test_outputs': num_tests,
                    'attempts': []
                }

                for attempt_idx, grid in enumerate(attempts):
                    total_attempts += 1
                    attempt_name = f"attempt_{attempt_idx + 1}"

                    # Check if grid is empty (all zeros or trivial)
                    is_empty = True
                    if grid and len(grid) > 0:
                        for row in grid:
                            if any(cell != 0 for cell in row):
                                is_empty = False
                                break

                    if is_empty:
                        empty_attempts += 1
                    else:
                        non_empty_attempts += 1

                    # Analyze grid size
                    grid_size = 'empty'
                    if grid:
                        grid_size = f"{len(grid)}x{len(grid[0]) if grid[0] else 0}"
                        grid_sizes[grid_size] += 1

                    task_info['attempts'].append({
                        'test_idx': 0,
                        'attempt': attempt_name,
                        'grid_size': grid_size,
                        'is_empty': is_empty
                    })

        task_details.append(task_info)

    # Print summary statistics
    print("SUBMISSION STATISTICS:")
    print("-" * 80)
    print(f"Total Tasks:                    {total_tasks}")
    print(f"Total Test Outputs:             {total_test_outputs}")
    print(f"Total Attempts:                 {total_attempts}")
    print(f"Tasks with Multiple Tests:      {tasks_with_multiple_tests}")
    print()
    print(f"Empty/Trivial Attempts:         {empty_attempts} ({empty_attempts/total_attempts*100:.2f}%)")
    print(f"Non-Empty Attempts:             {non_empty_attempts} ({non_empty_attempts/total_attempts*100:.2f}%)")
    print()

    # Grid size distribution
    print("GRID SIZE DISTRIBUTION:")
    print("-" * 80)
    sorted_sizes = sorted(grid_sizes.items(), key=lambda x: x[1], reverse=True)
    for size, count in sorted_sizes[:10]:  # Top 10 sizes
        print(f"{size:>10}: {count:>4} ({count/total_attempts*100:>5.2f}%)")
    print()

    # Scoring potential analysis
    print("SCORING ANALYSIS:")
    print("-" * 80)
    print(f"Maximum Possible Score:         100% ({total_test_outputs} correct out of {total_test_outputs})")
    print(f"Current Non-Empty Rate:         {non_empty_attempts/total_attempts*100:.2f}%")
    print()

    if solutions:
        print("CORRECTNESS ANALYSIS (vs Ground Truth):")
        print("-" * 80)
        correct_first_attempt = 0
        correct_second_attempt = 0
        correct_either_attempt = 0
        tasks_evaluated = 0

        for task_id, attempts in submission.items():
            if task_id in solutions:
                ground_truth = solutions[task_id]
                tasks_evaluated += 1

                # Detect format
                if isinstance(attempts, list) and len(attempts) > 0:
                    if isinstance(attempts[0], dict) and 'attempt_1' in attempts[0]:
                        # Dict format
                        for test_idx, test_dict in enumerate(attempts):
                            if test_idx < len(ground_truth):
                                expected_output = ground_truth[test_idx]

                                attempt_1 = test_dict.get('attempt_1', [])
                                attempt_2 = test_dict.get('attempt_2', [])

                                match_1 = (attempt_1 == expected_output)
                                match_2 = (attempt_2 == expected_output)

                                if match_1:
                                    correct_first_attempt += 1
                                    correct_either_attempt += 1
                                elif match_2:
                                    correct_second_attempt += 1
                                    correct_either_attempt += 1
                    else:
                        # Simple array format [attempt1, attempt2]
                        # Ground truth is also an array, typically with one expected output per task
                        if len(ground_truth) > 0:
                            expected_output = ground_truth[0]  # First test output

                            if len(attempts) >= 1:
                                match_1 = (attempts[0] == expected_output)
                                if match_1:
                                    correct_first_attempt += 1
                                    correct_either_attempt += 1
                                elif len(attempts) >= 2:
                                    match_2 = (attempts[1] == expected_output)
                                    if match_2:
                                        correct_second_attempt += 1
                                        correct_either_attempt += 1

        print(f"Tasks Evaluated:                {tasks_evaluated}")
        print(f"Correct on First Attempt:       {correct_first_attempt} ({correct_first_attempt/total_test_outputs*100:.2f}%)")
        print(f"Correct on Second Attempt:      {correct_second_attempt} ({correct_second_attempt/total_test_outputs*100:.2f}%)")
        print(f"Correct on Either Attempt:      {correct_either_attempt} ({correct_either_attempt/total_test_outputs*100:.2f}%)")
        print()
        print(f"FINAL SCORE:                    {correct_either_attempt/total_test_outputs*100:.2f}%")
        print(f"                                ({correct_either_attempt} out of {total_test_outputs} test outputs correct)")
        print()

    # Sample task details
    print("SAMPLE TASK DETAILS (first 5 tasks):")
    print("-" * 80)
    for task_info in task_details[:5]:
        print(f"Task: {task_info['task_id']}")
        print(f"  Test Outputs: {task_info['num_test_outputs']}")
        for attempt_info in task_info['attempts']:
            status = "EMPTY" if attempt_info['is_empty'] else "NON-EMPTY"
            print(f"    Test {attempt_info['test_idx']} - {attempt_info['attempt']}: "
                  f"{attempt_info['grid_size']} [{status}]")
        print()

    return {
        'total_tasks': total_tasks,
        'total_test_outputs': total_test_outputs,
        'empty_attempts': empty_attempts,
        'non_empty_attempts': non_empty_attempts,
        'grid_sizes': dict(grid_sizes)
    }

if __name__ == "__main__":
    submission_file = "/home/user/HungryOrca/submission.json"
    evaluation_solutions_file = "/home/user/HungryOrca/arc-agi_evaluation_solutions.json"

    print(f"Analyzing: {submission_file}")
    print()

    analyze_submission(submission_file, evaluation_solutions_file)
