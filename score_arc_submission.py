#!/usr/bin/env python3
"""
ARC Prize 2025 Submission Scorer
Evaluates submission.json against the evaluation solutions
"""

import json
from typing import Dict, List, Any, Tuple
from collections import defaultdict

def load_json(filepath: str) -> Dict:
    """Load JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)

def compare_grids(grid1: List[List[int]], grid2: List[List[int]]) -> bool:
    """Compare two grids for exact match"""
    if len(grid1) != len(grid2):
        return False
    for row1, row2 in zip(grid1, grid2):
        if len(row1) != len(row2):
            return False
        if row1 != row2:
            return False
    return True

def score_task(submission_outputs: List[Dict], solution_outputs: List[List[List[int]]]) -> Tuple[bool, Dict]:
    """
    Score a single task by checking if any attempt matches the solution
    Returns: (is_correct, details)
    """
    details = {
        'num_test_cases': len(solution_outputs),
        'test_case_results': [],
        'solved': False
    }
    
    # For each test case in the solution
    for test_idx, solution_grid in enumerate(solution_outputs):
        test_result = {
            'test_index': test_idx,
            'attempt_1_correct': False,
            'attempt_2_correct': False,
            'any_attempt_correct': False
        }
        
        if test_idx < len(submission_outputs):
            submission_test = submission_outputs[test_idx]
            
            # Check attempt_1
            if 'attempt_1' in submission_test:
                test_result['attempt_1_correct'] = compare_grids(
                    submission_test['attempt_1'], 
                    solution_grid
                )
            
            # Check attempt_2
            if 'attempt_2' in submission_test:
                test_result['attempt_2_correct'] = compare_grids(
                    submission_test['attempt_2'], 
                    solution_grid
                )
            
            test_result['any_attempt_correct'] = (
                test_result['attempt_1_correct'] or 
                test_result['attempt_2_correct']
            )
        
        details['test_case_results'].append(test_result)
    
    # Task is solved if ANY attempt correctly solves ALL test cases
    all_test_cases_solved = all(
        result['any_attempt_correct'] 
        for result in details['test_case_results']
    )
    details['solved'] = all_test_cases_solved
    
    return details['solved'], details

def score_submission(submission_path: str, solutions_path: str) -> Dict:
    """Score entire submission against solutions"""
    print("Loading submission and solutions...")
    submission = load_json(submission_path)
    solutions = load_json(solutions_path)
    
    results = {
        'total_tasks_in_submission': len(submission),
        'total_tasks_in_solutions': len(solutions),
        'tasks_evaluated': 0,
        'tasks_solved': 0,
        'tasks_unsolved': 0,
        'tasks_missing_from_submission': [],
        'tasks_extra_in_submission': [],
        'detailed_results': {},
        'solved_task_ids': [],
        'unsolved_task_ids': []
    }
    
    # Find tasks in solutions but not in submission
    solution_task_ids = set(solutions.keys())
    submission_task_ids = set(submission.keys())
    
    results['tasks_missing_from_submission'] = list(solution_task_ids - submission_task_ids)
    results['tasks_extra_in_submission'] = list(submission_task_ids - solution_task_ids)
    
    print(f"\nEvaluating {len(solution_task_ids)} tasks from solutions...")
    
    # Score each task that has a solution
    for task_id in solution_task_ids:
        if task_id in submission:
            is_solved, details = score_task(
                submission[task_id], 
                solutions[task_id]
            )
            
            results['detailed_results'][task_id] = details
            results['tasks_evaluated'] += 1
            
            if is_solved:
                results['tasks_solved'] += 1
                results['solved_task_ids'].append(task_id)
            else:
                results['tasks_unsolved'] += 1
                results['unsolved_task_ids'].append(task_id)
        else:
            # Task missing from submission - counts as unsolved
            results['tasks_unsolved'] += 1
            results['unsolved_task_ids'].append(task_id)
            results['detailed_results'][task_id] = {
                'solved': False,
                'error': 'Missing from submission'
            }
    
    # Calculate score percentage
    results['score_percentage'] = (
        (results['tasks_solved'] / len(solution_task_ids)) * 100 
        if len(solution_task_ids) > 0 else 0
    )
    
    return results

def print_summary(results: Dict):
    """Print a formatted summary of results"""
    print("\n" + "="*60)
    print(" ARC PRIZE 2025 SUBMISSION SCORE ")
    print("="*60)
    
    print(f"\nTOTAL SCORE: {results['tasks_solved']}/{results['total_tasks_in_solutions']} tasks solved")
    print(f"PERCENTAGE: {results['score_percentage']:.2f}%")
    
    print(f"\n📊 BREAKDOWN:")
    print(f"  • Tasks in evaluation set: {results['total_tasks_in_solutions']}")
    print(f"  • Tasks in submission: {results['total_tasks_in_submission']}")
    print(f"  • Tasks evaluated: {results['tasks_evaluated']}")
    print(f"  • Tasks solved: {results['tasks_solved']} ✅")
    print(f"  • Tasks unsolved: {results['tasks_unsolved']} ❌")
    
    if results['tasks_missing_from_submission']:
        print(f"\n⚠️  MISSING FROM SUBMISSION: {len(results['tasks_missing_from_submission'])} tasks")
        print(f"    First 5: {results['tasks_missing_from_submission'][:5]}")
    
    if results['tasks_extra_in_submission']:
        print(f"\n📝 EXTRA IN SUBMISSION (not in evaluation): {len(results['tasks_extra_in_submission'])} tasks")
        print(f"    First 5: {results['tasks_extra_in_submission'][:5]}")
    
    if results['solved_task_ids']:
        print(f"\n✅ SOLVED TASKS (First 10):")
        for task_id in results['solved_task_ids'][:10]:
            print(f"    • {task_id}")
    
    if results['unsolved_task_ids']:
        print(f"\n❌ UNSOLVED TASKS (First 10):")
        for task_id in results['unsolved_task_ids'][:10]:
            if task_id in results['detailed_results']:
                detail = results['detailed_results'][task_id]
                if 'error' in detail:
                    print(f"    • {task_id}: {detail['error']}")
                else:
                    correct_tests = sum(1 for r in detail.get('test_case_results', []) 
                                      if r['any_attempt_correct'])
                    total_tests = detail.get('num_test_cases', 0)
                    print(f"    • {task_id}: {correct_tests}/{total_tests} test cases correct")
    
    # Analyze partial successes
    partial_successes = []
    for task_id, details in results['detailed_results'].items():
        if not details.get('solved', False) and 'test_case_results' in details:
            correct_count = sum(1 for r in details['test_case_results'] 
                              if r['any_attempt_correct'])
            total_count = len(details['test_case_results'])
            if correct_count > 0:
                partial_successes.append((task_id, correct_count, total_count))
    
    if partial_successes:
        print(f"\n📈 PARTIAL SUCCESSES (solved some but not all test cases):")
        partial_successes.sort(key=lambda x: x[1]/x[2], reverse=True)
        for task_id, correct, total in partial_successes[:10]:
            percentage = (correct/total) * 100
            print(f"    • {task_id}: {correct}/{total} test cases ({percentage:.1f}%)")
    
    print("\n" + "="*60)
    
    # Competition context
    print("\n📌 ARC PRIZE 2025 CONTEXT:")
    print("  • Passing threshold: Usually ~85% of tasks solved")
    print("  • Top performance: ~95%+ of tasks solved")
    print("  • Your current performance places you at:", end=" ")
    
    if results['score_percentage'] >= 95:
        print("ELITE TIER 🏆")
    elif results['score_percentage'] >= 85:
        print("COMPETITIVE TIER 🥇")
    elif results['score_percentage'] >= 70:
        print("STRONG PROGRESS 🥈")
    elif results['score_percentage'] >= 50:
        print("SOLID FOUNDATION 🥉")
    else:
        print("DEVELOPMENT PHASE 🚀")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    submission_path = "/mnt/user-data/uploads/submission.json"
    solutions_path = "/mnt/user-data/uploads/arc-agi_evaluation_solutions.json"
    
    results = score_submission(submission_path, solutions_path)
    print_summary(results)
    
    # Save detailed results
    output_path = "/mnt/user-data/outputs/arc_score_detailed.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Detailed results saved to: {output_path}")
