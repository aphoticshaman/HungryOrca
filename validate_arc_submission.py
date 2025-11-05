#!/usr/bin/env python3
"""
ARC Prize 2025 Test Submission Validator
Validates submission.json format and completeness for the test set
"""

import json
import numpy as np
from typing import Dict, List, Any, Set, Tuple
from collections import defaultdict, Counter

def load_json(filepath: str) -> Dict:
    """Load JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)

def analyze_grid(grid: List[List[int]]) -> Dict:
    """Analyze a single grid's properties"""
    if not grid or not grid[0]:
        return {'height': 0, 'width': 0, 'colors': set(), 'is_valid': False}
    
    height = len(grid)
    width = len(grid[0])
    colors = set()
    is_rectangular = True
    
    for row in grid:
        if len(row) != width:
            is_rectangular = False
        colors.update(row)
    
    return {
        'height': height,
        'width': width,
        'colors': colors,
        'num_colors': len(colors),
        'is_rectangular': is_rectangular,
        'is_valid': is_rectangular and 0 < height <= 30 and 0 < width <= 30,
        'total_cells': height * width
    }

def validate_submission_format(submission: Dict, test_challenges: Dict) -> Dict:
    """Validate submission format and completeness"""
    results = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'stats': {
            'total_tasks': len(test_challenges),
            'tasks_present': 0,
            'tasks_missing': 0,
            'tasks_with_valid_format': 0,
            'total_predictions': 0,
            'predictions_per_task': defaultdict(int),
            'grid_sizes': [],
            'color_usage': Counter(),
            'attempt_consistency': {'consistent': 0, 'different': 0}
        },
        'missing_tasks': [],
        'invalid_tasks': []
    }
    
    # Check each test task
    for task_id in test_challenges:
        task_data = test_challenges[task_id]
        num_test_cases = len(task_data['test'])
        
        if task_id not in submission:
            results['errors'].append(f"Missing task: {task_id}")
            results['missing_tasks'].append(task_id)
            results['stats']['tasks_missing'] += 1
            results['valid'] = False
            continue
        
        results['stats']['tasks_present'] += 1
        submission_task = submission[task_id]
        
        # Validate structure
        if not isinstance(submission_task, list):
            results['errors'].append(f"Task {task_id}: Should be a list, got {type(submission_task)}")
            results['invalid_tasks'].append(task_id)
            results['valid'] = False
            continue
        
        if len(submission_task) != num_test_cases:
            results['errors'].append(
                f"Task {task_id}: Expected {num_test_cases} test cases, got {len(submission_task)}"
            )
            results['invalid_tasks'].append(task_id)
            results['valid'] = False
            continue
        
        task_valid = True
        
        # Check each test case
        for test_idx, test_output in enumerate(submission_task):
            if not isinstance(test_output, dict):
                results['errors'].append(
                    f"Task {task_id}, test {test_idx}: Should be a dict, got {type(test_output)}"
                )
                task_valid = False
                continue
            
            # Check for required attempts
            if 'attempt_1' not in test_output or 'attempt_2' not in test_output:
                results['errors'].append(
                    f"Task {task_id}, test {test_idx}: Missing attempt_1 or attempt_2"
                )
                task_valid = False
                continue
            
            # Analyze each attempt
            for attempt_name in ['attempt_1', 'attempt_2']:
                attempt = test_output[attempt_name]
                grid_analysis = analyze_grid(attempt)
                
                if not grid_analysis['is_valid']:
                    results['warnings'].append(
                        f"Task {task_id}, test {test_idx}, {attempt_name}: "
                        f"Invalid grid (size: {grid_analysis['height']}x{grid_analysis['width']})"
                    )
                
                results['stats']['grid_sizes'].append(
                    (grid_analysis['height'], grid_analysis['width'])
                )
                results['stats']['color_usage'].update(grid_analysis.get('colors', set()))
                results['stats']['total_predictions'] += 1
            
            # Check if attempts are identical
            if test_output['attempt_1'] == test_output['attempt_2']:
                results['stats']['attempt_consistency']['consistent'] += 1
            else:
                results['stats']['attempt_consistency']['different'] += 1
        
        if task_valid:
            results['stats']['tasks_with_valid_format'] += 1
        else:
            results['invalid_tasks'].append(task_id)
    
    results['stats']['predictions_per_task'] = results['stats']['total_predictions'] / len(test_challenges) if test_challenges else 0
    
    return results

def analyze_prediction_patterns(submission: Dict, test_challenges: Dict) -> Dict:
    """Analyze patterns in the predictions"""
    analysis = {
        'grid_size_distribution': Counter(),
        'color_distribution': Counter(),
        'prediction_diversity': [],
        'suspicious_patterns': [],
        'test_input_similarity': []
    }
    
    for task_id in submission:
        if task_id not in test_challenges:
            continue
        
        task_outputs = submission[task_id]
        task_inputs = test_challenges[task_id]['test']
        
        for test_idx, test_output in enumerate(task_outputs):
            if test_idx >= len(task_inputs):
                continue
            
            input_grid = task_inputs[test_idx]['input']
            
            for attempt_name in ['attempt_1', 'attempt_2']:
                if attempt_name not in test_output:
                    continue
                
                output_grid = test_output[attempt_name]
                
                # Analyze grid sizes
                if output_grid and output_grid[0]:
                    size_key = f"{len(output_grid)}x{len(output_grid[0])}"
                    analysis['grid_size_distribution'][size_key] += 1
                
                # Check if output is identical to input (suspicious)
                if output_grid == input_grid:
                    analysis['suspicious_patterns'].append(
                        f"Task {task_id}, test {test_idx}, {attempt_name}: Output identical to input"
                    )
                
                # Check prediction diversity
                if len(output_grid) > 0 and len(output_grid[0]) > 0:
                    unique_values = set()
                    for row in output_grid:
                        unique_values.update(row)
                    analysis['color_distribution'].update(unique_values)
    
    return analysis

def calculate_quality_metrics(submission: Dict, test_challenges: Dict) -> Dict:
    """Calculate quality metrics for the submission"""
    metrics = {
        'completeness_score': 0,
        'format_validity_score': 0,
        'prediction_diversity_score': 0,
        'attempt_diversity_score': 0,
        'overall_quality_score': 0,
        'grid_validity_rate': 0
    }
    
    validation = validate_submission_format(submission, test_challenges)
    
    # Completeness score
    metrics['completeness_score'] = validation['stats']['tasks_present'] / len(test_challenges) * 100
    
    # Format validity score
    if validation['stats']['tasks_present'] > 0:
        metrics['format_validity_score'] = (
            validation['stats']['tasks_with_valid_format'] / 
            validation['stats']['tasks_present'] * 100
        )
    
    # Attempt diversity score
    total_attempts = (validation['stats']['attempt_consistency']['consistent'] + 
                     validation['stats']['attempt_consistency']['different'])
    if total_attempts > 0:
        metrics['attempt_diversity_score'] = (
            validation['stats']['attempt_consistency']['different'] / total_attempts * 100
        )
    
    # Grid validity rate
    valid_grids = sum(1 for h, w in validation['stats']['grid_sizes'] if 0 < h <= 30 and 0 < w <= 30)
    if validation['stats']['grid_sizes']:
        metrics['grid_validity_rate'] = valid_grids / len(validation['stats']['grid_sizes']) * 100
    
    # Overall quality score (weighted average)
    metrics['overall_quality_score'] = (
        metrics['completeness_score'] * 0.4 +
        metrics['format_validity_score'] * 0.3 +
        metrics['attempt_diversity_score'] * 0.2 +
        metrics['grid_validity_rate'] * 0.1
    )
    
    return metrics

def print_validation_report(submission: Dict, test_challenges: Dict):
    """Print comprehensive validation report"""
    print("\n" + "="*70)
    print(" ARC PRIZE 2025 TEST SUBMISSION VALIDATION REPORT ")
    print("="*70)
    
    # Basic validation
    validation = validate_submission_format(submission, test_challenges)
    
    print(f"\n📋 SUBMISSION OVERVIEW:")
    print(f"  • Expected tasks (test set): {validation['stats']['total_tasks']}")
    print(f"  • Tasks present: {validation['stats']['tasks_present']}")
    print(f"  • Tasks missing: {validation['stats']['tasks_missing']}")
    print(f"  • Tasks with valid format: {validation['stats']['tasks_with_valid_format']}")
    print(f"  • Total predictions: {validation['stats']['total_predictions']}")
    
    # Pattern analysis
    patterns = analyze_prediction_patterns(submission, test_challenges)
    
    print(f"\n🔍 PREDICTION ANALYSIS:")
    print(f"  • Unique grid sizes used: {len(patterns['grid_size_distribution'])}")
    print(f"  • Most common grid sizes:")
    for size, count in patterns['grid_size_distribution'].most_common(5):
        print(f"    - {size}: {count} occurrences")
    
    print(f"  • Colors used: {sorted(patterns['color_distribution'].keys())}")
    print(f"  • Attempt diversity: {validation['stats']['attempt_consistency']['different']} different, "
          f"{validation['stats']['attempt_consistency']['consistent']} identical")
    
    # Quality metrics
    metrics = calculate_quality_metrics(submission, test_challenges)
    
    print(f"\n📊 QUALITY METRICS:")
    print(f"  • Completeness: {metrics['completeness_score']:.1f}%")
    print(f"  • Format validity: {metrics['format_validity_score']:.1f}%")
    print(f"  • Attempt diversity: {metrics['attempt_diversity_score']:.1f}%")
    print(f"  • Grid validity rate: {metrics['grid_validity_rate']:.1f}%")
    print(f"  • Overall quality score: {metrics['overall_quality_score']:.1f}%")
    
    # Validation status
    print(f"\n✅ VALIDATION STATUS:")
    if validation['valid']:
        print("  ✓ Submission format is VALID for competition")
    else:
        print("  ✗ Submission has ERRORS that need fixing:")
        for error in validation['errors'][:10]:  # Show first 10 errors
            print(f"    - {error}")
        if len(validation['errors']) > 10:
            print(f"    ... and {len(validation['errors']) - 10} more errors")
    
    if validation['warnings']:
        print(f"\n⚠️  WARNINGS ({len(validation['warnings'])} total):")
        for warning in validation['warnings'][:5]:
            print(f"    - {warning}")
        if len(validation['warnings']) > 5:
            print(f"    ... and {len(validation['warnings']) - 5} more warnings")
    
    if patterns['suspicious_patterns']:
        print(f"\n🔴 SUSPICIOUS PATTERNS DETECTED:")
        for pattern in patterns['suspicious_patterns'][:5]:
            print(f"    - {pattern}")
        if len(patterns['suspicious_patterns']) > 5:
            print(f"    ... and {len(patterns['suspicious_patterns']) - 5} more")
    
    print("\n" + "="*70)
    print("\n💡 IMPORTANT NOTES:")
    print("  • This validation checks FORMAT only, not correctness")
    print("  • Actual scoring requires ground truth (held by competition organizers)")
    print("  • The test set (240 tasks) is what gets submitted for official scoring")
    print("  • The evaluation set (120 tasks) is for local development only")
    
    if not validation['valid']:
        print("\n🚨 ACTION REQUIRED:")
        print("  Your submission has format errors that must be fixed before submitting!")
    elif metrics['overall_quality_score'] < 50:
        print("\n⚠️  QUALITY CONCERNS:")
        print("  Your submission has low quality metrics. Consider reviewing your approach.")
    else:
        print("\n✅ READY FOR SUBMISSION:")
        print("  Your submission appears properly formatted for ARC Prize 2025!")
    
    print("\n" + "="*70)
    
    return validation, patterns, metrics

if __name__ == "__main__":
    # Load files
    submission = load_json("/mnt/user-data/uploads/submission.json")
    test_challenges = load_json("/mnt/user-data/uploads/arc-agi_test_challenges.json")
    
    # Generate report
    validation, patterns, metrics = print_validation_report(submission, test_challenges)
    
    # Save detailed results
    detailed_results = {
        'validation': validation,
        'patterns': {
            'grid_size_distribution': dict(patterns['grid_size_distribution']),
            'color_distribution': dict(patterns['color_distribution']),
            'suspicious_patterns': patterns['suspicious_patterns']
        },
        'metrics': metrics
    }
    
    output_path = "/mnt/user-data/outputs/arc_validation_report.json"
    with open(output_path, 'w') as f:
        json.dump(detailed_results, f, indent=2)
    
    print(f"\n💾 Detailed validation report saved to: {output_path}")
