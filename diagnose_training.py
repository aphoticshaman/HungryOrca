#!/usr/bin/env python3
"""
Emergency Training Diagnostic
==============================

Analyzes the training output to diagnose why accuracy is so low (3-5%)
despite generous timeouts (60-90s).
"""

import re
from typing import List, Tuple

def parse_training_output(output_lines: List[str]) -> dict:
    """
    Parse training output and extract key metrics

    Example line:
    [ATTEMPT1] [  10/1000] Last 10: 1/10 ( 10.000%) | Overall:  10.000% | Timeout:  86.0s | Time:    15s
    """

    pattern = r'\[ATTEMPT1\]\s+\[\s*(\d+)/\d+\]\s+Last 10:\s+(\d+)/10\s+\(\s*([\d.]+)%\)\s+\|\s+Overall:\s+([\d.]+)%\s+\|\s+Timeout:\s+([\d.]+)s\s+\|\s+Time:\s+(\d+)s'

    data_points = []

    for line in output_lines:
        match = re.search(pattern, line)
        if match:
            task_num = int(match.group(1))
            last_10_solved = int(match.group(2))
            last_10_pct = float(match.group(3))
            overall_pct = float(match.group(4))
            timeout = float(match.group(5))
            elapsed_time = int(match.group(6))

            data_points.append({
                'task_num': task_num,
                'last_10_solved': last_10_solved,
                'last_10_pct': last_10_pct,
                'overall_pct': overall_pct,
                'timeout': timeout,
                'elapsed_time': elapsed_time
            })

    return analyze_data_points(data_points)


def analyze_data_points(data_points: List[dict]) -> dict:
    """Deep analysis of training data points"""

    if not data_points:
        return {'error': 'No data points found'}

    # Calculate key metrics
    total_tasks = data_points[-1]['task_num']
    final_accuracy = data_points[-1]['overall_pct']
    total_time = data_points[-1]['elapsed_time']

    # Average time per task
    avg_time_per_task = total_time / total_tasks if total_tasks > 0 else 0

    # Average timeout allocated
    avg_timeout = sum(d['timeout'] for d in data_points) / len(data_points)

    # Timeout utilization
    timeout_utilization = (avg_time_per_task / avg_timeout) * 100 if avg_timeout > 0 else 0

    # Success rate trend
    accuracies = [d['overall_pct'] for d in data_points]
    is_improving = accuracies[-1] > accuracies[0] if len(accuracies) > 1 else False
    improvement_rate = (accuracies[-1] - accuracies[0]) / len(data_points) if len(data_points) > 1 else 0

    # Rate of progress
    tasks_per_second = total_tasks / total_time if total_time > 0 else 0

    # Projected completion
    if tasks_per_second > 0:
        projected_time_for_1000 = 1000 / tasks_per_second
        projected_remaining = projected_time_for_1000 - total_time
    else:
        projected_time_for_1000 = 0
        projected_remaining = 0

    # Analysis
    analysis = {
        'summary': {
            'total_tasks_processed': total_tasks,
            'final_accuracy': final_accuracy,
            'total_time_elapsed': total_time,
            'tasks_per_second': tasks_per_second
        },
        'timing': {
            'avg_time_per_task': avg_time_per_task,
            'avg_timeout_allocated': avg_timeout,
            'timeout_utilization_pct': timeout_utilization,
            'projected_time_for_1000': projected_time_for_1000,
            'projected_remaining_time': projected_remaining
        },
        'performance': {
            'is_improving': is_improving,
            'improvement_rate_per_task': improvement_rate,
            'accuracy_trend': 'improving' if is_improving else 'flat/declining'
        },
        'diagnostics': []
    }

    # Diagnostic warnings
    if final_accuracy < 10:
        analysis['diagnostics'].append({
            'severity': 'CRITICAL',
            'issue': 'Accuracy < 10%',
            'description': 'Fundamental solver issue - primitives not matching tasks',
            'recommendations': [
                'Check if _train_task() is actually running strategies',
                'Verify eigenform/bootstrap are being called',
                'Check if simple transforms are being tested',
                'Look for early exits or exceptions being swallowed'
            ]
        })

    if timeout_utilization < 10:
        analysis['diagnostics'].append({
            'severity': 'CRITICAL',
            'issue': f'Timeout utilization only {timeout_utilization:.1f}%',
            'description': 'Tasks completing WAY faster than allocated time',
            'recommendations': [
                'Tasks are timing out or erroring early',
                'Check timeout enforcement in _train_task()',
                'Look for exceptions being caught silently',
                'Verify strategies are actually running full duration'
            ]
        })

    if avg_time_per_task < 1.0:
        analysis['diagnostics'].append({
            'severity': 'WARNING',
            'issue': f'Tasks completing in {avg_time_per_task:.2f}s on average',
            'description': 'Tasks finishing too quickly to learn anything',
            'recommendations': [
                'Increase minimum time per strategy',
                'Check if early stopping is too aggressive',
                'Verify timeout parameter is being used'
            ]
        })

    if not is_improving and len(data_points) > 10:
        analysis['diagnostics'].append({
            'severity': 'WARNING',
            'issue': 'No improvement trend detected',
            'description': 'Accuracy not increasing as easier tasks are processed',
            'recommendations': [
                'Difficulty estimation may be incorrect',
                'Easiest tasks may not actually be easy',
                'Re-sort tasks by different metrics'
            ]
        })

    return analysis


def print_diagnostic_report(analysis: dict):
    """Print formatted diagnostic report"""

    print("\n" + "="*70)
    print("🔬 EMERGENCY TRAINING DIAGNOSTIC")
    print("="*70)

    # Summary
    print("\n📊 SUMMARY:")
    for key, value in analysis['summary'].items():
        print(f"   {key}: {value}")

    # Timing
    print("\n⏱️  TIMING ANALYSIS:")
    for key, value in analysis['timing'].items():
        if 'pct' in key:
            print(f"   {key}: {value:.1f}%")
        elif 'time' in key:
            print(f"   {key}: {value:.1f}s ({value/60:.1f} min)")
        else:
            print(f"   {key}: {value:.2f}")

    # Performance
    print("\n📈 PERFORMANCE:")
    for key, value in analysis['performance'].items():
        print(f"   {key}: {value}")

    # Diagnostics
    print("\n🚨 DIAGNOSTICS:")
    if analysis['diagnostics']:
        for i, diag in enumerate(analysis['diagnostics'], 1):
            print(f"\n   Issue #{i}: [{diag['severity']}] {diag['issue']}")
            print(f"   Description: {diag['description']}")
            print(f"   Recommendations:")
            for rec in diag['recommendations']:
                print(f"      • {rec}")
    else:
        print("   ✅ No critical issues detected")

    print("\n" + "="*70)


# Example usage with the provided output
if __name__ == "__main__":
    sample_output = """
  [ATTEMPT1] [  10/1000] Last 10: 1/10 ( 10.000%) | Overall:  10.000% | Timeout:  86.0s | Time:    15s
  [ATTEMPT1] [  20/1000] Last 10: 1/10 ( 10.000%) | Overall:  10.000% | Timeout:  81.8s | Time:    31s
  [ATTEMPT1] [  30/1000] Last 10: 0/10 (  0.000%) | Overall:   6.667% | Timeout:  77.9s | Time:    47s
  [ATTEMPT1] [  40/1000] Last 10: 1/10 ( 10.000%) | Overall:   7.500% | Timeout:  74.1s | Time:    62s
  [ATTEMPT1] [  50/1000] Last 10: 1/10 ( 10.000%) | Overall:   8.000% | Timeout:  70.4s | Time:    79s
  [ATTEMPT1] [  60/1000] Last 10: 0/10 (  0.000%) | Overall:   6.667% | Timeout:  67.0s | Time:    94s
  [ATTEMPT1] [  70/1000] Last 10: 0/10 (  0.000%) | Overall:   5.714% | Timeout:  63.7s | Time:   109s
  [ATTEMPT1] [  80/1000] Last 10: 1/10 ( 10.000%) | Overall:   6.250% | Timeout:  60.6s | Time:   126s
  [ATTEMPT1] [  90/1000] Last 10: 0/10 (  0.000%) | Overall:   5.556% | Timeout:  57.7s | Time:   141s
  [ATTEMPT1] [ 100/1000] Last 10: 0/10 (  0.000%) | Overall:   5.000% | Timeout:  54.9s | Time:    158s
  [ATTEMPT1] [ 240/1000] Last 10: 0/10 (  0.000%) | Overall:   3.750% | Timeout:  27.2s | Time:   382s
    """

    lines = sample_output.strip().split('\n')
    analysis = parse_training_output(lines)
    print_diagnostic_report(analysis)
