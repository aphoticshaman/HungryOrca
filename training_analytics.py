#!/usr/bin/env python3
"""
HungryOrca Training Analytics - Deep Statistical Analysis
==========================================================

Multi-dimensional pivot table system for analyzing training performance
across difficulty, time, strategy, and task characteristics.
"""

import numpy as np
import json
from collections import defaultdict
from typing import Dict, List, Tuple
from dataclasses import dataclass, field
import time


@dataclass
class TaskAttempt:
    """Single task attempt record"""
    task_id: str
    attempt_num: int  # 1 or 2
    task_index: int  # Position in sorted difficulty
    difficulty_score: float
    timeout_allocated: float
    time_taken: float
    success: bool

    # Task characteristics
    grid_size_input: int
    grid_size_output: int
    num_examples: int
    num_colors: int
    shape_changes: bool

    # Strategy used
    strategy: str  # 'eigenform', 'bootstrap', 'simple', 'timeout', 'error'
    error_type: str = ""


class TrainingAnalytics:
    """
    Comprehensive training analytics with pivot tables and statistical analysis

    Tracks:
    - Performance by difficulty range
    - Performance by timeout range
    - Performance by task characteristics
    - Strategy effectiveness
    - Time-series trends
    """

    def __init__(self):
        self.attempts: List[TaskAttempt] = []
        self.start_time = time.time()

        # Real-time aggregations
        self.accuracy_by_difficulty_bin = defaultdict(lambda: {'solved': 0, 'total': 0})
        self.accuracy_by_timeout_bin = defaultdict(lambda: {'solved': 0, 'total': 0})
        self.accuracy_by_strategy = defaultdict(lambda: {'solved': 0, 'total': 0})
        self.accuracy_by_grid_size = defaultdict(lambda: {'solved': 0, 'total': 0})

    def record_attempt(self, attempt: TaskAttempt):
        """Record a task attempt and update aggregations"""
        self.attempts.append(attempt)

        # Bin by difficulty (0-5, 5-10, 10-15, 15-20, 20+)
        diff_bin = int(attempt.difficulty_score // 5) * 5
        diff_key = f"{diff_bin}-{diff_bin+5}"
        self.accuracy_by_difficulty_bin[diff_key]['total'] += 1
        if attempt.success:
            self.accuracy_by_difficulty_bin[diff_key]['solved'] += 1

        # Bin by timeout (0-10s, 10-30s, 30-60s, 60-90s)
        if attempt.timeout_allocated < 10:
            timeout_key = "0-10s"
        elif attempt.timeout_allocated < 30:
            timeout_key = "10-30s"
        elif attempt.timeout_allocated < 60:
            timeout_key = "30-60s"
        else:
            timeout_key = "60-90s"

        self.accuracy_by_timeout_bin[timeout_key]['total'] += 1
        if attempt.success:
            self.accuracy_by_timeout_bin[timeout_key]['solved'] += 1

        # By strategy
        self.accuracy_by_strategy[attempt.strategy]['total'] += 1
        if attempt.success:
            self.accuracy_by_strategy[attempt.strategy]['solved'] += 1

        # By grid size
        size_key = f"{attempt.grid_size_input}->{attempt.grid_size_output}"
        self.accuracy_by_grid_size[size_key]['total'] += 1
        if attempt.success:
            self.accuracy_by_grid_size[size_key]['solved'] += 1

    def generate_pivot_table_difficulty(self) -> str:
        """Pivot Table: Accuracy by Difficulty Range"""
        output = []
        output.append("\n" + "="*70)
        output.append("📊 PIVOT TABLE: Accuracy by Difficulty Range")
        output.append("="*70)
        output.append(f"{'Difficulty':<15} {'Total':<10} {'Solved':<10} {'Accuracy':<15} {'Avg Time'}")
        output.append("-"*70)

        sorted_bins = sorted(self.accuracy_by_difficulty_bin.keys(),
                            key=lambda x: float(x.split('-')[0]))

        for bin_name in sorted_bins:
            data = self.accuracy_by_difficulty_bin[bin_name]
            total = data['total']
            solved = data['solved']
            acc = (solved / total * 100) if total > 0 else 0

            # Calculate avg time for this bin
            times = [a.time_taken for a in self.attempts
                    if int(a.difficulty_score // 5) * 5 == int(bin_name.split('-')[0])]
            avg_time = np.mean(times) if times else 0

            output.append(f"{bin_name:<15} {total:<10} {solved:<10} {acc:>6.3f}%       {avg_time:>6.2f}s")

        output.append("="*70)
        return "\n".join(output)

    def generate_pivot_table_timeout(self) -> str:
        """Pivot Table: Accuracy by Timeout Range"""
        output = []
        output.append("\n" + "="*70)
        output.append("📊 PIVOT TABLE: Accuracy by Timeout Allocation")
        output.append("="*70)
        output.append(f"{'Timeout Range':<15} {'Total':<10} {'Solved':<10} {'Accuracy':<15}")
        output.append("-"*70)

        timeout_order = ["0-10s", "10-30s", "30-60s", "60-90s"]
        for bin_name in timeout_order:
            if bin_name in self.accuracy_by_timeout_bin:
                data = self.accuracy_by_timeout_bin[bin_name]
                total = data['total']
                solved = data['solved']
                acc = (solved / total * 100) if total > 0 else 0
                output.append(f"{bin_name:<15} {total:<10} {solved:<10} {acc:>6.3f}%")

        output.append("="*70)
        return "\n".join(output)

    def generate_pivot_table_strategy(self) -> str:
        """Pivot Table: Accuracy by Strategy Used"""
        output = []
        output.append("\n" + "="*70)
        output.append("📊 PIVOT TABLE: Accuracy by Strategy")
        output.append("="*70)
        output.append(f"{'Strategy':<20} {'Total':<10} {'Solved':<10} {'Accuracy':<15}")
        output.append("-"*70)

        sorted_strategies = sorted(self.accuracy_by_strategy.items(),
                                  key=lambda x: x[1]['total'], reverse=True)

        for strategy, data in sorted_strategies:
            total = data['total']
            solved = data['solved']
            acc = (solved / total * 100) if total > 0 else 0
            output.append(f"{strategy:<20} {total:<10} {solved:<10} {acc:>6.3f}%")

        output.append("="*70)
        return "\n".join(output)

    def generate_meta_analysis(self) -> str:
        """Meta-Analysis: Statistical insights and diagnostics"""
        output = []
        output.append("\n" + "="*70)
        output.append("🔬 META-ANALYSIS: Statistical Diagnostics")
        output.append("="*70)

        if not self.attempts:
            output.append("No data yet")
            return "\n".join(output)

        # Overall statistics
        total_attempts = len(self.attempts)
        total_solved = sum(1 for a in self.attempts if a.success)
        overall_acc = (total_solved / total_attempts * 100) if total_attempts > 0 else 0

        output.append(f"\n📈 Overall Performance:")
        output.append(f"   Total Attempts: {total_attempts}")
        output.append(f"   Solved: {total_solved}")
        output.append(f"   Accuracy: {overall_acc:.3f}%")

        # Time analysis
        avg_time = np.mean([a.time_taken for a in self.attempts])
        med_time = np.median([a.time_taken for a in self.attempts])
        elapsed = time.time() - self.start_time

        output.append(f"\n⏱️  Time Analysis:")
        output.append(f"   Avg time per task: {avg_time:.2f}s")
        output.append(f"   Median time: {med_time:.2f}s")
        output.append(f"   Total elapsed: {elapsed:.0f}s ({elapsed/60:.1f} min)")
        output.append(f"   Rate: {total_attempts/elapsed:.2f} tasks/sec")

        # Difficulty correlation
        difficulties = [a.difficulty_score for a in self.attempts]
        successes = [1 if a.success else 0 for a in self.attempts]

        if len(difficulties) > 10:
            correlation = np.corrcoef(difficulties, successes)[0, 1]
            output.append(f"\n🎯 Difficulty Correlation:")
            output.append(f"   Correlation (difficulty ↔ failure): {correlation:+.3f}")
            if abs(correlation) > 0.3:
                output.append(f"   ⚠️  STRONG correlation - difficulty estimation working")
            else:
                output.append(f"   ⚠️  WEAK correlation - difficulty estimation may be wrong!")

        # Success pattern analysis
        solved_indices = [a.task_index for a in self.attempts if a.success]
        failed_indices = [a.task_index for a in self.attempts if not a.success]

        if solved_indices:
            output.append(f"\n✅ Success Pattern:")
            output.append(f"   Solved task indices: {sorted(solved_indices)[:20]}")
            output.append(f"   Avg difficulty of solved: {np.mean([a.difficulty_score for a in self.attempts if a.success]):.2f}")

        if len(failed_indices) > 0:
            output.append(f"\n❌ Failure Pattern:")
            output.append(f"   Avg difficulty of failed: {np.mean([a.difficulty_score for a in self.attempts if not a.success]):.2f}")

        # Error type breakdown
        error_counts = defaultdict(int)
        for a in self.attempts:
            if not a.success and a.error_type:
                error_counts[a.error_type] += 1

        if error_counts:
            output.append(f"\n🐛 Error Breakdown:")
            for error, count in sorted(error_counts.items(), key=lambda x: x[1], reverse=True):
                output.append(f"   {error}: {count} occurrences")

        # Diagnostic warnings
        output.append(f"\n⚠️  Diagnostics:")

        if overall_acc < 10:
            output.append(f"   🚨 CRITICAL: <10% accuracy suggests fundamental solver issue!")
            output.append(f"      Check: primitives, eigenform logic, timeout handling")

        if avg_time < 0.5:
            output.append(f"   🚨 WARNING: Tasks completing too fast (<0.5s avg)")
            output.append(f"      Likely: early exits, timeouts not being used")

        timeout_utilization = avg_time / np.mean([a.timeout_allocated for a in self.attempts])
        if timeout_utilization < 0.1:
            output.append(f"   🚨 WARNING: Only using {timeout_utilization*100:.1f}% of timeout!")
            output.append(f"      Solver may be giving up too early")

        output.append("="*70)
        return "\n".join(output)

    def generate_full_report(self) -> str:
        """Generate comprehensive analytics report"""
        report = []
        report.append(self.generate_meta_analysis())
        report.append(self.generate_pivot_table_difficulty())
        report.append(self.generate_pivot_table_timeout())
        report.append(self.generate_pivot_table_strategy())
        return "\n".join(report)

    def save_report(self, filepath: str):
        """Save analytics report to file"""
        with open(filepath, 'w') as f:
            f.write(self.generate_full_report())

    def export_data(self, filepath: str):
        """Export raw data as JSON for external analysis"""
        data = {
            'attempts': [
                {
                    'task_id': a.task_id,
                    'attempt_num': a.attempt_num,
                    'task_index': a.task_index,
                    'difficulty_score': a.difficulty_score,
                    'timeout_allocated': a.timeout_allocated,
                    'time_taken': a.time_taken,
                    'success': a.success,
                    'grid_size_input': a.grid_size_input,
                    'grid_size_output': a.grid_size_output,
                    'num_examples': a.num_examples,
                    'num_colors': a.num_colors,
                    'shape_changes': a.shape_changes,
                    'strategy': a.strategy,
                    'error_type': a.error_type
                }
                for a in self.attempts
            ],
            'summary': {
                'total_attempts': len(self.attempts),
                'total_solved': sum(1 for a in self.attempts if a.success),
                'overall_accuracy': sum(1 for a in self.attempts if a.success) / len(self.attempts) * 100 if self.attempts else 0,
                'elapsed_time': time.time() - self.start_time
            }
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)


# Example usage
if __name__ == "__main__":
    # Create analytics instance
    analytics = TrainingAnalytics()

    # Simulate some attempts (would be real data in actual use)
    for i in range(100):
        attempt = TaskAttempt(
            task_id=f"task_{i}",
            attempt_num=1,
            task_index=i,
            difficulty_score=i * 0.2,
            timeout_allocated=90 - i * 0.5,
            time_taken=np.random.uniform(0.5, 10),
            success=np.random.random() < 0.1,  # 10% success rate
            grid_size_input=np.random.randint(20, 400),
            grid_size_output=np.random.randint(20, 400),
            num_examples=np.random.randint(2, 5),
            num_colors=np.random.randint(2, 10),
            shape_changes=np.random.random() < 0.5,
            strategy=np.random.choice(['eigenform', 'bootstrap', 'simple']),
            error_type=""
        )
        analytics.record_attempt(attempt)

    # Generate report
    print(analytics.generate_full_report())
