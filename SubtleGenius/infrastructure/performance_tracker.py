"""
Performance Tracker - Built-in Coverage + Accuracy Reporting

Tracks solver performance in real-time:
- Coverage: % of tasks where solver triggers
- Accuracy: % correct when solver triggers (if ground truth available)
- Contribution: Coverage × Accuracy

LAELD Rule #5: COVERAGE + ACCURACY REPORTING BUILT-IN

Usage:
    from performance_tracker import PerformanceTracker, SolverStats

    # Initialize tracker
    tracker = PerformanceTracker()

    # During solving
    tracker.record_trigger('pattern_rotate_90', task_id)
    result = apply_pattern(...)
    tracker.record_attempt('pattern_rotate_90', task_id, correct=(result == ground_truth))

    # After run
    tracker.print_report()
"""

import time
from typing import Dict, Optional, List
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class SolverStats:
    """Statistics for a single solver"""
    name: str
    triggers: int = 0
    attempts: int = 0
    successes: int = 0
    failures: int = 0
    errors: int = 0
    task_ids: List[str] = field(default_factory=list)
    correct_tasks: List[str] = field(default_factory=list)
    incorrect_tasks: List[str] = field(default_factory=list)

    @property
    def coverage(self) -> float:
        """Coverage rate (requires total_tasks from tracker)"""
        return 0.0  # Calculated by tracker

    @property
    def accuracy(self) -> float:
        """Accuracy when solver triggers"""
        total = self.successes + self.failures
        if total == 0:
            return 0.0
        return self.successes / total

    @property
    def contribution(self) -> float:
        """Overall contribution (coverage × accuracy)"""
        return 0.0  # Calculated by tracker

    def to_dict(self) -> Dict:
        """Export to dictionary"""
        return {
            'name': self.name,
            'triggers': self.triggers,
            'attempts': self.attempts,
            'successes': self.successes,
            'failures': self.failures,
            'errors': self.errors,
            'accuracy': self.accuracy,
            'task_count': len(self.task_ids)
        }


class PerformanceTracker:
    """Track performance of all solvers across tasks"""

    def __init__(self):
        """Initialize performance tracker"""
        self.solvers: Dict[str, SolverStats] = {}
        self.total_tasks = 0
        self.total_test_cases = 0
        self.start_time = time.time()
        self.task_solvers: Dict[str, str] = {}  # task_id -> solver_name

    def record_trigger(self, solver_name: str, task_id: str):
        """Record that a solver triggered on a task"""
        if solver_name not in self.solvers:
            self.solvers[solver_name] = SolverStats(name=solver_name)

        stats = self.solvers[solver_name]
        stats.triggers += 1

        if task_id not in stats.task_ids:
            stats.task_ids.append(task_id)
            self.task_solvers[task_id] = solver_name

    def record_attempt(self,
                      solver_name: str,
                      task_id: str,
                      correct: Optional[bool] = None):
        """
        Record a solver attempt

        Args:
            solver_name: Name of the solver
            task_id: Task ID
            correct: True if correct, False if incorrect, None if unknown
        """
        if solver_name not in self.solvers:
            self.solvers[solver_name] = SolverStats(name=solver_name)

        stats = self.solvers[solver_name]
        stats.attempts += 1

        if correct is not None:
            if correct:
                stats.successes += 1
                if task_id not in stats.correct_tasks:
                    stats.correct_tasks.append(task_id)
            else:
                stats.failures += 1
                if task_id not in stats.incorrect_tasks:
                    stats.incorrect_tasks.append(task_id)

    def record_error(self, solver_name: str, task_id: str):
        """Record that a solver encountered an error"""
        if solver_name not in self.solvers:
            self.solvers[solver_name] = SolverStats(name=solver_name)

        self.solvers[solver_name].errors += 1

    def set_total_tasks(self, total: int):
        """Set total number of tasks (for coverage calculation)"""
        self.total_tasks = total

    def set_total_test_cases(self, total: int):
        """Set total number of test cases"""
        self.total_test_cases = total

    def get_stats(self, solver_name: str) -> Optional[SolverStats]:
        """Get stats for a specific solver"""
        return self.solvers.get(solver_name)

    def get_all_stats(self) -> Dict[str, SolverStats]:
        """Get stats for all solvers"""
        return self.solvers

    def get_coverage(self, solver_name: str) -> float:
        """Get coverage for a solver"""
        if solver_name not in self.solvers or self.total_tasks == 0:
            return 0.0
        return self.solvers[solver_name].triggers / self.total_tasks

    def get_contribution(self, solver_name: str) -> float:
        """Get contribution for a solver (coverage × accuracy)"""
        coverage = self.get_coverage(solver_name)
        stats = self.solvers.get(solver_name)
        if not stats:
            return 0.0
        return coverage * stats.accuracy

    def print_report(self, top_n: int = 20):
        """Print comprehensive performance report"""
        print("\n" + "="*70)
        print("PERFORMANCE TRACKER REPORT")
        print("="*70)

        elapsed = time.time() - self.start_time

        # Overall stats
        print(f"\n📊 OVERALL STATISTICS")
        print(f"  Total tasks: {self.total_tasks}")
        print(f"  Total test cases: {self.total_test_cases}")
        print(f"  Elapsed time: {elapsed:.1f}s")
        if elapsed > 0:
            print(f"  Tasks/second: {self.total_tasks / elapsed:.1f}")

        # Active solvers
        print(f"  Active solvers: {len(self.solvers)}")

        # Per-solver breakdown
        if self.solvers:
            print(f"\n🔧 SOLVER PERFORMANCE")
            print(f"  {'Solver':<25} {'Triggers':>8} {'Coverage':>10} "
                  f"{'Accuracy':>10} {'Contribution':>13} {'Errors':>7}")
            print(f"  {'-'*25} {'-'*8} {'-'*10} {'-'*10} {'-'*13} {'-'*7}")

            # Sort by contribution
            solver_list = []
            for name, stats in self.solvers.items():
                coverage = self.get_coverage(name)
                contribution = self.get_contribution(name)
                solver_list.append((name, stats, coverage, contribution))

            solver_list.sort(key=lambda x: x[3], reverse=True)

            for name, stats, coverage, contribution in solver_list[:top_n]:
                print(f"  {name:<25} "
                      f"{stats.triggers:>8} "
                      f"{coverage*100:>9.1f}% "
                      f"{stats.accuracy*100:>9.1f}% "
                      f"{contribution*100:>12.1f}% "
                      f"{stats.errors:>7}")

        # Summary
        total_contribution = sum(self.get_contribution(name) for name in self.solvers.keys())
        print(f"\n💡 INSIGHTS")
        print(f"  Total contribution: {total_contribution*100:.1f}%")

        if self.solvers:
            best_solver = max(self.solvers.items(),
                            key=lambda x: self.get_contribution(x[0]))
            best_name = best_solver[0]
            best_contribution = self.get_contribution(best_name)
            print(f"  Best solver: {best_name} ({best_contribution*100:.1f}% contribution)")

        print("\n" + "="*70)
        print("✅ PERFORMANCE REPORT COMPLETE")
        print("="*70 + "\n")

    def export_json(self) -> Dict:
        """Export all stats as JSON-serializable dict"""
        return {
            'total_tasks': self.total_tasks,
            'total_test_cases': self.total_test_cases,
            'elapsed_seconds': time.time() - self.start_time,
            'solvers': {
                name: {
                    **stats.to_dict(),
                    'coverage': self.get_coverage(name),
                    'contribution': self.get_contribution(name)
                }
                for name, stats in self.solvers.items()
            }
        }


# Example usage
if __name__ == "__main__":
    print("Performance Tracker - Infrastructure Test")
    print("="*70)
    print("\nThis tracker monitors solver performance in real-time.")
    print("\nUsage:")
    print("  tracker = PerformanceTracker()")
    print("  tracker.set_total_tasks(240)")
    print("  ")
    print("  # During solving:")
    print("  tracker.record_trigger('pattern_rotate_90', task_id)")
    print("  result = apply_pattern(...)")
    print("  tracker.record_attempt('pattern_rotate_90', task_id, correct=True)")
    print("  ")
    print("  # After run:")
    print("  tracker.print_report()")
    print("\n" + "="*70)
    print("\n📈 Example output:")

    # Demo
    tracker = PerformanceTracker()
    tracker.set_total_tasks(100)

    # Simulate some solver activity
    for i in range(15):
        tracker.record_trigger('pattern_rotate_90', f'task_{i}')
        tracker.record_attempt('pattern_rotate_90', f'task_{i}', correct=(i < 12))

    for i in range(30):
        tracker.record_trigger('rule_color_map', f'task_{i+20}')
        tracker.record_attempt('rule_color_map', f'task_{i+20}', correct=(i < 25))

    tracker.print_report()
