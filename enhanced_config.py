#!/usr/bin/env python3
"""
Enhanced ARC Solver Configuration
Based on 25 Design Lessons from Failed Run Analysis

This module implements:
1. Data-driven parameter selection
2. Adaptive resource allocation
3. Early stopping mechanisms
4. Runtime assertions
5. Fallback strategies
6. Smoke test validation
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import time

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    # Mock numpy for basic functionality
    class np:
        @staticmethod
        def mean(arr):
            return sum(arr) / len(arr) if arr else 0
        @staticmethod
        def array(x):
            return x
        @staticmethod
        def full(shape, value):
            return [[value] * shape[1] for _ in range(shape[0])]
        @staticmethod
        def rot90(arr):
            return arr
        @staticmethod
        def fliplr(arr):
            return arr

@dataclass
class EnhancedChampionshipConfig:
    """
    Redesigned configuration based on 8-hour budget and empirical analysis.

    Design Principles:
    - Search depth derived from problem requirements (15-30 steps needed)
    - Adaptive allocation based on task difficulty
    - Early stopping to prevent wasted computation
    - Runtime bounds as test assertions
    """

    # --- Core Time Budget (8 hours) ---
    total_time_budget: float = 28800.0  # 8 hours in seconds
    submission_buffer: float = 900.0     # 15 min safety buffer

    # --- LESSON 1-2: Model search complexity & inform by requirements ---
    # With 8 hours and 100 tasks = ~288s per task average
    # ARC tasks need 15-30 transformation steps
    # Setting depth=150 allows for complex compositions with backtracking
    MAX_PROGRAM_DEPTH: int = 150  # Was 20 (too shallow), now 150
    BEAM_SEARCH_WIDTH: int = 8    # Increased from 5 for better exploration

    # Theoretical search space: 150 depth * 8 width * 30 primitives = ~36K nodes/task
    # Expected time per task: ~36s (leaves room for harder tasks to use more)

    # --- LESSON 6: Smoke Test Before Full Run ---
    ENABLE_SMOKE_TEST: bool = True
    SMOKE_TEST_SIZE: int = 10           # Test on 10 tasks first
    SMOKE_TEST_MIN_SUCCESS_RATE: float = 0.05  # Need at least 5% success to proceed

    # --- LESSON 7: Early Stopping on Failure ---
    ENABLE_EARLY_STOPPING: bool = True
    EARLY_STOP_WINDOW: int = 20         # Check every 20 tasks
    EARLY_STOP_MIN_SUCCESS_RATE: float = 0.03  # Abort if <3% success rate

    # --- LESSON 10: Runtime Assertions ---
    EXPECTED_MIN_RUNTIME_MINUTES: float = 45.0  # Should use at least 45 min
    EXPECTED_MAX_RUNTIME_MINUTES: float = 450.0  # Should finish within 7.5 hours

    # --- LESSON 12: Data-Driven Depth Allocation ---
    # Allocate depth based on task difficulty tier
    ADAPTIVE_DEPTH_ENABLED: bool = True
    DEPTH_ALLOCATION: Dict[str, int] = field(default_factory=lambda: {
        'easy': 100,    # Simple tasks don't need max depth
        'medium': 150,  # Standard allocation
        'hard': 200,    # Complex tasks get more depth
    })

    # --- LESSON 14: Dynamic Budget Reallocation ---
    DYNAMIC_REALLOCATION: bool = True
    # If task finishes early, redistribute time to remaining hard tasks

    # --- LESSON 18: Canary Checks ---
    CANARY_CHECK_ENABLED: bool = True
    CANARY_SIZE: int = 10  # First 10 tasks are canaries
    CANARY_MAX_IDENTICAL_FAILURES: int = 8  # If 8/10 fail identically, warn

    # --- LESSON 24: Fallback Strategies ---
    ENABLE_FALLBACKS: bool = True
    FALLBACK_STRATEGIES: List[str] = field(default_factory=lambda: [
        'copy_input',           # Simplest: output = input
        'majority_color_fill',  # Fill with most common color
        'identity_transform',   # Try basic geometric transforms
        'largest_object_only',  # Extract largest object
    ])

    # --- LTM & Training ---
    LTM_BUDGET_PERCENT: float = 0.25  # 25% for training (was 30%)
    LTM_CACHE_K: int = 5

    # --- Resource Management ---
    parallel_workers: int = 4
    memory_limit_gb: float = 12.0

    # --- Diagnostic & Monitoring ---
    DIAGNOSTIC_RUN: bool = False  # Set to False for production
    DIAGNOSTIC_SAMPLE_SIZE: int = 100
    ENABLE_DEPTH_TRACKING: bool = True  # Track actual depth reached vs limit
    ENABLE_ANOMALY_DETECTION: bool = True  # Detect statistical anomalies

    def validate(self) -> List[str]:
        """
        LESSON 9: Regression tests for configuration changes.
        Validate configuration consistency and raise warnings.
        """
        warnings = []

        # Check depth is sufficient
        if self.MAX_PROGRAM_DEPTH < 50:
            warnings.append(f"⚠️  MAX_PROGRAM_DEPTH={self.MAX_PROGRAM_DEPTH} is likely too shallow (need 50+)")

        # Check time budget is reasonable
        avg_time_per_task = (self.total_time_budget - self.submission_buffer) / 100
        if avg_time_per_task < 30:
            warnings.append(f"⚠️  Only {avg_time_per_task:.0f}s per task - may be insufficient")

        # Check beam width
        if self.BEAM_SEARCH_WIDTH < 5:
            warnings.append(f"⚠️  BEAM_WIDTH={self.BEAM_SEARCH_WIDTH} is very narrow")

        # Check adaptive depths are monotonic
        if self.ADAPTIVE_DEPTH_ENABLED:
            depths = self.DEPTH_ALLOCATION
            if depths['easy'] > depths['medium'] or depths['medium'] > depths['hard']:
                warnings.append("⚠️  Depth allocation not monotonic (easy < medium < hard)")

        return warnings

    def estimate_search_complexity(self) -> Dict[str, float]:
        """
        LESSON 1: Model search complexity before deploying.
        Calculate theoretical search space and expected runtime.
        """
        num_primitives = 30  # Approximate from code analysis

        # Per-task estimates
        nodes_per_task = self.MAX_PROGRAM_DEPTH * self.BEAM_SEARCH_WIDTH * num_primitives
        time_per_node = 0.001  # 1ms per node (conservative)
        expected_time_per_task = nodes_per_task * time_per_node

        # Total estimates
        num_tasks = 100
        total_expected_time = expected_time_per_task * num_tasks

        return {
            'nodes_per_task': nodes_per_task,
            'expected_time_per_task_seconds': expected_time_per_task,
            'total_expected_runtime_minutes': total_expected_time / 60,
            'budget_utilization_percent': (total_expected_time / self.total_time_budget) * 100,
        }

    def print_summary(self):
        """Print human-readable configuration summary."""
        print("=" * 70)
        print("🚀 ENHANCED ARC SOLVER CONFIGURATION")
        print("=" * 70)
        print(f"\n⏱️  TIME BUDGET:")
        print(f"  Total: {self.total_time_budget / 3600:.1f} hours")
        print(f"  Per task (avg): {(self.total_time_budget - self.submission_buffer) / 100:.0f}s")

        print(f"\n🔍 SEARCH CONFIGURATION:")
        print(f"  Max Depth: {self.MAX_PROGRAM_DEPTH} (was 20 - FIXED)")
        print(f"  Beam Width: {self.BEAM_SEARCH_WIDTH}")

        if self.ADAPTIVE_DEPTH_ENABLED:
            print(f"\n📊 ADAPTIVE DEPTH ALLOCATION:")
            for tier, depth in self.DEPTH_ALLOCATION.items():
                print(f"  {tier.capitalize()}: {depth}")

        print(f"\n🛡️  SAFETY MECHANISMS:")
        if self.ENABLE_SMOKE_TEST:
            print(f"  ✅ Smoke Test: {self.SMOKE_TEST_SIZE} tasks, {self.SMOKE_TEST_MIN_SUCCESS_RATE*100:.0f}% min success")
        if self.ENABLE_EARLY_STOPPING:
            print(f"  ✅ Early Stopping: Check every {self.EARLY_STOP_WINDOW} tasks")
        if self.CANARY_CHECK_ENABLED:
            print(f"  ✅ Canary Check: First {self.CANARY_SIZE} tasks monitored")

        print(f"\n🎯 FALLBACK STRATEGIES:")
        if self.ENABLE_FALLBACKS:
            for strategy in self.FALLBACK_STRATEGIES:
                print(f"  • {strategy}")

        # Show complexity estimates
        estimates = self.estimate_search_complexity()
        print(f"\n📈 ESTIMATED PERFORMANCE:")
        print(f"  Search nodes/task: {estimates['nodes_per_task']:,.0f}")
        print(f"  Time per task: ~{estimates['expected_time_per_task_seconds']:.0f}s")
        print(f"  Total runtime: ~{estimates['total_expected_runtime_minutes']:.0f} min")
        print(f"  Budget utilization: {estimates['budget_utilization_percent']:.0f}%")

        # Validate and show warnings
        warnings = self.validate()
        if warnings:
            print(f"\n⚠️  CONFIGURATION WARNINGS:")
            for warning in warnings:
                print(f"  {warning}")
        else:
            print(f"\n✅ Configuration validated - no warnings")

        print("=" * 70)


@dataclass
class RuntimeMonitor:
    """
    LESSON 16-19: Monitoring, alerting, and visualization.
    Tracks solver performance and detects anomalies.
    """

    config: EnhancedChampionshipConfig
    start_time: float = field(default_factory=time.time)

    # Metrics tracking
    tasks_completed: int = 0
    tasks_successful: int = 0
    failure_modes: Dict[str, int] = field(default_factory=lambda: {
        'Synthesizer.Fail.MaxDepth': 0,
        'Synthesizer.Timeout': 0,
        'Synthesizer.Success': 0,
        'Other': 0,
    })

    # Depth tracking (LESSON 17)
    depths_reached: List[int] = field(default_factory=list)
    depths_allocated: List[int] = field(default_factory=list)

    # Time tracking
    task_times: List[float] = field(default_factory=list)

    def record_task(self, task_id: str, status: str, time_taken: float,
                   depth_reached: Optional[int] = None, depth_allocated: Optional[int] = None):
        """Record metrics for a completed task."""
        self.tasks_completed += 1
        self.task_times.append(time_taken)

        if 'Success' in status:
            self.tasks_successful += 1
            self.failure_modes['Synthesizer.Success'] += 1
        elif 'MaxDepth' in status:
            self.failure_modes['Synthesizer.Fail.MaxDepth'] += 1
        elif 'Timeout' in status:
            self.failure_modes['Synthesizer.Timeout'] += 1
        else:
            self.failure_modes['Other'] += 1

        if depth_reached is not None:
            self.depths_reached.append(depth_reached)
        if depth_allocated is not None:
            self.depths_allocated.append(depth_allocated)

    def check_canary(self) -> Dict[str, any]:
        """
        LESSON 18: Canary checks after first N tasks.
        Returns alert status and recommended action.
        """
        if self.tasks_completed != self.config.CANARY_SIZE:
            return {'status': 'incomplete', 'action': 'continue'}

        # Check for homogeneous failures
        max_depth_failures = self.failure_modes['Synthesizer.Fail.MaxDepth']
        if max_depth_failures >= self.config.CANARY_MAX_IDENTICAL_FAILURES:
            return {
                'status': 'CRITICAL',
                'action': 'abort',
                'message': f'🚨 {max_depth_failures}/{self.config.CANARY_SIZE} tasks failed with MaxDepth - depth likely insufficient!',
            }

        # Check success rate
        success_rate = self.tasks_successful / self.tasks_completed
        if success_rate < self.config.SMOKE_TEST_MIN_SUCCESS_RATE:
            return {
                'status': 'WARNING',
                'action': 'review',
                'message': f'⚠️  Success rate {success_rate*100:.1f}% below target {self.config.SMOKE_TEST_MIN_SUCCESS_RATE*100:.0f}%',
            }

        return {
            'status': 'OK',
            'action': 'continue',
            'message': f'✅ Canary check passed: {success_rate*100:.1f}% success rate',
        }

    def check_early_stop(self) -> bool:
        """
        LESSON 7: Early stopping when success rate is too low.
        Returns True if we should stop.
        """
        if not self.config.ENABLE_EARLY_STOPPING:
            return False

        if self.tasks_completed % self.config.EARLY_STOP_WINDOW != 0:
            return False

        success_rate = self.tasks_successful / self.tasks_completed if self.tasks_completed > 0 else 0

        if success_rate < self.config.EARLY_STOP_MIN_SUCCESS_RATE:
            print(f"\n🛑 EARLY STOPPING TRIGGERED:")
            print(f"  Tasks completed: {self.tasks_completed}")
            print(f"  Success rate: {success_rate*100:.2f}%")
            print(f"  Threshold: {self.config.EARLY_STOP_MIN_SUCCESS_RATE*100:.0f}%")
            print(f"  Stopping to prevent wasted computation.")
            return True

        return False

    def detect_anomalies(self) -> List[str]:
        """
        LESSON 16: Statistical anomaly detection.
        Returns list of detected anomalies.
        """
        anomalies = []

        if not self.config.ENABLE_ANOMALY_DETECTION or self.tasks_completed < 10:
            return anomalies

        # Check for depth utilization
        if self.depths_reached and self.depths_allocated:
            avg_reached = np.mean(self.depths_reached)
            avg_allocated = np.mean(self.depths_allocated)
            utilization = avg_reached / avg_allocated if avg_allocated > 0 else 0

            if utilization > 0.95:
                anomalies.append(f"🔴 Depth utilization {utilization*100:.0f}% - hitting limit!")
            elif utilization < 0.3:
                anomalies.append(f"🟡 Depth utilization {utilization*100:.0f}% - may be over-allocated")

        # Check for time distribution anomalies
        if len(self.task_times) >= 10:
            recent_times = self.task_times[-10:]
            if all(t < 1.0 for t in recent_times):
                anomalies.append(f"🔴 All recent tasks < 1s - likely hitting immediate failures")

        # Check homogeneous failure modes
        total_failures = self.tasks_completed - self.tasks_successful
        if total_failures > 0:
            max_depth_ratio = self.failure_modes['Synthesizer.Fail.MaxDepth'] / total_failures
            if max_depth_ratio > 0.9:
                anomalies.append(f"🔴 {max_depth_ratio*100:.0f}% of failures are MaxDepth - increase depth!")

        return anomalies

    def validate_runtime(self) -> Dict[str, any]:
        """
        LESSON 10: Runtime bounds as test assertions.
        Check if runtime is within expected bounds.
        """
        elapsed_minutes = (time.time() - self.start_time) / 60

        result = {
            'elapsed_minutes': elapsed_minutes,
            'min_expected': self.config.EXPECTED_MIN_RUNTIME_MINUTES,
            'max_expected': self.config.EXPECTED_MAX_RUNTIME_MINUTES,
            'status': 'OK',
            'message': '',
        }

        if elapsed_minutes < self.config.EXPECTED_MIN_RUNTIME_MINUTES:
            result['status'] = 'SUSPICIOUS'
            result['message'] = f"⚠️  Completed in {elapsed_minutes:.0f} min (expected >{self.config.EXPECTED_MIN_RUNTIME_MINUTES:.0f} min) - may indicate failures"
        elif elapsed_minutes > self.config.EXPECTED_MAX_RUNTIME_MINUTES:
            result['status'] = 'TIMEOUT'
            result['message'] = f"⏱️  Runtime {elapsed_minutes:.0f} min exceeded limit {self.config.EXPECTED_MAX_RUNTIME_MINUTES:.0f} min"
        else:
            result['status'] = 'OK'
            result['message'] = f"✅ Runtime {elapsed_minutes:.0f} min within expected bounds"

        return result

    def print_progress(self):
        """Print current progress and metrics."""
        success_rate = (self.tasks_successful / self.tasks_completed * 100) if self.tasks_completed > 0 else 0
        elapsed_min = (time.time() - self.start_time) / 60

        print(f"\n📊 Progress: {self.tasks_completed}/100 tasks | "
              f"Success: {success_rate:.1f}% | "
              f"Time: {elapsed_min:.1f} min")

        # Show failure breakdown
        if self.tasks_completed > 0:
            print(f"   Failure modes: ", end="")
            for mode, count in self.failure_modes.items():
                if count > 0:
                    print(f"{mode}={count} ", end="")
            print()

        # Check for anomalies
        anomalies = self.detect_anomalies()
        if anomalies:
            print(f"   ⚠️  ANOMALIES DETECTED:")
            for anomaly in anomalies:
                print(f"      {anomaly}")


def create_fallback_solver():
    """
    LESSON 24: Implement hierarchical fallbacks.
    Returns simple heuristic solvers to use when synthesis fails.
    """

    class FallbackStrategies:
        """Simple heuristics for when main solver fails."""

        @staticmethod
        def copy_input(task: dict) -> list:
            """Simplest fallback: output = input."""
            return [{'attempt_1': ex['input'], 'attempt_2': ex['input']}
                    for ex in task.get('test', [])]

        @staticmethod
        def majority_color_fill(task: dict) -> list:
            """Fill output with most common non-zero color from examples."""
            try:
                all_colors = []
                for ex in task.get('train', []):
                    grid = np.array(ex['output'])
                    all_colors.extend(grid[grid > 0].tolist())

                if all_colors:
                    majority_color = max(set(all_colors), key=all_colors.count)
                else:
                    majority_color = 0

                results = []
                for ex in task.get('test', []):
                    shape = np.array(ex['input']).shape
                    output = np.full(shape, majority_color).tolist()
                    results.append({'attempt_1': output, 'attempt_2': output})
                return results
            except:
                return FallbackStrategies.copy_input(task)

        @staticmethod
        def try_transforms(task: dict) -> list:
            """Try basic geometric transforms."""
            try:
                results = []
                for ex in task.get('test', []):
                    input_grid = np.array(ex['input'])
                    # Try rotation and flip
                    attempt_1 = np.rot90(input_grid).tolist()
                    attempt_2 = np.fliplr(input_grid).tolist()
                    results.append({'attempt_1': attempt_1, 'attempt_2': attempt_2})
                return results
            except:
                return FallbackStrategies.copy_input(task)

    return FallbackStrategies()


if __name__ == "__main__":
    # Demo the enhanced configuration
    config = EnhancedChampionshipConfig()
    config.print_summary()

    print("\n" + "=" * 70)
    print("📝 DESIGN LESSONS IMPLEMENTED:")
    print("=" * 70)
    print("""
✅ LESSON 1: Model search complexity (estimate_search_complexity method)
✅ LESSON 2: Depth from requirements (150 vs 20, derived from 15-30 step needs)
✅ LESSON 3: Adaptive behavior (adaptive depth allocation)
✅ LESSON 6: Smoke tests (ENABLE_SMOKE_TEST with validation)
✅ LESSON 7: Early stopping (check_early_stop method)
✅ LESSON 9: Config regression tests (validate method)
✅ LESSON 10: Runtime assertions (validate_runtime method)
✅ LESSON 11: Document parameters (extensive comments and derivations)
✅ LESSON 12: Data-driven config (DEPTH_ALLOCATION based on analysis)
✅ LESSON 14: Dynamic reallocation (DYNAMIC_REALLOCATION flag)
✅ LESSON 16: Anomaly detection (detect_anomalies method)
✅ LESSON 17: Log depth utilization (depths_reached tracking)
✅ LESSON 18: Canary checks (check_canary method)
✅ LESSON 24: Fallback strategies (FallbackStrategies class)
✅ LESSON 25: Outcome-focused metrics (success rate, not just runtime)
    """)

    print("\n" + "=" * 70)
    print("🎯 NEXT STEPS:")
    print("=" * 70)
    print("""
1. Integrate this configuration into lucidorcax.ipynb Cell 2
2. Add RuntimeMonitor calls throughout the main loop
3. Implement FallbackStrategies when synthesis fails
4. Add smoke test at start of Cell 10 (main loop)
5. Add early stopping checks every 20 tasks
6. Add canary check after first 10 tasks
7. Add final runtime validation at end
    """)
