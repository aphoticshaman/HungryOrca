#!/usr/bin/env python3
"""
🧪 3-ROUND INTEGRATION TESTING FRAMEWORK
Comprehensive testing protocol with test-refactor-test x3 cycles per round.

TESTING PHILOSOPHY:
Round 1 (Underfit): Test basic functionality - ensure core features work
Round 2 (Overfit): Test edge cases - find breaking points
Round 3 (Sweet Spot): Find optimal balance - tune for production

Each round: Test → Refactor → Test → Refactor → Test (x3 cycle)
"""

import numpy as np
import json
import time
from typing import List, Dict, Tuple, Optional, Callable, Any
from dataclasses import dataclass, field
from pathlib import Path
import traceback

# Import all components to test
try:
    from vision_ebnf_hybrid import VisionEBNFHybridSolver, VisionModelEncoder, BeamSearchLLM
except ImportError:
    VisionEBNFHybridSolver = None
    print("⚠️  vision_ebnf_hybrid not available")

try:
    from interactive_arc_ui import InteractiveARCSession, GridEditor, TransformationToolkit
except ImportError:
    InteractiveARCSession = None
    print("⚠️  interactive_arc_ui not available")


# ═══════════════════════════════════════════════════════════════
# TEST DATA GENERATORS
# ═══════════════════════════════════════════════════════════════

class TestDataGenerator:
    """Generate test cases for different complexity levels"""

    @staticmethod
    def generate_simple_tasks() -> List[Dict]:
        """Round 1: Simple, basic transformations"""

        tasks = []

        # Task 1: Simple rotation
        tasks.append({
            'name': 'simple_rotation',
            'train': [
                {
                    'input': [[1, 0], [0, 0]],
                    'output': [[0, 1], [0, 0]]
                },
                {
                    'input': [[2, 0], [0, 0]],
                    'output': [[0, 2], [0, 0]]
                }
            ],
            'test': [
                {'input': [[3, 0], [0, 0]]}
            ],
            'expected_pattern': 'rotate_90'
        })

        # Task 2: Simple flip
        tasks.append({
            'name': 'simple_flip',
            'train': [
                {
                    'input': [[1, 2, 3]],
                    'output': [[3, 2, 1]]
                }
            ],
            'test': [
                {'input': [[4, 5, 6]]}
            ],
            'expected_pattern': 'flip_horizontal'
        })

        # Task 3: Color inversion
        tasks.append({
            'name': 'color_inversion',
            'train': [
                {
                    'input': [[1, 2], [3, 4]],
                    'output': [[8, 7], [6, 5]]
                }
            ],
            'test': [
                {'input': [[5, 6], [7, 8]]}
            ],
            'expected_pattern': 'invert_colors'
        })

        return tasks

    @staticmethod
    def generate_complex_tasks() -> List[Dict]:
        """Round 2: Complex, edge-case transformations"""

        tasks = []

        # Task 1: Multi-step transformation
        tasks.append({
            'name': 'multi_step',
            'train': [
                {
                    'input': [[1, 0, 0], [0, 0, 0], [0, 0, 0]],
                    'output': [[0, 0, 1], [0, 0, 0], [0, 0, 0]]
                }
            ],
            'test': [
                {'input': [[2, 0, 0], [0, 0, 0], [0, 0, 0]]}
            ],
            'expected_pattern': 'rotate_90_then_flip'
        })

        # Task 2: Large grid
        tasks.append({
            'name': 'large_grid',
            'train': [
                {
                    'input': np.random.randint(0, 5, (15, 15)).tolist(),
                    'output': np.rot90(np.random.randint(0, 5, (15, 15))).tolist()
                }
            ],
            'test': [
                {'input': np.random.randint(0, 5, (15, 15)).tolist()}
            ],
            'expected_pattern': 'rotation_large'
        })

        # Task 3: Sparse pattern
        tasks.append({
            'name': 'sparse_pattern',
            'train': [
                {
                    'input': [[0, 0, 0, 0, 1, 0, 0, 0, 0, 0]],
                    'output': [[0, 0, 0, 0, 0, 0, 1, 0, 0, 0]]
                }
            ],
            'test': [
                {'input': [[0, 0, 0, 1, 0, 0, 0, 0, 0, 0]]}
            ],
            'expected_pattern': 'shift_right'
        })

        # Task 4: Empty grids
        tasks.append({
            'name': 'empty_grids',
            'train': [
                {
                    'input': [[0, 0], [0, 0]],
                    'output': [[0, 0], [0, 0]]
                }
            ],
            'test': [
                {'input': [[0, 0, 0], [0, 0, 0], [0, 0, 0]]}
            ],
            'expected_pattern': 'identity'
        })

        return tasks

    @staticmethod
    def generate_realistic_tasks() -> List[Dict]:
        """Round 3: Realistic ARC-style tasks"""

        tasks = []

        # Task 1: Object manipulation
        tasks.append({
            'name': 'object_extraction',
            'train': [
                {
                    'input': [[0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0], [2, 2, 0, 0]],
                    'output': [[1, 1], [1, 1]]
                }
            ],
            'test': [
                {'input': [[0, 0, 0, 0], [3, 3, 0, 0], [3, 3, 0, 0], [0, 0, 0, 0]]}
            ],
            'expected_pattern': 'extract_largest_object'
        })

        # Task 2: Pattern completion
        tasks.append({
            'name': 'pattern_completion',
            'train': [
                {
                    'input': [[1, 0, 1], [0, 0, 0], [1, 0, 1]],
                    'output': [[1, 0, 1], [0, 1, 0], [1, 0, 1]]
                }
            ],
            'test': [
                {'input': [[2, 0, 2], [0, 0, 0], [2, 0, 2]]}
            ],
            'expected_pattern': 'fill_center'
        })

        return tasks


# ═══════════════════════════════════════════════════════════════
# TEST RESULT TRACKING
# ═══════════════════════════════════════════════════════════════

@dataclass
class TestResult:
    """Result of a single test"""
    test_name: str
    round_number: int
    cycle_number: int
    passed: bool
    execution_time: float
    error_message: str = ""
    confidence_score: float = 0.0
    metadata: Dict = field(default_factory=dict)


class TestResultAggregator:
    """Aggregate and analyze test results"""

    def __init__(self):
        self.results: List[TestResult] = []

    def add_result(self, result: TestResult):
        """Add test result"""
        self.results.append(result)

    def get_round_summary(self, round_number: int) -> Dict:
        """Get summary for specific round"""
        round_results = [r for r in self.results if r.round_number == round_number]

        if not round_results:
            return {'total': 0, 'passed': 0, 'failed': 0, 'pass_rate': 0.0}

        passed = sum(1 for r in round_results if r.passed)
        total = len(round_results)

        return {
            'round': round_number,
            'total': total,
            'passed': passed,
            'failed': total - passed,
            'pass_rate': passed / total if total > 0 else 0.0,
            'avg_time': np.mean([r.execution_time for r in round_results]),
            'avg_confidence': np.mean([r.confidence_score for r in round_results if r.passed]),
        }

    def get_full_report(self) -> str:
        """Generate comprehensive test report"""

        report = []
        report.append("\n" + "="*80)
        report.append("📊 INTEGRATION TESTING REPORT")
        report.append("="*80)

        # Overall statistics
        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)

        report.append(f"\n🎯 OVERALL RESULTS:")
        report.append(f"   Total Tests: {total}")
        report.append(f"   Passed: {passed} ✅")
        report.append(f"   Failed: {total - passed} ❌")
        report.append(f"   Pass Rate: {passed/total*100:.1f}%")

        # Round-by-round breakdown
        for round_num in [1, 2, 3]:
            summary = self.get_round_summary(round_num)

            if summary['total'] > 0:
                report.append(f"\n📋 ROUND {round_num} SUMMARY:")
                report.append(f"   Total: {summary['total']}")
                report.append(f"   Passed: {summary['passed']} ({summary['pass_rate']*100:.1f}%)")
                report.append(f"   Avg Time: {summary['avg_time']:.3f}s")
                if summary['passed'] > 0:
                    report.append(f"   Avg Confidence: {summary['avg_confidence']:.2f}")

        # Failed tests detail
        failed = [r for r in self.results if not r.passed]
        if failed:
            report.append(f"\n❌ FAILED TESTS ({len(failed)}):")
            for result in failed:
                report.append(f"   • {result.test_name} (Round {result.round_number}, Cycle {result.cycle_number})")
                if result.error_message:
                    report.append(f"     Error: {result.error_message[:100]}")

        report.append("\n" + "="*80)

        return '\n'.join(report)

    def save_report(self, filename: str):
        """Save report to file"""
        with open(filename, 'w') as f:
            f.write(self.get_full_report())

        print(f"📁 Report saved to: {filename}")


# ═══════════════════════════════════════════════════════════════
# TEST EXECUTORS
# ═══════════════════════════════════════════════════════════════

class ComponentTester:
    """Test individual components"""

    @staticmethod
    def test_vision_encoder(test_grid: np.ndarray) -> Tuple[bool, str, float]:
        """Test vision model encoder"""
        if VisionModelEncoder is None:
            return False, "VisionModelEncoder not available", 0.0

        try:
            start = time.time()
            encoder = VisionModelEncoder()
            features = encoder.encode_grid(test_grid)

            # Validate features
            assert features.shape_signature != ""
            assert len(features.color_histogram) == 10
            assert 0.0 <= features.complexity_score <= 1.0

            exec_time = time.time() - start
            return True, "", exec_time

        except Exception as e:
            return False, str(e), 0.0

    @staticmethod
    def test_beam_search(task: Dict) -> Tuple[bool, str, float, float]:
        """Test EBNF beam search"""
        if BeamSearchLLM is None:
            return False, "BeamSearchLLM not available", 0.0, 0.0

        try:
            start = time.time()

            llm = BeamSearchLLM(beam_width=5)

            # Get visual features
            encoder = VisionModelEncoder()
            input_grid = np.array(task['train'][0]['input'])
            features = encoder.encode_grid(input_grid)

            # Generate programs
            programs = llm.generate_program(features, task['train'], max_length=3)

            # Validate
            assert len(programs) > 0
            assert all(isinstance(p[0], str) and isinstance(p[1], float) for p in programs)

            exec_time = time.time() - start
            confidence = programs[0][1] if programs else 0.0

            return True, "", exec_time, confidence

        except Exception as e:
            return False, str(e), 0.0, 0.0

    @staticmethod
    def test_hybrid_solver(task: Dict) -> Tuple[bool, str, float, float]:
        """Test hybrid vision-EBNF solver"""
        if VisionEBNFHybridSolver is None:
            return False, "VisionEBNFHybridSolver not available", 0.0, 0.0

        try:
            start = time.time()

            solver = VisionEBNFHybridSolver(beam_width=5)
            predictions, confidence = solver.solve(task, timeout=5.0)

            # Validate
            if predictions is not None:
                assert isinstance(predictions, dict)
                success = True
                error_msg = ""
            else:
                success = False
                error_msg = "No predictions generated"

            exec_time = time.time() - start

            return success, error_msg, exec_time, confidence

        except Exception as e:
            return False, str(e), 0.0, 0.0


# ═══════════════════════════════════════════════════════════════
# MAIN TESTING ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════

class IntegrationTestOrchestrator:
    """
    Main orchestrator for 3-round testing protocol.

    Each round includes 3 test-refactor cycles.
    """

    def __init__(self):
        self.aggregator = TestResultAggregator()
        self.generator = TestDataGenerator()
        self.tester = ComponentTester()

    def run_all_rounds(self):
        """Execute all 3 rounds with test-refactor-test x3"""

        print("\n" + "="*80)
        print("🧪 STARTING 3-ROUND INTEGRATION TESTING")
        print("="*80)

        # Round 1: Underfit (Basic Functionality)
        print("\n🔵 ROUND 1: UNDERFIT - Testing Basic Functionality")
        print("-" * 80)
        self.run_round_1()

        # Round 2: Overfit (Edge Cases)
        print("\n🟠 ROUND 2: OVERFIT - Testing Edge Cases")
        print("-" * 80)
        self.run_round_2()

        # Round 3: Sweet Spot (Production Balance)
        print("\n🟢 ROUND 3: SWEET SPOT - Finding Optimal Balance")
        print("-" * 80)
        self.run_round_3()

        # Final report
        print("\n" + self.aggregator.get_full_report())

        # Save report
        self.aggregator.save_report('integration_test_report.txt')

    def run_round_1(self):
        """Round 1: Underfit - Basic functionality tests"""

        tasks = self.generator.generate_simple_tasks()

        for cycle in range(1, 4):  # 3 cycles
            print(f"\n  📌 Cycle {cycle}/3")

            for task in tasks:
                self._run_single_test(task, round_number=1, cycle=cycle)

            # Refactor checkpoint
            if cycle < 3:
                print(f"  🔧 Refactoring after cycle {cycle}...")
                time.sleep(0.5)  # Simulate refactoring time

    def run_round_2(self):
        """Round 2: Overfit - Edge case tests"""

        tasks = self.generator.generate_complex_tasks()

        for cycle in range(1, 4):
            print(f"\n  📌 Cycle {cycle}/3")

            for task in tasks:
                self._run_single_test(task, round_number=2, cycle=cycle)

            if cycle < 3:
                print(f"  🔧 Refactoring after cycle {cycle}...")
                time.sleep(0.5)

    def run_round_3(self):
        """Round 3: Sweet Spot - Realistic production tests"""

        tasks = self.generator.generate_realistic_tasks()

        for cycle in range(1, 4):
            print(f"\n  📌 Cycle {cycle}/3")

            for task in tasks:
                self._run_single_test(task, round_number=3, cycle=cycle)

            if cycle < 3:
                print(f"  🔧 Refactoring after cycle {cycle}...")
                time.sleep(0.5)

    def _run_single_test(self, task: Dict, round_number: int, cycle: int):
        """Run a single test case"""

        test_name = task.get('name', 'unnamed')

        print(f"    Testing: {test_name}...", end=' ')

        try:
            # Test components in sequence

            # 1. Vision encoder
            input_grid = np.array(task['train'][0]['input'])
            vision_pass, vision_err, vision_time = self.tester.test_vision_encoder(input_grid)

            if not vision_pass:
                result = TestResult(
                    test_name=f"{test_name}_vision",
                    round_number=round_number,
                    cycle_number=cycle,
                    passed=False,
                    execution_time=vision_time,
                    error_message=vision_err,
                )
                self.aggregator.add_result(result)
                print("❌ (Vision)")
                return

            # 2. Beam search
            beam_pass, beam_err, beam_time, beam_conf = self.tester.test_beam_search(task)

            if not beam_pass:
                result = TestResult(
                    test_name=f"{test_name}_beam",
                    round_number=round_number,
                    cycle_number=cycle,
                    passed=False,
                    execution_time=beam_time,
                    error_message=beam_err,
                )
                self.aggregator.add_result(result)
                print("❌ (Beam)")
                return

            # 3. Hybrid solver
            hybrid_pass, hybrid_err, hybrid_time, hybrid_conf = self.tester.test_hybrid_solver(task)

            result = TestResult(
                test_name=test_name,
                round_number=round_number,
                cycle_number=cycle,
                passed=hybrid_pass,
                execution_time=vision_time + beam_time + hybrid_time,
                error_message=hybrid_err,
                confidence_score=hybrid_conf,
            )

            self.aggregator.add_result(result)

            if hybrid_pass:
                print(f"✅ ({result.execution_time:.3f}s, conf={hybrid_conf:.2f})")
            else:
                print(f"❌ ({hybrid_err[:50]})")

        except Exception as e:
            result = TestResult(
                test_name=test_name,
                round_number=round_number,
                cycle_number=cycle,
                passed=False,
                execution_time=0.0,
                error_message=str(e),
            )
            self.aggregator.add_result(result)
            print(f"❌ (Exception: {str(e)[:50]})")


# ═══════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("🧪 INTEGRATION TESTING FRAMEWORK")
    print("=" * 80)
    print("\nThis will run 3 rounds of integration testing:")
    print("  Round 1: Underfit - Basic functionality (3 cycles)")
    print("  Round 2: Overfit - Edge cases (3 cycles)")
    print("  Round 3: Sweet Spot - Production balance (3 cycles)")
    print("\nTotal: 9 test cycles across all components")
    print("=" * 80)

    input("\nPress Enter to start testing...")

    orchestrator = IntegrationTestOrchestrator()
    orchestrator.run_all_rounds()

    print("\n✅ Testing complete!")
    print("📁 Full report saved to: integration_test_report.txt")
