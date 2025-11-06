#!/usr/bin/env python3
"""
🧪 ABLATION TEST - Test Each Reasoning Layer Separately

Tests 5 configurations to measure contribution of each layer:
- Config 0: Fallback only (baseline)
- Config 1: + Primitive brute-force
- Config 2: + Abstract reasoning
- Config 3: + Meta-cognitive reasoning
- Config 4: All layers combined

Run on 20 tasks from evaluation set.

Author: Ryan Cardwell & Claude
Date: November 2025
"""

import json
import numpy as np
import time
from pathlib import Path
from typing import Dict, List
from quantum_arc_exploiter import QuantumARCExploiter
from lucidorcavZ import LucidOrcaVZ, ChampionshipConfig


class AblationTester:
    """Test each reasoning layer's contribution"""

    def __init__(self):
        self.results = []

    def load_test_tasks(self, n: int = 5) -> Dict:
        """Load n tasks from evaluation set"""
        eval_path = Path("arc-agi_evaluation_challenges.json")

        if not eval_path.exists():
            print(f"⚠️  Evaluation file not found, using training set for ablation")
            eval_path = Path("arc-agi_training_challenges.json")

        if not eval_path.exists():
            print(f"❌ No task files found!")
            return {}

        with open(eval_path, 'r') as f:
            all_tasks = json.load(f)

        # Take first n tasks
        task_ids = list(all_tasks.keys())[:n]
        test_tasks = {tid: all_tasks[tid] for tid in task_ids}

        return test_tasks

    def load_solutions(self, tasks: Dict) -> Dict:
        """Load ground truth solutions"""
        solutions_path = Path("arc-agi_evaluation_solutions.json")

        if not solutions_path.exists():
            print(f"⚠️  Solutions not found, using training solutions")
            solutions_path = Path("arc-agi_training_solutions.json")

        if not solutions_path.exists():
            return {}

        with open(solutions_path, 'r') as f:
            all_solutions = json.load(f)

        # Get solutions for our test tasks
        return {tid: all_solutions[tid] for tid in tasks.keys() if tid in all_solutions}

    def test_config(self, config_name: str, enable_metacog: bool, enable_abstract: bool,
                    enable_primitives: bool, tasks: Dict, solutions: Dict) -> Dict:
        """
        Test a specific configuration

        Args:
            config_name: Name of configuration
            enable_metacog: Enable metacognitive reasoning
            enable_abstract: Enable abstract reasoning
            enable_primitives: Enable primitive brute-force
            tasks: Test tasks
            solutions: Ground truth solutions

        Returns:
            Dict with results
        """
        print(f"\n{'='*70}")
        print(f"🧪 Testing: {config_name}")
        print(f"   Metacognitive: {'✅' if enable_metacog else '❌'}")
        print(f"   Abstract: {'✅' if enable_abstract else '❌'}")
        print(f"   Primitives: {'✅' if enable_primitives else '❌'}")
        print('='*70)

        # Initialize solver with disabled layers
        config = ChampionshipConfig()
        base_solver = LucidOrcaVZ(config)
        quantum = QuantumARCExploiter(solvers={'full': base_solver})

        # Monkey-patch to disable layers
        original_metacog = quantum.metacognitive_reasoner.reason_and_solve
        original_abstract = quantum.abstract_reasoner.reason_and_solve
        original_primitives = quantum._try_geometric_primitives

        if not enable_metacog:
            quantum.metacognitive_reasoner.reason_and_solve = lambda task: (None, {})

        if not enable_abstract:
            quantum.abstract_reasoner.reason_and_solve = lambda task: None

        if not enable_primitives:
            quantum._try_geometric_primitives = lambda task: None

        # Test on all tasks
        correct = 0
        total = 0
        methods_used = []
        start_time = time.time()

        for task_id, task in tasks.items():
            if task_id not in solutions:
                continue

            try:
                # Solve
                result = quantum.solve_with_quantum_exploitation(task, timeout=2)

                # Check both attempts
                ground_truth = solutions[task_id]
                attempt_1 = np.array(result['attempt_1'])
                attempt_2 = np.array(result['attempt_2'])

                # Score: ANY match counts
                match = False
                for gt in ground_truth:
                    gt_arr = np.array(gt)
                    if np.array_equal(attempt_1, gt_arr) or np.array_equal(attempt_2, gt_arr):
                        match = True
                        break

                if match:
                    correct += 1

                total += 1
                methods_used.append(result.get('method', 'unknown'))

            except Exception as e:
                print(f"   ❌ {task_id}: {e}")
                total += 1

        # Calculate stats with 6 decimal precision
        elapsed = time.time() - start_time
        accuracy = (correct / total * 100.0) if total > 0 else 0.000000

        # Count method usage
        from collections import Counter
        method_counts = Counter(methods_used)

        result_dict = {
            'config': config_name,
            'enabled': {
                'metacognitive': enable_metacog,
                'abstract': enable_abstract,
                'primitives': enable_primitives
            },
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'time': elapsed,
            'methods_used': dict(method_counts)
        }

        # Restore original methods
        quantum.metacognitive_reasoner.reason_and_solve = original_metacog
        quantum.abstract_reasoner.reason_and_solve = original_abstract
        quantum._try_geometric_primitives = original_primitives

        print(f"\n📊 Results:")
        print(f"   Accuracy: {accuracy:.6f}% ({correct}/{total})")
        print(f"   Time: {elapsed:.6f}s ({elapsed/total:.6f}s per task)")
        print(f"   Methods: {dict(method_counts)}")

        return result_dict

    def run_ablation(self):
        """Run full ablation study"""
        print("\n" + "="*70)
        print("🧪 ABLATION STUDY - Testing Each Reasoning Layer")
        print("="*70)

        # Load test tasks
        print("\n📂 Loading test tasks...")
        tasks = self.load_test_tasks(n=5)
        solutions = self.load_solutions(tasks)

        print(f"   ✅ Loaded {len(tasks)} tasks")
        print(f"   ✅ Loaded {len(solutions)} solutions")

        if len(solutions) == 0:
            print("\n⚠️  No ground truth solutions available!")
            print("   Will test but cannot measure accuracy")

        # Configuration matrix
        configs = [
            ("Config 0: Fallback Only", False, False, False),
            ("Config 1: + Primitives", False, False, True),
            ("Config 2: + Abstract", False, True, True),
            ("Config 3: + Metacognitive", True, False, False),
            ("Config 4: All Layers", True, True, True),
        ]

        # Test each configuration
        results = []
        for config_name, enable_metacog, enable_abstract, enable_primitives in configs:
            result = self.test_config(
                config_name, enable_metacog, enable_abstract, enable_primitives,
                tasks, solutions
            )
            results.append(result)

        # Summary
        print("\n" + "="*70)
        print("📊 ABLATION STUDY RESULTS")
        print("="*70)

        print(f"\n{'Config':<30} {'Accuracy':<18} {'Time':<18} {'Primary Method':<20}")
        print("-"*80)

        for r in results:
            primary_method = max(r['methods_used'].items(), key=lambda x: x[1])[0] if r['methods_used'] else 'none'
            print(f"{r['config']:<30} {r['accuracy']:>12.6f}%     {r['time']:>12.6f}s      {primary_method:<20}")

        # Calculate deltas with 6 decimal precision
        print("\n📈 ACCURACY GAINS:")
        baseline = results[0]['accuracy']
        for r in results[1:]:
            delta = r['accuracy'] - baseline
            print(f"   {r['config']:<30} {delta:>+12.6f}%")

        # Best configuration
        best = max(results, key=lambda x: x['accuracy'])
        print(f"\n🏆 BEST CONFIGURATION: {best['config']}")
        print(f"   Accuracy: {best['accuracy']:.6f}%")
        print(f"   Gain: +{best['accuracy'] - baseline:.6f}% vs baseline")

        # Save results
        output_path = Path("ablation_results.json")
        with open(output_path, 'w') as f:
            json.dump({
                'summary': results,
                'best_config': best,
                'baseline_accuracy': baseline,
                'tasks_tested': len(tasks)
            }, f, indent=2)

        print(f"\n💾 Results saved: {output_path}")
        print("="*70)

        return results


def main():
    """Run ablation test"""
    tester = AblationTester()
    tester.run_ablation()


if __name__ == "__main__":
    main()
