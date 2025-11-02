#!/usr/bin/env python3
"""
RIGOROUS 5× ABLATION STUDY

Tests three conditions with 5 runs each:
1. Baseline: Best individual BOLT-ON (Pattern solver)
2. Fuzzy-Integrated: Fuzzy controller + all BOLT-ONs
3. Statistical comparison with paired t-tests

Methodology: User's guidance - "x5 rounds of testing"

Success criteria:
- p < 0.05 (statistical significance)
- Improvement ≥ 5% accuracy

Author: HungryOrca Ablation Framework
Date: 2025-11-02
"""

import json
import numpy as np
import sys
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass
import math
sys.path.insert(0, str(Path(__file__).parent / 'SubtleGenius' / 'arc_solvers'))


# Manual statistical functions (no scipy dependency)
def manual_ttest_rel(a, b):
    """Paired t-test without scipy."""
    diffs = np.array(a) - np.array(b)
    n = len(diffs)
    mean_diff = np.mean(diffs)
    std_diff = np.std(diffs, ddof=1)
    t_stat = mean_diff / (std_diff / math.sqrt(n)) if std_diff > 0 else 0

    # Approximate p-value using t-distribution
    # For n-1 degrees of freedom
    df = n - 1
    # Two-tailed p-value approximation
    abs_t = abs(t_stat)

    # Rough p-value estimate (for df >= 4)
    if df >= 4:
        p_value = 2 * (1 - 0.5 * (1 + math.erf(abs_t / math.sqrt(2))))
    else:
        p_value = 0.5  # Conservative

    return t_stat, p_value


def manual_sem(data):
    """Standard error of mean."""
    n = len(data)
    std = np.std(data, ddof=1)
    return std / math.sqrt(n) if n > 0 else 0


def manual_confidence_interval(data, confidence=0.95):
    """Compute confidence interval."""
    n = len(data)
    mean = np.mean(data)
    sem = manual_sem(data)

    # t-value for 95% CI (approximate for common df values)
    t_values = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
                6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}

    df = n - 1
    t_val = t_values.get(df, 2.0)  # Default to ~2 for large df

    margin = t_val * sem
    return (mean - margin, mean + margin)

from pattern_transformation_solver import PatternTransformationSolver
from fuzzy_arc_integrated_solver import FuzzyIntegratedARCSolver


@dataclass
class RunResult:
    """Results from a single run."""
    condition: str
    run_number: int
    tasks_solved: int
    total_tasks: int
    accuracy: float
    solved_task_ids: List[str]


class AblationStudy:
    """Manages 5× ablation study with statistical analysis."""

    def __init__(self, task_ids: List[str], num_runs: int = 5):
        self.task_ids = task_ids
        self.num_runs = num_runs
        self.results: List[RunResult] = []

        # Load data
        with open('arc-agi_training_challenges.json', 'r') as f:
            self.challenges = json.load(f)

        with open('arc-agi_training_solutions.json', 'r') as f:
            self.solutions = json.load(f)

    def run_condition(self, condition_name: str, solver) -> List[RunResult]:
        """Run one condition 5 times."""
        print(f"\n{'='*80}")
        print(f"CONDITION: {condition_name}")
        print(f"{'='*80}")

        condition_results = []

        for run_num in range(1, self.num_runs + 1):
            print(f"\nRun {run_num}/{self.num_runs}...")

            solved = 0
            total = 0
            solved_task_ids = []

            for task_id in self.task_ids:
                task = self.challenges[task_id]
                train_pairs = [(np.array(ex['input']), np.array(ex['output']))
                               for ex in task['train']]

                for test_idx, test_ex in enumerate(task['test']):
                    test_input = np.array(test_ex['input'])

                    if task_id in self.solutions and test_idx < len(self.solutions[task_id]):
                        expected = np.array(self.solutions[task_id][test_idx])

                        try:
                            predicted = solver.solve(train_pairs, test_input)

                            if predicted is not None and np.array_equal(predicted, expected):
                                solved += 1
                                if task_id not in solved_task_ids:
                                    solved_task_ids.append(task_id)
                        except:
                            pass

                        total += 1

            accuracy = solved / total if total > 0 else 0

            result = RunResult(
                condition=condition_name,
                run_number=run_num,
                tasks_solved=len(solved_task_ids),
                total_tasks=len(self.task_ids),
                accuracy=accuracy,
                solved_task_ids=solved_task_ids
            )

            condition_results.append(result)
            self.results.append(result)

            print(f"  Tasks solved: {result.tasks_solved}/{result.total_tasks}")
            print(f"  Accuracy: {result.accuracy:.2%}")

        return condition_results

    def analyze_results(self):
        """Statistical analysis of results."""
        print("\n" + "="*80)
        print("STATISTICAL ANALYSIS")
        print("="*80)

        # Group by condition
        conditions = {}
        for result in self.results:
            if result.condition not in conditions:
                conditions[result.condition] = []
            conditions[result.condition].append(result.accuracy)

        # Compute summary statistics
        print("\n### SUMMARY STATISTICS ###\n")

        for condition, accuracies in conditions.items():
            mean = np.mean(accuracies)
            std = np.std(accuracies, ddof=1)
            sem = manual_sem(accuracies)
            ci = manual_confidence_interval(accuracies)

            print(f"{condition}:")
            print(f"  Mean accuracy: {mean:.2%} ± {std:.2%} (SD)")
            print(f"  95% CI: [{ci[0]:.2%}, {ci[1]:.2%}]")
            print(f"  Runs: {accuracies}")
            print()

        # Paired t-tests
        print("\n### PAIRED T-TESTS ###\n")

        condition_names = list(conditions.keys())

        for i, cond1 in enumerate(condition_names):
            for cond2 in condition_names[i+1:]:
                acc1 = conditions[cond1]
                acc2 = conditions[cond2]

                # Paired t-test
                t_stat, p_value = manual_ttest_rel(acc1, acc2)

                mean_diff = np.mean(acc2) - np.mean(acc1)
                improvement_pct = (mean_diff / (np.mean(acc1) + 1e-10)) * 100

                print(f"{cond2} vs {cond1}:")
                print(f"  t-statistic: {t_stat:.3f}")
                print(f"  p-value: {p_value:.4f}")
                print(f"  Mean difference: {mean_diff:.2%}")
                print(f"  Relative improvement: {improvement_pct:.1f}%")

                if p_value < 0.05:
                    if mean_diff > 0:
                        print(f"  ✅ STATISTICALLY SIGNIFICANT IMPROVEMENT (p < 0.05)")
                    else:
                        print(f"  ⚠️  STATISTICALLY SIGNIFICANT REGRESSION (p < 0.05)")
                else:
                    print(f"  ❌ No statistical significance (p >= 0.05)")

                print()

        # Effect size (Cohen's d)
        print("\n### EFFECT SIZES (Cohen's d) ###\n")

        for i, cond1 in enumerate(condition_names):
            for cond2 in condition_names[i+1:]:
                acc1 = conditions[cond1]
                acc2 = conditions[cond2]

                mean1, mean2 = np.mean(acc1), np.mean(acc2)
                std1, std2 = np.std(acc1, ddof=1), np.std(acc2, ddof=1)
                pooled_std = np.sqrt((std1**2 + std2**2) / 2)

                cohens_d = (mean2 - mean1) / pooled_std if pooled_std > 0 else 0

                print(f"{cond2} vs {cond1}:")
                print(f"  Cohen's d: {cohens_d:.3f}")

                if abs(cohens_d) < 0.2:
                    effect = "negligible"
                elif abs(cohens_d) < 0.5:
                    effect = "small"
                elif abs(cohens_d) < 0.8:
                    effect = "medium"
                else:
                    effect = "large"

                print(f"  Effect size: {effect}")
                print()

    def export_results(self, filename: str):
        """Export results to JSON."""
        data = {
            'study_parameters': {
                'num_runs': self.num_runs,
                'num_tasks': len(self.task_ids),
                'task_ids': self.task_ids
            },
            'results': [
                {
                    'condition': r.condition,
                    'run_number': r.run_number,
                    'tasks_solved': r.tasks_solved,
                    'total_tasks': r.total_tasks,
                    'accuracy': r.accuracy,
                    'solved_task_ids': r.solved_task_ids
                }
                for r in self.results
            ]
        }

        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"\n✓ Results exported to: {filename}")


def main():
    """Run complete 5× ablation study."""
    print("="*80)
    print("5× ABLATION STUDY: Baseline vs Fuzzy-Integrated")
    print("="*80)
    print("\nMethodology: User guidance - 'x5 rounds of testing'")
    print("Success criteria: p < 0.05, improvement ≥ 5%\n")

    # Test on 10 tasks
    with open('arc-agi_training_challenges.json', 'r') as f:
        challenges = json.load(f)

    task_ids = list(challenges.keys())[:10]

    study = AblationStudy(task_ids, num_runs=5)

    # Condition 1: Baseline (Pattern solver - best individual component)
    print("\nPreparing Condition 1: BASELINE (Pattern Solver)...")
    baseline_solver = PatternTransformationSolver()
    baseline_results = study.run_condition('Baseline-Pattern', baseline_solver)

    # Condition 2: Fuzzy-Integrated
    print("\nPreparing Condition 2: FUZZY-INTEGRATED...")
    fuzzy_solver = FuzzyIntegratedARCSolver()
    fuzzy_results = study.run_condition('Fuzzy-Integrated', fuzzy_solver)

    # Statistical analysis
    study.analyze_results()

    # Export results
    study.export_results('ablation_study_5x_results.json')

    print("\n" + "="*80)
    print("ABLATION STUDY COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
