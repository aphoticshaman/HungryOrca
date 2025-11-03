#!/usr/bin/env python3
"""
ABLATION TEST: META COACH + SPECIALISTS

x5 testing to prove MetaCoach provides value!

Conditions (x5 each):
A. Specialists alone (no coach)
B. MetaCoach orchestrating specialists

Hypothesis: Coach orchestration improves performance by identifying
synergies and calling audibles.

ONLY keep the coach if it passes ablation!
"""

import json
import numpy as np
from collaborative_multi_agent_solver import CollaborativeSolver
from meta_coach_solver import MetaCoachSolver

# Load data
with open('arc-agi_training_challenges.json', 'r') as f:
    challenges = json.load(f)

with open('arc-agi_training_solutions.json', 'r') as f:
    solutions = json.load(f)

task_ids = list(challenges.keys())[:10]


def run_solver(solver, num_runs=5):
    """Run solver multiple times."""
    all_runs = []

    for run_num in range(1, num_runs + 1):
        solved = []
        partial_scores = []

        for task_id in task_ids:
            task = challenges[task_id]
            train_pairs = [(np.array(ex['input']), np.array(ex['output']))
                          for ex in task['train']]

            test_input = np.array(task['test'][0]['input'])
            expected = np.array(solutions[task_id][0]) if task_id in solutions else None

            try:
                predicted = solver.solve(train_pairs, test_input)

                if predicted is not None and expected is not None and predicted.shape == expected.shape:
                    score = np.sum(predicted == expected) / predicted.size
                    partial_scores.append(score)

                    if np.array_equal(predicted, expected):
                        solved.append(task_id)
                else:
                    partial_scores.append(0.0)
            except:
                partial_scores.append(0.0)

        avg_partial = np.mean(partial_scores) * 100

        all_runs.append({
            'run': run_num,
            'exact_matches': len(solved),
            'avg_partial_match': avg_partial,
            'solved_tasks': solved
        })

    return all_runs


print("="*80)
print("ABLATION TEST: META COACH vs SPECIALISTS ALONE")
print("="*80)
print("\nCondition A: Specialists only (no coach) - 5 runs")
print("Condition B: MetaCoach orchestrating - 5 runs")
print("\n" + "="*80)

# Condition A: Specialists alone
print("\nRUNNING CONDITION A (Specialists alone)...")
solver_a = CollaborativeSolver(verbose=False)
results_a = run_solver(solver_a, num_runs=5)

for r in results_a:
    print(f"  Run {r['run']}: {r['exact_matches']}/10 exact, {r['avg_partial_match']:.1f}% partial")

# Condition B: MetaCoach
print("\nRUNNING CONDITION B (MetaCoach orchestrating)...")
solver_b = MetaCoachSolver(verbose=False)
results_b = run_solver(solver_b, num_runs=5)

for r in results_b:
    print(f"  Run {r['run']}: {r['exact_matches']}/10 exact, {r['avg_partial_match']:.1f}% partial")

# Analysis
print("\n" + "="*80)
print("STATISTICAL ANALYSIS")
print("="*80)

exact_a = [r['exact_matches'] for r in results_a]
exact_b = [r['exact_matches'] for r in results_b]
partial_a = [r['avg_partial_match'] for r in results_a]
partial_b = [r['avg_partial_match'] for r in results_b]

mean_exact_a = np.mean(exact_a)
mean_exact_b = np.mean(exact_b)
mean_partial_a = np.mean(partial_a)
mean_partial_b = np.mean(partial_b)

print(f"\nExact Matches:")
print(f"  Specialists alone: {mean_exact_a:.1f}/10")
print(f"  With MetaCoach:    {mean_exact_b:.1f}/10")
print(f"  Improvement:       {mean_exact_b - mean_exact_a:+.1f}")

print(f"\nPartial Match:")
print(f"  Specialists alone: {mean_partial_a:.1f}%")
print(f"  With MetaCoach:    {mean_partial_b:.1f}%")
print(f"  Improvement:       {mean_partial_b - mean_partial_a:+.1f}%")

# Decision
print("\n" + "="*80)
print("GO/NO-GO DECISION")
print("="*80)

improvement = mean_partial_b - mean_partial_a

if mean_exact_b > mean_exact_a:
    decision = "✅ GO - MetaCoach delivers exact matches!"
elif improvement >= 2.0:
    decision = "✅ GO - MetaCoach improves partial match by ≥2%"
elif improvement > 0:
    decision = "⚠️ MARGINAL - Slight improvement but < 2%"
else:
    decision = "❌ NO-GO - No improvement, remove MetaCoach"

print(f"\n{decision}")

# Export
results = {
    'condition_a': results_a,
    'condition_b': results_b,
    'mean_exact_a': mean_exact_a,
    'mean_exact_b': mean_exact_b,
    'mean_partial_a': mean_partial_a,
    'mean_partial_b': mean_partial_b,
    'improvement': improvement,
    'decision': decision
}

with open('ablation_meta_coach_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n✓ Results exported to: ablation_meta_coach_results.json")
print("="*80)
