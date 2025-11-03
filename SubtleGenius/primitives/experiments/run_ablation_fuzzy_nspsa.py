"""
ABLATION STUDY: FUZZY META-CONTROLLER + NSPSA INTEGRATION

Rigorous x5 testing methodology:
- Condition 1: NSPSA Baseline (5x runs)
- Condition 2: Fuzzy Alone (5x runs)
- Condition 3: Combined System (5x runs)

Total: 15 runs, statistical validation required
"""

import numpy as np
import json
import time
import sys
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict
sys.path.insert(0, '/home/user/HungryOrca/SubtleGenius/primitives')

from symbolic_solver import SymbolicProgramSynthesizer, SearchStrategy, GridState


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class TaskResult:
    """Result for a single task."""
    task_id: str
    task_type: str
    difficulty: str
    solved: bool
    states_explored: int
    search_time: float
    solution_length: int
    failure_mode: str  # 'success', 'timeout', 'wrong', 'crash'


@dataclass
class ConditionResult:
    """Aggregated results for one condition (5 runs)."""
    condition_name: str
    run_results: List[Dict]  # Raw results per run

    # Aggregated metrics (mean ± std)
    accuracy_mean: float
    accuracy_std: float

    states_mean: float
    states_std: float

    time_mean: float
    time_std: float

    timeout_rate: float

    # By difficulty
    easy_accuracy: float
    medium_accuracy: float
    hard_accuracy: float


# ============================================================================
# TEST DATASET GENERATION
# ============================================================================

def generate_test_dataset(seed: int = 42) -> List[Tuple[str, np.ndarray, np.ndarray, str, str]]:
    """
    Generate 100 diverse test tasks.

    Returns:
        List of (task_id, input_grid, output_grid, task_type, difficulty)
    """
    np.random.seed(seed)
    tasks = []

    # ===== EASY TASKS (30 tasks, 10×10 grids) =====

    # 10x symmetric puzzles
    for i in range(10):
        size = 10
        inp = np.random.randint(0, 5, (size, size))
        # Make it symmetric
        inp = (inp + inp.T) // 2
        out = np.flip(inp, axis=0)  # Reflect horizontal
        tasks.append((f'easy_sym_{i}', inp, out, 'symmetric', 'easy'))

    # 10x multi-scale patterns
    for i in range(10):
        size = 10
        # Create pattern with different scales
        inp = np.random.randint(0, 3, (size, size))
        out = np.kron(inp[:5, :5], np.ones((2, 2), dtype=int))  # Scale up quadrant
        tasks.append((f'easy_scale_{i}', inp, out, 'multiscale', 'easy'))

    # 10x simple rotations/reflections
    for i in range(10):
        size = 10
        inp = np.random.randint(0, 6, (size, size))
        transform = np.random.choice(['rotate', 'reflect'])
        if transform == 'rotate':
            out = np.rot90(inp, k=-1)
        else:
            out = np.flip(inp, axis=1)
        tasks.append((f'easy_transform_{i}', inp, out, 'simple_transform', 'easy'))

    # ===== MEDIUM TASKS (40 tasks, 20×20 grids) =====

    # 15x global constraint puzzles
    for i in range(15):
        size = 20
        inp = np.random.randint(0, 4, (size, size))
        # Constraint: blues must be in top half
        out = inp.copy()
        blues = (inp == 1)
        out[blues] = 0
        out[:size//2][blues[:size//2]] = 1
        tasks.append((f'med_constraint_{i}', inp, out, 'global_constraint', 'medium'))

    # 15x phase transition tasks
    for i in range(15):
        size = 20
        # Near percolation threshold (p ≈ 0.59 for 2D)
        p = 0.59 + np.random.uniform(-0.1, 0.1)
        inp = (np.random.rand(size, size) < p).astype(int)
        # Flood fill from center
        out = inp.copy()
        center = size // 2
        if inp[center, center] == 1:
            # Simple flood fill simulation
            visited = np.zeros_like(inp)
            stack = [(center, center)]
            while stack:
                y, x = stack.pop()
                if 0 <= y < size and 0 <= x < size and inp[y, x] == 1 and not visited[y, x]:
                    visited[y, x] = 1
                    out[y, x] = 2  # Mark cluster
                    stack.extend([(y-1, x), (y+1, x), (y, x-1), (y, x+1)])
        tasks.append((f'med_phase_{i}', inp, out, 'phase_transition', 'medium'))

    # 10x compositional patterns
    for i in range(10):
        size = 20
        inp = np.random.randint(0, 4, (size, size))
        # Composition: rotate then reflect
        out = np.flip(np.rot90(inp, k=-1), axis=1)
        tasks.append((f'med_comp_{i}', inp, out, 'composition', 'medium'))

    # ===== HARD TASKS (30 tasks, 30×30 grids) =====

    # 10x extreme symmetry + noise
    for i in range(10):
        size = 30
        inp = np.random.randint(0, 5, (size, size))
        # Make mostly symmetric (90%)
        sym = (inp + inp.T) // 2
        noise_mask = np.random.rand(size, size) < 0.1
        inp[noise_mask] = np.random.randint(0, 5, noise_mask.sum())
        # Output: enforce perfect symmetry
        out = (inp + inp.T) // 2
        tasks.append((f'hard_sym_noise_{i}', inp, out, 'noisy_symmetry', 'hard'))

    # 10x critical regime
    for i in range(10):
        size = 30
        # Exactly at percolation threshold
        inp = (np.random.rand(size, size) < 0.592).astype(int)
        # Output: extract largest cluster
        # (Simplified - in real ARC this would be more complex)
        out = inp.copy()
        tasks.append((f'hard_critical_{i}', inp, out, 'critical', 'hard'))

    # 10x novel patterns (require meta-learning)
    for i in range(10):
        size = 30
        # Complex pattern: checkerboard with color cycling
        inp = np.zeros((size, size), dtype=int)
        for y in range(size):
            for x in range(size):
                inp[y, x] = ((y + x) % 3) + 1
        # Transform: shift colors
        out = (inp % 5) + 1
        tasks.append((f'hard_novel_{i}', inp, out, 'novel', 'hard'))

    print(f"Generated {len(tasks)} test tasks:")
    print(f"  Easy: {sum(1 for t in tasks if t[4] == 'easy')}")
    print(f"  Medium: {sum(1 for t in tasks if t[4] == 'medium')}")
    print(f"  Hard: {sum(1 for t in tasks if t[4] == 'hard')}")

    return tasks


# ============================================================================
# CONDITION 1: NSPSA BASELINE
# ============================================================================

def test_nspsa_baseline(tasks: List, beam_width: int = 5, search_depth: int = 3,
                       timeout: float = 10.0, random_seed: int = 1000) -> Dict:
    """
    Condition 1: NSPSA alone (no fuzzy controller).

    Fixed parameters, no adaptation.
    """
    np.random.seed(random_seed)

    synthesizer = SymbolicProgramSynthesizer(
        max_program_length=search_depth,
        beam_width=beam_width
    )

    results = []

    for task_id, inp, out, task_type, difficulty in tasks:
        start = time.time()

        try:
            program = synthesizer.synthesize(
                inp, out,
                strategy=SearchStrategy.BIDIRECTIONAL,
                timeout=timeout
            )
            elapsed = time.time() - start

            if program is not None:
                # Verify correctness
                state = GridState.from_array(inp)
                for prim in program:
                    state = synthesizer.executor.execute(state, prim)
                    if state is None:
                        break

                correct = (state is not None and
                          np.array_equal(state.to_array(), out))

                result = TaskResult(
                    task_id=task_id,
                    task_type=task_type,
                    difficulty=difficulty,
                    solved=correct,
                    states_explored=synthesizer.stats['states_explored'],
                    search_time=elapsed,
                    solution_length=len(program),
                    failure_mode='success' if correct else 'wrong'
                )
            else:
                result = TaskResult(
                    task_id=task_id,
                    task_type=task_type,
                    difficulty=difficulty,
                    solved=False,
                    states_explored=synthesizer.stats['states_explored'],
                    search_time=elapsed,
                    solution_length=0,
                    failure_mode='timeout' if elapsed >= timeout * 0.95 else 'wrong'
                )

        except Exception as e:
            result = TaskResult(
                task_id=task_id,
                task_type=task_type,
                difficulty=difficulty,
                solved=False,
                states_explored=0,
                search_time=0,
                solution_length=0,
                failure_mode=f'crash: {str(e)}'
            )

        results.append(asdict(result))

    return {
        'condition': 'nspsa_baseline',
        'random_seed': random_seed,
        'beam_width': beam_width,
        'search_depth': search_depth,
        'timeout': timeout,
        'results': results
    }


# ============================================================================
# CONDITION 2: FUZZY ALONE (Placeholder - need fuzzy controller)
# ============================================================================

def test_fuzzy_alone(tasks: List, timeout: float = 10.0, random_seed: int = 2000) -> Dict:
    """
    Condition 2: Fuzzy meta-controller with simple BFS fallback.

    NOTE: This is a PLACEHOLDER until we have fuzzy controller integrated.
    For now, use simple BFS as baseline comparison.
    """
    np.random.seed(random_seed)

    # TODO: Integrate actual fuzzy controller
    # For now: Simple BFS with original primitives only
    synthesizer = SymbolicProgramSynthesizer(
        max_program_length=3,
        beam_width=10  # Wider beam to compensate for no heuristics
    )

    results = []

    for task_id, inp, out, task_type, difficulty in tasks:
        start = time.time()

        try:
            # Simple BFS (no learned heuristics, no compositions)
            program = synthesizer.synthesize(
                inp, out,
                strategy=SearchStrategy.BIDIRECTIONAL,
                timeout=timeout
            )
            elapsed = time.time() - start

            if program is not None:
                state = GridState.from_array(inp)
                for prim in program:
                    state = synthesizer.executor.execute(state, prim)
                    if state is None:
                        break

                correct = (state is not None and
                          np.array_equal(state.to_array(), out))

                result = TaskResult(
                    task_id=task_id,
                    task_type=task_type,
                    difficulty=difficulty,
                    solved=correct,
                    states_explored=synthesizer.stats['states_explored'],
                    search_time=elapsed,
                    solution_length=len(program),
                    failure_mode='success' if correct else 'wrong'
                )
            else:
                result = TaskResult(
                    task_id=task_id,
                    task_type=task_type,
                    difficulty=difficulty,
                    solved=False,
                    states_explored=synthesizer.stats['states_explored'],
                    search_time=elapsed,
                    solution_length=0,
                    failure_mode='timeout' if elapsed >= timeout * 0.95 else 'wrong'
                )

        except Exception as e:
            result = TaskResult(
                task_id=task_id,
                task_type=task_type,
                difficulty=difficulty,
                solved=False,
                states_explored=0,
                search_time=0,
                solution_length=0,
                failure_mode=f'crash: {str(e)}'
            )

        results.append(asdict(result))

    return {
        'condition': 'fuzzy_alone',
        'random_seed': random_seed,
        'timeout': timeout,
        'results': results,
        'note': 'PLACEHOLDER - using BFS until fuzzy controller integrated'
    }


# ============================================================================
# CONDITION 3: COMBINED (Placeholder)
# ============================================================================

def test_nspsa_fuzzy_combined(tasks: List, timeout: float = 10.0, random_seed: int = 3000) -> Dict:
    """
    Condition 3: NSPSA + Fuzzy meta-controller combined.

    NOTE: PLACEHOLDER until fuzzy controller integrated.
    """
    # TODO: Implement actual fuzzy + NSPSA integration
    # For now: Same as baseline (will update after integration)

    return test_nspsa_baseline(tasks, beam_width=5, search_depth=3,
                               timeout=timeout, random_seed=random_seed)


# ============================================================================
# STATISTICAL ANALYSIS
# ============================================================================

def analyze_condition_results(condition_results: List[Dict]) -> ConditionResult:
    """Aggregate results from 5 runs of one condition."""

    all_accuracies = []
    all_states = []
    all_times = []
    all_timeouts = []

    easy_acc = []
    med_acc = []
    hard_acc = []

    for run_data in condition_results:
        results = run_data['results']

        # Accuracy
        solved = sum(1 for r in results if r['solved'])
        total = len(results)
        accuracy = solved / total if total > 0 else 0
        all_accuracies.append(accuracy)

        # States explored (only solved tasks)
        solved_results = [r for r in results if r['solved']]
        if solved_results:
            avg_states = np.mean([r['states_explored'] for r in solved_results])
            all_states.append(avg_states)

        # Time
        avg_time = np.mean([r['search_time'] for r in results])
        all_times.append(avg_time)

        # Timeout rate
        timeouts = sum(1 for r in results if 'timeout' in r['failure_mode'])
        timeout_rate = timeouts / total if total > 0 else 0
        all_timeouts.append(timeout_rate)

        # By difficulty
        easy = [r for r in results if r['difficulty'] == 'easy']
        medium = [r for r in results if r['difficulty'] == 'medium']
        hard = [r for r in results if r['difficulty'] == 'hard']

        easy_acc.append(sum(1 for r in easy if r['solved']) / len(easy) if easy else 0)
        med_acc.append(sum(1 for r in medium if r['solved']) / len(medium) if medium else 0)
        hard_acc.append(sum(1 for r in hard if r['solved']) / len(hard) if hard else 0)

    return ConditionResult(
        condition_name=condition_results[0]['condition'],
        run_results=condition_results,
        accuracy_mean=np.mean(all_accuracies),
        accuracy_std=np.std(all_accuracies),
        states_mean=np.mean(all_states) if all_states else 0,
        states_std=np.std(all_states) if all_states else 0,
        time_mean=np.mean(all_times),
        time_std=np.std(all_times),
        timeout_rate=np.mean(all_timeouts),
        easy_accuracy=np.mean(easy_acc),
        medium_accuracy=np.mean(med_acc),
        hard_accuracy=np.mean(hard_acc)
    )


def print_comparison(cond1: ConditionResult, cond2: ConditionResult, cond3: ConditionResult):
    """Print comparison table."""
    print("\n" + "="*80)
    print("ABLATION STUDY RESULTS (5x runs per condition)")
    print("="*80)

    print(f"\n{'Metric':<30} {'Baseline':<20} {'Fuzzy Alone':<20} {'Combined':<20}")
    print("-"*90)

    print(f"{'Accuracy (mean ± std)':<30} "
          f"{cond1.accuracy_mean:.3f} ± {cond1.accuracy_std:.3f}    "
          f"{cond2.accuracy_mean:.3f} ± {cond2.accuracy_std:.3f}    "
          f"{cond3.accuracy_mean:.3f} ± {cond3.accuracy_std:.3f}")

    print(f"{'States explored (mean)':<30} "
          f"{cond1.states_mean:6.1f}              "
          f"{cond2.states_mean:6.1f}              "
          f"{cond3.states_mean:6.1f}")

    print(f"{'Time (mean, seconds)':<30} "
          f"{cond1.time_mean:6.3f}              "
          f"{cond2.time_mean:6.3f}              "
          f"{cond3.time_mean:6.3f}")

    print(f"{'Timeout rate':<30} "
          f"{cond1.timeout_rate:6.1%}              "
          f"{cond2.timeout_rate:6.1%}              "
          f"{cond3.timeout_rate:6.1%}")

    print("\nBy Difficulty:")
    print(f"{'  Easy tasks':<30} "
          f"{cond1.easy_accuracy:6.1%}              "
          f"{cond2.easy_accuracy:6.1%}              "
          f"{cond3.easy_accuracy:6.1%}")

    print(f"{'  Medium tasks':<30} "
          f"{cond1.medium_accuracy:6.1%}              "
          f"{cond2.medium_accuracy:6.1%}              "
          f"{cond3.medium_accuracy:6.1%}")

    print(f"{'  Hard tasks':<30} "
          f"{cond1.hard_accuracy:6.1%}              "
          f"{cond2.hard_accuracy:6.1%}              "
          f"{cond3.hard_accuracy:6.1%}")

    # Statistical significance
    from scipy import stats

    baseline_acc = [r['results'] for r in cond1.run_results]
    combined_acc = [r['results'] for r in cond3.run_results]

    baseline_scores = [np.mean([t['solved'] for t in run['results']]) for run in cond1.run_results]
    combined_scores = [np.mean([t['solved'] for t in run['results']]) for run in cond3.run_results]

    t_stat, p_value = stats.ttest_rel(combined_scores, baseline_scores)

    print("\n" + "="*80)
    print("STATISTICAL SIGNIFICANCE")
    print("="*80)
    print(f"Paired t-test (Combined vs Baseline):")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value: {p_value:.4f}")
    print(f"  Significant (p < 0.05): {'YES ✅' if p_value < 0.05 else 'NO ❌'}")

    improvement = cond3.accuracy_mean - cond1.accuracy_mean
    print(f"\nAbsolute improvement: {improvement:+.1%}")
    print(f"Relative improvement: {improvement/cond1.accuracy_mean:+.1%}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == '__main__':
    print("="*80)
    print("ABLATION STUDY: FUZZY META-CONTROLLER + NSPSA")
    print("="*80)
    print("\nRigorous x5 testing methodology")
    print("NOTE: Currently running with PLACEHOLDERS for fuzzy controller")
    print()

    # Generate test dataset
    print("Generating test dataset...")
    tasks = generate_test_dataset(seed=42)

    # Run Condition 1: NSPSA Baseline (5x)
    print("\n" + "="*80)
    print("CONDITION 1: NSPSA BASELINE (5 runs)")
    print("="*80)

    cond1_results = []
    for run in range(5):
        print(f"\n  Run {run+1}/5...")
        result = test_nspsa_baseline(tasks, random_seed=1000+run)
        cond1_results.append(result)

        # Quick stats
        solved = sum(1 for r in result['results'] if r['solved'])
        print(f"    Accuracy: {solved}/{len(tasks)} = {100*solved/len(tasks):.1f}%")

    # Aggregate Condition 1
    cond1_summary = analyze_condition_results(cond1_results)

    # Run Condition 2: Fuzzy Alone (5x)
    print("\n" + "="*80)
    print("CONDITION 2: FUZZY ALONE (5 runs) - PLACEHOLDER")
    print("="*80)

    cond2_results = []
    for run in range(5):
        print(f"\n  Run {run+1}/5...")
        result = test_fuzzy_alone(tasks, random_seed=2000+run)
        cond2_results.append(result)

        solved = sum(1 for r in result['results'] if r['solved'])
        print(f"    Accuracy: {solved}/{len(tasks)} = {100*solved/len(tasks):.1f}%")

    cond2_summary = analyze_condition_results(cond2_results)

    # Run Condition 3: Combined (5x)
    print("\n" + "="*80)
    print("CONDITION 3: COMBINED NSPSA + FUZZY (5 runs) - PLACEHOLDER")
    print("="*80)

    cond3_results = []
    for run in range(5):
        print(f"\n  Run {run+1}/5...")
        result = test_nspsa_fuzzy_combined(tasks, random_seed=3000+run)
        cond3_results.append(result)

        solved = sum(1 for r in result['results'] if r['solved'])
        print(f"    Accuracy: {solved}/{len(tasks)} = {100*solved/len(tasks):.1f}%")

    cond3_summary = analyze_condition_results(cond3_results)

    # Print comparison
    print_comparison(cond1_summary, cond2_summary, cond3_summary)

    # Save results
    print("\n" + "="*80)
    print("Saving results...")

    with open('ablation_results_condition1.json', 'w') as f:
        json.dump(cond1_results, f, indent=2)

    with open('ablation_results_condition2.json', 'w') as f:
        json.dump(cond2_results, f, indent=2)

    with open('ablation_results_condition3.json', 'w') as f:
        json.dump(cond3_results, f, indent=2)

    with open('ablation_summary.json', 'w') as f:
        json.dump({
            'condition1': asdict(cond1_summary),
            'condition2': asdict(cond2_summary),
            'condition3': asdict(cond3_summary)
        }, f, indent=2)

    print("✅ Results saved to ablation_results_*.json")
    print("\n" + "="*80)
    print("ABLATION STUDY COMPLETE")
    print("="*80)
    print("\nNEXT STEPS:")
    print("1. Review results and statistical significance")
    print("2. Document 3x3 lessons (pros, cons, actions)")
    print("3. ONLY integrate fuzzy if p < 0.05 and improvement ≥ 5%")
    print("4. Refine fuzzy controller based on failure analysis")
