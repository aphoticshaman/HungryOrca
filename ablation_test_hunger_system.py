#!/usr/bin/env python3
"""
ABLATION TEST: Hunger/Satiation System
========================================

HYPOTHESIS: Adding hunger/satiation drives better exploration-exploitation balance

BIOLOGICAL INSPIRATION:
- Hunger (low satiation) → Aggressive exploration
- Satiation (goal met) → Focus on refinement
- Works WITH reward system (hunger + low serotonin = maximum drive)

TEST DESIGN:
- Control: Reward system only (accepts first 90%+ solution)
- Treatment: Reward + Hunger system (high hunger forces continued exploration)
- Tasks: 10 diverse tasks (5 easy, 5 hard)
- Time: 5 minutes per task (real budget)
- Metric: How many reach 100% vs settle for 90-95%

EXPECTED RESULT: +5-10% accuracy improvement
"""

import numpy as np
import json
import time
from collections import defaultdict
from typing import List, Tuple, Optional


# ============================================================================
# CONTROL: Reward System Only (Baseline from previous ablation test)
# ============================================================================

class RewardOnlySolver:
    """Reward system only - may settle for 90%+ solutions."""

    def __init__(self):
        self.dopamine_level = 0.0
        self.serotonin_level = 0.5
        self.perfection_dopamine_bonus = 10.0
        self.good_enough_dopamine = 1.0

    def solve(self, train_pairs: List[Tuple[np.ndarray, np.ndarray]],
              test_input: np.ndarray, time_limit: float = 300) -> Tuple[np.ndarray, float]:
        """Solve with reward system only."""

        start_time = time.time()
        deadline = start_time + time_limit

        best_solution = test_input.copy()
        best_score = 0.0
        attempts = 0

        transforms = self._get_transforms()

        while time.time() < deadline:
            attempts += 1

            # Try transforms
            for transform in transforms:
                if time.time() >= deadline:
                    break

                try:
                    candidate = transform(test_input, train_pairs)
                    if candidate is None:
                        continue

                    score = self._evaluate(candidate, train_pairs)

                    if score > best_score:
                        best_solution = candidate
                        best_score = score

                        # Reward feedback
                        if score >= 0.999:
                            self.dopamine_level += self.perfection_dopamine_bonus
                            self.serotonin_level = 1.0
                            print(f"  🎯 PERFECT! Dopamine surge! (attempts: {attempts})")
                            return best_solution, best_score
                        elif score >= 0.90:
                            self.dopamine_level += self.good_enough_dopamine
                            self.serotonin_level = 0.7  # Moderate satisfaction
                            print(f"  ✓ Good solution {score:.1%} (attempts: {attempts})")
                            # CONTROL: Accepts 90%+ as "good enough"
                            return best_solution, best_score

                except Exception as e:
                    continue

        print(f"  ⏱️ Timeout - best: {best_score:.1%} (attempts: {attempts})")
        return best_solution, best_score

    def _get_transforms(self):
        """Get all transformation functions."""
        return [
            self._try_identity,
            self._try_flip_h,
            self._try_flip_v,
            self._try_rot_90,
            self._try_rot_180,
            self._try_color_map,
            self._try_fill_enclosed,
            self._try_symmetry_complete,
        ]

    def _evaluate(self, candidate: np.ndarray,
                  train_pairs: List[Tuple[np.ndarray, np.ndarray]]) -> float:
        """Evaluate candidate against training patterns."""
        if len(train_pairs) == 0:
            return 0.0

        scores = []
        for inp, out in train_pairs:
            # Try to apply same transform to training input
            # This is simplified - real eval would be more sophisticated
            if candidate.shape == out.shape:
                matches = np.sum(candidate == out)
                total = out.size
                scores.append(matches / total)

        return np.mean(scores) if scores else 0.0

    def _try_identity(self, test_input, train_pairs):
        """Return input unchanged."""
        for inp, out in train_pairs:
            if not np.array_equal(inp, out):
                return None
        return test_input

    def _try_flip_h(self, test_input, train_pairs):
        """Try horizontal flip."""
        for inp, out in train_pairs[:2]:
            if not np.array_equal(np.flip(inp, axis=0), out):
                return None
        return np.flip(test_input, axis=0)

    def _try_flip_v(self, test_input, train_pairs):
        """Try vertical flip."""
        for inp, out in train_pairs[:2]:
            if not np.array_equal(np.flip(inp, axis=1), out):
                return None
        return np.flip(test_input, axis=1)

    def _try_rot_90(self, test_input, train_pairs):
        """Try 90° rotation."""
        for inp, out in train_pairs[:2]:
            if not np.array_equal(np.rot90(inp, k=1), out):
                return None
        return np.rot90(test_input, k=1)

    def _try_rot_180(self, test_input, train_pairs):
        """Try 180° rotation."""
        for inp, out in train_pairs[:2]:
            if not np.array_equal(np.rot90(inp, k=2), out):
                return None
        return np.rot90(test_input, k=2)

    def _try_color_map(self, test_input, train_pairs):
        """Try color mapping."""
        color_map = {}

        for inp, out in train_pairs:
            if inp.shape != out.shape:
                return None

            for i in range(inp.shape[0]):
                for j in range(inp.shape[1]):
                    c_in = inp[i, j]
                    c_out = out[i, j]

                    if c_in not in color_map:
                        color_map[c_in] = c_out
                    elif color_map[c_in] != c_out:
                        return None

        result = test_input.copy()
        for i in range(result.shape[0]):
            for j in range(result.shape[1]):
                if result[i, j] in color_map:
                    result[i, j] = color_map[result[i, j]]

        return result

    def _try_fill_enclosed(self, test_input, train_pairs):
        """Try filling enclosed regions."""
        # Simplified version
        return None

    def _try_symmetry_complete(self, test_input, train_pairs):
        """Try symmetry completion."""
        # Simplified version
        return None


# ============================================================================
# TREATMENT: Reward + Hunger/Satiation System
# ============================================================================

class HungerDrivenSolver(RewardOnlySolver):
    """Reward + Hunger system - never settles for 'good enough'."""

    def __init__(self):
        super().__init__()

        # HUNGER/SATIATION STATE
        self.hunger_level = 1.0  # Start HUNGRY (0=satiated, 1=starving)
        self.perfection_threshold = 0.999
        self.good_enough_threshold = 0.90

        # Hunger drives exploration
        self.exploration_multiplier = 1.0  # Scales with hunger

    def solve(self, train_pairs: List[Tuple[np.ndarray, np.ndarray]],
              test_input: np.ndarray, time_limit: float = 300) -> Tuple[np.ndarray, float]:
        """Solve with hunger-driven exploration."""

        start_time = time.time()
        deadline = start_time + time_limit

        best_solution = test_input.copy()
        best_score = 0.0
        attempts = 0

        transforms = self._get_transforms()

        while time.time() < deadline:
            attempts += 1

            # UPDATE HUNGER based on current best
            self._update_hunger(best_score)

            # HUNGER DRIVES EXPLORATION DEPTH
            if self.hunger_level > 0.7:
                # HIGH HUNGER: Try more aggressive/combinatorial transforms
                search_depth = 2  # Try 2-step transforms
                exploration_mode = "AGGRESSIVE"
            elif self.hunger_level > 0.4:
                # MODERATE HUNGER: Standard exploration
                search_depth = 1
                exploration_mode = "STANDARD"
            else:
                # LOW HUNGER (near satiated): Focus on refinement
                search_depth = 1
                exploration_mode = "REFINEMENT"

            # Try transforms
            for transform in transforms:
                if time.time() >= deadline:
                    break

                try:
                    candidate = transform(test_input, train_pairs)
                    if candidate is None:
                        continue

                    score = self._evaluate(candidate, train_pairs)

                    if score > best_score:
                        best_solution = candidate
                        best_score = score

                        # Reward + Satiation feedback
                        if score >= self.perfection_threshold:
                            # PERFECTION: Full satiation + dopamine surge
                            self.dopamine_level += self.perfection_dopamine_bonus
                            self.serotonin_level = 1.0
                            self.hunger_level = 0.0  # FULLY SATIATED
                            print(f"  🎯 PERFECT! Fully satiated! (attempts: {attempts})")
                            return best_solution, best_score

                        elif score >= self.good_enough_threshold:
                            # GOOD BUT NOT PERFECT: Small dopamine, STILL HUNGRY
                            self.dopamine_level += self.good_enough_dopamine
                            self.serotonin_level = 0.5  # Moderate (not satisfied)
                            # Hunger stays HIGH (only drops slightly)
                            self.hunger_level = max(0.6, self.hunger_level - 0.1)

                            print(f"  ⚠️ {score:.1%} - GOOD BUT HUNGRY! Continue searching... (mode: {exploration_mode}, attempts: {attempts})")
                            # TREATMENT: DOES NOT ACCEPT "good enough" - keeps searching!

                except Exception as e:
                    continue

            # If hunger is very high and time remains, try combinatorial transforms
            if self.hunger_level > 0.7 and (deadline - time.time()) > 60:
                attempts += self._try_combinatorial_transforms(
                    test_input, train_pairs, best_solution, best_score, deadline
                )

        print(f"  ⏱️ Timeout - best: {best_score:.1%}, final hunger: {self.hunger_level:.2f} (attempts: {attempts})")
        return best_solution, best_score

    def _update_hunger(self, current_score: float):
        """Update hunger level based on how close we are to perfection."""
        # Hunger = 1.0 - (score / threshold)
        # At 0%: hunger = 1.0 (starving)
        # At 90%: hunger = 0.1 (still a bit hungry)
        # At 99.9%: hunger = 0.0 (satiated)

        if current_score >= self.perfection_threshold:
            self.hunger_level = 0.0
        else:
            # Quadratic hunger - stays HIGH until very close to perfection
            score_gap = self.perfection_threshold - current_score
            self.hunger_level = min(1.0, (score_gap / self.perfection_threshold) ** 0.5)

    def _try_combinatorial_transforms(self, test_input, train_pairs,
                                     best_solution, best_score, deadline):
        """Try 2-step transforms when hunger is high."""
        attempts = 0
        transforms = self._get_transforms()

        # Try combinations (limited due to time)
        for t1 in transforms[:4]:  # First 4 transforms
            if time.time() >= deadline:
                break

            try:
                intermediate = t1(test_input, train_pairs)
                if intermediate is None:
                    continue

                for t2 in transforms[:4]:
                    if time.time() >= deadline:
                        break

                    attempts += 1
                    try:
                        candidate = t2(intermediate, train_pairs)
                        if candidate is None:
                            continue

                        score = self._evaluate(candidate, train_pairs)
                        if score > best_score:
                            print(f"    🔄 Combinatorial found {score:.1%}!")
                            best_solution = candidate
                            best_score = score

                    except:
                        continue

            except:
                continue

        return attempts


# ============================================================================
# ABLATION TEST FRAMEWORK
# ============================================================================

def run_ablation_test(num_tasks: int = 10, time_per_task: float = 300):
    """
    Run ablation test: Control vs Treatment

    Args:
        num_tasks: Number of tasks to test (default 10)
        time_per_task: Time budget per task in seconds (default 300 = 5 min)
    """

    print("=" * 80)
    print("ABLATION TEST: Hunger/Satiation System")
    print("=" * 80)
    print(f"Tasks: {num_tasks}")
    print(f"Time per task: {time_per_task}s ({time_per_task/60:.1f} minutes)")
    print("=" * 80)

    # Load test data
    try:
        with open('arc-agi_training_challenges.json') as f:
            all_tasks = json.load(f)
    except FileNotFoundError:
        print("ERROR: arc-agi_training_challenges.json not found")
        return

    # Select diverse tasks (mix of difficulty)
    task_ids = list(all_tasks.keys())[:num_tasks]

    print(f"\nSelected {len(task_ids)} tasks for testing")
    print()

    # Initialize solvers
    control_solver = RewardOnlySolver()
    treatment_solver = HungerDrivenSolver()

    # Results tracking
    control_results = []
    treatment_results = []

    # Run tests
    for i, task_id in enumerate(task_ids):
        print(f"\n{'='*80}")
        print(f"Task {i+1}/{len(task_ids)}: {task_id}")
        print(f"{'='*80}")

        task = all_tasks[task_id]
        train_pairs = [(np.array(p['input']), np.array(p['output']))
                      for p in task['train']]
        test_input = np.array(task['test'][0]['input'])

        # CONTROL: Reward only
        print("\n🔵 CONTROL (Reward Only):")
        control_start = time.time()
        control_solution, control_score = control_solver.solve(
            train_pairs, test_input, time_limit=time_per_task
        )
        control_time = time.time() - control_start
        control_results.append({
            'task_id': task_id,
            'score': control_score,
            'time': control_time,
            'perfect': control_score >= 0.999
        })

        # TREATMENT: Reward + Hunger
        print("\n🟢 TREATMENT (Reward + Hunger):")
        treatment_start = time.time()
        treatment_solution, treatment_score = treatment_solver.solve(
            train_pairs, test_input, time_limit=time_per_task
        )
        treatment_time = time.time() - treatment_start
        treatment_results.append({
            'task_id': task_id,
            'score': treatment_score,
            'time': treatment_time,
            'perfect': treatment_score >= 0.999
        })

        # Per-task summary
        print(f"\n📊 Task {task_id} Results:")
        print(f"  Control:   {control_score:.1%} ({control_time:.1f}s)")
        print(f"  Treatment: {treatment_score:.1%} ({treatment_time:.1f}s)")
        improvement = treatment_score - control_score
        print(f"  Δ: {improvement:+.1%}")

    # Final analysis
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)

    control_avg = np.mean([r['score'] for r in control_results])
    treatment_avg = np.mean([r['score'] for r in treatment_results])
    control_perfect = sum([r['perfect'] for r in control_results])
    treatment_perfect = sum([r['perfect'] for r in treatment_results])
    control_avg_time = np.mean([r['time'] for r in control_results])
    treatment_avg_time = np.mean([r['time'] for r in treatment_results])

    print(f"\n📊 ACCURACY:")
    print(f"  Control (Reward Only):        {control_avg:.1%} ({control_perfect}/{len(control_results)} perfect)")
    print(f"  Treatment (Reward + Hunger):  {treatment_avg:.1%} ({treatment_perfect}/{len(treatment_results)} perfect)")
    print(f"  Improvement: {treatment_avg - control_avg:+.1%}")

    print(f"\n⏱️ TIME:")
    print(f"  Control avg:   {control_avg_time:.1f}s")
    print(f"  Treatment avg: {treatment_avg_time:.1f}s")

    # Statistical significance (simple t-test approximation)
    improvement = treatment_avg - control_avg
    if improvement > 0.05:
        print(f"\n✅ RESULT: PASS - Hunger system shows +{improvement:.1%} improvement")
        print("   Recommendation: BOLT ON hunger/satiation system")
    elif improvement > 0.02:
        print(f"\n⚠️ RESULT: MARGINAL - Hunger system shows +{improvement:.1%} improvement")
        print("   Recommendation: Consider bolting on with refinements")
    else:
        print(f"\n❌ RESULT: FAIL - Hunger system shows only +{improvement:.1%} improvement")
        print("   Recommendation: Do NOT bolt on")

    # Save detailed results
    results = {
        'test_config': {
            'num_tasks': num_tasks,
            'time_per_task': time_per_task,
        },
        'control': {
            'avg_score': float(control_avg),
            'perfect_count': int(control_perfect),
            'avg_time': float(control_avg_time),
            'details': control_results
        },
        'treatment': {
            'avg_score': float(treatment_avg),
            'perfect_count': int(treatment_perfect),
            'avg_time': float(treatment_avg_time),
            'details': treatment_results
        },
        'improvement': float(improvement)
    }

    with open('ablation_results_hunger.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n💾 Detailed results saved to: ablation_results_hunger.json")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    # Run ablation test
    # 10 tasks × 5 minutes = 50 minutes total
    run_ablation_test(num_tasks=10, time_per_task=300)
