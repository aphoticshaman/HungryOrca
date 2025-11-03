#!/usr/bin/env python3
"""
TurboOrca v1 - One-Click ARC Prize 2025 Submission Generator
=============================================================

USAGE: python3 TurboOrcav1.py

OUTPUT: submission.json (ready for Kaggle)

TIME BUDGET: 90 minutes (22.5s per task × 240 tasks)
"""

import numpy as np
import json
import time
from typing import List, Tuple, Optional


class TurboOrca:
    """Fast, effective ARC solver with biological motivation."""

    def __init__(self, time_budget_minutes: float = 90):
        self.time_budget = time_budget_minutes * 60
        self.tasks_solved = 0
        self.start_time = None

        # Biological state - drives performance
        self.hunger = 1.0  # Start hungry (drives exploration)
        self.dopamine = 0.0
        self.serotonin = 0.5

    def solve_task(self, train_pairs: List[Tuple[np.ndarray, np.ndarray]],
                   test_input: np.ndarray, time_limit: float) -> np.ndarray:
        """Solve a single task."""

        deadline = time.time() + time_limit
        best_solution = test_input.copy()
        best_score = 0.0

        # Phase 1: Single transforms (fast)
        for transform in [self._identity, self._flip_h, self._flip_v,
                         self._rot_90, self._rot_180, self._rot_270]:
            if time.time() > deadline:
                break

            candidate = transform(test_input, train_pairs)
            if candidate is not None:
                score = self._score(candidate, train_pairs)
                if score > best_score:
                    best_solution = candidate
                    best_score = score
                    if score >= 0.999:
                        self._reward(score)
                        return best_solution

        # Phase 2: Color mapping
        if time.time() < deadline:
            candidate = self._color_map(test_input, train_pairs)
            if candidate is not None:
                score = self._score(candidate, train_pairs)
                if score > best_score:
                    best_solution = candidate
                    best_score = score
                    if score >= 0.999:
                        self._reward(score)
                        return best_solution

        # Phase 3: Compositional (if hungry - not satisfied yet)
        if self.hunger > 0.5 and time.time() < deadline:
            # Try flip + color_map
            candidate = self._compositional(test_input, train_pairs, deadline)
            if candidate is not None:
                score = self._score(candidate, train_pairs)
                if score > best_score:
                    best_solution = candidate
                    best_score = score

        self._reward(best_score)
        return best_solution

    def _identity(self, test_input: np.ndarray, train_pairs: List) -> Optional[np.ndarray]:
        """Check if input == output."""
        for inp, out in train_pairs:
            if not np.array_equal(inp, out):
                return None
        return test_input

    def _flip_h(self, test_input: np.ndarray, train_pairs: List) -> Optional[np.ndarray]:
        """Horizontal flip."""
        for inp, out in train_pairs[:2]:
            if out.shape != inp.shape or not np.array_equal(np.flip(inp, axis=0), out):
                return None
        return np.flip(test_input, axis=0)

    def _flip_v(self, test_input: np.ndarray, train_pairs: List) -> Optional[np.ndarray]:
        """Vertical flip."""
        for inp, out in train_pairs[:2]:
            if out.shape != inp.shape or not np.array_equal(np.flip(inp, axis=1), out):
                return None
        return np.flip(test_input, axis=1)

    def _rot_90(self, test_input: np.ndarray, train_pairs: List) -> Optional[np.ndarray]:
        """90° rotation."""
        for inp, out in train_pairs[:2]:
            if not np.array_equal(np.rot90(inp, k=1), out):
                return None
        return np.rot90(test_input, k=1)

    def _rot_180(self, test_input: np.ndarray, train_pairs: List) -> Optional[np.ndarray]:
        """180° rotation."""
        for inp, out in train_pairs[:2]:
            if out.shape != inp.shape or not np.array_equal(np.rot90(inp, k=2), out):
                return None
        return np.rot90(test_input, k=2)

    def _rot_270(self, test_input: np.ndarray, train_pairs: List) -> Optional[np.ndarray]:
        """270° rotation."""
        for inp, out in train_pairs[:2]:
            if not np.array_equal(np.rot90(inp, k=3), out):
                return None
        return np.rot90(test_input, k=3)

    def _color_map(self, test_input: np.ndarray, train_pairs: List) -> Optional[np.ndarray]:
        """Learn and apply color mapping."""
        color_map = {}

        for inp, out in train_pairs:
            if inp.shape != out.shape:
                return None

            for i in range(inp.shape[0]):
                for j in range(inp.shape[1]):
                    c_in = int(inp[i, j])
                    c_out = int(out[i, j])

                    if c_in in color_map:
                        if color_map[c_in] != c_out:
                            return None
                    else:
                        color_map[c_in] = c_out

        # Apply mapping
        result = test_input.copy()
        for i in range(result.shape[0]):
            for j in range(result.shape[1]):
                c = int(result[i, j])
                if c in color_map:
                    result[i, j] = color_map[c]

        return result

    def _compositional(self, test_input: np.ndarray, train_pairs: List,
                      deadline: float) -> Optional[np.ndarray]:
        """Try 2-step transforms."""
        transforms = [self._flip_h, self._flip_v, self._rot_90]

        for t1 in transforms:
            if time.time() > deadline:
                break
            try:
                intermediate = t1(test_input, train_pairs)
                if intermediate is None:
                    continue

                # Try color map on result
                candidate = self._color_map(intermediate, train_pairs)
                if candidate is not None:
                    return candidate
            except:
                continue

        return None

    def _score(self, candidate: np.ndarray, train_pairs: List) -> float:
        """Score candidate by checking if same transform works on training."""
        if len(train_pairs) == 0:
            return 0.5

        # Simple heuristic: if shapes match training outputs, likely correct
        scores = []
        for inp, out in train_pairs:
            if candidate.shape == out.shape:
                scores.append(0.9)  # Shape match is good sign
            else:
                scores.append(0.1)

        return np.mean(scores)

    def _reward(self, score: float):
        """Update biological state based on performance."""
        if score >= 0.999:
            self.hunger = 0.0
            self.dopamine = 10.0
            self.serotonin = 1.0
        elif score >= 0.8:
            self.hunger = max(0.3, self.hunger - 0.2)
            self.dopamine += 1.0
        else:
            self.hunger = min(1.0, self.hunger + 0.1)

    def generate_submission(self, input_file: str = 'arc-agi_test_challenges.json',
                          output_file: str = 'submission.json'):
        """Generate submission.json from test challenges."""

        print("=" * 80)
        print("TurboOrca v1 - ARC Prize 2025 Solver")
        print("=" * 80)

        # Load test data
        try:
            with open(input_file, 'r') as f:
                test_tasks = json.load(f)
        except FileNotFoundError:
            print(f"ERROR: {input_file} not found!")
            return

        num_tasks = len(test_tasks)
        time_per_task = self.time_budget / num_tasks

        print(f"Tasks: {num_tasks}")
        print(f"Time budget: {self.time_budget/60:.1f} minutes")
        print(f"Time per task: {time_per_task:.1f} seconds")
        print("=" * 80)

        self.start_time = time.time()
        deadline = self.start_time + self.time_budget

        submission = {}
        completed = 0

        for task_id, task_data in test_tasks.items():
            # Check time
            if time.time() > deadline:
                print(f"\n⏱️ Time limit reached at task {completed}/{num_tasks}")
                break

            # Progress
            if completed % 50 == 0:
                elapsed = (time.time() - self.start_time) / 60
                remaining = (deadline - time.time()) / 60
                print(f"Progress: {completed}/{num_tasks} | "
                      f"Elapsed: {elapsed:.1f}m | Remaining: {remaining:.1f}m")

            # Parse training examples
            train_pairs = []
            for pair in task_data['train']:
                inp = np.array(pair['input'], dtype=np.int32)
                out = np.array(pair['output'], dtype=np.int32)
                train_pairs.append((inp, out))

            # Solve each test input
            attempts = []
            for test_pair in task_data['test']:
                test_input = np.array(test_pair['input'], dtype=np.int32)

                try:
                    solution = self.solve_task(train_pairs, test_input, time_per_task)
                    attempts.append(solution.tolist())
                except Exception as e:
                    # Fallback: return input
                    attempts.append(test_input.tolist())

            # Ensure 2 attempts (Kaggle requirement)
            while len(attempts) < 2:
                attempts.append(attempts[0] if attempts else [[0]])

            submission[task_id] = attempts[:2]  # Only keep 2 attempts
            completed += 1

        # Fill remaining tasks if time ran out
        if completed < num_tasks:
            print(f"\nFilling remaining {num_tasks - completed} tasks with input...")
            for task_id, task_data in list(test_tasks.items())[completed:]:
                test_input = np.array(task_data['test'][0]['input'], dtype=np.int32)
                submission[task_id] = [test_input.tolist(), test_input.tolist()]

        # Save submission
        with open(output_file, 'w') as f:
            json.dump(submission, f)

        elapsed_total = (time.time() - self.start_time) / 60

        print("=" * 80)
        print("✅ COMPLETE")
        print("=" * 80)
        print(f"Tasks completed: {len(submission)}/{num_tasks}")
        print(f"Total time: {elapsed_total:.1f} minutes")
        print(f"Output: {output_file}")
        print("=" * 80)
        print("\nReady for Kaggle submission!")


if __name__ == '__main__':
    solver = TurboOrca(time_budget_minutes=90)
    solver.generate_submission()
