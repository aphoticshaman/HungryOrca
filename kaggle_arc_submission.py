#!/usr/bin/env python3
"""
🏆 ARC Prize 2025 - Production Kaggle Submission Script

CRITICAL REQUIREMENTS:
- Complete ALL 100 test tasks (no exceptions!)
- Generate submission.json in correct format
- Handle errors gracefully with fallbacks
- Respect 9-hour Kaggle time limit

Author: Ryan Cardwell & Claude
Date: November 2025
"""

import json
import numpy as np
import time
from pathlib import Path
from typing import Dict, List
import sys

# Import quantum solver
from lucidorca_quantum import LucidOrcaQuantum


def ensure_all_tasks_solved(test_tasks: Dict, solutions: Dict, fallback_fn) -> Dict:
    """
    CRITICAL: Ensure every test task has a solution

    If any task is missing, generate fallback solution
    This prevents Kaggle rejection for incomplete submissions
    """
    missing = []
    for task_id in test_tasks:
        if task_id not in solutions:
            missing.append(task_id)
            # Generate fallback
            solutions[task_id] = fallback_fn(test_tasks[task_id])

    if missing:
        print(f"\n⚠️  Generated fallback for {len(missing)} missing tasks:")
        for task_id in missing[:5]:
            print(f"   - {task_id}")
        if len(missing) > 5:
            print(f"   ... and {len(missing)-5} more")

    return solutions


def validate_submission(submission: Dict, expected_count: int) -> bool:
    """
    Validate submission meets Kaggle requirements

    Returns True if valid, False otherwise
    """
    print(f"\n{'='*60}")
    print("VALIDATING SUBMISSION")
    print('='*60)

    errors = []

    # Check task count
    if len(submission) != expected_count:
        errors.append(f"Wrong task count: {len(submission)} (expected {expected_count})")

    # Check each task
    for task_id, outputs in submission.items():
        if not isinstance(outputs, list):
            errors.append(f"{task_id}: Not a list")
            continue

        for i, output in enumerate(outputs):
            if not isinstance(output, dict):
                errors.append(f"{task_id}[{i}]: Not a dict")
                continue

            if 'attempt_1' not in output or 'attempt_2' not in output:
                errors.append(f"{task_id}[{i}]: Missing attempts")
                continue

            # Check grid validity
            for attempt in ['attempt_1', 'attempt_2']:
                grid = output[attempt]
                if not grid or len(grid) == 0:
                    errors.append(f"{task_id}[{i}].{attempt}: Empty grid")
                    continue

                try:
                    arr = np.array(grid)
                    if arr.min() < 0 or arr.max() > 9:
                        errors.append(f"{task_id}[{i}].{attempt}: Values out of range")
                except:
                    errors.append(f"{task_id}[{i}].{attempt}: Invalid grid data")

    if errors:
        print(f"❌ VALIDATION FAILED: {len(errors)} errors")
        for err in errors[:10]:
            print(f"   ✗ {err}")
        if len(errors) > 10:
            print(f"   ... and {len(errors)-10} more")
        return False
    else:
        print(f"✅ VALIDATION PASSED")
        print(f"   - {len(submission)} tasks")
        print(f"   - All tasks have attempt_1 and attempt_2")
        print(f"   - All grids valid (0-9 range)")
        return True


def main():
    """Production Kaggle submission"""

    print("\n" + "="*70)
    print("🏆 ARC PRIZE 2025 - PRODUCTION SUBMISSION")
    print("="*70)
    print("Target: 85%+ accuracy = $700K Grand Prize")
    print("Strategy: Progressive Overload + Quantum Exploitation")
    print("="*70)

    start_time = time.time()

    # Detect environment
    if Path("/kaggle/input").exists():
        data_dir = Path("/kaggle/input/arc-prize-2025")
        output_dir = Path("/kaggle/working")
        print("\n📍 Environment: Kaggle")
    else:
        data_dir = Path(".")
        output_dir = Path(".")
        print("\n📍 Environment: Local")

    # Load test set
    test_path = data_dir / "arc-agi_test_challenges.json"

    if not test_path.exists():
        print(f"\n❌ ERROR: Test file not found: {test_path}")
        print("   Available files:")
        for f in data_dir.glob("*.json"):
            print(f"      - {f.name}")
        sys.exit(1)

    print(f"\n📂 Loading test set: {test_path}")
    with open(test_path, 'r') as f:
        test_tasks = json.load(f)

    print(f"   ✅ Loaded {len(test_tasks)} test tasks")

    # Initialize solver
    print(f"\n🌊⚛️ Initializing LucidOrca Quantum...")
    solver = LucidOrcaQuantum()

    # Solve with progressive overload
    # Budget: 6 hours (leave 3 hours buffer for Kaggle's 9hr limit)
    time_budget = 6 * 3600  # 6 hours in seconds

    print(f"\n🏋️ Solving {len(test_tasks)} tasks...")
    print(f"   Time budget: {time_budget/3600:.1f} hours")
    print(f"   Strategy: Progressive Overload (75% easy, 20% medium, 5% hard)")

    try:
        solutions = solver.solve_test_set(test_tasks, time_budget=time_budget)
    except Exception as e:
        print(f"\n❌ SOLVER CRASHED: {e}")
        print("   Using fallback for all tasks...")
        solutions = {}

    # CRITICAL: Ensure ALL tasks have solutions
    def generate_fallback(task):
        """Fallback: return input + rotated input"""
        test_input = np.array(task['test'][0]['input'])
        num_outputs = len(task['test'])

        task_solutions = []
        for _ in range(num_outputs):
            task_solutions.append({
                'attempt_1': test_input.tolist(),
                'attempt_2': np.rot90(test_input).tolist()
            })
        return task_solutions

    solutions = ensure_all_tasks_solved(test_tasks, solutions, generate_fallback)

    # Validate before saving
    if not validate_submission(solutions, len(test_tasks)):
        print("\n❌ SUBMISSION INVALID - ABORTING")
        sys.exit(1)

    # Save submission
    output_path = output_dir / "submission.json"
    print(f"\n💾 Saving submission: {output_path}")

    with open(output_path, 'w') as f:
        json.dump(solutions, f)

    file_size = output_path.stat().st_size
    print(f"   ✅ Saved {len(solutions)} tasks ({file_size/1024:.1f} KB)")

    # Final summary
    total_time = time.time() - start_time

    print("\n" + "="*70)
    print("🏆 SUBMISSION COMPLETE")
    print("="*70)
    print(f"   Tasks: {len(solutions)}/{len(test_tasks)}")
    print(f"   Time: {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"   File: {output_path}")
    print(f"   Size: {file_size/1024:.1f} KB")
    print("="*70)
    print("\n✅ Ready for Kaggle submission!")
    print("   Expected accuracy: 10-85% (depending on solver performance)")
    print("   Target: 85%+ for Grand Prize")
    print("="*70)


if __name__ == "__main__":
    main()
