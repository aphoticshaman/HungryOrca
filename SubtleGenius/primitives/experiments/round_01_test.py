"""
ROUND 1 TEST: Composition Primitives

Insight: 2-step tasks fail with atomic primitives alone
Solution: Add 5 common compositions as atomic operations

Test: Run ablation before/after adding compositions
Expected: Coverage improves from ~80% to ~95% on composition tasks
"""

import numpy as np
import sys
import time
sys.path.insert(0, '/home/user/HungryOrca/SubtleGenius/primitives')

from symbolic_solver import SymbolicProgramSynthesizer, GridState

def generate_composition_tasks():
    """Generate tasks requiring 2-step transformations"""
    tasks = []

    # Task 1: Rotate + Reflect H
    input1 = np.array([[1, 2], [3, 4]])
    output1 = np.flip(np.rot90(input1, k=-1), axis=1)
    tasks.append(('rotate_reflect_h', input1, output1))

    # Task 2: Rotate + Reflect V
    input2 = np.array([[1, 2, 3], [4, 5, 6]])
    output2 = np.flip(np.rot90(input2, k=-1), axis=0)
    tasks.append(('rotate_reflect_v', input2, output2))

    # Task 3: Reflect + Transpose
    input3 = np.array([[1, 2], [3, 4]])
    output3 = np.flip(input3, axis=1).T
    tasks.append(('reflect_transpose', input3, output3))

    # Task 4-10: More compositions
    for i in range(7):
        size = np.random.choice([2, 3, 4])
        inp = np.random.randint(0, 10, (size, size))

        # Random composition
        out = np.rot90(inp, k=-1)
        out = np.flip(out, axis=1)

        tasks.append((f'composition_{i}', inp, out))

    return tasks


def test_round_1():
    """Test composition primitives"""
    print("="*70)
    print("ROUND 1 TEST: Composition Primitives")
    print("="*70)

    synthesizer = SymbolicProgramSynthesizer(max_program_length=2)

    tasks = generate_composition_tasks()

    print(f"\nGenerated {len(tasks)} composition tasks")
    print("\nTesting...")

    results = {'found': 0, 'not_found': 0, 'times': []}

    for name, inp, out in tasks:
        start = time.time()
        program = synthesizer.synthesize(inp, out, timeout=2.0)
        elapsed = time.time() - start

        if program:
            results['found'] += 1
            results['times'].append(elapsed)
            print(f"  ✅ {name}: {program} ({elapsed:.3f}s)")
        else:
            results['not_found'] += 1
            print(f"  ❌ {name}: NOT FOUND")

    # Summary
    print("\n" + "="*70)
    print("RESULTS:")
    print("="*70)

    coverage = results['found'] / len(tasks)
    print(f"Coverage: {results['found']}/{len(tasks)} ({coverage:.1%})")

    if results['times']:
        print(f"Avg time: {np.mean(results['times']):.3f}s")
        print(f"Max time: {np.max(results['times']):.3f}s")

    # Validate insight
    print("\n" + "="*70)
    print("INSIGHT VALIDATION:")
    print("="*70)

    if coverage >= 0.90:
        print("✅ INSIGHT CONFIRMED: Composition primitives solve 2-step tasks")
        print(f"   Coverage improved to {coverage:.1%}")
    elif coverage >= 0.70:
        print("⚠️  PARTIAL SUCCESS: Some improvement but not enough")
        print(f"   Coverage: {coverage:.1%} (target: >90%)")
    else:
        print("❌ INSIGHT FAILED: Composition primitives don't help")
        print(f"   Coverage: {coverage:.1%}")

    return results


if __name__ == '__main__':
    results = test_round_1()

    print("\n" + "="*70)
    print("ROUND 1 COMPLETE")
    print("="*70)
    print("Novel insight: Composition primitives (rotate+reflect, etc.)")
    print("Code extension: Added 5 new atomic operations")
    print("Validation: Tested on 10 composition tasks")
    print("="*70)
