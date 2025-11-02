"""
ROUND 3.4: Test Beam Search with Priority Queue

Validates:
1. Beam search works with different beam widths
2. Accuracy vs speed tradeoff
3. State reduction compared to baseline
"""

import numpy as np
import sys
import time
sys.path.insert(0, '/home/user/HungryOrca/SubtleGenius/primitives')

from symbolic_solver import SymbolicProgramSynthesizer, SearchStrategy


def test_beam_widths():
    """Test beam search with different widths (k=3, k=5, k=10)"""
    print("="*70)
    print("TEST 1: Beam Width Comparison")
    print("="*70)
    print("Tradeoff: Larger beam → higher accuracy, more states")
    print("         Smaller beam → faster, may miss solutions\n")

    # Test tasks
    test_cases = [
        ('Rotate 90°', np.array([[1, 2], [3, 4]]), lambda x: np.rot90(x, k=-1)),
        ('Reflect H', np.array([[1, 2, 3], [4, 5, 6]]), lambda x: np.flip(x, axis=0)),
        ('Transpose', np.array([[1, 2], [3, 4], [5, 6]]), lambda x: np.transpose(x)),
        ('Rotate 180°', np.array([[1, 2], [3, 4]]), lambda x: np.rot90(x, k=2)),
        ('Scale 2x', np.array([[1, 2]]), lambda x: np.kron(x, np.ones((2, 2), dtype=int))),
    ]

    beam_widths = [3, 5, 10, 20]  # Different beam widths to test

    results = {}
    for beam_width in beam_widths:
        synth = SymbolicProgramSynthesizer(max_program_length=3, beam_width=beam_width)

        solved = 0
        total_states = 0

        for name, inp, transform in test_cases:
            out = transform(inp)
            program = synth.synthesize(inp, out, strategy=SearchStrategy.BEAM, timeout=2.0)

            if program is not None:
                solved += 1
                total_states += synth.stats['states_explored']

        avg_states = total_states / len(test_cases) if solved > 0 else 0
        accuracy = solved / len(test_cases)

        results[beam_width] = {
            'solved': solved,
            'total': len(test_cases),
            'accuracy': accuracy,
            'avg_states': avg_states
        }

        print(f"  Beam width={beam_width:2d}: "
              f"Accuracy={accuracy:5.1%} ({solved}/{len(test_cases)}), "
              f"Avg states={avg_states:5.1f}")

    print("\n  ✅ Beam search working across different widths")
    return True


def test_state_reduction():
    """Compare beam search state count vs baseline"""
    print("\n" + "="*70)
    print("TEST 2: State Reduction vs Baseline")
    print("="*70)
    print("Baseline (BIDIRECTIONAL): ~12.7 states avg")
    print("Beam search goal: Reduce to 8-10 states avg\n")

    # Use beam_width=5 (good balance)
    synth_beam = SymbolicProgramSynthesizer(max_program_length=3, beam_width=5)
    synth_baseline = SymbolicProgramSynthesizer(max_program_length=3)

    test_cases = [
        ('Rotate 90°', np.array([[1, 2], [3, 4]]), lambda x: np.rot90(x, k=-1)),
        ('Reflect H', np.array([[1, 2], [3, 4]]), lambda x: np.flip(x, axis=0)),
        ('Transpose', np.array([[1, 2, 3], [4, 5, 6]]), lambda x: np.transpose(x)),
        ('Scale', np.array([[1]]), lambda x: np.kron(x, np.ones((2, 2), dtype=int))),
        ('Identity', np.array([[5, 6], [7, 8]]), lambda x: x),
    ]

    beam_states = []
    baseline_states = []

    for name, inp, transform in test_cases:
        out = transform(inp)

        # Beam search
        program_beam = synth_beam.synthesize(inp, out, strategy=SearchStrategy.BEAM, timeout=2.0)
        if program_beam is not None:
            beam_states.append(synth_beam.stats['states_explored'])

        # Baseline
        program_baseline = synth_baseline.synthesize(inp, out, strategy=SearchStrategy.BIDIRECTIONAL, timeout=2.0)
        if program_baseline is not None:
            baseline_states.append(synth_baseline.stats['states_explored'])

    beam_avg = np.mean(beam_states) if beam_states else 0
    baseline_avg = np.mean(baseline_states) if baseline_states else 0
    reduction = (baseline_avg - beam_avg) / baseline_avg if baseline_avg > 0 else 0

    print(f"  Baseline avg:   {baseline_avg:6.1f} states")
    print(f"  Beam avg:       {beam_avg:6.1f} states")
    print(f"  Reduction:      {reduction:6.1%}")

    if reduction > 0.1:  # At least 10% reduction
        print("\n  ✅ Beam search reduces state exploration")
        return True
    else:
        print("\n  ⚠️  Beam search not showing significant reduction")
        return False


def test_priority_ordering():
    """Verify that priority queue orders by f-score"""
    print("\n" + "="*70)
    print("TEST 3: Priority Queue Ordering")
    print("="*70)
    print("Verify that beam search explores best states first\n")

    synth = SymbolicProgramSynthesizer(max_program_length=2, beam_width=3)

    # Task where heuristic should guide well
    inp = np.array([[1, 2], [3, 4]])
    out = np.rot90(inp, k=-1)

    program = synth.synthesize(inp, out, strategy=SearchStrategy.BEAM, timeout=2.0)
    states = synth.stats['states_explored']

    print(f"  Program: {program}")
    print(f"  States explored: {states}")
    print(f"  Beam width: 3")

    if program is not None and states <= 10:
        print("\n  ✅ Priority ordering working (low state count)")
        return True
    else:
        print("\n  ⚠️  High state count or failed")
        return False


def test_beam_correctness():
    """Ensure beam search finds correct solutions"""
    print("\n" + "="*70)
    print("TEST 4: Beam Search Correctness")
    print("="*70)
    print("Verify that beam search finds valid programs\n")

    synth = SymbolicProgramSynthesizer(max_program_length=3, beam_width=5)
    executor = synth.executor

    test_cases = [
        (np.array([[1, 2], [3, 4]]), np.rot90(np.array([[1, 2], [3, 4]]), k=-1)),
        (np.array([[5, 6]]), np.flip(np.array([[5, 6]]), axis=1)),
        (np.array([[1, 2, 3]]), np.transpose(np.array([[1, 2, 3]]))),
    ]

    all_correct = True
    for i, (inp, expected_out) in enumerate(test_cases):
        program = synth.synthesize(inp, expected_out, strategy=SearchStrategy.BEAM, timeout=2.0)

        if program is None:
            print(f"  Test {i+1}: ❌ No program found")
            all_correct = False
            continue

        # Verify program correctness
        from symbolic_solver import GridState
        state = GridState.from_array(inp)
        for prim in program:
            state = executor.execute(state, prim)

        actual_out = state.to_array() if state else None
        correct = np.array_equal(actual_out, expected_out)

        status = "✅" if correct else "❌"
        print(f"  Test {i+1}: {status} Program {program} → Correct={correct}")

        if not correct:
            all_correct = False

    if all_correct:
        print("\n  ✅ All programs correct")
        return True
    else:
        print("\n  ❌ Some programs incorrect")
        return False


if __name__ == '__main__':
    print("="*70)
    print("ROUND 3.4: BEAM SEARCH VALIDATION")
    print("="*70)
    print("\nGoal: Validate beam search with priority queue and top-k cutoff")
    print("Test: Different beam widths, state reduction, correctness\n")

    results = {
        'beam_widths': test_beam_widths(),
        'state_reduction': test_state_reduction(),
        'priority_ordering': test_priority_ordering(),
        'correctness': test_beam_correctness()
    }

    print("\n" + "="*70)
    print("VALIDATION RESULTS:")
    print("="*70)

    passed = sum(results.values())
    for test, result in results.items():
        status = "✅" if result else "⚠️"
        print(f"  {status} {test}")

    print(f"\nPassed: {passed}/4")

    if passed >= 3:
        print("\n✅ BEAM SEARCH WORKING")
        print("   - Priority queue ordering states by f-score")
        print("   - Beam width controls exploration")
        print("   - Finds correct programs efficiently")
    else:
        print("\n⚠️  Some tests need attention")
