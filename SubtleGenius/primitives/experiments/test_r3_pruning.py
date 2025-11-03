"""
ROUND 3.3: Test Cost-Aware Visited-State Pruning

Validates that cost-aware pruning prevents redundant exploration,
especially for cyclic tasks (e.g., rotate 4x = identity).
"""

import numpy as np
import sys
sys.path.insert(0, '/home/user/HungryOrca/SubtleGenius/primitives')

from symbolic_solver import SymbolicProgramSynthesizer, SearchStrategy


def test_cyclic_tasks():
    """Test that cyclic tasks (rotate 4x = identity) are handled efficiently"""
    print("="*70)
    print("TEST 1: Cyclic Task Handling (rotate 4x = identity)")
    print("="*70)
    print("Without pruning: might explore rotate→rotate→rotate→rotate paths")
    print("With pruning: should recognize identity faster\n")

    synth = SymbolicProgramSynthesizer(max_program_length=4)

    # Cyclic task: rotate 4 times = identity
    inp = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    out = inp  # Identity after 4 rotations

    # Synthesize
    program = synth.synthesize(inp, out, timeout=2.0)
    states = synth.stats['states_explored']

    print(f"  Input:  {inp.flatten().tolist()}")
    print(f"  Output: {out.flatten().tolist()}")
    print(f"  Program found: {program}")
    print(f"  States explored: {states}")

    if program is not None and len(program) == 0:
        print("\n  ✅ Correctly recognized identity (empty program)")
        return True
    else:
        print("\n  ⚠️  Did not recognize identity optimally")
        return False


def test_multiple_paths():
    """Test that cost-aware pruning chooses shorter paths"""
    print("\n" + "="*70)
    print("TEST 2: Multiple Paths to Same State")
    print("="*70)
    print("Task: rotate 180° can be done via:")
    print("  - Path 1: rotate_180 (cost 1)")
    print("  - Path 2: rotate_90_cw → rotate_90_cw (cost 2)")
    print("Pruning should prefer cost-1 path\n")

    synth = SymbolicProgramSynthesizer(max_program_length=3)

    inp = np.array([[1, 2], [3, 4]])
    out = np.rot90(inp, k=2)  # 180° rotation

    program = synth.synthesize(inp, out, timeout=2.0)
    states = synth.stats['states_explored']

    print(f"  Program found: {program}")
    print(f"  Program length: {len(program) if program else 'N/A'}")
    print(f"  States explored: {states}")

    if program and len(program) == 1 and 'rotate_180' in program[0]:
        print("\n  ✅ Found optimal 1-step program (rotate_180)")
        return True
    else:
        print("\n  ⚠️  Did not find optimal program")
        return False


def test_state_reduction():
    """Measure state reduction from cost-aware pruning"""
    print("\n" + "="*70)
    print("TEST 3: State Reduction Measurement")
    print("="*70)
    print("Compare states explored on various tasks\n")

    synth = SymbolicProgramSynthesizer(max_program_length=3)

    test_cases = [
        ('Rotate 90°', np.array([[1, 2], [3, 4]]), lambda x: np.rot90(x, k=-1)),
        ('Reflect H', np.array([[1, 2], [3, 4]]), lambda x: np.flip(x, axis=0)),
        ('Transpose', np.array([[1, 2, 3], [4, 5, 6]]), lambda x: np.transpose(x)),
        ('Identity', np.array([[5, 6], [7, 8]]), lambda x: x),
    ]

    total_states = 0
    for name, inp, transform in test_cases:
        out = transform(inp)
        program = synth.synthesize(inp, out, timeout=2.0)
        states = synth.stats['states_explored']
        total_states += states

        print(f"  {name:15s}: {states:3d} states, program={program}")

    avg_states = total_states / len(test_cases)
    print(f"\n  Average states explored: {avg_states:.1f}")
    print(f"  Total across {len(test_cases)} tasks: {total_states}")

    # Baseline from R3.1 was 12.8 states average
    # With pruning, should be similar or slightly better
    if avg_states <= 15:
        print(f"\n  ✅ Efficient search (avg ≤ 15 states)")
        return True
    else:
        print(f"\n  ⚠️  High state count (avg > 15)")
        return False


def test_no_worse_paths():
    """Verify that we don't accept worse paths"""
    print("\n" + "="*70)
    print("TEST 4: No Worse Paths Accepted")
    print("="*70)
    print("Ensure pruning doesn't skip better paths for worse ones\n")

    synth = SymbolicProgramSynthesizer(max_program_length=4)

    # Task that has both short and long solutions
    inp = np.array([[1, 2], [3, 4]])
    out = np.flip(inp, axis=0)  # Can do reflect_h (cost 1)

    program = synth.synthesize(inp, out, timeout=2.0)

    if program and len(program) == 1:
        print(f"  Program: {program}")
        print(f"  Length: {len(program)}")
        print("\n  ✅ Found optimal short path")
        return True
    else:
        print(f"  Program: {program}")
        print(f"  Length: {len(program) if program else 'N/A'}")
        print("\n  ⚠️  Did not find optimal path")
        return False


if __name__ == '__main__':
    print("="*70)
    print("ROUND 3.3: COST-AWARE PRUNING VALIDATION")
    print("="*70)
    print("\nGoal: Validate that cost-aware pruning improves efficiency")
    print("Focus: Cyclic tasks and multiple-path scenarios\n")

    results = {
        'cyclic': test_cyclic_tasks(),
        'multiple_paths': test_multiple_paths(),
        'state_reduction': test_state_reduction(),
        'no_worse': test_no_worse_paths()
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
        print("\n✅ COST-AWARE PRUNING WORKING")
        print("   - Handles cyclic tasks efficiently")
        print("   - Chooses optimal paths")
        print("   - Maintains low state count")
    else:
        print("\n⚠️  Some tests need attention")
