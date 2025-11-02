"""
ROUND 3.2: Test A* Heuristic Function

Validates:
1. Admissibility: h(s,g) ≤ true_cost(s,g) for all s,g
2. Consistency: h(s,g) ≤ cost(s,s') + h(s',g)
3. Informativeness: h provides useful guidance

"""

import numpy as np
import sys
sys.path.insert(0, '/home/user/HungryOrca/SubtleGenius/primitives')

from symbolic_solver import SymbolicProgramSynthesizer, GridState


def test_heuristic_identity():
    """Test that h(s,s) = 0"""
    print("="*70)
    print("TEST 1: Heuristic Identity Property")
    print("="*70)

    synth = SymbolicProgramSynthesizer()

    # Test several grids
    test_cases = [
        np.array([[1, 2], [3, 4]]),
        np.array([[5]]),
        np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
    ]

    all_pass = True
    for i, grid in enumerate(test_cases):
        state = GridState.from_array(grid)
        h = synth.compute_heuristic(state, state)

        status = "✅" if h == 0.0 else "❌"
        print(f"  {status} Test case {i+1}: h(s,s) = {h:.4f} (expect 0.0)")

        if h != 0.0:
            all_pass = False

    return all_pass


def test_heuristic_admissibility():
    """Test that h(s,g) ≤ true_cost(s,g)"""
    print("\n" + "="*70)
    print("TEST 2: Heuristic Admissibility")
    print("="*70)
    print("Property: h(s,g) must never overestimate true cost\n")

    synth = SymbolicProgramSynthesizer(max_program_length=3)

    # Test cases where we know the true cost
    test_cases = [
        {
            'name': '1-step: rotate 90',
            'input': np.array([[1, 2], [3, 4]]),
            'output': np.array([[3, 1], [4, 2]]),
            'true_cost': 1
        },
        {
            'name': '1-step: reflect horizontal',
            'input': np.array([[1, 2], [3, 4]]),
            'output': np.array([[3, 4], [1, 2]]),
            'true_cost': 1
        },
        {
            'name': '2-step: rotate + reflect',
            'input': np.array([[1, 2], [3, 4]]),
            'output': np.flip(np.rot90(np.array([[1, 2], [3, 4]]), k=-1), axis=1),
            'true_cost': 1  # We have composition primitive!
        },
        {
            'name': '1-step: scale up 2x',
            'input': np.array([[1, 2]]),
            'output': np.array([[1, 1, 2, 2], [1, 1, 2, 2]]),
            'true_cost': 1
        }
    ]

    all_pass = True
    for case in test_cases:
        inp_state = GridState.from_array(case['input'])
        out_state = GridState.from_array(case['output'])

        h = synth.compute_heuristic(inp_state, out_state)
        true_cost = case['true_cost']

        admissible = h <= true_cost
        status = "✅" if admissible else "❌"

        print(f"  {status} {case['name']:30s}")
        print(f"      h={h:.4f}, true_cost={true_cost}, admissible={admissible}")

        if not admissible:
            all_pass = False
            print(f"      ⚠️  VIOLATION: Heuristic overestimates!")

    return all_pass


def test_heuristic_consistency():
    """Test that h(s,g) ≤ cost(s,s') + h(s',g) (triangle inequality)"""
    print("\n" + "="*70)
    print("TEST 3: Heuristic Consistency (Triangle Inequality)")
    print("="*70)
    print("Property: h(s,g) ≤ cost(s,s') + h(s',g) for any intermediate state s'\n")

    synth = SymbolicProgramSynthesizer()

    # Test: s → s' → g
    s = np.array([[1, 2], [3, 4]])
    s_prime = np.rot90(s, k=-1)  # Intermediate state
    g = np.flip(s_prime, axis=1)  # Goal

    s_state = GridState.from_array(s)
    s_prime_state = GridState.from_array(s_prime)
    g_state = GridState.from_array(g)

    h_s_g = synth.compute_heuristic(s_state, g_state)
    h_s_prime_g = synth.compute_heuristic(s_prime_state, g_state)
    cost_s_s_prime = 1  # One primitive: rotate

    consistent = h_s_g <= cost_s_s_prime + h_s_prime_g

    status = "✅" if consistent else "❌"
    print(f"  {status} Consistency check:")
    print(f"      h(s, g) = {h_s_g:.4f}")
    print(f"      cost(s, s') + h(s', g) = {cost_s_s_prime} + {h_s_prime_g:.4f} = {cost_s_s_prime + h_s_prime_g:.4f}")
    print(f"      Consistent: {consistent}")

    return consistent


def test_heuristic_informativeness():
    """Test that heuristic provides useful information (not always 0 or 1)"""
    print("\n" + "="*70)
    print("TEST 4: Heuristic Informativeness")
    print("="*70)
    print("Property: h should vary meaningfully based on state-goal distance\n")

    synth = SymbolicProgramSynthesizer()

    inp = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    test_cases = [
        ('Identity', inp),
        ('Rotate 90°', np.rot90(inp, k=-1)),
        ('Reflect H', np.flip(inp, axis=0)),
        ('Completely different', np.random.randint(0, 10, (3, 3))),
        ('Different size', np.kron(inp, np.ones((2, 2), dtype=int)))
    ]

    heuristics = []
    for name, out in test_cases:
        inp_state = GridState.from_array(inp)
        out_state = GridState.from_array(out)

        h = synth.compute_heuristic(inp_state, out_state)
        heuristics.append(h)

        print(f"  {name:25s}: h = {h:.4f}")

    # Check that heuristics vary
    unique_values = len(set(heuristics))
    informative = unique_values >= 3  # At least 3 different values

    print(f"\n  {'Unique heuristic values:'} {unique_values}")

    if informative:
        print("  ✅ Heuristic is informative (varies based on distance)")
        return True
    else:
        print("  ❌ Heuristic is not informative (too similar values)")
        return False


def test_heuristic_bounds():
    """Test that heuristic is bounded appropriately"""
    print("\n" + "="*70)
    print("TEST 5: Heuristic Bounds")
    print("="*70)
    print("Property: 0 ≤ h(s,g) ≤ max_program_length\n")

    synth = SymbolicProgramSynthesizer(max_program_length=3)

    # Generate 20 random test cases
    all_pass = True
    for i in range(20):
        size1 = np.random.choice([2, 3, 4])
        size2 = np.random.choice([2, 3, 4])

        inp = np.random.randint(0, 10, (size1, size1))
        out = np.random.randint(0, 10, (size2, size2))

        inp_state = GridState.from_array(inp)
        out_state = GridState.from_array(out)

        h = synth.compute_heuristic(inp_state, out_state)

        bounded = 0.0 <= h <= synth.max_program_length

        if not bounded:
            print(f"  ❌ Test {i+1}: h = {h:.4f} (out of bounds!)")
            all_pass = False

    if all_pass:
        print("  ✅ All 20 random tests: 0 ≤ h ≤ max_program_length")

    return all_pass


if __name__ == '__main__':
    print("="*70)
    print("ROUND 3.2: A* HEURISTIC VALIDATION")
    print("="*70)
    print("\nGoal: Validate heuristic properties before integrating into search\n")

    results = {
        'identity': test_heuristic_identity(),
        'admissibility': test_heuristic_admissibility(),
        'consistency': test_heuristic_consistency(),
        'informativeness': test_heuristic_informativeness(),
        'bounds': test_heuristic_bounds()
    }

    print("\n" + "="*70)
    print("VALIDATION RESULTS:")
    print("="*70)

    passed = sum(results.values())
    for test, result in results.items():
        status = "✅" if result else "❌"
        print(f"  {status} {test}")

    print(f"\nPassed: {passed}/5")

    if passed == 5:
        print("\n✅ ALL TESTS PASSED")
        print("   Heuristic is admissible, consistent, and informative")
        print("   Ready to integrate into A* search")
    else:
        print(f"\n⚠️  {5-passed} tests failed - need debugging")
