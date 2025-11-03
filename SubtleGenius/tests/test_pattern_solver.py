#!/usr/bin/env python3
"""
Test harness for SubtleGenius Pattern Matching Solver (Iteration 1)
Tests basic geometric and color transformations
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'notebooks'))

import numpy as np
from typing import List, Dict

# Import solver functions
from cell5_iteration1_patterns import *

# ═══════════════════════════════════════════════════════════════════════════════
# TEST CASES
# ═══════════════════════════════════════════════════════════════════════════════

def create_test_task(input_grids: List[List[List[int]]],
                     output_grids: List[List[List[int]]],
                     test_input: List[List[int]]) -> Dict:
    """Create a test task structure"""
    train_examples = []
    for inp, out in zip(input_grids, output_grids):
        train_examples.append({'input': inp, 'output': out})

    return {
        'train': train_examples,
        'test': [{'input': test_input}]
    }

def test_rotate_90_cw():
    """Test 90-degree clockwise rotation detection"""
    print("\n🧪 Test 1: Rotate 90° Clockwise")

    # Create simple test case
    input1 = [[1, 2], [3, 4]]
    output1 = [[3, 1], [4, 2]]  # Rotated 90 CW

    input2 = [[5, 6], [7, 8]]
    output2 = [[7, 5], [8, 6]]  # Rotated 90 CW

    test_input = [[9, 0], [1, 2]]
    expected_output = [[1, 9], [2, 0]]  # Should rotate 90 CW

    task = create_test_task([input1, input2], [output1, output2], test_input)

    # Detect pattern
    pattern_result = detect_combined_pattern(task)

    if pattern_result:
        pattern_name, transform_func = pattern_result
        print(f"   ✅ Pattern detected: {pattern_name}")

        result = transform_func(test_input)
        if grids_equal(result, expected_output):
            print(f"   ✅ Transform correct!")
            return True
        else:
            print(f"   ❌ Transform incorrect")
            print(f"      Expected: {expected_output}")
            print(f"      Got: {result}")
            return False
    else:
        print(f"   ❌ No pattern detected")
        return False

def test_flip_horizontal():
    """Test horizontal flip detection"""
    print("\n🧪 Test 2: Flip Horizontal")

    input1 = [[1, 2, 3], [4, 5, 6]]
    output1 = [[3, 2, 1], [6, 5, 4]]  # Flipped horizontally

    input2 = [[7, 8], [9, 0]]
    output2 = [[8, 7], [0, 9]]  # Flipped horizontally

    test_input = [[1, 2], [3, 4]]
    expected_output = [[2, 1], [4, 3]]

    task = create_test_task([input1, input2], [output1, output2], test_input)

    pattern_result = detect_combined_pattern(task)

    if pattern_result:
        pattern_name, transform_func = pattern_result
        print(f"   ✅ Pattern detected: {pattern_name}")

        result = transform_func(test_input)
        if grids_equal(result, expected_output):
            print(f"   ✅ Transform correct!")
            return True
        else:
            print(f"   ❌ Transform incorrect")
            return False
    else:
        print(f"   ❌ No pattern detected")
        return False

def test_color_mapping():
    """Test color mapping detection"""
    print("\n🧪 Test 3: Color Mapping")

    # Simple color swap: 1->5, 2->7, 0->0
    input1 = [[1, 2], [0, 1]]
    output1 = [[5, 7], [0, 5]]

    input2 = [[2, 1], [1, 0]]
    output2 = [[7, 5], [5, 0]]

    test_input = [[1, 0], [2, 2]]
    expected_output = [[5, 0], [7, 7]]

    task = create_test_task([input1, input2], [output1, output2], test_input)

    pattern_result = detect_combined_pattern(task)

    if pattern_result:
        pattern_name, transform_func = pattern_result
        print(f"   ✅ Pattern detected: {pattern_name}")

        result = transform_func(test_input)
        if grids_equal(result, expected_output):
            print(f"   ✅ Transform correct!")
            return True
        else:
            print(f"   ❌ Transform incorrect")
            print(f"      Expected: {expected_output}")
            print(f"      Got: {result}")
            return False
    else:
        print(f"   ❌ No pattern detected")
        return False

def test_identity():
    """Test identity (no transformation)"""
    print("\n🧪 Test 4: Identity (No Pattern)")

    # Random grids with no pattern
    input1 = [[1, 2], [3, 4]]
    output1 = [[9, 7], [2, 1]]

    input2 = [[5, 6], [7, 8]]
    output2 = [[3, 0], [4, 5]]

    test_input = [[1, 2], [3, 4]]

    task = create_test_task([input1, input2], [output1, output2], test_input)

    pattern_result = detect_combined_pattern(task)

    if pattern_result is None:
        print(f"   ✅ Correctly detected no pattern")
        # Should fall back to identity
        result = pattern_matching_solver(test_input, task, attempt=1)
        if grids_equal(result, test_input):
            print(f"   ✅ Fallback to identity correct!")
            return True
    else:
        print(f"   ❌ False positive pattern detection")
        return False

def test_flip_vertical():
    """Test vertical flip detection"""
    print("\n🧪 Test 5: Flip Vertical")

    input1 = [[1, 2], [3, 4]]
    output1 = [[3, 4], [1, 2]]  # Flipped vertically

    input2 = [[5, 6], [7, 8]]
    output2 = [[7, 8], [5, 6]]  # Flipped vertically

    test_input = [[9, 0], [1, 2]]
    expected_output = [[1, 2], [9, 0]]

    task = create_test_task([input1, input2], [output1, output2], test_input)

    pattern_result = detect_combined_pattern(task)

    if pattern_result:
        pattern_name, transform_func = pattern_result
        print(f"   ✅ Pattern detected: {pattern_name}")

        result = transform_func(test_input)
        if grids_equal(result, expected_output):
            print(f"   ✅ Transform correct!")
            return True
        else:
            print(f"   ❌ Transform incorrect")
            return False
    else:
        print(f"   ❌ No pattern detected")
        return False

# ═══════════════════════════════════════════════════════════════════════════════
# RUN ALL TESTS
# ═══════════════════════════════════════════════════════════════════════════════

def run_all_tests():
    """Run complete test suite"""

    print("="*70)
    print("SubtleGenius Pattern Matching Solver - Test Suite")
    print("Iteration 1: Basic Geometric and Color Patterns")
    print("="*70)

    tests = [
        test_rotate_90_cw,
        test_flip_horizontal,
        test_color_mapping,
        test_identity,
        test_flip_vertical
    ]

    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"   ❌ Test crashed: {e}")
            results.append(False)

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total} ({passed/total*100:.1f}%)")

    if passed == total:
        print("🎉 All tests passed!")
    else:
        print(f"⚠️  {total - passed} test(s) failed")

    print("="*70 + "\n")

    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
