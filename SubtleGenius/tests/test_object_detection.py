#!/usr/bin/env python3
"""
Test harness for SubtleGenius Object Detection Solver (Iteration 2)
Tests connected component analysis, spatial relationships, object transformations
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'notebooks'))

# Import solver functions
from cell5_iteration2_objects import *

# ═══════════════════════════════════════════════════════════════════════════════
# TEST CASES
# ═══════════════════════════════════════════════════════════════════════════════

def test_connected_components_4():
    """Test 4-connected component analysis"""
    print("\n🧪 Test 1: Connected Components (4-connectivity)")

    # Simple grid with 2 objects
    grid = [
        [1, 1, 0, 2],
        [1, 0, 0, 2],
        [0, 0, 2, 2]
    ]

    objects = find_connected_components(grid, connectivity=4, background_color=0)

    # Should find 2 objects (color 1 and color 2)
    if len(objects) == 2:
        print(f"   ✅ Found {len(objects)} objects (expected 2)")

        # Check object 1 (color 1)
        obj1 = [o for o in objects if o.color == 1][0]
        if obj1.area == 3:
            print(f"   ✅ Object 1 (color 1): area {obj1.area} (expected 3)")
        else:
            print(f"   ❌ Object 1 area {obj1.area}, expected 3")
            return False

        # Check object 2 (color 2)
        obj2 = [o for o in objects if o.color == 2][0]
        if obj2.area == 4:
            print(f"   ✅ Object 2 (color 2): area {obj2.area} (expected 4)")
        else:
            print(f"   ❌ Object 2 area {obj2.area}, expected 4")
            return False

        return True
    else:
        print(f"   ❌ Found {len(objects)} objects, expected 2")
        return False

def test_object_properties():
    """Test object property extraction"""
    print("\n🧪 Test 2: Object Properties")

    # Rectangle object
    grid = [
        [0, 0, 0],
        [0, 1, 1],
        [0, 1, 1]
    ]

    objects = find_connected_components(grid, connectivity=4)

    if len(objects) == 1:
        obj = objects[0]
        print(f"   ✅ Found 1 object")

        # Check properties
        if obj.color == 1:
            print(f"   ✅ Color: {obj.color}")
        else:
            print(f"   ❌ Color {obj.color}, expected 1")
            return False

        if obj.area == 4:
            print(f"   ✅ Area: {obj.area}")
        else:
            print(f"   ❌ Area {obj.area}, expected 4")
            return False

        if obj.width == 2 and obj.height == 2:
            print(f"   ✅ Dimensions: {obj.width}x{obj.height}")
        else:
            print(f"   ❌ Dimensions {obj.width}x{obj.height}, expected 2x2")
            return False

        if obj.is_rectangle():
            print(f"   ✅ Shape: rectangle")
        else:
            print(f"   ❌ Not recognized as rectangle")
            return False

        return True
    else:
        print(f"   ❌ Found {len(objects)} objects, expected 1")
        return False

def test_spatial_adjacency():
    """Test spatial relationship detection - adjacency"""
    print("\n🧪 Test 3: Spatial Adjacency")

    # Two adjacent objects
    grid = [
        [1, 1, 2, 2],
        [1, 1, 2, 2]
    ]

    objects = find_connected_components(grid, connectivity=4)

    if len(objects) == 2:
        print(f"   ✅ Found 2 objects")

        obj1, obj2 = objects[0], objects[1]

        if objects_are_adjacent(obj1, obj2):
            print(f"   ✅ Objects are adjacent")
            return True
        else:
            print(f"   ❌ Objects not detected as adjacent")
            return False
    else:
        print(f"   ❌ Found {len(objects)} objects, expected 2")
        return False

def test_object_color_change_pattern():
    """Test object color change pattern detection"""
    print("\n🧪 Test 4: Object Color Change Pattern")

    # Create task with consistent color change: 1->5, 2->7
    task = {
        'train': [
            {
                'input': [[1, 2], [1, 2]],
                'output': [[5, 7], [5, 7]]
            },
            {
                'input': [[2, 1], [2, 1]],
                'output': [[7, 5], [7, 5]]
            }
        ],
        'test': [
            {'input': [[1, 1], [2, 2]]}
        ]
    }

    pattern = detect_object_transformation_pattern(task)

    if pattern and pattern.get('type') == 'object_color_change':
        print(f"   ✅ Detected pattern: {pattern['type']}")
        color_mapping = pattern.get('color_mapping', {})

        if color_mapping.get(1) == 5 and color_mapping.get(2) == 7:
            print(f"   ✅ Color mapping correct: {color_mapping}")
            return True
        else:
            print(f"   ❌ Wrong color mapping: {color_mapping}")
            return False
    else:
        print(f"   ❌ Pattern not detected or wrong type")
        return False

def test_object_to_grid_conversion():
    """Test converting object back to grid"""
    print("\n🧪 Test 5: Object to Grid Conversion")

    # Create grid with object
    grid = [
        [0, 0, 0, 0],
        [0, 1, 1, 0],
        [0, 1, 1, 0],
        [0, 0, 0, 0]
    ]

    objects = find_connected_components(grid, connectivity=4)

    if len(objects) == 1:
        obj = objects[0]
        print(f"   ✅ Found 1 object")

        # Convert back to grid
        obj_grid = obj.to_grid(background_color=0)

        # Should be 2x2 grid with all 1s
        expected = [[1, 1], [1, 1]]

        if obj_grid == expected:
            print(f"   ✅ Object grid conversion correct")
            return True
        else:
            print(f"   ❌ Wrong conversion")
            print(f"      Expected: {expected}")
            print(f"      Got: {obj_grid}")
            return False
    else:
        print(f"   ❌ Found {len(objects)} objects, expected 1")
        return False

def test_combined_solver():
    """Test combined solver with object + pattern matching"""
    print("\n🧪 Test 6: Combined Solver (Object + Pattern)")

    # Task with object color change
    task = {
        'train': [
            {
                'input': [[1, 1], [1, 1]],
                'output': [[5, 5], [5, 5]]
            },
            {
                'input': [[1, 1, 1], [1, 1, 1]],
                'output': [[5, 5, 5], [5, 5, 5]]
            }
        ],
        'test': [
            {'input': [[1, 1], [1, 1]]}
        ]
    }

    test_input = [[1, 1], [1, 1]]

    result = combined_solver(test_input, task, attempt=1)

    # Should change 1s to 5s
    expected = [[5, 5], [5, 5]]

    if result == expected:
        print(f"   ✅ Combined solver produced correct result")
        return True
    else:
        print(f"   ❌ Wrong result")
        print(f"      Expected: {expected}")
        print(f"      Got: {result}")
        return False

# ═══════════════════════════════════════════════════════════════════════════════
# RUN ALL TESTS
# ═══════════════════════════════════════════════════════════════════════════════

def run_all_tests():
    """Run complete test suite"""

    print("="*70)
    print("SubtleGenius Object Detection Solver - Test Suite")
    print("Iteration 2: Object-Based Pattern Recognition")
    print("="*70)

    tests = [
        test_connected_components_4,
        test_object_properties,
        test_spatial_adjacency,
        test_object_color_change_pattern,
        test_object_to_grid_conversion,
        test_combined_solver
    ]

    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"   ❌ Test crashed: {e}")
            import traceback
            traceback.print_exc()
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
