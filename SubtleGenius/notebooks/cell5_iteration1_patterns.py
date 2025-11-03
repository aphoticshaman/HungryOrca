# ═══════════════════════════════════════════════════════════════════════════════
# CELL 5 - ITERATION 1: BASIC PATTERN MATCHING
# Phase 1: Geometric and Color Transformations (Target: 10-15% accuracy)
# ═══════════════════════════════════════════════════════════════════════════════

import numpy as np
from typing import List, Dict, Tuple, Optional, Callable

# ═══════════════════════════════════════════════════════════════════════════════
# PATTERN DETECTION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def grid_to_array(grid: List[List[int]]) -> np.ndarray:
    """Convert grid to numpy array for easier manipulation"""
    return np.array(grid, dtype=np.int32)

def array_to_grid(arr: np.ndarray) -> List[List[int]]:
    """Convert numpy array back to grid (2D list)"""
    return arr.tolist()

# ═══════════════════════════════════════════════════════════════════════════════
# GEOMETRIC TRANSFORMATIONS
# ═══════════════════════════════════════════════════════════════════════════════

def rotate_90_cw(grid: List[List[int]]) -> List[List[int]]:
    """Rotate grid 90 degrees clockwise"""
    arr = grid_to_array(grid)
    rotated = np.rot90(arr, k=-1)  # k=-1 for clockwise
    return array_to_grid(rotated)

def rotate_90_ccw(grid: List[List[int]]) -> List[List[int]]:
    """Rotate grid 90 degrees counter-clockwise"""
    arr = grid_to_array(grid)
    rotated = np.rot90(arr, k=1)  # k=1 for counter-clockwise
    return array_to_grid(rotated)

def rotate_180(grid: List[List[int]]) -> List[List[int]]:
    """Rotate grid 180 degrees"""
    arr = grid_to_array(grid)
    rotated = np.rot90(arr, k=2)
    return array_to_grid(rotated)

def rotate_270_cw(grid: List[List[int]]) -> List[List[int]]:
    """Rotate grid 270 degrees clockwise (same as 90 ccw)"""
    return rotate_90_ccw(grid)

def flip_horizontal(grid: List[List[int]]) -> List[List[int]]:
    """Flip grid horizontally (left-right mirror)"""
    arr = grid_to_array(grid)
    flipped = np.fliplr(arr)
    return array_to_grid(flipped)

def flip_vertical(grid: List[List[int]]) -> List[List[int]]:
    """Flip grid vertically (top-bottom mirror)"""
    arr = grid_to_array(grid)
    flipped = np.flipud(arr)
    return array_to_grid(flipped)

def flip_diagonal_main(grid: List[List[int]]) -> List[List[int]]:
    """Flip along main diagonal (transpose)"""
    arr = grid_to_array(grid)
    transposed = np.transpose(arr)
    return array_to_grid(transposed)

def flip_diagonal_anti(grid: List[List[int]]) -> List[List[int]]:
    """Flip along anti-diagonal"""
    arr = grid_to_array(grid)
    # Anti-diagonal = rotate 90, then transpose
    flipped = np.transpose(np.rot90(arr, k=-1))
    return array_to_grid(flipped)

# ═══════════════════════════════════════════════════════════════════════════════
# COLOR TRANSFORMATIONS
# ═══════════════════════════════════════════════════════════════════════════════

def detect_color_mapping(input_grid: List[List[int]],
                         output_grid: List[List[int]]) -> Optional[Dict[int, int]]:
    """
    Detect consistent color mapping from input to output.
    Returns mapping dict if consistent, None otherwise.
    """
    if len(input_grid) != len(output_grid):
        return None
    if len(input_grid) == 0 or len(input_grid[0]) != len(output_grid[0]):
        return None

    mapping = {}
    inp_arr = grid_to_array(input_grid)
    out_arr = grid_to_array(output_grid)

    for i in range(inp_arr.shape[0]):
        for j in range(inp_arr.shape[1]):
            in_color = int(inp_arr[i, j])
            out_color = int(out_arr[i, j])

            if in_color in mapping:
                if mapping[in_color] != out_color:
                    return None  # Inconsistent mapping
            else:
                mapping[in_color] = out_color

    return mapping

def apply_color_mapping(grid: List[List[int]],
                        mapping: Dict[int, int]) -> List[List[int]]:
    """Apply color mapping to grid"""
    arr = grid_to_array(grid)
    result = np.copy(arr)

    for old_color, new_color in mapping.items():
        result[arr == old_color] = new_color

    return array_to_grid(result)

# ═══════════════════════════════════════════════════════════════════════════════
# PATTERN MATCHING ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def grids_equal(g1: List[List[int]], g2: List[List[int]]) -> bool:
    """Check if two grids are equal"""
    if len(g1) != len(g2):
        return False
    if len(g1) == 0:
        return True
    if len(g1[0]) != len(g2[0]):
        return False

    arr1 = grid_to_array(g1)
    arr2 = grid_to_array(g2)
    return np.array_equal(arr1, arr2)

def test_transformation(input_grid: List[List[int]],
                       output_grid: List[List[int]],
                       transform_func: Callable) -> bool:
    """Test if transformation matches input->output"""
    try:
        result = transform_func(input_grid)
        return grids_equal(result, output_grid)
    except:
        return False

def detect_geometric_pattern(task_data: Dict) -> Optional[Tuple[str, Callable]]:
    """
    Detect geometric transformation pattern from training examples.
    Returns (pattern_name, transform_function) if found, None otherwise.
    """

    # All geometric transformations to try
    transformations = [
        ('rotate_90_cw', rotate_90_cw),
        ('rotate_90_ccw', rotate_90_ccw),
        ('rotate_180', rotate_180),
        ('flip_horizontal', flip_horizontal),
        ('flip_vertical', flip_vertical),
        ('flip_diagonal_main', flip_diagonal_main),
        ('flip_diagonal_anti', flip_diagonal_anti),
    ]

    train_examples = task_data.get('train', [])
    if not train_examples:
        return None

    # Test each transformation
    for pattern_name, transform_func in transformations:
        # Check if this transformation works for ALL training examples
        matches_all = True

        for example in train_examples:
            input_grid = example['input']
            output_grid = example['output']

            if not test_transformation(input_grid, output_grid, transform_func):
                matches_all = False
                break

        if matches_all:
            return (pattern_name, transform_func)

    return None

def detect_color_pattern(task_data: Dict) -> Optional[Dict[int, int]]:
    """
    Detect color mapping pattern from training examples.
    Returns mapping dict if consistent across all examples, None otherwise.
    """

    train_examples = task_data.get('train', [])
    if not train_examples:
        return None

    # Get mapping from first example
    first_mapping = detect_color_mapping(
        train_examples[0]['input'],
        train_examples[0]['output']
    )

    if first_mapping is None:
        return None

    # Verify same mapping works for all examples
    for example in train_examples[1:]:
        test_mapping = detect_color_mapping(
            example['input'],
            example['output']
        )

        if test_mapping != first_mapping:
            return None  # Different mappings, not consistent

    return first_mapping

# ═══════════════════════════════════════════════════════════════════════════════
# COMBINED PATTERN DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def detect_combined_pattern(task_data: Dict) -> Optional[Tuple[str, Callable]]:
    """
    Detect patterns that combine geometric + color transformations.
    Returns (pattern_name, combined_function) if found.
    """

    # Try geometric first
    geo_result = detect_geometric_pattern(task_data)
    if geo_result:
        return geo_result

    # Try color mapping
    color_mapping = detect_color_pattern(task_data)
    if color_mapping:
        # Create closure with the mapping
        def color_transform(grid):
            return apply_color_mapping(grid, color_mapping)
        return ('color_mapping', color_transform)

    # Try geometric + color combination
    # For each geometric transformation
    transformations = [
        ('rotate_90_cw', rotate_90_cw),
        ('rotate_90_ccw', rotate_90_ccw),
        ('rotate_180', rotate_180),
        ('flip_horizontal', flip_horizontal),
        ('flip_vertical', flip_vertical),
    ]

    train_examples = task_data.get('train', [])
    if not train_examples:
        return None

    for geo_name, geo_func in transformations:
        # Try this geometric transform + color mapping
        # Apply geometric to first example
        first_geo = geo_func(train_examples[0]['input'])
        first_mapping = detect_color_mapping(first_geo, train_examples[0]['output'])

        if first_mapping is None:
            continue

        # Check if this combo works for all examples
        matches_all = True
        for example in train_examples:
            geo_result = geo_func(example['input'])
            color_result = apply_color_mapping(geo_result, first_mapping)

            if not grids_equal(color_result, example['output']):
                matches_all = False
                break

        if matches_all:
            # Create combined function
            def combined_transform(grid, gf=geo_func, cm=first_mapping):
                return apply_color_mapping(gf(grid), cm)
            return (f'{geo_name}_+_color', combined_transform)

    return None

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN SOLVER (ITERATION 1)
# ═══════════════════════════════════════════════════════════════════════════════

def pattern_matching_solver(test_input: List[List[int]],
                            task_data: Dict,
                            attempt: int = 1) -> List[List[int]]:
    """
    Phase 1 solver: Basic pattern matching.

    Strategy:
    1. Detect pattern from training examples
    2. Apply pattern to test input
    3. For attempt 2, try alternative patterns

    Target: 10-15% accuracy on ARC tasks
    """

    # Detect pattern from training data
    pattern_result = detect_combined_pattern(task_data)

    if pattern_result is not None:
        pattern_name, transform_func = pattern_result

        try:
            # Apply detected pattern
            result = transform_func(test_input)

            # For attempt 2, try a variation
            if attempt == 2:
                # Try opposite transformation as variation
                if 'flip_horizontal' in pattern_name:
                    result = flip_vertical(test_input)
                elif 'flip_vertical' in pattern_name:
                    result = flip_horizontal(test_input)
                elif 'rotate_90_cw' in pattern_name:
                    result = rotate_90_ccw(test_input)
                elif 'rotate_90_ccw' in pattern_name:
                    result = rotate_90_cw(test_input)
                else:
                    # Use original pattern for attempt 2 as well
                    pass

            return result

        except Exception as e:
            # Pattern application failed, return input
            pass

    # No pattern detected or pattern failed
    # Fallback: return input (identity transform)
    return test_input

# ═══════════════════════════════════════════════════════════════════════════════
# SOLVER STATISTICS
# ═══════════════════════════════════════════════════════════════════════════════

class PatternStats:
    """Track pattern detection statistics"""

    def __init__(self):
        self.patterns_detected = {}
        self.total_tasks = 0
        self.pattern_found_count = 0

    def record_pattern(self, pattern_name: Optional[str]):
        """Record detected pattern"""
        self.total_tasks += 1

        if pattern_name:
            self.pattern_found_count += 1
            if pattern_name in self.patterns_detected:
                self.patterns_detected[pattern_name] += 1
            else:
                self.patterns_detected[pattern_name] = 1

    def print_stats(self):
        """Print pattern detection statistics"""
        print(f"\n{'='*70}")
        print("PATTERN DETECTION STATISTICS")
        print(f"{'='*70}")
        print(f"Total tasks: {self.total_tasks}")
        print(f"Patterns found: {self.pattern_found_count} ({self.pattern_found_count/self.total_tasks*100:.1f}%)")
        print(f"\nPattern breakdown:")
        for pattern, count in sorted(self.patterns_detected.items(), key=lambda x: x[1], reverse=True):
            print(f"  {pattern}: {count}")
        print(f"{'='*70}\n")

# Global stats tracker
pattern_stats = PatternStats()

# ═══════════════════════════════════════════════════════════════════════════════
# ENHANCED SOLVER WITH STATS
# ═══════════════════════════════════════════════════════════════════════════════

def enhanced_pattern_solver(test_input: List[List[int]],
                            task_data: Dict,
                            attempt: int = 1,
                            track_stats: bool = False) -> List[List[int]]:
    """
    Enhanced solver with statistics tracking.
    Use this as the main solver function.
    """

    # Detect pattern
    pattern_result = detect_combined_pattern(task_data)

    # Track statistics
    if track_stats and attempt == 1:  # Only track once per task
        pattern_name = pattern_result[0] if pattern_result else None
        pattern_stats.record_pattern(pattern_name)

    # Apply pattern
    if pattern_result is not None:
        pattern_name, transform_func = pattern_result

        try:
            result = transform_func(test_input)

            # Attempt 2 variation
            if attempt == 2:
                if 'flip_horizontal' in pattern_name:
                    result = flip_vertical(test_input)
                elif 'flip_vertical' in pattern_name:
                    result = flip_horizontal(test_input)
                elif 'rotate_90_cw' in pattern_name:
                    result = rotate_90_ccw(test_input)
                elif 'rotate_90_ccw' in pattern_name:
                    result = rotate_90_cw(test_input)

            return result

        except Exception:
            pass

    # Fallback: identity
    return test_input

print("✅ Pattern Matching Solver (Iteration 1) loaded")
print("   Patterns: 7 geometric + color mapping + combinations")
print("   Strategy: Detect from training, apply to test")
print("   Target: 10-15% accuracy improvement")
