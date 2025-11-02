# ═══════════════════════════════════════════════════════════════════════════════
# CELL 5 - ITERATION 2: OBJECT DETECTION & SPATIAL REASONING
# Phase 2: Object-based pattern recognition (Target: 20-30% accuracy)
# Builds on Iteration 1 (pattern matching) with object-level intelligence
# ═══════════════════════════════════════════════════════════════════════════════

import numpy as np
from typing import List, Dict, Tuple, Optional, Set, Callable
from dataclasses import dataclass
from collections import defaultdict

# Import Iteration 1 pattern matching as foundation
# In production: include cell5_iteration1_patterns.py first, then this file
# For standalone testing: uncomment the import below
# from cell5_iteration1_patterns import *

# ═══════════════════════════════════════════════════════════════════════════════
# OBJECT REPRESENTATION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class DetectedObject:
    """Represents a discrete object detected in a grid"""

    id: int                                    # Unique identifier
    color: int                                 # Primary color value
    pixels: List[Tuple[int, int]]              # List of (row, col) coordinates
    bounding_box: Tuple[int, int, int, int]    # (min_row, min_col, max_row, max_col)

    @property
    def area(self) -> int:
        """Number of pixels in object"""
        return len(self.pixels)

    @property
    def width(self) -> int:
        """Width of bounding box"""
        return self.bounding_box[3] - self.bounding_box[1] + 1

    @property
    def height(self) -> int:
        """Height of bounding box"""
        return self.bounding_box[2] - self.bounding_box[0] + 1

    @property
    def center(self) -> Tuple[float, float]:
        """Center of mass (row, col)"""
        rows = [p[0] for p in self.pixels]
        cols = [p[1] for p in self.pixels]
        return (np.mean(rows), np.mean(cols))

    @property
    def shape_type(self) -> str:
        """Classify object shape"""
        if self.is_rectangle():
            if self.width == 1 or self.height == 1:
                return "line"
            elif self.width == self.height:
                return "square"
            return "rectangle"
        elif self.is_single_pixel():
            return "point"
        else:
            return "irregular"

    def is_rectangle(self) -> bool:
        """Check if object forms perfect rectangle"""
        expected_area = self.width * self.height
        return self.area == expected_area

    def is_single_pixel(self) -> bool:
        """Check if object is a single pixel"""
        return self.area == 1

    def to_grid(self, background_color: int = 0) -> List[List[int]]:
        """Convert object to minimal grid representation"""
        min_r, min_c, max_r, max_c = self.bounding_box
        height = max_r - min_r + 1
        width = max_c - min_c + 1

        grid = [[background_color for _ in range(width)] for _ in range(height)]

        for r, c in self.pixels:
            local_r = r - min_r
            local_c = c - min_c
            grid[local_r][local_c] = self.color

        return grid

# ═══════════════════════════════════════════════════════════════════════════════
# CONNECTED COMPONENT ANALYSIS (No scipy dependency - pure numpy)
# ═══════════════════════════════════════════════════════════════════════════════

def find_connected_components(grid: List[List[int]],
                              connectivity: int = 4,
                              background_color: int = 0) -> List[DetectedObject]:
    """
    Find connected components using flood-fill algorithm.
    No external dependencies - pure numpy implementation.

    Args:
        grid: 2D list of integers
        connectivity: 4 or 8 (4-connected or 8-connected)
        background_color: Color to treat as background (not objects)

    Returns:
        List of DetectedObject instances
    """

    if not grid or not grid[0]:
        return []

    arr = np.array(grid, dtype=np.int32)
    rows, cols = arr.shape
    visited = np.zeros((rows, cols), dtype=bool)
    objects = []
    object_id = 0

    # Neighbor offsets for connectivity
    if connectivity == 4:
        neighbors = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    else:  # 8-connected
        neighbors = [(0, 1), (1, 0), (0, -1), (-1, 0),
                     (1, 1), (1, -1), (-1, 1), (-1, -1)]

    def flood_fill(start_r: int, start_c: int, color: int) -> List[Tuple[int, int]]:
        """Flood-fill to find all pixels in connected component"""
        stack = [(start_r, start_c)]
        pixels = []

        while stack:
            r, c = stack.pop()

            if r < 0 or r >= rows or c < 0 or c >= cols:
                continue
            if visited[r, c]:
                continue
            if arr[r, c] != color:
                continue

            visited[r, c] = True
            pixels.append((r, c))

            # Add neighbors to stack
            for dr, dc in neighbors:
                stack.append((r + dr, c + dc))

        return pixels

    # Find all objects
    for r in range(rows):
        for c in range(cols):
            if visited[r, c]:
                continue

            color = int(arr[r, c])

            # Skip background
            if color == background_color:
                visited[r, c] = True
                continue

            # Flood-fill to find connected component
            pixels = flood_fill(r, c, color)

            if pixels:
                # Calculate bounding box
                rows_list = [p[0] for p in pixels]
                cols_list = [p[1] for p in pixels]
                bbox = (min(rows_list), min(cols_list),
                       max(rows_list), max(cols_list))

                # Create object
                obj = DetectedObject(
                    id=object_id,
                    color=color,
                    pixels=pixels,
                    bounding_box=bbox
                )
                objects.append(obj)
                object_id += 1

    return objects

# ═══════════════════════════════════════════════════════════════════════════════
# SPATIAL RELATIONSHIP ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def objects_are_adjacent(obj1: DetectedObject, obj2: DetectedObject) -> bool:
    """Check if two objects are touching (adjacent)"""
    # Expand each object's bounding box by 1 and check for overlap
    min_r1, min_c1, max_r1, max_c1 = obj1.bounding_box
    min_r2, min_c2, max_r2, max_c2 = obj2.bounding_box

    # Check if bounding boxes are adjacent or overlapping
    h_adjacent = (max_c1 + 1 >= min_c2 and min_c1 - 1 <= max_c2)
    v_adjacent = (max_r1 + 1 >= min_r2 and min_r1 - 1 <= max_r2)

    if not (h_adjacent and v_adjacent):
        return False

    # Check actual pixel adjacency
    for r1, c1 in obj1.pixels:
        for r2, c2 in obj2.pixels:
            if abs(r1 - r2) + abs(c1 - c2) == 1:  # Manhattan distance = 1
                return True

    return False

def object_contains(outer: DetectedObject, inner: DetectedObject) -> bool:
    """Check if outer object contains inner object"""
    min_r_out, min_c_out, max_r_out, max_c_out = outer.bounding_box
    min_r_in, min_c_in, max_r_in, max_c_in = inner.bounding_box

    return (min_r_out <= min_r_in and max_r_out >= max_r_in and
            min_c_out <= min_c_in and max_c_out >= max_c_in)

def objects_aligned_horizontal(obj1: DetectedObject, obj2: DetectedObject,
                               tolerance: int = 1) -> bool:
    """Check if objects are horizontally aligned"""
    center1 = obj1.center
    center2 = obj2.center
    return abs(center1[0] - center2[0]) <= tolerance

def objects_aligned_vertical(obj1: DetectedObject, obj2: DetectedObject,
                            tolerance: int = 1) -> bool:
    """Check if objects are vertically aligned"""
    center1 = obj1.center
    center2 = obj2.center
    return abs(center1[1] - center2[1]) <= tolerance

def analyze_spatial_relationships(objects: List[DetectedObject]) -> Dict[str, List[Tuple[int, int]]]:
    """
    Analyze spatial relationships between all objects.

    Returns:
        Dictionary mapping relationship types to pairs of object IDs
    """
    relationships = {
        'adjacent': [],
        'contains': [],
        'aligned_h': [],
        'aligned_v': []
    }

    for i, obj1 in enumerate(objects):
        for j, obj2 in enumerate(objects):
            if i >= j:
                continue

            # Check adjacency
            if objects_are_adjacent(obj1, obj2):
                relationships['adjacent'].append((obj1.id, obj2.id))

            # Check containment
            if object_contains(obj1, obj2):
                relationships['contains'].append((obj1.id, obj2.id))
            elif object_contains(obj2, obj1):
                relationships['contains'].append((obj2.id, obj1.id))

            # Check alignment
            if objects_aligned_horizontal(obj1, obj2):
                relationships['aligned_h'].append((obj1.id, obj2.id))

            if objects_aligned_vertical(obj1, obj2):
                relationships['aligned_v'].append((obj1.id, obj2.id))

    return relationships

# ═══════════════════════════════════════════════════════════════════════════════
# OBJECT TRANSFORMATION DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def match_objects(input_objects: List[DetectedObject],
                 output_objects: List[DetectedObject]) -> List[Tuple[Optional[DetectedObject],
                                                                     Optional[DetectedObject]]]:
    """
    Match objects from input to output based on similarity.
    Returns list of (input_obj, output_obj) pairs.
    None in tuple means object was created/deleted.
    """
    matches = []
    used_outputs = set()

    # Try to match by color first
    for inp_obj in input_objects:
        best_match = None
        best_score = -1

        for out_obj in output_objects:
            if out_obj.id in used_outputs:
                continue

            # Matching score based on color, size, position
            score = 0

            # Color match
            if inp_obj.color == out_obj.color:
                score += 10

            # Size similarity
            size_diff = abs(inp_obj.area - out_obj.area)
            score += max(0, 5 - size_diff)

            # Position similarity
            center_dist = np.sqrt((inp_obj.center[0] - out_obj.center[0])**2 +
                                 (inp_obj.center[1] - out_obj.center[1])**2)
            score += max(0, 5 - center_dist)

            if score > best_score:
                best_score = score
                best_match = out_obj

        if best_match and best_score > 5:  # Minimum threshold
            matches.append((inp_obj, best_match))
            used_outputs.add(best_match.id)
        else:
            matches.append((inp_obj, None))  # Object deleted

    # Add created objects (in output but not matched)
    for out_obj in output_objects:
        if out_obj.id not in used_outputs:
            matches.append((None, out_obj))  # Object created

    return matches

def detect_object_transformation_pattern(task_data: Dict) -> Optional[Dict]:
    """
    Detect object-level transformation patterns from training examples.

    Returns:
        Pattern description dict if consistent pattern found, None otherwise
    """

    train_examples = task_data.get('train', [])
    if not train_examples:
        return None

    # Analyze first example to establish pattern hypothesis
    first_input = train_examples[0]['input']
    first_output = train_examples[0]['output']

    input_objects = find_connected_components(first_input)
    output_objects = find_connected_components(first_output)

    # Check for simple object-level patterns

    # Pattern 1: Object deletion (fewer objects in output)
    if len(output_objects) < len(input_objects):
        # Check if pattern is consistent across all examples
        consistent = True
        for example in train_examples[1:]:
            inp_objs = find_connected_components(example['input'])
            out_objs = find_connected_components(example['output'])
            if len(out_objs) >= len(inp_objs):
                consistent = False
                break

        if consistent:
            return {'type': 'object_deletion', 'description': 'Removes some objects'}

    # Pattern 2: Object creation (more objects in output)
    if len(output_objects) > len(input_objects):
        consistent = True
        for example in train_examples[1:]:
            inp_objs = find_connected_components(example['input'])
            out_objs = find_connected_components(example['output'])
            if len(out_objs) <= len(inp_objs):
                consistent = False
                break

        if consistent:
            return {'type': 'object_creation', 'description': 'Creates new objects'}

    # Pattern 3: Object color change
    if len(input_objects) == len(output_objects):
        # Check if same number of objects with color changes
        matches = match_objects(input_objects, output_objects)

        color_changes = {}
        for inp_obj, out_obj in matches:
            if inp_obj and out_obj:
                if inp_obj.color != out_obj.color:
                    color_changes[inp_obj.color] = out_obj.color

        if color_changes:
            # Verify pattern across all examples
            consistent = True
            for example in train_examples[1:]:
                inp_objs = find_connected_components(example['input'])
                out_objs = find_connected_components(example['output'])
                example_matches = match_objects(inp_objs, out_objs)

                for inp_obj, out_obj in example_matches:
                    if inp_obj and out_obj:
                        expected_color = color_changes.get(inp_obj.color)
                        if expected_color and out_obj.color != expected_color:
                            consistent = False
                            break

            if consistent:
                return {
                    'type': 'object_color_change',
                    'color_mapping': color_changes,
                    'description': f'Changes object colors: {color_changes}'
                }

    return None

# ═══════════════════════════════════════════════════════════════════════════════
# OBJECT-BASED SOLVER
# ═══════════════════════════════════════════════════════════════════════════════

def apply_object_transformation(grid: List[List[int]],
                                pattern: Dict) -> List[List[int]]:
    """Apply detected object transformation pattern to grid"""

    pattern_type = pattern.get('type')

    if pattern_type == 'object_color_change':
        # Apply color mapping to objects
        color_mapping = pattern.get('color_mapping', {})
        arr = np.array(grid, dtype=np.int32)
        result = np.copy(arr)

        for old_color, new_color in color_mapping.items():
            result[arr == old_color] = new_color

        return result.tolist()

    elif pattern_type == 'object_deletion':
        # For now, return input (more complex logic needed)
        return grid

    elif pattern_type == 'object_creation':
        # For now, return input (more complex logic needed)
        return grid

    else:
        return grid

def object_detection_solver(test_input: List[List[int]],
                           task_data: Dict,
                           attempt: int = 1) -> List[List[int]]:
    """
    Phase 2 solver: Object detection and spatial reasoning.

    Strategy:
    1. Detect objects in input
    2. Analyze spatial relationships
    3. Detect object-level transformation pattern
    4. Apply pattern to test input
    5. Fallback to pattern matching (Iteration 1) if no object pattern

    Target: 20-30% accuracy
    """

    # Try object-level pattern detection
    obj_pattern = detect_object_transformation_pattern(task_data)

    if obj_pattern:
        try:
            result = apply_object_transformation(test_input, obj_pattern)

            # For attempt 2, could try variation
            # For now, use same transformation

            return result
        except Exception:
            pass

    # Fallback: try pattern matching from Iteration 1
    # NOTE: In production, this would call enhanced_pattern_solver from Iteration 1
    # For standalone testing, return identity
    return test_input

# ═══════════════════════════════════════════════════════════════════════════════
# COMBINED SOLVER (Iteration 1 + Iteration 2)
# ═══════════════════════════════════════════════════════════════════════════════

def combined_solver(test_input: List[List[int]],
                   task_data: Dict,
                   attempt: int = 1) -> List[List[int]]:
    """
    Combined solver using both pattern matching and object detection.

    Strategy hierarchy:
    1. Try object-level transformations (Iteration 2)
    2. Fallback to pattern matching (Iteration 1)
    3. Ultimate fallback to identity

    This is the main solver to use for Cell 5.
    """

    # Try object detection first
    obj_pattern = detect_object_transformation_pattern(task_data)

    if obj_pattern:
        try:
            result = apply_object_transformation(test_input, obj_pattern)
            # Validate result is a valid grid
            if result and isinstance(result, list) and len(result) > 0:
                return result
        except Exception:
            pass

    # Fallback to pattern matching (Iteration 1)
    # NOTE: Assumes enhanced_pattern_solver is available from Iteration 1
    # If not available, will fall through to identity
    try:
        # This will use Iteration 1's pattern matching
        from cell5_iteration1_patterns import enhanced_pattern_solver
        return enhanced_pattern_solver(test_input, task_data, attempt)
    except ImportError:
        # Iteration 1 not available, use identity fallback
        return test_input

# ═══════════════════════════════════════════════════════════════════════════════
# STATISTICS TRACKING
# ═══════════════════════════════════════════════════════════════════════════════

class ObjectDetectionStats:
    """Track object detection statistics"""

    def __init__(self):
        self.total_tasks = 0
        self.object_patterns_found = 0
        self.pattern_types = defaultdict(int)
        self.avg_objects_per_task = []

    def record_task(self, objects_count: int, pattern: Optional[Dict]):
        """Record statistics for a task"""
        self.total_tasks += 1
        self.avg_objects_per_task.append(objects_count)

        if pattern:
            self.object_patterns_found += 1
            pattern_type = pattern.get('type', 'unknown')
            self.pattern_types[pattern_type] += 1

    def print_stats(self):
        """Print object detection statistics"""
        print(f"\n{'='*70}")
        print("OBJECT DETECTION STATISTICS")
        print(f"{'='*70}")
        print(f"Total tasks: {self.total_tasks}")
        print(f"Object patterns found: {self.object_patterns_found} "
              f"({self.object_patterns_found/max(1,self.total_tasks)*100:.1f}%)")

        if self.avg_objects_per_task:
            print(f"Avg objects per task: {np.mean(self.avg_objects_per_task):.1f}")

        if self.pattern_types:
            print(f"\nPattern types detected:")
            for ptype, count in sorted(self.pattern_types.items(),
                                      key=lambda x: x[1], reverse=True):
                print(f"  {ptype}: {count}")

        print(f"{'='*70}\n")

# Global stats tracker
object_stats = ObjectDetectionStats()

print("✅ Object Detection Solver (Iteration 2) loaded")
print("   Capabilities: Connected components, spatial analysis, object transformations")
print("   Strategy: Object patterns → Pattern matching → Identity fallback")
print("   Target: 20-30% accuracy improvement")
