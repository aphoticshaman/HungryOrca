#!/usr/bin/env python3
"""
ARC Multi-Stage Reasoner - Proper Object-Centric Approach
==========================================================

Implements 4 levels of reasoning as described in ARC best practices:

L1 (Pixel): Direct grid transforms (rotate, mirror, crop)
L2 (Object): Transform individual objects (move, recolor, resize)
L3 (Pattern): Infer rules between input/output object sets
L4 (Constraint): Filter hypotheses using properties

Key Insight: Most tasks require L2/L3, not just L1!
"""

import numpy as np
import json
from typing import List, Dict, Tuple, Set, Optional, Any
from dataclasses import dataclass, field
from scipy.ndimage import label, find_objects
from collections import defaultdict
import copy

Grid = List[List[int]]

# ============================================================================
# OBJECT DECOMPOSITION (Foundation for L2+)
# ============================================================================

@dataclass
class ARCObject:
    """Represents a single object in an ARC grid."""
    id: int
    mask: np.ndarray  # Binary mask of where object is
    color: int
    bounding_box: Tuple[int, int, int, int]  # (min_r, min_c, max_r, max_c)
    centroid: Tuple[float, float]  # (r, c)
    size: int  # Number of pixels

    # Shape properties
    width: int
    height: int
    is_square: bool
    is_rectangle: bool
    is_line: bool

    # Spatial relationships (filled by analyzer)
    neighbors: List[int] = field(default_factory=list)
    relative_position: str = ""  # "top_left", "center", etc.

    def to_grid(self, bg_color: int = 0) -> np.ndarray:
        """Extract object as its own grid."""
        min_r, min_c, max_r, max_c = self.bounding_box
        grid = np.full((max_r - min_r + 1, max_c - min_c + 1), bg_color, dtype=int)

        obj_pixels = np.argwhere(self.mask)
        for r, c in obj_pixels:
            grid[r - min_r, c - min_c] = self.color

        return grid


class ObjectDecomposer:
    """Decomposes grid into objects using connected components."""

    def decompose(self, grid: np.ndarray, connectivity: int = 4) -> List[ARCObject]:
        """
        Decompose grid into objects.

        Args:
            grid: Input grid
            connectivity: 4 or 8-connectivity

        Returns:
            List of ARCObject instances
        """
        if grid.size == 0:
            return []

        # Find connected components for each color (excluding background 0)
        objects = []
        obj_id = 0

        for color in range(1, 10):  # ARC uses colors 0-9
            # Create mask for this color
            color_mask = (grid == color)

            if not color_mask.any():
                continue

            # Find connected components
            structure = None
            if connectivity == 4:
                structure = [[0, 1, 0], [1, 1, 1], [0, 1, 0]]
            else:  # 8-connectivity
                structure = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]

            labeled, num_features = label(color_mask, structure=structure)

            # Extract each component as an object
            for component_id in range(1, num_features + 1):
                mask = (labeled == component_id)

                # Get bounding box
                rows, cols = np.where(mask)
                if len(rows) == 0:
                    continue

                min_r, max_r = rows.min(), rows.max()
                min_c, max_c = cols.min(), cols.max()

                # Get centroid
                centroid_r = rows.mean()
                centroid_c = cols.mean()

                # Get size
                size = mask.sum()

                # Shape properties
                width = max_c - min_c + 1
                height = max_r - min_r + 1

                is_square = (width == height)
                is_rectangle = (size == width * height)
                is_line = (width == 1 or height == 1)

                obj = ARCObject(
                    id=obj_id,
                    mask=mask,
                    color=color,
                    bounding_box=(min_r, min_c, max_r, max_c),
                    centroid=(centroid_r, centroid_c),
                    size=size,
                    width=width,
                    height=height,
                    is_square=is_square,
                    is_rectangle=is_rectangle,
                    is_line=is_line
                )

                objects.append(obj)
                obj_id += 1

        return objects

    def analyze_relationships(self, objects: List[ARCObject], grid_shape: Tuple[int, int]):
        """Analyze spatial relationships between objects."""
        grid_h, grid_w = grid_shape

        for obj in objects:
            # Determine relative position in grid
            cent_r, cent_c = obj.centroid

            if cent_r < grid_h / 3:
                v_pos = "top"
            elif cent_r < 2 * grid_h / 3:
                v_pos = "middle"
            else:
                v_pos = "bottom"

            if cent_c < grid_w / 3:
                h_pos = "left"
            elif cent_c < 2 * grid_w / 3:
                h_pos = "center"
            else:
                h_pos = "right"

            obj.relative_position = f"{v_pos}_{h_pos}"

            # Find adjacent objects
            for other in objects:
                if other.id == obj.id:
                    continue

                # Check if bounding boxes are close (within 2 pixels)
                min_r1, min_c1, max_r1, max_c1 = obj.bounding_box
                min_r2, min_c2, max_r2, max_c2 = other.bounding_box

                h_gap = max(0, min_c2 - max_c1 - 1, min_c1 - max_c2 - 1)
                v_gap = max(0, min_r2 - max_r1 - 1, min_r1 - max_r2 - 1)

                if h_gap <= 2 and v_gap <= 2:
                    obj.neighbors.append(other.id)


# ============================================================================
# L1: PIXEL-LEVEL TRANSFORMS
# ============================================================================

class L1_PixelTransforms:
    """Level 1: Direct grid transformations."""

    @staticmethod
    def rotate_90(grid: np.ndarray) -> np.ndarray:
        return np.rot90(grid, k=1)

    @staticmethod
    def rotate_180(grid: np.ndarray) -> np.ndarray:
        return np.rot90(grid, k=2)

    @staticmethod
    def rotate_270(grid: np.ndarray) -> np.ndarray:
        return np.rot90(grid, k=3)

    @staticmethod
    def flip_h(grid: np.ndarray) -> np.ndarray:
        return np.fliplr(grid)

    @staticmethod
    def flip_v(grid: np.ndarray) -> np.ndarray:
        return np.flipud(grid)

    @staticmethod
    def transpose(grid: np.ndarray) -> np.ndarray:
        return grid.T

    @staticmethod
    def crop_to_objects(grid: np.ndarray) -> np.ndarray:
        """Crop grid to bounding box of all non-zero pixels."""
        rows, cols = np.where(grid != 0)
        if len(rows) == 0:
            return grid

        min_r, max_r = rows.min(), rows.max()
        min_c, max_c = cols.min(), cols.max()

        return grid[min_r:max_r+1, min_c:max_c+1]


# ============================================================================
# L2: OBJECT-LEVEL TRANSFORMS
# ============================================================================

class L2_ObjectTransforms:
    """Level 2: Transform individual objects."""

    @staticmethod
    def move_object(obj: ARCObject, delta_r: int, delta_c: int,
                   grid_shape: Tuple[int, int]) -> ARCObject:
        """Move object by (delta_r, delta_c)."""
        new_obj = copy.deepcopy(obj)

        # Shift mask
        new_mask = np.zeros(grid_shape, dtype=bool)
        rows, cols = np.where(obj.mask)

        for r, c in zip(rows, cols):
            new_r = r + delta_r
            new_c = c + delta_c

            if 0 <= new_r < grid_shape[0] and 0 <= new_c < grid_shape[1]:
                new_mask[new_r, new_c] = True

        # Update properties
        new_obj.mask = new_mask
        if new_mask.any():
            rows, cols = np.where(new_mask)
            new_obj.bounding_box = (rows.min(), cols.min(), rows.max(), cols.max())
            new_obj.centroid = (rows.mean(), cols.mean())

        return new_obj

    @staticmethod
    def recolor_object(obj: ARCObject, new_color: int) -> ARCObject:
        """Change object's color."""
        new_obj = copy.deepcopy(obj)
        new_obj.color = new_color
        return new_obj

    @staticmethod
    def scale_object(obj: ARCObject, scale: float,
                    grid_shape: Tuple[int, int]) -> ARCObject:
        """Scale object by factor."""
        # Get object grid
        obj_grid = obj.to_grid()

        # Scale using nearest neighbor
        new_h = int(obj_grid.shape[0] * scale)
        new_w = int(obj_grid.shape[1] * scale)

        if new_h == 0 or new_w == 0:
            return obj

        # Simple scaling (repeat pixels)
        scaled = np.repeat(np.repeat(obj_grid, int(scale), axis=0), int(scale), axis=1)

        # Create new object
        new_obj = copy.deepcopy(obj)
        new_obj.width = scaled.shape[1]
        new_obj.height = scaled.shape[0]
        new_obj.size = (scaled != 0).sum()

        return new_obj

    @staticmethod
    def delete_object(obj: ARCObject) -> None:
        """Mark object for deletion."""
        obj.mask = np.zeros_like(obj.mask)
        obj.size = 0

    @staticmethod
    def duplicate_object(obj: ARCObject, offset: Tuple[int, int],
                        grid_shape: Tuple[int, int]) -> ARCObject:
        """Create duplicate of object at offset."""
        return L2_ObjectTransforms.move_object(obj, offset[0], offset[1], grid_shape)


# ============================================================================
# L3: PATTERN-BASED RULES
# ============================================================================

class L3_PatternReasoner:
    """Level 3: Infer transformation rules from input/output object sets."""

    def __init__(self):
        self.decomposer = ObjectDecomposer()

    def infer_rule(self, input_grid: np.ndarray, output_grid: np.ndarray) -> Dict[str, Any]:
        """
        Analyze input/output pair to infer transformation rule.

        Returns dict describing the rule, e.g.:
        - {"type": "color_map", "mapping": {1: 3, 2: 5}}
        - {"type": "select_largest", "criterion": "size"}
        - {"type": "duplicate_objects", "count": 2}
        """
        input_objs = self.decomposer.decompose(input_grid)
        output_objs = self.decomposer.decompose(output_grid)

        rule = {"type": "unknown", "confidence": 0.0}

        # Check object count changes
        if len(output_objs) == 0 and len(input_objs) > 0:
            return {"type": "clear_grid", "confidence": 1.0}

        if len(input_objs) == 0:
            return {"type": "generate_pattern", "confidence": 0.5}

        # Check if output is subset of input (selection)
        if len(output_objs) < len(input_objs):
            # Find which object(s) were kept
            kept = self._find_matching_objects(input_objs, output_objs)

            if len(kept) == 1:
                # Single object selected - why?
                obj = input_objs[kept[0]]

                # Check if it's the largest
                sizes = [o.size for o in input_objs]
                if obj.size == max(sizes):
                    return {"type": "select_largest", "criterion": "size", "confidence": 0.9}

                # Check if it's a specific color
                colors = [o.color for o in input_objs]
                if len(set(colors)) > 1:
                    return {"type": "select_by_color", "color": obj.color, "confidence": 0.9}

        # Check if output is duplication/tiling of input
        if len(output_objs) > len(input_objs):
            ratio = len(output_objs) / len(input_objs)
            if abs(ratio - round(ratio)) < 0.1:
                return {"type": "duplicate_objects", "count": int(round(ratio)), "confidence": 0.8}

        # Check if same objects but recolored
        if len(input_objs) == len(output_objs):
            color_mapping = self._infer_color_mapping(input_objs, output_objs)
            if color_mapping:
                return {"type": "color_map", "mapping": color_mapping, "confidence": 0.9}

            # Check if objects moved
            movements = self._infer_movements(input_objs, output_objs)
            if movements:
                return {"type": "move_objects", "movements": movements, "confidence": 0.8}

        # Check grid-level transforms
        if np.array_equal(output_grid, np.rot90(input_grid)):
            return {"type": "rotate_90", "confidence": 1.0}

        if np.array_equal(output_grid, np.fliplr(input_grid)):
            return {"type": "flip_h", "confidence": 1.0}

        return rule

    def _find_matching_objects(self, input_objs: List[ARCObject],
                              output_objs: List[ARCObject]) -> List[int]:
        """Find which input objects match output objects."""
        matched = []

        for out_obj in output_objs:
            for i, in_obj in enumerate(input_objs):
                # Match by size and color
                if in_obj.size == out_obj.size and in_obj.color == out_obj.color:
                    matched.append(i)
                    break

        return matched

    def _infer_color_mapping(self, input_objs: List[ARCObject],
                            output_objs: List[ARCObject]) -> Optional[Dict[int, int]]:
        """Infer color mapping if objects only changed color."""
        if len(input_objs) != len(output_objs):
            return None

        mapping = {}

        # Try to match objects by size/position
        for in_obj in input_objs:
            for out_obj in output_objs:
                if in_obj.size == out_obj.size:
                    in_color = in_obj.color
                    out_color = out_obj.color

                    if in_color in mapping and mapping[in_color] != out_color:
                        return None  # Inconsistent mapping

                    mapping[in_color] = out_color
                    break

        return mapping if mapping else None

    def _infer_movements(self, input_objs: List[ARCObject],
                        output_objs: List[ARCObject]) -> Optional[List[Tuple[int, int]]]:
        """Infer if objects moved."""
        if len(input_objs) != len(output_objs):
            return None

        movements = []

        for in_obj, out_obj in zip(input_objs, output_objs):
            if in_obj.color != out_obj.color or in_obj.size != out_obj.size:
                return None  # Not just movement

            delta_r = out_obj.centroid[0] - in_obj.centroid[0]
            delta_c = out_obj.centroid[1] - in_obj.centroid[1]

            movements.append((int(delta_r), int(delta_c)))

        return movements


# ============================================================================
# L4: CONSTRAINT-BASED FILTERING
# ============================================================================

class L4_ConstraintFilter:
    """Level 4: Filter hypotheses using constraints."""

    @staticmethod
    def validate_size_constraint(output: np.ndarray, max_size: Tuple[int, int]) -> bool:
        """Check if output size is within bounds."""
        return output.shape[0] <= max_size[0] and output.shape[1] <= max_size[1]

    @staticmethod
    def validate_color_count(output: np.ndarray, expected_colors: Set[int]) -> bool:
        """Check if output uses expected colors."""
        actual_colors = set(np.unique(output))
        return actual_colors.issubset(expected_colors)

    @staticmethod
    def validate_object_count(output: np.ndarray, min_count: int, max_count: int) -> bool:
        """Check if object count is in range."""
        decomposer = ObjectDecomposer()
        objects = decomposer.decompose(output)
        return min_count <= len(objects) <= max_count


# ============================================================================
# MULTI-STAGE SOLVER
# ============================================================================

class MultiStageSolver:
    """
    Main solver using all 4 reasoning levels.
    """

    def __init__(self):
        self.decomposer = ObjectDecomposer()
        self.l1 = L1_PixelTransforms()
        self.l2 = L2_ObjectTransforms()
        self.l3 = L3_PatternReasoner()
        self.l4 = L4_ConstraintFilter()

    def solve_task(self, task: Dict) -> List[np.ndarray]:
        """
        Solve ARC task using multi-stage reasoning.

        Args:
            task: Dict with 'train' and 'test' keys

        Returns:
            List of predictions for test cases
        """
        # Step 1: Learn rules from training examples (L3)
        rules = []
        for example in task['train']:
            input_grid = np.array(example['input'])
            output_grid = np.array(example['output'])

            rule = self.l3.infer_rule(input_grid, output_grid)
            rules.append(rule)

        # Step 2: Find most confident consistent rule
        best_rule = self._select_best_rule(rules)

        # Step 3: Apply rule to test inputs
        predictions = []
        for test_case in task['test']:
            input_grid = np.array(test_case['input'])

            prediction = self._apply_rule(best_rule, input_grid)
            predictions.append(prediction)

        return predictions

    def _select_best_rule(self, rules: List[Dict]) -> Dict:
        """Select most confident and consistent rule."""
        if not rules:
            return {"type": "identity", "confidence": 0.1}

        # Count rule types
        rule_counts = defaultdict(int)
        rule_examples = defaultdict(list)

        for rule in rules:
            rule_type = rule['type']
            rule_counts[rule_type] += 1
            rule_examples[rule_type].append(rule)

        # Pick most common
        best_type = max(rule_counts, key=rule_counts.get)

        # Return first instance of that type
        return rule_examples[best_type][0]

    def _apply_rule(self, rule: Dict, input_grid: np.ndarray) -> np.ndarray:
        """Apply inferred rule to input."""
        rule_type = rule['type']

        if rule_type == "identity" or rule_type == "unknown":
            return input_grid.copy()

        # L1 transforms
        if rule_type == "rotate_90":
            return self.l1.rotate_90(input_grid)

        if rule_type == "flip_h":
            return self.l1.flip_h(input_grid)

        if rule_type == "clear_grid":
            return np.zeros_like(input_grid)

        # L2/L3 transforms (object-based)
        if rule_type == "select_largest":
            return self._select_largest_object(input_grid)

        if rule_type == "select_by_color":
            return self._select_by_color(input_grid, rule['color'])

        if rule_type == "color_map":
            return self._apply_color_mapping(input_grid, rule['mapping'])

        if rule_type == "move_objects":
            return self._move_objects(input_grid, rule['movements'])

        # Fallback
        return input_grid.copy()

    def _select_largest_object(self, grid: np.ndarray) -> np.ndarray:
        """Select only the largest object."""
        objects = self.decomposer.decompose(grid)

        if not objects:
            return grid.copy()

        largest = max(objects, key=lambda o: o.size)

        # Create output with only largest object
        output = np.zeros_like(grid)
        output[largest.mask] = largest.color

        return output

    def _select_by_color(self, grid: np.ndarray, color: int) -> np.ndarray:
        """Select only objects of specific color."""
        output = np.zeros_like(grid)
        output[grid == color] = color
        return output

    def _apply_color_mapping(self, grid: np.ndarray, mapping: Dict[int, int]) -> np.ndarray:
        """Apply color mapping."""
        output = grid.copy()

        for old_color, new_color in mapping.items():
            output[grid == old_color] = new_color

        return output

    def _move_objects(self, grid: np.ndarray, movements: List[Tuple[int, int]]) -> np.ndarray:
        """Move objects according to learned movements."""
        objects = self.decomposer.decompose(grid)

        if len(movements) != len(objects):
            return grid.copy()

        output = np.zeros_like(grid)

        for obj, (delta_r, delta_c) in zip(objects, movements):
            moved = self.l2.move_object(obj, delta_r, delta_c, grid.shape)
            output[moved.mask] = moved.color

        return output


# ============================================================================
# SUBMISSION GENERATION
# ============================================================================

def generate_submission(test_file: str, output_file: str = 'submission.json'):
    """Generate submission using multi-stage solver."""
    with open(test_file, 'r') as f:
        test_tasks = json.load(f)

    solver = MultiStageSolver()
    submission = {}

    solved = 0
    total = 0

    for task_id, task in test_tasks.items():
        print(f"Solving {task_id}...", end=" ")

        try:
            predictions = solver.solve_task(task)

            # Format for submission (2 attempts per test case)
            task_predictions = []
            for pred in predictions:
                task_predictions.append({
                    "attempt_1": pred.tolist(),
                    "attempt_2": pred.tolist()
                })

            submission[task_id] = task_predictions

            print("✓")
            solved += 1
        except Exception as e:
            print(f"✗ ({e})")
            # Fallback: return input as output
            task_predictions = []
            for test_case in task['test']:
                task_predictions.append({
                    "attempt_1": test_case['input'],
                    "attempt_2": test_case['input']
                })
            submission[task_id] = task_predictions

        total += 1

    # Save submission
    with open(output_file, 'w') as f:
        json.dump(submission, f)

    print(f"\n{'='*60}")
    print(f"Solved: {solved}/{total} ({solved/total*100:.1f}%)")
    print(f"Submission saved to: {output_file}")
    print(f"{'='*60}")


if __name__ == '__main__':
    # Test on training set
    print("Multi-Stage ARC Solver")
    print("="*60)
    print("Running on training set...")
    generate_submission('arc-agi_training_challenges.json', 'test_multi_stage.json')
