"""
Iteration 4: Rule Induction Solver
Learn transformation rules from training examples, apply to test cases.

Based on: SubtleGenius/docs/RULE_INDUCTION_SOLVER.md
Expected Coverage: 15-25% of tasks
Expected Accuracy: 70-85% (when rules detected)
Total Contribution: 10-20% overall accuracy
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Callable, Any
from dataclasses import dataclass
from collections import Counter


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def shape(grid: List[List[int]]) -> Tuple[int, int]:
    """Get shape of grid (height, width)"""
    if not grid:
        return (0, 0)
    return (len(grid), len(grid[0]) if grid[0] else 0)


def flatten(grid: List[List[int]]) -> List[int]:
    """Flatten 2D grid to 1D list"""
    return [val for row in grid for val in row]


def is_valid_grid(grid: Any) -> bool:
    """Check if grid is valid ARC format"""
    if not isinstance(grid, list) or not grid:
        return False
    if not all(isinstance(row, list) for row in grid):
        return False
    if not grid[0]:
        return False
    width = len(grid[0])
    if not all(len(row) == width for row in grid):
        return False
    if not all(all(isinstance(v, (int, np.integer)) and 0 <= v <= 9 for v in row) for row in grid):
        return False
    return True


def convert_to_python_list(arr: Any) -> List[List[int]]:
    """Convert numpy array or nested structure to pure Python list"""
    if isinstance(arr, np.ndarray):
        return arr.tolist()
    if isinstance(arr, list):
        return [[int(v) for v in row] for row in arr]
    return arr


# ============================================================================
# RULE TYPE 1: COLOR MAPPING RULES (10-15% coverage)
# ============================================================================

def detect_color_mapping_rule(train_pairs: List[Dict]) -> Optional[Dict]:
    """
    Detect if transformation is: color X → color Y

    Returns: {'type': 'color_mapping', 'mapping': {old: new}} or None
    """
    if not train_pairs:
        return None

    try:
        # Extract mapping from first pair
        inp = train_pairs[0]['input']
        out = train_pairs[0]['output']

        # Must be same shape
        if shape(inp) != shape(out):
            return None

        # Build mapping
        mapping = {}
        for (in_val, out_val) in zip(flatten(inp), flatten(out)):
            if in_val in mapping:
                if mapping[in_val] != out_val:
                    return None  # Inconsistent mapping
            else:
                mapping[in_val] = out_val

        # Must actually change something
        if all(k == v for k, v in mapping.items()):
            return None

        # Verify mapping on all other training pairs
        for pair in train_pairs[1:]:
            inp = pair['input']
            out = pair['output']

            if shape(inp) != shape(out):
                return None

            for (in_val, out_val) in zip(flatten(inp), flatten(out)):
                expected = mapping.get(in_val, in_val)
                if expected != out_val:
                    return None  # Mapping doesn't hold

        return {
            'type': 'color_mapping',
            'mapping': mapping,
            'confidence': 0.90
        }

    except Exception:
        return None


def apply_color_mapping(grid: List[List[int]], mapping: Dict[int, int]) -> List[List[int]]:
    """Apply color mapping to grid"""
    result = []
    for row in grid:
        new_row = [mapping.get(val, val) for val in row]
        result.append(new_row)
    return result


# ============================================================================
# RULE TYPE 2: SIZE/SHAPE RULES (3-5% coverage)
# ============================================================================

def detect_size_rules(train_pairs: List[Dict]) -> Optional[Dict]:
    """
    Detect rules about grid size changes:
    - Output size = input size × N
    - Output size = fixed size regardless of input
    """
    if not train_pairs:
        return None

    try:
        # Check if output size is related to input size by consistent ratio
        ratios = []
        for pair in train_pairs:
            inp_h, inp_w = shape(pair['input'])
            out_h, out_w = shape(pair['output'])

            if inp_h == 0 or inp_w == 0:
                return None

            h_ratio = out_h / inp_h
            w_ratio = out_w / inp_w

            ratios.append((h_ratio, w_ratio))

        # All ratios must be identical
        if not all(r == ratios[0] for r in ratios):
            return None

        h_ratio, w_ratio = ratios[0]

        # Must be integer ratios and not 1:1 (that's identity)
        if h_ratio != int(h_ratio) or w_ratio != int(w_ratio):
            return None
        if h_ratio == 1.0 and w_ratio == 1.0:
            return None

        # For simple scaling, content should repeat
        # Check first pair to see if it's a simple tile/repeat
        inp = np.array(train_pairs[0]['input'])
        out = np.array(train_pairs[0]['output'])

        h_scale = int(h_ratio)
        w_scale = int(w_ratio)

        # Test if output is input tiled
        expected = np.tile(inp, (h_scale, w_scale))
        if expected.shape == out.shape and np.array_equal(expected, out):
            # Verify on all pairs
            for pair in train_pairs[1:]:
                inp = np.array(pair['input'])
                out = np.array(pair['output'])
                expected = np.tile(inp, (h_scale, w_scale))
                if not np.array_equal(expected, out):
                    return None

            return {
                'type': 'size_tile',
                'h_scale': h_scale,
                'w_scale': w_scale,
                'confidence': 0.95
            }

    except Exception:
        pass

    return None


def apply_size_tile(grid: List[List[int]], h_scale: int, w_scale: int) -> List[List[int]]:
    """Tile grid by h_scale × w_scale"""
    arr = np.array(grid)
    result = np.tile(arr, (h_scale, w_scale))
    return convert_to_python_list(result)


# ============================================================================
# RULE TYPE 3: CONDITIONAL RULES (5-8% coverage)
# ============================================================================

def detect_conditional_rule(train_pairs: List[Dict]) -> Optional[Dict]:
    """
    Detect conditional rules like:
    - If rare color appears → isolate only that color
    - If majority color exists → fill entire grid with it
    - If background is X → change to Y
    """
    if not train_pairs:
        return None

    try:
        # Rule: "Keep only rare colors (appearing in <20% of cells)"
        rare_color_rule = detect_rare_color_isolation(train_pairs)
        if rare_color_rule:
            return rare_color_rule

        # Rule: "Fill with majority color"
        majority_rule = detect_majority_fill(train_pairs)
        if majority_rule:
            return majority_rule

        # Rule: "Background color change"
        background_rule = detect_background_change(train_pairs)
        if background_rule:
            return background_rule

    except Exception:
        pass

    return None


def detect_rare_color_isolation(train_pairs: List[Dict]) -> Optional[Dict]:
    """Detect if rule is: keep only rare colors, zero out common ones"""
    try:
        for pair in train_pairs:
            inp = np.array(pair['input'])
            out = np.array(pair['output'])

            if inp.shape != out.shape:
                return None

            # Find color frequencies in input
            flat_inp = inp.flatten()
            color_counts = Counter(flat_inp)
            total_cells = len(flat_inp)

            # Identify rare colors (< 20% of cells)
            rare_colors = {c for c, count in color_counts.items() if count / total_cells < 0.2}

            if not rare_colors:
                return None

            # Check if output keeps rare colors and zeros common ones
            for i in range(inp.shape[0]):
                for j in range(inp.shape[1]):
                    in_val = inp[i, j]
                    out_val = out[i, j]

                    if in_val in rare_colors:
                        if out_val != in_val:
                            return None
                    else:
                        if out_val != 0:
                            return None

        return {
            'type': 'rare_color_isolation',
            'threshold': 0.2,
            'confidence': 0.75
        }

    except Exception:
        return None


def detect_majority_fill(train_pairs: List[Dict]) -> Optional[Dict]:
    """Detect if rule is: fill entire grid with majority color"""
    try:
        for pair in train_pairs:
            inp = np.array(pair['input'])
            out = np.array(pair['output'])

            if inp.shape != out.shape:
                return None

            # Find majority color in input
            color_counts = Counter(inp.flatten())
            majority_color = color_counts.most_common(1)[0][0]

            # Check if output is filled with majority color
            if not np.all(out == majority_color):
                return None

        return {
            'type': 'majority_fill',
            'confidence': 0.80
        }

    except Exception:
        return None


def detect_background_change(train_pairs: List[Dict]) -> Optional[Dict]:
    """Detect if rule is: change background color X to Y"""
    try:
        # Background is assumed to be most common color
        inp0 = np.array(train_pairs[0]['input'])
        out0 = np.array(train_pairs[0]['output'])

        if inp0.shape != out0.shape:
            return None

        # Find background color (most common)
        bg_color_in = Counter(inp0.flatten()).most_common(1)[0][0]
        bg_color_out = Counter(out0.flatten()).most_common(1)[0][0]

        if bg_color_in == bg_color_out:
            return None

        # Verify: all bg pixels change, all non-bg pixels stay same
        for pair in train_pairs:
            inp = np.array(pair['input'])
            out = np.array(pair['output'])

            if inp.shape != out.shape:
                return None

            for i in range(inp.shape[0]):
                for j in range(inp.shape[1]):
                    if inp[i, j] == bg_color_in:
                        if out[i, j] != bg_color_out:
                            return None
                    else:
                        if out[i, j] != inp[i, j]:
                            return None

        return {
            'type': 'background_change',
            'old_bg': int(bg_color_in),
            'new_bg': int(bg_color_out),
            'confidence': 0.85
        }

    except Exception:
        return None


def apply_rare_color_isolation(grid: List[List[int]], threshold: float = 0.2) -> List[List[int]]:
    """Keep only rare colors, zero out common ones"""
    arr = np.array(grid)
    color_counts = Counter(arr.flatten())
    total_cells = arr.size

    rare_colors = {c for c, count in color_counts.items() if count / total_cells < threshold}

    result = np.zeros_like(arr)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if arr[i, j] in rare_colors:
                result[i, j] = arr[i, j]

    return convert_to_python_list(result)


def apply_majority_fill(grid: List[List[int]]) -> List[List[int]]:
    """Fill entire grid with majority color"""
    arr = np.array(grid)
    majority_color = Counter(arr.flatten()).most_common(1)[0][0]
    result = np.full_like(arr, majority_color)
    return convert_to_python_list(result)


def apply_background_change(grid: List[List[int]], old_bg: int, new_bg: int) -> List[List[int]]:
    """Change background color from old_bg to new_bg"""
    result = []
    for row in grid:
        new_row = [new_bg if val == old_bg else val for val in row]
        result.append(new_row)
    return result


# ============================================================================
# MAIN RULE INDUCTION SOLVER
# ============================================================================

@dataclass
class DetectedRule:
    """Container for a detected rule"""
    rule_type: str
    apply_func: Callable
    confidence: float
    params: Dict[str, Any]


class RuleInductionSolver:
    """
    Learn transformation rules from training examples, apply to test cases.

    This is the HIGHEST PRIORITY layer in the cascading architecture because
    rules are learned from THIS SPECIFIC TASK's training examples.
    """

    def __init__(self):
        self.rule_detectors = [
            ('color_mapping', detect_color_mapping_rule),
            ('size_rules', detect_size_rules),
            ('conditional_rules', detect_conditional_rule),
        ]

    def detect_rules(self, train_pairs: List[Dict]) -> List[DetectedRule]:
        """Try all rule detectors, return detected rules"""
        detected_rules = []

        for detector_name, detector_func in self.rule_detectors:
            try:
                rule_data = detector_func(train_pairs)

                if rule_data is not None:
                    # Map rule type to apply function
                    rule_type = rule_data['type']
                    confidence = rule_data.get('confidence', 0.70)

                    apply_func = self._get_apply_function(rule_type, rule_data)

                    if apply_func:
                        detected_rules.append(DetectedRule(
                            rule_type=rule_type,
                            apply_func=apply_func,
                            confidence=confidence,
                            params=rule_data
                        ))

            except Exception:
                continue

        return detected_rules

    def _get_apply_function(self, rule_type: str, rule_data: Dict) -> Optional[Callable]:
        """Get the appropriate apply function for a rule type"""
        if rule_type == 'color_mapping':
            mapping = rule_data['mapping']
            return lambda grid: apply_color_mapping(grid, mapping)

        elif rule_type == 'size_tile':
            h_scale = rule_data['h_scale']
            w_scale = rule_data['w_scale']
            return lambda grid: apply_size_tile(grid, h_scale, w_scale)

        elif rule_type == 'rare_color_isolation':
            threshold = rule_data.get('threshold', 0.2)
            return lambda grid: apply_rare_color_isolation(grid, threshold)

        elif rule_type == 'majority_fill':
            return apply_majority_fill

        elif rule_type == 'background_change':
            old_bg = rule_data['old_bg']
            new_bg = rule_data['new_bg']
            return lambda grid: apply_background_change(grid, old_bg, new_bg)

        return None

    def solve(self, test_input: List[List[int]], task_data: Dict, attempt: int = 1) -> Optional[Tuple]:
        """
        Main solving function

        Returns: (result_grid, confidence, solver_name) or None
        """
        train_pairs = task_data.get('train', [])

        if not train_pairs:
            return None

        # Detect rules from training examples
        rules = self.detect_rules(train_pairs)

        if not rules:
            return None

        # Sort by confidence (highest first)
        rules.sort(key=lambda r: r.confidence, reverse=True)

        # For attempt 1, use highest confidence rule
        # For attempt 2, use second-best rule (if exists)
        if attempt == 2 and len(rules) > 1:
            rule = rules[1]
        else:
            rule = rules[0]

        try:
            result = rule.apply_func(test_input)

            if is_valid_grid(result):
                return (result, rule.confidence, f"rule_{rule.rule_type}")

        except Exception:
            pass

        return None


# ============================================================================
# INTEGRATION FUNCTION
# ============================================================================

def apply_rule_induction(test_input: List[List[int]],
                        task_data: Dict,
                        attempt: int = 1) -> Optional[Tuple[List[List[int]], float, str]]:
    """
    Apply rule induction solver (for integration into ensemble)

    Returns: (grid, confidence, solver_name) or None
    """
    solver = RuleInductionSolver()
    result = solver.solve(test_input, task_data, attempt)
    return result


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    print("Rule Induction Solver - Iteration 4")
    print("=" * 60)

    # Test 1: Color Mapping Rule
    print("\n[Test 1: Color Mapping Rule]")
    task_data = {
        'train': [
            {
                'input': [[1, 1, 2], [1, 2, 2]],
                'output': [[3, 3, 4], [3, 4, 4]]
            },
            {
                'input': [[2, 1, 1], [2, 2, 1]],
                'output': [[4, 3, 3], [4, 4, 3]]
            }
        ]
    }

    solver = RuleInductionSolver()
    rules = solver.detect_rules(task_data['train'])
    print(f"Detected {len(rules)} rule(s):")
    for rule in rules:
        print(f"  - {rule.rule_type} (confidence: {rule.confidence:.2f})")

    test_input = [[1, 2, 1], [2, 2, 2]]
    result = solver.solve(test_input, task_data, attempt=1)
    if result:
        grid, conf, name = result
        print(f"Applied {name} → {grid}")
        expected = [[3, 4, 3], [4, 4, 4]]
        print(f"Expected: {expected}")
        print(f"Match: {grid == expected}")

    # Test 2: Size Tile Rule
    print("\n[Test 2: Size Tile Rule]")
    task_data = {
        'train': [
            {
                'input': [[1, 2]],
                'output': [[1, 2], [1, 2]]
            },
            {
                'input': [[3, 4]],
                'output': [[3, 4], [3, 4]]
            }
        ]
    }

    rules = solver.detect_rules(task_data['train'])
    print(f"Detected {len(rules)} rule(s):")
    for rule in rules:
        print(f"  - {rule.rule_type} (confidence: {rule.confidence:.2f})")

    test_input = [[5, 6]]
    result = solver.solve(test_input, task_data, attempt=1)
    if result:
        grid, conf, name = result
        print(f"Applied {name} → {grid}")
        expected = [[5, 6], [5, 6]]
        print(f"Expected: {expected}")
        print(f"Match: {grid == expected}")

    # Test 3: Background Change Rule
    print("\n[Test 3: Background Change Rule]")
    task_data = {
        'train': [
            {
                'input': [[0, 0, 1], [0, 0, 0]],
                'output': [[5, 5, 1], [5, 5, 5]]
            },
            {
                'input': [[0, 2, 0], [0, 0, 0]],
                'output': [[5, 2, 5], [5, 5, 5]]
            }
        ]
    }

    rules = solver.detect_rules(task_data['train'])
    print(f"Detected {len(rules)} rule(s):")
    for rule in rules:
        print(f"  - {rule.rule_type} (confidence: {rule.confidence:.2f})")

    test_input = [[0, 0, 3], [0, 3, 0]]
    result = solver.solve(test_input, task_data, attempt=1)
    if result:
        grid, conf, name = result
        print(f"Applied {name} → {grid}")
        expected = [[5, 5, 3], [5, 3, 5]]
        print(f"Expected: {expected}")
        print(f"Match: {grid == expected}")

    print("\n" + "=" * 60)
    print("All tests complete!")
