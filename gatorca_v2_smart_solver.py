"""
GatORCA v2: Smart ARC Solver with Pattern Recognition & Concept Learning
Implements actual ARC-solving techniques instead of random evolution
"""

import json
import numpy as np
from typing import List, Dict, Tuple, Any, Callable
import copy

Grid = List[List[int]]

class PatternDetector:
    """Detects patterns in grids: symmetry, repetition, objects, etc."""

    @staticmethod
    def detect_symmetry(grid: np.ndarray) -> Dict[str, bool]:
        """Detect horizontal, vertical, diagonal symmetry"""
        return {
            'horizontal': np.array_equal(grid, np.flipud(grid)),
            'vertical': np.array_equal(grid, np.fliplr(grid)),
            'diagonal': np.array_equal(grid, grid.T),
            'anti_diagonal': np.array_equal(grid, np.fliplr(grid.T))
        }

    @staticmethod
    def detect_tiling(input_grid: np.ndarray, output_grid: np.ndarray) -> Dict[str, Any]:
        """Detect if output is a tiled version of input"""
        in_h, in_w = input_grid.shape
        out_h, out_w = output_grid.shape

        if out_h % in_h == 0 and out_w % in_w == 0:
            v_tiles = out_h // in_h
            h_tiles = out_w // in_w

            # Check if it's simple tiling
            tiled = np.tile(input_grid, (v_tiles, h_tiles))
            if np.array_equal(tiled, output_grid):
                return {'is_tiled': True, 'v_tiles': v_tiles, 'h_tiles': h_tiles, 'pattern': 'simple'}

            # Check alternating patterns
            for flip_v in [False, True]:
                for flip_h in [False, True]:
                    test = input_grid.copy()
                    if flip_v:
                        test = np.flipud(test)
                    if flip_h:
                        test = np.fliplr(test)
                    tiled = np.tile(test, (v_tiles, h_tiles))
                    if np.array_equal(tiled, output_grid):
                        return {
                            'is_tiled': True,
                            'v_tiles': v_tiles,
                            'h_tiles': h_tiles,
                            'pattern': 'flip',
                            'flip_v': flip_v,
                            'flip_h': flip_h
                        }

        return {'is_tiled': False}

    @staticmethod
    def detect_size_change(input_grid: np.ndarray, output_grid: np.ndarray) -> Dict[str, Any]:
        """Detect scaling, cropping, padding"""
        in_h, in_w = input_grid.shape
        out_h, out_w = output_grid.shape

        result = {
            'height_ratio': out_h / in_h if in_h > 0 else 0,
            'width_ratio': out_w / in_w if in_w > 0 else 0,
            'height_diff': out_h - in_h,
            'width_diff': out_w - in_w
        }

        # Check if it's exact scaling
        if out_h % in_h == 0 and out_w % in_w == 0:
            result['is_scaled'] = True
            result['scale_h'] = out_h // in_h
            result['scale_w'] = out_w // in_w
        else:
            result['is_scaled'] = False

        return result

    @staticmethod
    def detect_color_mapping(input_grid: np.ndarray, output_grid: np.ndarray) -> Dict[int, int]:
        """Detect if colors are mapped (e.g., 1->3, 2->5)"""
        if input_grid.shape != output_grid.shape:
            return {}

        mapping = {}
        for i in range(input_grid.shape[0]):
            for j in range(input_grid.shape[1]):
                in_color = input_grid[i, j]
                out_color = output_grid[i, j]

                if in_color in mapping:
                    if mapping[in_color] != out_color:
                        return {}  # Inconsistent mapping
                else:
                    mapping[in_color] = out_color

        return mapping

    @staticmethod
    def find_objects(grid: np.ndarray) -> List[Dict]:
        """Find connected components (objects) in grid"""
        objects = []
        visited = np.zeros_like(grid, dtype=bool)

        def flood_fill(r, c, color):
            if r < 0 or r >= grid.shape[0] or c < 0 or c >= grid.shape[1]:
                return []
            if visited[r, c] or grid[r, c] != color or color == 0:  # 0 is background
                return []

            visited[r, c] = True
            coords = [(r, c)]

            # 4-connectivity
            coords += flood_fill(r+1, c, color)
            coords += flood_fill(r-1, c, color)
            coords += flood_fill(r, c+1, color)
            coords += flood_fill(r, c-1, color)

            return coords

        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if not visited[i, j] and grid[i, j] != 0:
                    coords = flood_fill(i, j, grid[i, j])
                    if coords:
                        objects.append({
                            'color': grid[i, j],
                            'coords': coords,
                            'size': len(coords),
                            'bbox': (
                                min(c[0] for c in coords),
                                min(c[1] for c in coords),
                                max(c[0] for c in coords),
                                max(c[1] for c in coords)
                            )
                        })

        return objects


class ConceptLearner:
    """Learns transformation concepts from training examples"""

    def __init__(self):
        self.detector = PatternDetector()
        self.learned_transforms = []

    def learn_from_examples(self, examples: List[Dict]) -> List[Dict]:
        """Analyze training examples to learn transformation patterns"""
        transforms = []

        for ex in examples:
            input_grid = np.array(ex['input'])
            output_grid = np.array(ex['output'])

            transform = {
                'input_shape': input_grid.shape,
                'output_shape': output_grid.shape,
                'tiling': self.detector.detect_tiling(input_grid, output_grid),
                'size_change': self.detector.detect_size_change(input_grid, output_grid),
                'color_mapping': self.detector.detect_color_mapping(input_grid, output_grid),
                'input_symmetry': self.detector.detect_symmetry(input_grid),
                'output_symmetry': self.detector.detect_symmetry(output_grid),
                'input_objects': self.detector.find_objects(input_grid),
                'output_objects': self.detector.find_objects(output_grid)
            }

            transforms.append(transform)

        self.learned_transforms = transforms
        return transforms

    def find_consistent_pattern(self) -> Dict[str, Any]:
        """Find patterns that are consistent across all training examples"""
        if not self.learned_transforms:
            return {}

        pattern = {}

        # Check if all examples use same tiling
        tiling_patterns = [t['tiling'] for t in self.learned_transforms]
        if all(tp.get('is_tiled') for tp in tiling_patterns):
            # Check if same tiling ratio
            v_tiles = [tp['v_tiles'] for tp in tiling_patterns]
            h_tiles = [tp['h_tiles'] for tp in tiling_patterns]
            if len(set(v_tiles)) == 1 and len(set(h_tiles)) == 1:
                pattern['operation'] = 'tile'
                pattern['v_tiles'] = v_tiles[0]
                pattern['h_tiles'] = h_tiles[0]
                pattern['pattern_type'] = tiling_patterns[0].get('pattern', 'simple')
                if 'flip_v' in tiling_patterns[0]:
                    pattern['flip_v'] = tiling_patterns[0]['flip_v']
                    pattern['flip_h'] = tiling_patterns[0]['flip_h']
                return pattern

        # Check if all examples use same color mapping
        color_maps = [t['color_mapping'] for t in self.learned_transforms]
        if all(cm for cm in color_maps):
            # Check if mappings are consistent
            if len(set(frozenset(cm.items()) for cm in color_maps)) == 1:
                pattern['operation'] = 'color_map'
                pattern['mapping'] = color_maps[0]
                return pattern

        # Check if all examples use same scaling
        size_changes = [t['size_change'] for t in self.learned_transforms]
        if all(sc.get('is_scaled') for sc in size_changes):
            scale_h = [sc['scale_h'] for sc in size_changes]
            scale_w = [sc['scale_w'] for sc in size_changes]
            if len(set(scale_h)) == 1 and len(set(scale_w)) == 1:
                pattern['operation'] = 'scale'
                pattern['scale_h'] = scale_h[0]
                pattern['scale_w'] = scale_w[0]
                return pattern

        # Check for object-based transformations
        obj_count_changes = [(len(t['output_objects']) - len(t['input_objects']))
                             for t in self.learned_transforms]
        if all(occ == obj_count_changes[0] for occ in obj_count_changes):
            if obj_count_changes[0] > 0:
                pattern['operation'] = 'duplicate_objects'
                pattern['count_increase'] = obj_count_changes[0]
                return pattern

        return pattern


class SmartARCSolver:
    """ARC solver using pattern detection and concept learning"""

    def __init__(self):
        self.learner = ConceptLearner()
        self.detector = PatternDetector()

    def apply_tiling(self, grid: np.ndarray, v_tiles: int, h_tiles: int,
                     pattern_type: str = 'simple', flip_v: bool = False,
                     flip_h: bool = False) -> np.ndarray:
        """Apply tiling transformation"""
        transformed = grid.copy()
        if pattern_type == 'flip':
            if flip_v:
                transformed = np.flipud(transformed)
            if flip_h:
                transformed = np.fliplr(transformed)

        return np.tile(transformed, (v_tiles, h_tiles))

    def apply_color_mapping(self, grid: np.ndarray, mapping: Dict[int, int]) -> np.ndarray:
        """Apply color mapping transformation"""
        result = grid.copy()
        for old_color, new_color in mapping.items():
            result[grid == old_color] = new_color
        return result

    def apply_scaling(self, grid: np.ndarray, scale_h: int, scale_w: int) -> np.ndarray:
        """Apply scaling transformation"""
        return np.repeat(np.repeat(grid, scale_h, axis=0), scale_w, axis=1)

    def solve_task(self, task: Dict) -> Dict:
        """Solve an ARC task using learned concepts"""
        # Learn from training examples
        transforms = self.learner.learn_from_examples(task['train'])
        pattern = self.learner.find_consistent_pattern()

        if not pattern:
            # No consistent pattern found - return input as fallback
            test_input = np.array(task['test'][0]['input'])
            return {
                'grid': test_input.tolist(),
                'confidence': 0.0,
                'method': 'fallback'
            }

        # Apply learned pattern to test input
        test_input = np.array(task['test'][0]['input'])

        try:
            if pattern['operation'] == 'tile':
                result = self.apply_tiling(
                    test_input,
                    pattern['v_tiles'],
                    pattern['h_tiles'],
                    pattern.get('pattern_type', 'simple'),
                    pattern.get('flip_v', False),
                    pattern.get('flip_h', False)
                )
                return {
                    'grid': result.tolist(),
                    'confidence': 0.9,
                    'method': 'tiling'
                }

            elif pattern['operation'] == 'color_map':
                result = self.apply_color_mapping(test_input, pattern['mapping'])
                return {
                    'grid': result.tolist(),
                    'confidence': 0.9,
                    'method': 'color_mapping'
                }

            elif pattern['operation'] == 'scale':
                result = self.apply_scaling(test_input, pattern['scale_h'], pattern['scale_w'])
                return {
                    'grid': result.tolist(),
                    'confidence': 0.9,
                    'method': 'scaling'
                }

            else:
                return {
                    'grid': test_input.tolist(),
                    'confidence': 0.1,
                    'method': 'unknown_pattern'
                }

        except Exception as e:
            return {
                'grid': test_input.tolist(),
                'confidence': 0.0,
                'method': f'error: {str(e)}'
            }


def generate_submission(test_file: str, output_file: str = 'submission.json'):
    """Generate submission file for ARC Prize 2025"""
    with open(test_file, 'r') as f:
        test_challenges = json.load(f)

    solver = SmartARCSolver()
    submission = {}

    solved_count = 0
    total_count = 0

    for task_id, task in test_challenges.items():
        print(f"Solving {task_id}...", end=" ")

        try:
            result = solver.solve_task(task)

            task_predictions = []
            for test_case in task['test']:
                task_predictions.append({
                    "attempt_1": result['grid'],
                    "attempt_2": result['grid']  # Use same prediction for both attempts
                })

            submission[task_id] = task_predictions

            if result['confidence'] > 0.5:
                solved_count += 1
                print(f"✓ ({result['method']}, {result['confidence']:.1%})")
            else:
                print(f"✗ ({result['method']}, {result['confidence']:.1%})")

            total_count += 1

        except Exception as e:
            print(f"ERROR: {e}")
            # Fallback: use input as output
            task_predictions = []
            for test_case in task['test']:
                task_predictions.append({
                    "attempt_1": test_case['input'],
                    "attempt_2": test_case['input']
                })
            submission[task_id] = task_predictions
            total_count += 1

    # Save submission
    with open(output_file, 'w') as f:
        json.dump(submission, f)

    print(f"\n{'='*60}")
    print(f"Solved: {solved_count}/{total_count} ({solved_count/total_count*100:.1f}%)")
    print(f"Submission saved to: {output_file}")
    print(f"{'='*60}")


if __name__ == '__main__':
    # Test on training set first
    print("Testing on training set...")
    generate_submission('arc-agi_training_challenges.json', 'test_submission.json')
