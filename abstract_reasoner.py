"""
🧠 Abstract Reasoning Module for ARC

Instead of brute-force, UNDERSTAND the transformation:
1. Analyze what CHANGED between input→output
2. Formulate HYPOTHESIS about the rule
3. Test hypothesis on all training examples
4. Apply the understood rule to test input

Author: Ryan Cardwell & Claude
Date: November 2025
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import Counter


class AbstractReasoner:
    """Reasons about transformations instead of brute-forcing"""

    def analyze_transformation(self, input_grid: np.ndarray, output_grid: np.ndarray) -> Dict:
        """
        Analyze what changed from input to output

        Returns dict describing the transformation
        """
        analysis = {
            'shape_changed': input_grid.shape != output_grid.shape,
            'colors_changed': not np.array_equal(np.unique(input_grid), np.unique(output_grid)),
            'same_shape': input_grid.shape == output_grid.shape,
            'size_ratio': None,
            'transformation_type': None
        }

        # Detect shape transformations
        if input_grid.shape == output_grid.shape:
            # Same shape - could be color mapping, rotation, etc.
            if np.array_equal(input_grid, output_grid):
                analysis['transformation_type'] = 'identity'
            elif np.array_equal(np.rot90(input_grid), output_grid):
                analysis['transformation_type'] = 'rotate_90'
            elif np.array_equal(np.rot90(input_grid, 2), output_grid):
                analysis['transformation_type'] = 'rotate_180'
            elif np.array_equal(np.rot90(input_grid, 3), output_grid):
                analysis['transformation_type'] = 'rotate_270'
            elif np.array_equal(np.fliplr(input_grid), output_grid):
                analysis['transformation_type'] = 'flip_horizontal'
            elif np.array_equal(np.flipud(input_grid), output_grid):
                analysis['transformation_type'] = 'flip_vertical'
            elif input_grid.shape[0] == input_grid.shape[1] and np.array_equal(input_grid.T, output_grid):
                analysis['transformation_type'] = 'transpose'
            else:
                # Check for color mapping
                color_map = self._detect_color_mapping(input_grid, output_grid)
                if color_map:
                    analysis['transformation_type'] = 'color_mapping'
                    analysis['color_map'] = color_map
        else:
            # Shape changed - cropping, scaling, tiling, etc.
            in_h, in_w = input_grid.shape
            out_h, out_w = output_grid.shape

            if out_h < in_h or out_w < in_w:
                analysis['transformation_type'] = 'crop'
                analysis['crop_info'] = self._detect_crop(input_grid, output_grid)
            elif out_h > in_h or out_w > in_w:
                if out_h % in_h == 0 and out_w % in_w == 0:
                    analysis['transformation_type'] = 'tile'
                    analysis['tile_factor'] = (out_h // in_h, out_w // in_w)
                else:
                    analysis['transformation_type'] = 'scale_or_pad'

        return analysis

    def _detect_color_mapping(self, input_grid: np.ndarray, output_grid: np.ndarray) -> Optional[Dict]:
        """Detect if there's a consistent color mapping"""
        if input_grid.shape != output_grid.shape:
            return None

        color_map = {}
        for i in range(input_grid.shape[0]):
            for j in range(input_grid.shape[1]):
                in_color = input_grid[i, j]
                out_color = output_grid[i, j]

                if in_color in color_map:
                    if color_map[in_color] != out_color:
                        return None  # Inconsistent mapping
                else:
                    color_map[in_color] = out_color

        return color_map

    def _detect_crop(self, input_grid: np.ndarray, output_grid: np.ndarray) -> Optional[Dict]:
        """Detect crop parameters"""
        in_h, in_w = input_grid.shape
        out_h, out_w = output_grid.shape

        # Try to find where output appears in input
        for i in range(in_h - out_h + 1):
            for j in range(in_w - out_w + 1):
                if np.array_equal(input_grid[i:i+out_h, j:j+out_w], output_grid):
                    return {'top': i, 'left': j, 'height': out_h, 'width': out_w}

        return None

    def formulate_hypothesis(self, training_examples: List[Dict]) -> Optional[Dict]:
        """
        Analyze all training examples and formulate hypothesis about the rule

        Returns hypothesis that works on ALL training examples
        """
        if not training_examples:
            return None

        # Analyze first example
        first_input = np.array(training_examples[0]['input'])
        first_output = np.array(training_examples[0]['output'])

        hypothesis = self.analyze_transformation(first_input, first_output)

        # Verify hypothesis on all other examples
        for example in training_examples[1:]:
            input_grid = np.array(example['input'])
            output_grid = np.array(example['output'])

            analysis = self.analyze_transformation(input_grid, output_grid)

            # Check if consistent with hypothesis
            if analysis['transformation_type'] != hypothesis['transformation_type']:
                return None  # Inconsistent pattern

            # For color mapping, verify it's the same mapping
            if hypothesis['transformation_type'] == 'color_mapping':
                if analysis.get('color_map') != hypothesis.get('color_map'):
                    return None

            # For tiling, verify same factor
            if hypothesis['transformation_type'] == 'tile':
                if analysis.get('tile_factor') != hypothesis.get('tile_factor'):
                    return None

            # For crop, verify same crop region
            if hypothesis['transformation_type'] == 'crop':
                # Crop might be relative, not absolute - just verify it works
                pass

        return hypothesis

    def apply_hypothesis(self, hypothesis: Dict, test_input: np.ndarray) -> Optional[np.ndarray]:
        """Apply the understood rule to test input"""
        try:
            transform_type = hypothesis.get('transformation_type')

            if transform_type == 'identity':
                return test_input
            elif transform_type == 'rotate_90':
                return np.rot90(test_input)
            elif transform_type == 'rotate_180':
                return np.rot90(test_input, 2)
            elif transform_type == 'rotate_270':
                return np.rot90(test_input, 3)
            elif transform_type == 'flip_horizontal':
                return np.fliplr(test_input)
            elif transform_type == 'flip_vertical':
                return np.flipud(test_input)
            elif transform_type == 'transpose':
                if test_input.shape[0] == test_input.shape[1]:
                    return test_input.T
            elif transform_type == 'color_mapping':
                color_map = hypothesis.get('color_map', {})
                result = test_input.copy()
                for old_color, new_color in color_map.items():
                    result[test_input == old_color] = new_color
                return result
            elif transform_type == 'tile':
                tile_h, tile_w = hypothesis.get('tile_factor', (1, 1))
                return np.tile(test_input, (tile_h, tile_w))
            elif transform_type == 'crop':
                crop_info = hypothesis.get('crop_info')
                if crop_info:
                    top = crop_info['top']
                    left = crop_info['left']
                    height = crop_info['height']
                    width = crop_info['width']
                    return test_input[top:top+height, left:left+width]
        except:
            pass

        return None

    def reason_and_solve(self, task: Dict) -> Optional[np.ndarray]:
        """
        Main reasoning pipeline:
        1. Analyze training examples
        2. Formulate hypothesis
        3. Apply to test input
        """
        training_examples = task.get('train', [])
        if not training_examples:
            return None

        # Formulate hypothesis from training
        hypothesis = self.formulate_hypothesis(training_examples)

        if hypothesis is None:
            return None

        # Apply hypothesis to test
        test_input = np.array(task['test'][0]['input'])
        result = self.apply_hypothesis(hypothesis, test_input)

        return result
