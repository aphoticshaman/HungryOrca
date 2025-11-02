#!/usr/bin/env python3
"""
ENHANCED SPECIALIST LIBRARY - MORE PRIMITIVES!

Add MORE specialists to eke out the last 10% accuracy:
- GridSpecialist (grid/line detection)
- TilingSpecialist (pattern repetition)
- ObjectSpecialist (connected components)
- MaskSpecialist (template overlay)
- ReflectionSpecialist (mirror operations)
- CropSpecialist (bounding box extraction)

Each specialist is ATOMIC and FAST - runs in <1 second.

Author: HungryOrca - Final Push
Date: 2025-11-02
"""

import numpy as np
from typing import List, Tuple, Optional, Set
from collections import Counter
from dataclasses import dataclass


@dataclass
class SpecialistReport:
    """What specialist found."""
    name: str
    result: Optional[np.ndarray]
    confidence: float
    insight: str


class GridSpecialist:
    """Detects and manipulates grid lines."""

    def __init__(self):
        self.name = "GridSpecialist"

    def solve(self, train_pairs, test_input) -> SpecialistReport:
        """Detect grid patterns."""
        # Check if output removes grid lines
        for inp, out in train_pairs:
            if inp.shape == out.shape:
                # Find most common color in input
                colors = Counter(inp.flatten())
                if len(colors) >= 2:
                    # Try removing each color (potential grid)
                    for color, count in colors.most_common():
                        test_result = test_input.copy()
                        test_result[test_result == color] = 0

                        # Score this
                        inp_test = inp.copy()
                        inp_test[inp_test == color] = 0

                        if np.array_equal(inp_test, out):
                            # Found it! Remove this color
                            return SpecialistReport(
                                name=self.name,
                                result=test_result,
                                confidence=0.95,
                                insight=f"Removes grid color {color}"
                            )

        return SpecialistReport(self.name, None, 0.0, "No grid pattern")


class TilingSpecialist:
    """Detects tiling/repetition patterns."""

    def __init__(self):
        self.name = "TilingSpecialist"

    def solve(self, train_pairs, test_input) -> SpecialistReport:
        """Find tiling patterns."""
        for inp, out in train_pairs:
            # Check if output is tiled version of input
            if out.shape[0] % inp.shape[0] == 0 and out.shape[1] % inp.shape[1] == 0:
                tile_h = out.shape[0] // inp.shape[0]
                tile_w = out.shape[1] // inp.shape[1]

                # Verify it's a tiling
                tiled = np.tile(inp, (tile_h, tile_w))
                if np.array_equal(tiled, out):
                    # Apply same tiling to test
                    result = np.tile(test_input, (tile_h, tile_w))
                    return SpecialistReport(
                        name=self.name,
                        result=result,
                        confidence=0.98,
                        insight=f"Tiles {tile_h}x{tile_w}"
                    )

        return SpecialistReport(self.name, None, 0.0, "No tiling")


class ObjectSpecialist:
    """Connected component detection."""

    def __init__(self):
        self.name = "ObjectSpecialist"

    def find_objects(self, grid, bg=0):
        """Find connected components."""
        objects = []
        visited = set()

        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if grid[i,j] != bg and (i,j) not in visited:
                    # Flood fill to find object
                    obj_cells = set()
                    stack = [(i,j)]
                    color = grid[i,j]

                    while stack:
                        ci, cj = stack.pop()
                        if (ci, cj) in visited:
                            continue
                        if not (0 <= ci < grid.shape[0] and 0 <= cj < grid.shape[1]):
                            continue
                        if grid[ci, cj] != color:
                            continue

                        visited.add((ci, cj))
                        obj_cells.add((ci, cj))

                        for di, dj in [(0,1),(1,0),(0,-1),(-1,0)]:
                            stack.append((ci+di, cj+dj))

                    if obj_cells:
                        objects.append((color, obj_cells))

        return objects

    def solve(self, train_pairs, test_input) -> SpecialistReport:
        """Find and manipulate objects."""
        # Check if task moves/colors objects
        for inp, out in train_pairs:
            if inp.shape == out.shape:
                inp_objs = self.find_objects(inp)
                out_objs = self.find_objects(out)

                # Check if object count/sizes match
                if len(inp_objs) == len(out_objs):
                    # Might be color remapping of objects
                    result = test_input.copy()

                    # Try simple: keep structure, change colors
                    test_objs = self.find_objects(test_input)
                    if len(test_objs) == len(inp_objs):
                        return SpecialistReport(
                            name=self.name,
                            result=result,
                            confidence=0.6,
                            insight=f"Found {len(test_objs)} objects"
                        )

        return SpecialistReport(self.name, None, 0.0, "No object pattern")


class MaskSpecialist:
    """Template/mask overlay operations."""

    def __init__(self):
        self.name = "MaskSpecialist"

    def solve(self, train_pairs, test_input) -> SpecialistReport:
        """Apply mask/template."""
        # Check if all outputs share a common template
        if len(train_pairs) >= 2:
            outputs = [out for _, out in train_pairs]

            # Find common shape
            if all(out.shape == outputs[0].shape for out in outputs):
                # Find cells that are ALWAYS the same color
                template = outputs[0].copy()
                mask = np.ones(template.shape, dtype=bool)

                for out in outputs[1:]:
                    mask &= (template == out)

                # If >50% of cells are constant, it's a template
                if np.sum(mask) > template.size * 0.5:
                    # Apply template to test (if same size)
                    if test_input.shape == template.shape:
                        result = test_input.copy()
                        result[mask] = template[mask]

                        return SpecialistReport(
                            name=self.name,
                            result=result,
                            confidence=0.7,
                            insight=f"Applied template ({np.sum(mask)} cells)"
                        )

        return SpecialistReport(self.name, None, 0.0, "No template")


class ReflectionSpecialist:
    """Mirror/reflection operations."""

    def __init__(self):
        self.name = "ReflectionSpecialist"

    def solve(self, train_pairs, test_input) -> SpecialistReport:
        """Try reflection operations."""
        for inp, out in train_pairs:
            # Horizontal reflection
            if np.array_equal(np.flip(inp, axis=0), out):
                result = np.flip(test_input, axis=0)
                return SpecialistReport(
                    name=self.name,
                    result=result,
                    confidence=0.99,
                    insight="Horizontal flip"
                )

            # Vertical reflection
            if np.array_equal(np.flip(inp, axis=1), out):
                result = np.flip(test_input, axis=1)
                return SpecialistReport(
                    name=self.name,
                    result=result,
                    confidence=0.99,
                    insight="Vertical flip"
                )

            # Diagonal flip
            if inp.shape[0] == inp.shape[1]:
                if np.array_equal(inp.T, out):
                    result = test_input.T if test_input.shape[0] == test_input.shape[1] else test_input
                    return SpecialistReport(
                        name=self.name,
                        result=result,
                        confidence=0.99,
                        insight="Diagonal flip"
                    )

        return SpecialistReport(self.name, None, 0.0, "No reflection")


class CropSpecialist:
    """Bounding box extraction."""

    def __init__(self):
        self.name = "CropSpecialist"

    def crop_to_content(self, grid, bg=0):
        """Crop to non-bg bounding box."""
        mask = grid != bg
        if not mask.any():
            return grid

        rows = np.where(mask.any(axis=1))[0]
        cols = np.where(mask.any(axis=0))[0]

        if len(rows) == 0 or len(cols) == 0:
            return grid

        return grid[rows[0]:rows[-1]+1, cols[0]:cols[-1]+1]

    def solve(self, train_pairs, test_input) -> SpecialistReport:
        """Try cropping operations."""
        for inp, out in train_pairs:
            # Check if output is cropped input
            cropped = self.crop_to_content(inp)
            if np.array_equal(cropped, out):
                result = self.crop_to_content(test_input)
                return SpecialistReport(
                    name=self.name,
                    result=result,
                    confidence=0.95,
                    insight="Crops to content"
                )

        return SpecialistReport(self.name, None, 0.0, "No crop")


# Export all
ALL_ENHANCED_SPECIALISTS = [
    GridSpecialist,
    TilingSpecialist,
    ObjectSpecialist,
    MaskSpecialist,
    ReflectionSpecialist,
    CropSpecialist,
]
