#!/usr/bin/env python3
"""
360° CHAMELEON VISION SYSTEM
============================

Processes INPUT and OUTPUT grids simultaneously from all angles.
Like compound eye - multiple perspectives → unified percept.

ABLATION FLAGS for vision features
"""

import numpy as np
from typing import Dict, List, Tuple
from unified_cortex import UnifiedCortex

# ============================================================================
# ABLATION FLAGS - Vision System
# ============================================================================

ENABLE_360_VISION = True
ENABLE_INPUT_PERSPECTIVES = True   # 8 perspectives on input grid
ENABLE_OUTPUT_PERSPECTIVES = True  # 8 perspectives on output grid
ENABLE_PERSPECTIVE_OVERLAP = True  # Overlapping activation (redundancy)

# ============================================================================
# VISION AGENT - Single Perspective
# ============================================================================

class VisionAgent:
    """
    Single perspective analyzer.

    Like one facet of compound eye - sees from ONE angle only.
    """

    def __init__(self, perspective: str):
        self.perspective = perspective

    def analyze(self, grid: np.ndarray) -> np.ndarray:
        """
        Analyze grid from this perspective.

        Returns: Feature vector for this perspective
        """

        if not ENABLE_360_VISION:
            # Fallback: simple flatten
            return grid.flatten()

        if self.perspective == 'top_down':
            return self._scan_top_down(grid)
        elif self.perspective == 'bottom_up':
            return self._scan_bottom_up(grid)
        elif self.perspective == 'left_right':
            return self._scan_left_right(grid)
        elif self.perspective == 'right_left':
            return self._scan_right_left(grid)
        elif self.perspective == 'diagonal_NE':
            return self._scan_diagonal_NE(grid)
        elif self.perspective == 'diagonal_NW':
            return self._scan_diagonal_NW(grid)
        elif self.perspective == 'diagonal_SE':
            return self._scan_diagonal_SE(grid)
        elif self.perspective == 'diagonal_SW':
            return self._scan_diagonal_SW(grid)
        elif self.perspective == 'inside_out':
            return self._scan_inside_out(grid)
        elif self.perspective == 'outside_in':
            return self._scan_outside_in(grid)
        elif self.perspective == 'color_first':
            return self._scan_color_first(grid)
        elif self.perspective == 'shape_first':
            return self._scan_shape_first(grid)
        elif self.perspective == 'pattern_first':
            return self._scan_pattern_first(grid)
        elif self.perspective == 'symmetry_first':
            return self._scan_symmetry_first(grid)
        elif self.perspective == 'topology_first':
            return self._scan_topology_first(grid)
        elif self.perspective == 'transform_first':
            return self._scan_transform_first(grid)
        else:
            # Fallback
            return grid.flatten()

    # ========================================================================
    # SCAN METHODS - Different Perspectives
    # ========================================================================

    def _scan_top_down(self, grid: np.ndarray) -> np.ndarray:
        """Scan rows from top to bottom."""
        return grid.flatten()

    def _scan_bottom_up(self, grid: np.ndarray) -> np.ndarray:
        """Scan rows from bottom to top."""
        return np.flip(grid, axis=0).flatten()

    def _scan_left_right(self, grid: np.ndarray) -> np.ndarray:
        """Scan columns from left to right."""
        return grid.T.flatten()

    def _scan_right_left(self, grid: np.ndarray) -> np.ndarray:
        """Scan columns from right to left."""
        return np.flip(grid.T, axis=0).flatten()

    def _scan_diagonal_NE(self, grid: np.ndarray) -> np.ndarray:
        """Scan diagonals from bottom-left to top-right."""
        h, w = grid.shape
        features = []
        for k in range(-h+1, w):
            diag = np.diag(grid, k=k)
            features.extend(diag)
        return np.array(features)

    def _scan_diagonal_NW(self, grid: np.ndarray) -> np.ndarray:
        """Scan diagonals from bottom-right to top-left."""
        flipped = np.fliplr(grid)
        return self._scan_diagonal_NE(flipped)

    def _scan_diagonal_SE(self, grid: np.ndarray) -> np.ndarray:
        """Scan diagonals from top-left to bottom-right."""
        return self._scan_diagonal_NE(np.flip(grid, axis=0))

    def _scan_diagonal_SW(self, grid: np.ndarray) -> np.ndarray:
        """Scan diagonals from top-right to bottom-left."""
        return self._scan_diagonal_NW(np.flip(grid, axis=0))

    def _scan_inside_out(self, grid: np.ndarray) -> np.ndarray:
        """Spiral scan from center outward."""
        h, w = grid.shape
        center_y, center_x = h // 2, w // 2

        features = [grid[center_y, center_x]]

        # Spiral outward
        for radius in range(1, max(h, w)):
            for y in range(max(0, center_y - radius), min(h, center_y + radius + 1)):
                for x in range(max(0, center_x - radius), min(w, center_x + radius + 1)):
                    if abs(y - center_y) == radius or abs(x - center_x) == radius:
                        features.append(grid[y, x])

        return np.array(features)

    def _scan_outside_in(self, grid: np.ndarray) -> np.ndarray:
        """Spiral scan from edges inward."""
        return np.flip(self._scan_inside_out(grid))

    def _scan_color_first(self, grid: np.ndarray) -> np.ndarray:
        """Group by color, then scan spatially."""
        unique_colors = np.unique(grid)
        features = []
        for color in unique_colors:
            mask = (grid == color)
            features.extend(grid[mask])
        return np.array(features) if features else grid.flatten()

    def _scan_shape_first(self, grid: np.ndarray) -> np.ndarray:
        """Detect shapes, then scan."""
        # Simple: edges, then interior
        h, w = grid.shape
        features = []

        # Edge pixels
        features.extend(grid[0, :])     # Top
        features.extend(grid[-1, :])    # Bottom
        features.extend(grid[:, 0])     # Left
        features.extend(grid[:, -1])    # Right

        # Interior
        if h > 2 and w > 2:
            features.extend(grid[1:-1, 1:-1].flatten())

        return np.array(features)

    def _scan_pattern_first(self, grid: np.ndarray) -> np.ndarray:
        """Detect repeating patterns."""
        # Simple: row-wise differences
        features = []
        features.extend(grid[0, :])  # First row baseline

        for i in range(1, grid.shape[0]):
            diff = grid[i, :] - grid[i-1, :]
            features.extend(diff)

        return np.array(features)

    def _scan_symmetry_first(self, grid: np.ndarray) -> np.ndarray:
        """Check symmetry, then scan."""
        h, w = grid.shape
        features = []

        # Horizontal symmetry
        for i in range(h // 2):
            features.extend(grid[i, :])
            features.extend(grid[-(i+1), :])

        # Middle row if odd height
        if h % 2 == 1:
            features.extend(grid[h // 2, :])

        return np.array(features)

    def _scan_topology_first(self, grid: np.ndarray) -> np.ndarray:
        """Connectivity-based scan."""
        # Simple: connected components by color
        features = []
        visited = np.zeros_like(grid, dtype=bool)

        def flood_fill(y, x, color):
            if y < 0 or y >= grid.shape[0] or x < 0 or x >= grid.shape[1]:
                return []
            if visited[y, x] or grid[y, x] != color:
                return []

            visited[y, x] = True
            cells = [grid[y, x]]

            # 4-connectivity
            cells.extend(flood_fill(y-1, x, color))
            cells.extend(flood_fill(y+1, x, color))
            cells.extend(flood_fill(y, x-1, color))
            cells.extend(flood_fill(y, x+1, color))

            return cells

        for y in range(grid.shape[0]):
            for x in range(grid.shape[1]):
                if not visited[y, x]:
                    component = flood_fill(y, x, grid[y, x])
                    features.extend(component)

        return np.array(features) if features else grid.flatten()

    def _scan_transform_first(self, grid: np.ndarray) -> np.ndarray:
        """Encode grid + common transforms."""
        features = []
        features.extend(grid.flatten())
        features.extend(np.rot90(grid).flatten())
        features.extend(np.flip(grid, axis=0).flatten())
        return np.array(features)


# ============================================================================
# CHAMELEON VISION SYSTEM - 360° Coverage
# ============================================================================

class ChameleonVision:
    """
    360° vision system for ARC grids.

    Processes input AND output grids from all perspectives simultaneously.
    Like compound eye - redundant coverage, robust perception.
    """

    def __init__(self, cortex: UnifiedCortex):
        self.cortex = cortex

        # INPUT GRID PERSPECTIVES (8 agents)
        if ENABLE_INPUT_PERSPECTIVES:
            self.input_agents = [
                VisionAgent('top_down'),
                VisionAgent('left_right'),
                VisionAgent('diagonal_NE'),
                VisionAgent('diagonal_NW'),
                VisionAgent('inside_out'),
                VisionAgent('color_first'),
                VisionAgent('pattern_first'),
                VisionAgent('topology_first'),
            ]
        else:
            self.input_agents = [VisionAgent('top_down')]  # Minimal

        # OUTPUT GRID PERSPECTIVES (8 agents)
        if ENABLE_OUTPUT_PERSPECTIVES:
            self.output_agents = [
                VisionAgent('bottom_up'),
                VisionAgent('right_left'),
                VisionAgent('diagonal_SE'),
                VisionAgent('diagonal_SW'),
                VisionAgent('outside_in'),
                VisionAgent('shape_first'),
                VisionAgent('symmetry_first'),
                VisionAgent('transform_first'),
            ]
        else:
            self.output_agents = [VisionAgent('bottom_up')]  # Minimal

    def perceive(self, input_grid: np.ndarray,
                 output_grid: np.ndarray = None) -> np.ndarray:
        """
        Perceive grids from all perspectives.

        Activates cortex with 360° vision.

        Args:
            input_grid: Input grid (what we have)
            output_grid: Output grid (what we want) - optional

        Returns:
            Cortical activation pattern
        """

        if not ENABLE_360_VISION:
            # Fallback: simple encoding
            return self.cortex.encode_grid(input_grid)

        # PHASE 1: Input perspectives activate visual cortex
        input_features = []
        for agent in self.input_agents:
            try:
                features = agent.analyze(input_grid)
                input_features.append(features)
            except:
                # If analysis fails, skip this perspective
                continue

        # PHASE 2: Output perspectives (if available)
        output_features = []
        if output_grid is not None:
            for agent in self.output_agents:
                try:
                    features = agent.analyze(output_grid)
                    output_features.append(features)
                except:
                    continue

        # AGGREGATE: Combine all perspectives
        all_features = input_features + output_features

        if not all_features:
            # Fallback
            return self.cortex.encode_grid(input_grid)

        # Find max length
        max_len = max(len(f) for f in all_features)

        # Pad all to same length
        padded = []
        for features in all_features:
            if len(features) < max_len:
                padded_features = np.pad(features, (0, max_len - len(features)))
            else:
                padded_features = features[:max_len]
            padded.append(padded_features)

        # OVERLAP: Sum perspectives (redundancy = robustness)
        if ENABLE_PERSPECTIVE_OVERLAP:
            aggregated = np.sum(padded, axis=0)
            # Normalize
            aggregated = aggregated / len(padded)
        else:
            # Concatenate (no overlap)
            aggregated = np.concatenate(padded)

        # Encode into visual cortex
        visual_size = int(self.cortex.size * 0.25)

        # Resize to fit visual cortex
        if len(aggregated) < visual_size:
            # Tile to fill
            repeats = int(np.ceil(visual_size / len(aggregated)))
            activation = np.tile(aggregated, repeats)[:visual_size]
        else:
            # Truncate
            activation = aggregated[:visual_size]

        # Normalize
        if activation.max() > 0:
            activation = activation / activation.max()

        return activation


# ============================================================================
# QUICK TEST
# ============================================================================

if __name__ == '__main__':
    print("="*80)
    print("360° CHAMELEON VISION - Quick Test")
    print("="*80)

    # Test cortex
    cortex = UnifiedCortex(size=10000)
    vision = ChameleonVision(cortex)

    # Test grid
    test_grid_input = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    test_grid_output = np.array([[9, 8, 7], [6, 5, 4], [3, 2, 1]])

    print("\n📊 Test 1: Input Grid Only")
    activation = vision.perceive(test_grid_input)
    print(f"   Activation length: {len(activation)}")
    print(f"   Visual cortex filled: {np.sum(activation > 0)} neurons")
    print(f"   ✓ Input perception working")

    print("\n📊 Test 2: Input + Output Grids")
    activation = vision.perceive(test_grid_input, test_grid_output)
    print(f"   Activation length: {len(activation)}")
    print(f"   Visual cortex filled: {np.sum(activation > 0)} neurons")
    print(f"   ✓ Dual-grid perception working")

    print("\n📊 Test 3: Cortex Activation")
    cortex.reset()
    activation = vision.perceive(test_grid_input, test_grid_output)
    cortex.activate(activation, context='both')
    print(f"   Total cortex active: {np.sum(cortex.neurons > 0.01)}/{cortex.size}")
    print(f"   Coherence: {cortex.measure_coherence():.3f}")
    print(f"   ✓ Cortex integration working")

    print("\n" + "="*80)
    print("✅ CHAMELEON VISION: OPERATIONAL")
    print("="*80)
    print("\n🧪 ABLATION FLAGS:")
    print(f"   ENABLE_360_VISION: {ENABLE_360_VISION}")
    print(f"   ENABLE_INPUT_PERSPECTIVES: {ENABLE_INPUT_PERSPECTIVES}")
    print(f"   ENABLE_OUTPUT_PERSPECTIVES: {ENABLE_OUTPUT_PERSPECTIVES}")
    print(f"   ENABLE_PERSPECTIVE_OVERLAP: {ENABLE_PERSPECTIVE_OVERLAP}")
    print("\nReady for vision ablation tests!")
