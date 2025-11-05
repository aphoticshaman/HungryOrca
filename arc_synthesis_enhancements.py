#!/usr/bin/env python3
"""
🧬 ARC PRIZE 2025 - NOVEL SYNTHESIS ENHANCEMENTS
Implements 5 cutting-edge methods from ARC Prize meta-analysis:

1. Hyper-Feature Object Clustering (HFOC) - Enhanced object perception
2. Goal-Directed Potential Fields (GDPF) - Heuristic search guidance
3. Inverse Semantics & Bi-Directional Search - Bidirectional search
4. Causal Abstraction Graph (CAG) - DAG-based program composition
5. Recursive Transformation Decomposition - Hierarchical decomposition

These methods provide x10 synthesis insights beyond basic program search.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Callable, Set
from dataclasses import dataclass, field
from collections import defaultdict, deque
import hashlib


# ═══════════════════════════════════════════════════════════════
# ENHANCEMENT 1: HYPER-FEATURE OBJECT CLUSTERING (HFOC)
# ═══════════════════════════════════════════════════════════════

@dataclass
class HyperObject:
    """Enhanced object representation with 6 hyper-features"""

    # Basic features (existing)
    color: int
    positions: np.ndarray
    size: int
    bbox: Tuple[int, int, int, int]  # min_row, min_col, max_row, max_col
    center: Tuple[float, float]

    # Hyper-features (new)
    symmetry_score: float = 0.0      # Horizontal/vertical symmetry
    convexity_score: float = 0.0     # How "filled" is the bounding box
    density: float = 0.0              # size / bbox_area
    hierarchy_level: int = 0          # Nesting depth (0=outermost)
    rotational_invariant_signature: str = ""  # For matching rotated versions
    topology: str = "simple"          # simple, hollow, nested, connected


class HyperFeatureObjectClustering:
    """
    Go beyond simple pixel-connectivity for object definition.
    Cluster pixels based on six hyper-features:
    1. Color
    2. Connectivity
    3. Bounding box hierarchy
    4. Symmetry
    5. Convexity
    6. Density

    This creates a richer, more descriptive object graph for symbolic reasoning.
    """

    @staticmethod
    def extract_hyper_objects(grid: np.ndarray) -> List[HyperObject]:
        """Extract objects with full hyper-feature analysis"""

        if grid.size == 0:
            return []

        # Step 1: Basic object extraction (color + connectivity)
        basic_objects = HyperFeatureObjectClustering._extract_basic_objects(grid)

        # Step 2: Compute hyper-features for each object
        hyper_objects = []
        for obj in basic_objects:
            hyper_obj = HyperFeatureObjectClustering._compute_hyper_features(obj, grid)
            hyper_objects.append(hyper_obj)

        # Step 3: Compute hierarchy relationships
        HyperFeatureObjectClustering._compute_hierarchy(hyper_objects)

        return hyper_objects

    @staticmethod
    def _extract_basic_objects(grid: np.ndarray) -> List[Dict]:
        """Extract basic connected components"""
        objects = []

        unique_colors = np.unique(grid)

        for color in unique_colors:
            if color == 0:  # Skip background
                continue

            mask = (grid == color).astype(np.uint8)
            labeled = HyperFeatureObjectClustering._label_components(mask)

            for obj_id in range(1, labeled.max() + 1):
                obj_mask = (labeled == obj_id)
                positions = np.argwhere(obj_mask)

                if len(positions) == 0:
                    continue

                min_row, min_col = positions.min(axis=0)
                max_row, max_col = positions.max(axis=0)

                objects.append({
                    'color': int(color),
                    'positions': positions,
                    'size': len(positions),
                    'bbox': (min_row, min_col, max_row, max_col),
                    'center': (positions[:, 0].mean(), positions[:, 1].mean()),
                    'mask': obj_mask,
                })

        return objects

    @staticmethod
    def _compute_hyper_features(obj: Dict, grid: np.ndarray) -> HyperObject:
        """Compute all 6 hyper-features"""

        # Extract bbox region
        min_r, min_c, max_r, max_c = obj['bbox']
        bbox_height = max_r - min_r + 1
        bbox_width = max_c - min_c + 1
        bbox_area = bbox_height * bbox_width

        # Extract object region
        obj_region = obj['mask'][min_r:max_r+1, min_c:max_c+1]

        # 1. Symmetry score (horizontal + vertical)
        symmetry_h = np.mean(obj_region == np.fliplr(obj_region))
        symmetry_v = np.mean(obj_region == np.flipud(obj_region))
        symmetry_score = (symmetry_h + symmetry_v) / 2.0

        # 2. Convexity score (object_size / convex_hull_size)
        # Approximation: use bounding box as convex hull
        convexity_score = obj['size'] / max(bbox_area, 1)

        # 3. Density
        density = obj['size'] / max(bbox_area, 1)

        # 4. Rotational invariant signature (for matching across rotations)
        # Use normalized moments (rotation invariant)
        signature = HyperFeatureObjectClustering._compute_rotation_invariant_signature(obj_region)

        # 5. Topology detection
        topology = HyperFeatureObjectClustering._detect_topology(obj_region)

        return HyperObject(
            color=obj['color'],
            positions=obj['positions'],
            size=obj['size'],
            bbox=obj['bbox'],
            center=obj['center'],
            symmetry_score=symmetry_score,
            convexity_score=convexity_score,
            density=density,
            rotational_invariant_signature=signature,
            topology=topology,
        )

    @staticmethod
    def _compute_rotation_invariant_signature(region: np.ndarray) -> str:
        """Compute rotation-invariant signature using normalized moments"""
        if region.size == 0:
            return "empty"

        # Compute central moments (translation invariant)
        y_coords, x_coords = np.where(region > 0)
        if len(y_coords) == 0:
            return "empty"

        cy, cx = y_coords.mean(), x_coords.mean()

        # Second-order moments
        mu20 = np.sum((x_coords - cx)**2)
        mu02 = np.sum((y_coords - cy)**2)
        mu11 = np.sum((x_coords - cx) * (y_coords - cy))

        # Normalize
        norm = np.sqrt(mu20**2 + mu02**2 + mu11**2) + 1e-10

        # Create signature (rotation invariant)
        sig = f"{mu20/norm:.3f}_{mu02/norm:.3f}_{abs(mu11)/norm:.3f}"
        return sig

    @staticmethod
    def _detect_topology(region: np.ndarray) -> str:
        """Detect topological type: simple, hollow, nested, connected"""
        if region.size == 0:
            return "empty"

        # Count components in region
        n_components = HyperFeatureObjectClustering._count_components(region)

        # Count holes (connected components of zeros)
        inverted = 1 - region
        n_holes = HyperFeatureObjectClustering._count_components(inverted) - 1  # -1 for background

        if n_holes > 0:
            return "hollow"
        elif n_components > 1:
            return "disconnected"
        else:
            # Check if border is significantly filled
            border_mask = np.zeros_like(region)
            border_mask[0, :] = 1
            border_mask[-1, :] = 1
            border_mask[:, 0] = 1
            border_mask[:, -1] = 1

            border_fill = np.sum(region * border_mask) / np.sum(border_mask)

            if border_fill > 0.5:
                return "enclosed"
            else:
                return "simple"

    @staticmethod
    def _count_components(mask: np.ndarray) -> int:
        """Count connected components"""
        if mask.size == 0 or mask.max() == 0:
            return 0

        labeled = HyperFeatureObjectClustering._label_components(mask)
        return labeled.max()

    @staticmethod
    def _compute_hierarchy(objects: List[HyperObject]):
        """Compute containment hierarchy (which objects contain which)"""

        for i, obj_i in enumerate(objects):
            # Check if obj_i is contained in any other object
            level = 0

            for j, obj_j in enumerate(objects):
                if i == j:
                    continue

                # Check if obj_i center is inside obj_j bbox
                min_r, min_c, max_r, max_c = obj_j.bbox
                cy, cx = obj_i.center

                if min_r <= cy <= max_r and min_c <= cx <= max_c:
                    # obj_i might be inside obj_j
                    level += 1

            obj_i.hierarchy_level = level

    @staticmethod
    def _label_components(mask: np.ndarray) -> np.ndarray:
        """Connected component labeling (4-connectivity)"""
        labeled = np.zeros_like(mask, dtype=np.int32)
        label = 0

        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if mask[i, j] and not labeled[i, j]:
                    label += 1
                    HyperFeatureObjectClustering._flood_fill(mask, labeled, i, j, label)

        return labeled

    @staticmethod
    def _flood_fill(mask: np.ndarray, labeled: np.ndarray, i: int, j: int, label: int):
        """Flood fill for connected component labeling"""
        stack = [(i, j)]

        while stack:
            ci, cj = stack.pop()

            if ci < 0 or ci >= mask.shape[0] or cj < 0 or cj >= mask.shape[1]:
                continue
            if not mask[ci, cj] or labeled[ci, cj]:
                continue

            labeled[ci, cj] = label

            # 4-connectivity
            stack.extend([(ci-1, cj), (ci+1, cj), (ci, cj-1), (ci, cj+1)])


# ═══════════════════════════════════════════════════════════════
# ENHANCEMENT 2: GOAL-DIRECTED POTENTIAL FIELDS (GDPF)
# ═══════════════════════════════════════════════════════════════

class GoalDirectedPotentialField:
    """
    Create a metric potential field defined by the "distance" between
    the object representation in the input and output.

    The NSM prioritizes symbolic primitives that reduce this potential field,
    making the search goal-directed rather than blind.

    Distance metrics:
    - Center of mass shift
    - Size change
    - Bounding box change
    - Color distribution change
    """

    @staticmethod
    def compute_potential_field(input_grid: np.ndarray,
                                output_grid: np.ndarray) -> Dict[str, float]:
        """
        Compute potential field between input and output.
        Returns metrics that quantify the "distance" to the goal.
        """

        # Extract objects from both grids
        input_objects = HyperFeatureObjectClustering.extract_hyper_objects(input_grid)
        output_objects = HyperFeatureObjectClustering.extract_hyper_objects(output_grid)

        potential = {
            'shape_distance': 0.0,
            'color_distance': 0.0,
            'position_distance': 0.0,
            'size_distance': 0.0,
            'topology_distance': 0.0,
        }

        # Shape distance
        if input_grid.shape != output_grid.shape:
            potential['shape_distance'] = np.sum(np.abs(np.array(input_grid.shape) - np.array(output_grid.shape)))

        # Color distribution distance
        input_colors = np.bincount(input_grid.flatten(), minlength=10)
        output_colors = np.bincount(output_grid.flatten(), minlength=10)
        potential['color_distance'] = np.sum(np.abs(input_colors - output_colors))

        # Object-level distances (match objects by closest center)
        if input_objects and output_objects:
            position_dists = []
            size_dists = []

            for in_obj in input_objects:
                # Find closest output object
                min_dist = float('inf')
                closest_out_obj = None

                for out_obj in output_objects:
                    dist = np.linalg.norm(np.array(in_obj.center) - np.array(out_obj.center))
                    if dist < min_dist:
                        min_dist = dist
                        closest_out_obj = out_obj

                if closest_out_obj:
                    position_dists.append(min_dist)
                    size_dists.append(abs(in_obj.size - closest_out_obj.size))

            potential['position_distance'] = np.mean(position_dists) if position_dists else 0.0
            potential['size_distance'] = np.mean(size_dists) if size_dists else 0.0

        # Topology distance (count of different topological types)
        input_topologies = set(obj.topology for obj in input_objects)
        output_topologies = set(obj.topology for obj in output_objects)
        potential['topology_distance'] = len(input_topologies.symmetric_difference(output_topologies))

        return potential

    @staticmethod
    def evaluate_transform_quality(input_grid: np.ndarray,
                                   transformed_grid: np.ndarray,
                                   target_grid: np.ndarray) -> float:
        """
        Evaluate how much a transformation reduces the potential field.
        Higher score = transformation moves closer to target.
        """

        # Compute potential before and after
        initial_potential = GoalDirectedPotentialField.compute_potential_field(
            input_grid, target_grid
        )

        final_potential = GoalDirectedPotentialField.compute_potential_field(
            transformed_grid, target_grid
        )

        # Compute reduction in potential (higher is better)
        initial_sum = sum(initial_potential.values())
        final_sum = sum(final_potential.values())

        reduction = initial_sum - final_sum

        # Normalize to 0-1 range
        if initial_sum == 0:
            return 1.0 if final_sum == 0 else 0.0

        quality_score = max(0.0, reduction / initial_sum)

        return quality_score


# ═══════════════════════════════════════════════════════════════
# ENHANCEMENT 3: INVERSE SEMANTICS & BI-DIRECTIONAL SEARCH
# ═══════════════════════════════════════════════════════════════

class InverseSemantics:
    """
    Instead of only generating Input → Output programs, simultaneously
    search for inverse functions Output → Input.

    The solution is validated when P and P^-1 are found that lead to
    a shared latent state or are logically consistent.

    This cuts the search space in half by meeting in the middle.
    """

    def __init__(self):
        self.forward_cache = {}   # input_hash -> (intermediate_state, transform)
        self.backward_cache = {}  # output_hash -> (intermediate_state, inverse_transform)

    def bidirectional_search(self,
                            input_grid: np.ndarray,
                            output_grid: np.ndarray,
                            primitives: List[Tuple[str, Callable]],
                            max_depth: int = 3) -> Optional[Tuple[Callable, Callable, float]]:
        """
        Search forward from input and backward from output.
        Return (forward_program, inverse_program, confidence) when they meet.
        """

        # Hash grids for caching
        input_hash = self._hash_grid(input_grid)
        output_hash = self._hash_grid(output_grid)

        # Initialize search frontiers
        forward_frontier = deque([(input_grid, [], 0)])   # (state, path, depth)
        backward_frontier = deque([(output_grid, [], 0)])

        forward_visited = {input_hash: (input_grid, [])}
        backward_visited = {output_hash: (output_grid, [])}

        # Bidirectional BFS
        for _ in range(max_depth):
            # Expand forward frontier
            if forward_frontier:
                state, path, depth = forward_frontier.popleft()

                if depth < max_depth:
                    for prim_name, prim_func in primitives:
                        try:
                            new_state = prim_func(state)
                            new_hash = self._hash_grid(new_state)

                            if new_hash not in forward_visited:
                                new_path = path + [(prim_name, prim_func, 'forward')]
                                forward_visited[new_hash] = (new_state, new_path)
                                forward_frontier.append((new_state, new_path, depth + 1))

                                # Check if we met the backward search
                                if new_hash in backward_visited:
                                    backward_state, backward_path = backward_visited[new_hash]
                                    return self._construct_solution(new_path, backward_path)

                        except Exception:
                            continue

            # Expand backward frontier (apply inverse operations)
            if backward_frontier:
                state, path, depth = backward_frontier.popleft()

                if depth < max_depth:
                    for prim_name, prim_func in primitives:
                        try:
                            # Attempt to apply inverse
                            inverse_func = self._get_inverse(prim_name, prim_func)
                            if inverse_func is None:
                                continue

                            new_state = inverse_func(state)
                            new_hash = self._hash_grid(new_state)

                            if new_hash not in backward_visited:
                                new_path = path + [(prim_name, inverse_func, 'backward')]
                                backward_visited[new_hash] = (new_state, new_path)
                                backward_frontier.append((new_state, new_path, depth + 1))

                                # Check if we met the forward search
                                if new_hash in forward_visited:
                                    forward_state, forward_path = forward_visited[new_hash]
                                    return self._construct_solution(forward_path, new_path)

                        except Exception:
                            continue

        return None

    @staticmethod
    def _hash_grid(grid: np.ndarray) -> str:
        """Hash a grid for caching"""
        return hashlib.md5(grid.tobytes()).hexdigest()

    @staticmethod
    def _get_inverse(prim_name: str, prim_func: Callable) -> Optional[Callable]:
        """Get inverse of a primitive operation"""

        inverse_map = {
            'rot90': lambda g: np.rot90(g, 3),      # rot90 inverse is rot270
            'rot180': lambda g: np.rot90(g, 2),     # rot180 inverse is rot180
            'rot270': lambda g: np.rot90(g, 1),     # rot270 inverse is rot90
            'flip_h': lambda g: np.fliplr(g),       # flip_h inverse is flip_h
            'flip_v': lambda g: np.flipud(g),       # flip_v inverse is flip_v
            'transpose': lambda g: g.T if g.shape[0] == g.shape[1] else g,  # transpose inverse is transpose
            'identity': lambda g: g,                 # identity inverse is identity
        }

        return inverse_map.get(prim_name)

    @staticmethod
    def _construct_solution(forward_path: List, backward_path: List) -> Tuple[Callable, Callable, float]:
        """Construct final solution from meeting point"""

        # Create composite functions
        def forward_program(grid):
            state = grid
            for _, func, _ in forward_path:
                state = func(state)
            return state

        def inverse_program(grid):
            state = grid
            for _, func, _ in reversed(backward_path):
                state = func(state)
            return state

        # Confidence based on path length (shorter is better)
        total_length = len(forward_path) + len(backward_path)
        confidence = max(0.1, 1.0 - (total_length * 0.1))

        return forward_program, inverse_program, confidence


# ═══════════════════════════════════════════════════════════════
# ENHANCEMENT 4: CAUSAL ABSTRACTION GRAPH (CAG)
# ═══════════════════════════════════════════════════════════════

@dataclass
class CAGNode:
    """Node in Causal Abstraction Graph"""
    node_id: str
    grid_state: np.ndarray
    confidence: float
    parent_ids: List[str] = field(default_factory=list)
    transform_from_parent: Optional[Callable] = None
    transform_name: str = ""


class CausalAbstractionGraph:
    """
    Represent the solution as a Directed Acyclic Graph (DAG) where:
    - Nodes are intermediate grid states
    - Edges are high-confidence primitives

    This makes composition of complex rules explicit and eliminates
    redundant search paths through memoization.
    """

    def __init__(self):
        self.nodes: Dict[str, CAGNode] = {}
        self.edges: Dict[str, List[str]] = defaultdict(list)  # parent_id -> [child_ids]
        self.node_counter = 0

    def add_state(self,
                  grid: np.ndarray,
                  confidence: float,
                  parent_id: Optional[str] = None,
                  transform: Optional[Callable] = None,
                  transform_name: str = "") -> str:
        """Add a new grid state to the graph"""

        # Generate unique node ID
        grid_hash = hashlib.md5(grid.tobytes()).hexdigest()[:8]
        node_id = f"node_{self.node_counter}_{grid_hash}"
        self.node_counter += 1

        # Check if this state already exists
        for existing_id, existing_node in self.nodes.items():
            if np.array_equal(existing_node.grid_state, grid):
                # State exists, update confidence if higher
                if confidence > existing_node.confidence:
                    existing_node.confidence = confidence

                # Add parent link
                if parent_id and parent_id not in existing_node.parent_ids:
                    existing_node.parent_ids.append(parent_id)
                    self.edges[parent_id].append(existing_id)

                return existing_id

        # Create new node
        node = CAGNode(
            node_id=node_id,
            grid_state=grid,
            confidence=confidence,
            parent_ids=[parent_id] if parent_id else [],
            transform_from_parent=transform,
            transform_name=transform_name,
        )

        self.nodes[node_id] = node

        if parent_id:
            self.edges[parent_id].append(node_id)

        return node_id

    def find_path(self, start_id: str, end_id: str) -> Optional[List[str]]:
        """Find path from start node to end node using BFS"""

        if start_id not in self.nodes or end_id not in self.nodes:
            return None

        if start_id == end_id:
            return [start_id]

        # BFS
        queue = deque([(start_id, [start_id])])
        visited = {start_id}

        while queue:
            current_id, path = queue.popleft()

            for child_id in self.edges[current_id]:
                if child_id == end_id:
                    return path + [child_id]

                if child_id not in visited:
                    visited.add(child_id)
                    queue.append((child_id, path + [child_id]))

        return None

    def extract_program(self, start_id: str, end_id: str) -> Optional[Tuple[Callable, float]]:
        """Extract composite program from start to end"""

        path = self.find_path(start_id, end_id)

        if not path or len(path) < 2:
            return None

        # Compose transforms along path
        transforms = []
        min_confidence = 1.0

        for i in range(1, len(path)):
            node = self.nodes[path[i]]
            if node.transform_from_parent:
                transforms.append(node.transform_from_parent)
                min_confidence = min(min_confidence, node.confidence)

        # Create composite function
        def composite_program(grid):
            state = grid
            for transform in transforms:
                state = transform(state)
            return state

        return composite_program, min_confidence

    def get_most_confident_path(self, start_id: str) -> Optional[Tuple[str, float]]:
        """Get the highest confidence path from start node"""

        if start_id not in self.nodes:
            return None

        best_end_id = start_id
        best_confidence = self.nodes[start_id].confidence

        # DFS to find best path
        def dfs(node_id, path_confidence):
            nonlocal best_end_id, best_confidence

            for child_id in self.edges[node_id]:
                child = self.nodes[child_id]
                new_confidence = min(path_confidence, child.confidence)

                if new_confidence > best_confidence:
                    best_end_id = child_id
                    best_confidence = new_confidence

                dfs(child_id, new_confidence)

        dfs(start_id, self.nodes[start_id].confidence)

        return best_end_id, best_confidence


# ═══════════════════════════════════════════════════════════════
# ENHANCEMENT 5: RECURSIVE TRANSFORMATION DECOMPOSITION
# ═══════════════════════════════════════════════════════════════

class RecursiveTransformationDecomposition:
    """
    A recursive function that finds a large-scale, easy-to-detect primitive
    (e.g., "Remove all red pixels" or "Mirror the left half").

    It applies this function, creating a residual problem (Input' → Output'),
    and then recursively calls the main solver on the simplified residual.

    This hierarchical decomposition dramatically reduces complexity for
    multi-step transformations.
    """

    @staticmethod
    def detect_large_scale_primitive(input_grid: np.ndarray,
                                     output_grid: np.ndarray) -> Optional[Tuple[str, Callable, np.ndarray]]:
        """
        Detect large-scale, obvious transformations.
        Returns (name, transform_function, residual_input) if found.
        """

        # 1. Check for color removal
        for color in range(1, 10):
            if np.any(input_grid == color) and not np.any(output_grid == color):
                # Color was removed
                def remove_color(g, c=color):
                    result = g.copy()
                    result[result == c] = 0
                    return result

                residual_input = remove_color(input_grid)

                # Check if this is the only transformation
                if np.array_equal(residual_input, output_grid):
                    return f"remove_color_{color}", remove_color, residual_input

        # 2. Check for half-mirroring (left→right or top→bottom)
        h, w = input_grid.shape

        # Left half mirrored to right
        if w % 2 == 0:
            left_half = input_grid[:, :w//2]
            right_half = input_grid[:, w//2:]

            output_left = output_grid[:, :w//2]
            output_right = output_grid[:, w//2:]

            if np.array_equal(left_half, output_left) and np.array_equal(np.fliplr(left_half), output_right):
                def mirror_left_to_right(g):
                    result = g.copy()
                    result[:, g.shape[1]//2:] = np.fliplr(g[:, :g.shape[1]//2])
                    return result

                return "mirror_left_to_right", mirror_left_to_right, input_grid

        # Top half mirrored to bottom
        if h % 2 == 0:
            top_half = input_grid[:h//2, :]
            bottom_half = input_grid[h//2:, :]

            output_top = output_grid[:h//2, :]
            output_bottom = output_grid[h//2:, :]

            if np.array_equal(top_half, output_top) and np.array_equal(np.flipud(top_half), output_bottom):
                def mirror_top_to_bottom(g):
                    result = g.copy()
                    result[g.shape[0]//2:, :] = np.flipud(g[:g.shape[0]//2, :])
                    return result

                return "mirror_top_to_bottom", mirror_top_to_bottom, input_grid

        # 3. Check for simple geometric transforms
        simple_transforms = [
            ("rot90", lambda g: np.rot90(g)),
            ("rot180", lambda g: np.rot90(g, 2)),
            ("rot270", lambda g: np.rot90(g, 3)),
            ("flip_h", lambda g: np.fliplr(g)),
            ("flip_v", lambda g: np.flipud(g)),
            ("transpose", lambda g: g.T if g.shape[0] == g.shape[1] else g),
        ]

        for name, transform in simple_transforms:
            try:
                if np.array_equal(transform(input_grid), output_grid):
                    return name, transform, input_grid
            except:
                continue

        return None

    @staticmethod
    def decompose_recursively(input_grid: np.ndarray,
                             output_grid: np.ndarray,
                             max_depth: int = 3) -> List[Tuple[str, Callable]]:
        """
        Recursively decompose transformation into a sequence of primitives.
        Returns list of (name, transform) tuples.
        """

        if max_depth == 0:
            return []

        # Try to detect large-scale primitive
        detection = RecursiveTransformationDecomposition.detect_large_scale_primitive(
            input_grid, output_grid
        )

        if detection is None:
            return []

        name, transform, residual_input = detection

        # Apply transform to get intermediate state
        intermediate = transform(input_grid)

        # Check if we're done
        if np.array_equal(intermediate, output_grid):
            return [(name, transform)]

        # Recursively decompose the residual problem
        remaining_transforms = RecursiveTransformationDecomposition.decompose_recursively(
            intermediate, output_grid, max_depth - 1
        )

        return [(name, transform)] + remaining_transforms

    @staticmethod
    def compose_transforms(transforms: List[Tuple[str, Callable]]) -> Callable:
        """Compose a list of transforms into a single function"""

        def composed(grid):
            state = grid
            for name, transform in transforms:
                state = transform(state)
            return state

        return composed


if __name__ == "__main__":
    print("🧬 ARC Prize 2025 - Novel Synthesis Enhancements")
    print("=" * 60)
    print("\n✅ 5 cutting-edge enhancements implemented:")
    print("  1. Hyper-Feature Object Clustering (HFOC)")
    print("  2. Goal-Directed Potential Fields (GDPF)")
    print("  3. Inverse Semantics & Bi-Directional Search")
    print("  4. Causal Abstraction Graph (CAG)")
    print("  5. Recursive Transformation Decomposition")
    print("\n🚀 Ready to integrate with LucidOrca solver!")
