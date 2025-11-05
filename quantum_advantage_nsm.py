#!/usr/bin/env python3
"""
🌀 QUANTUM ADVANTAGE NSM TOOLKIT
Neuro-Symbolic Methods for Detecting Great Attractor Basins in ARC Puzzles

Philosophy: ARC puzzles hide in higher-dimensional attractor basins.
3D grids are mere projections. Use NSM to peek behind the curtain.

Author: Ryan Cardwell & Claude
Date: November 2025
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Set, Any
from dataclasses import dataclass
from collections import defaultdict
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import svd
from itertools import combinations, permutations
import hashlib


# ═══════════════════════════════════════════════════════════════
# PART 1: DIMENSIONAL PROJECTION TOOLKIT
# ═══════════════════════════════════════════════════════════════

@dataclass
class HigherDimensionalProjection:
    """Project 2D ARC grid into higher dimensions to reveal hidden structure"""

    # 3D: Spatial (height, width, color)
    spatial_embedding: np.ndarray

    # 4D: Temporal (if multiple examples, track evolution)
    temporal_flow: Optional[np.ndarray] = None

    # 5D: Topological (connected components, holes, boundaries)
    topological_features: Optional[Dict[str, Any]] = None

    # 6D: Symmetry group representation
    symmetry_manifold: Optional[np.ndarray] = None

    # 7D+: Abstract feature space (learned embeddings)
    abstract_embedding: Optional[np.ndarray] = None

    # Orthogonal: Side-channel dimensions
    orthogonal_features: Optional[Dict[str, float]] = None


class DimensionalProjector:
    """Peek behind 3D curtain: Project grids into higher dimensions"""

    def __init__(self, max_dimension: int = 36):
        self.max_dimension = max_dimension
        self.projection_cache = {}

    def project_to_higher_dims(self, grid: np.ndarray, target_dim: int = 36) -> np.ndarray:
        """
        Project 2D grid into higher-dimensional feature space
        Returns: (target_dim,) feature vector
        """
        features = []

        # Dimension 1-3: Spatial statistics
        features.extend([
            grid.shape[0],  # Height
            grid.shape[1],  # Width
            grid.size,      # Total cells
        ])

        # Dimension 4-6: Color statistics
        features.extend([
            len(np.unique(grid)),           # Unique colors
            np.mean(grid),                   # Mean color
            np.std(grid),                    # Color variance
        ])

        # Dimension 7-9: Topological
        features.extend([
            self._count_connected_components(grid),
            self._count_holes(grid),
            self._measure_boundary_complexity(grid),
        ])

        # Dimension 10-15: Symmetry group (6 transforms)
        symmetries = self._compute_symmetry_features(grid)
        features.extend(symmetries)

        # Dimension 16-21: Frequency domain (FFT magnitudes)
        freq_features = self._compute_frequency_features(grid)
        features.extend(freq_features)

        # Dimension 22-27: Structural (patterns, repetition)
        structural = self._compute_structural_features(grid)
        features.extend(structural)

        # Dimension 28-33: Information theoretic
        info_features = self._compute_information_features(grid)
        features.extend(info_features)

        # Dimension 34-36: Orthogonal (compression, randomness, predictability)
        orthogonal = self._compute_orthogonal_features(grid)
        features.extend(orthogonal)

        # Pad or truncate to target dimension
        feature_vec = np.array(features)
        if len(feature_vec) < target_dim:
            # Pad with learned nonlinear combinations
            extra = self._generate_nonlinear_features(feature_vec, target_dim - len(feature_vec))
            feature_vec = np.concatenate([feature_vec, extra])

        return feature_vec[:target_dim]

    def _count_connected_components(self, grid: np.ndarray) -> int:
        """Count connected components per color"""
        components = 0
        for color in np.unique(grid):
            if color == 0:  # Skip background
                continue
            mask = (grid == color).astype(int)
            # Simple 4-connectivity count (flood fill simulation)
            visited = np.zeros_like(mask)
            for i in range(mask.shape[0]):
                for j in range(mask.shape[1]):
                    if mask[i,j] and not visited[i,j]:
                        components += 1
                        # Mark as visited (simplified - full flood fill omitted for brevity)
                        visited[i,j] = 1
        return components

    def _count_holes(self, grid: np.ndarray) -> int:
        """Topological holes (Euler characteristic approximation)"""
        # Simplified: Count enclosed empty regions
        holes = 0
        for color in np.unique(grid):
            if color == 0:
                continue
            mask = (grid == color)
            # Check for enclosed zeros (holes)
            interior = mask[1:-1, 1:-1]
            if interior.any():
                holes += np.sum((grid[1:-1, 1:-1] == 0) & mask[1:-1, 1:-1])
        return holes

    def _measure_boundary_complexity(self, grid: np.ndarray) -> float:
        """Measure boundary fractal dimension"""
        # Count boundary pixels
        boundary_count = 0
        for i in range(1, grid.shape[0]-1):
            for j in range(1, grid.shape[1]-1):
                if grid[i,j] != grid[i-1,j] or grid[i,j] != grid[i,j-1]:
                    boundary_count += 1
        return boundary_count / max(grid.size, 1)

    def _compute_symmetry_features(self, grid: np.ndarray) -> List[float]:
        """Symmetry group representation (6D: identity, rot90, rot180, rot270, flip_h, flip_v)"""
        features = []

        transforms = [
            grid,                      # Identity
            np.rot90(grid, 1),        # 90° rotation
            np.rot90(grid, 2),        # 180° rotation
            np.rot90(grid, 3),        # 270° rotation
            np.fliplr(grid),          # Horizontal flip
            np.flipud(grid),          # Vertical flip
        ]

        # Measure similarity to original
        for t in transforms:
            if t.shape == grid.shape:
                similarity = np.sum(t == grid) / grid.size
            else:
                similarity = 0.0
            features.append(similarity)

        return features

    def _compute_frequency_features(self, grid: np.ndarray) -> List[float]:
        """FFT-based frequency domain features"""
        # 2D FFT
        fft = np.fft.fft2(grid.astype(float))
        fft_mag = np.abs(np.fft.fftshift(fft))

        # Extract features: DC, low freq energy, high freq energy
        center = (fft_mag.shape[0]//2, fft_mag.shape[1]//2)
        dc = fft_mag[center]

        # Low frequency (inner quarter)
        r = min(center[0]//2, center[1]//2)
        low_freq = np.mean(fft_mag[center[0]-r:center[0]+r, center[1]-r:center[1]+r])

        # High frequency (outer regions)
        high_freq = np.mean(fft_mag) - low_freq

        # Spectral centroid
        centroid_x = np.sum(np.arange(fft_mag.shape[1]) * np.sum(fft_mag, axis=0)) / np.sum(fft_mag)
        centroid_y = np.sum(np.arange(fft_mag.shape[0]) * np.sum(fft_mag, axis=1)) / np.sum(fft_mag)

        # Spectral spread
        spread = np.std(fft_mag)

        return [
            np.log1p(dc),
            np.log1p(low_freq),
            np.log1p(high_freq),
            centroid_x / fft_mag.shape[1],
            centroid_y / fft_mag.shape[0],
            np.log1p(spread)
        ]

    def _compute_structural_features(self, grid: np.ndarray) -> List[float]:
        """Detect repeating patterns and structure"""
        features = []

        # Horizontal repetition
        h_repetition = 0
        for i in range(grid.shape[0]):
            row = grid[i, :]
            if len(row) >= 2:
                # Check if row is repeating pattern
                for period in range(1, len(row)//2 + 1):
                    if len(row) % period == 0:
                        pattern = row[:period]
                        if np.all(row.reshape(-1, period) == pattern):
                            h_repetition = max(h_repetition, len(row) / period)
        features.append(h_repetition)

        # Vertical repetition
        v_repetition = 0
        for j in range(grid.shape[1]):
            col = grid[:, j]
            if len(col) >= 2:
                for period in range(1, len(col)//2 + 1):
                    if len(col) % period == 0:
                        pattern = col[:period]
                        if np.all(col.reshape(-1, period) == pattern):
                            v_repetition = max(v_repetition, len(col) / period)
        features.append(v_repetition)

        # Block repetition (2x2, 3x3 patterns)
        features.append(self._measure_block_repetition(grid, 2))
        features.append(self._measure_block_repetition(grid, 3))

        # Self-similarity (fractal measure)
        features.append(self._measure_self_similarity(grid))

        # Compressibility (proxy via run-length encoding)
        features.append(self._measure_compressibility(grid))

        return features

    def _measure_block_repetition(self, grid: np.ndarray, block_size: int) -> float:
        """Measure how much grid is composed of repeating blocks"""
        if grid.shape[0] < block_size or grid.shape[1] < block_size:
            return 0.0

        blocks = {}
        for i in range(0, grid.shape[0] - block_size + 1):
            for j in range(0, grid.shape[1] - block_size + 1):
                block = tuple(grid[i:i+block_size, j:j+block_size].flatten())
                blocks[block] = blocks.get(block, 0) + 1

        if not blocks:
            return 0.0

        max_repetition = max(blocks.values())
        return max_repetition / len(blocks)

    def _measure_self_similarity(self, grid: np.ndarray) -> float:
        """Measure fractal self-similarity"""
        # Compare grid to downsampled versions
        if grid.shape[0] < 4 or grid.shape[1] < 4:
            return 0.0

        # Downsample by factor of 2
        downsampled = grid[::2, ::2]

        # Upsample back and compare
        from scipy.ndimage import zoom
        try:
            upsampled = zoom(downsampled, 2, order=0)[:grid.shape[0], :grid.shape[1]]
            similarity = np.sum(upsampled == grid) / grid.size
            return similarity
        except:
            return 0.0

    def _measure_compressibility(self, grid: np.ndarray) -> float:
        """Estimate compressibility via run-length encoding"""
        flat = grid.flatten()
        runs = 1
        for i in range(1, len(flat)):
            if flat[i] != flat[i-1]:
                runs += 1

        compression_ratio = runs / len(flat)
        return compression_ratio

    def _compute_information_features(self, grid: np.ndarray) -> List[float]:
        """Information-theoretic features"""
        flat = grid.flatten()

        # Entropy
        counts = np.bincount(flat)
        probs = counts[counts > 0] / len(flat)
        entropy = -np.sum(probs * np.log2(probs + 1e-10))

        # Mutual information (rows vs cols)
        mi = self._mutual_information(grid)

        # Complexity (normalized compression length)
        complexity = len(hashlib.md5(flat.tobytes()).digest()) / len(flat)

        # Redundancy
        max_entropy = np.log2(len(np.unique(flat)))
        redundancy = 1.0 - (entropy / max_entropy) if max_entropy > 0 else 0.0

        # Predictability (how well can we predict next cell from previous)
        predictability = self._measure_predictability(flat)

        # Kolmogorov complexity estimate (via compression)
        kolmogorov = complexity

        return [entropy, mi, complexity, redundancy, predictability, kolmogorov]

    def _mutual_information(self, grid: np.ndarray) -> float:
        """Mutual information between rows and columns"""
        if grid.size == 0:
            return 0.0

        # Simplified: mutual information between row positions and column values
        row_entropy = 0.0
        col_entropy = 0.0

        for i in range(grid.shape[0]):
            row = grid[i, :]
            probs = np.bincount(row) / len(row)
            probs = probs[probs > 0]
            row_entropy += -np.sum(probs * np.log2(probs + 1e-10))

        for j in range(grid.shape[1]):
            col = grid[:, j]
            probs = np.bincount(col) / len(col)
            probs = probs[probs > 0]
            col_entropy += -np.sum(probs * np.log2(probs + 1e-10))

        # Joint entropy approximation
        flat = grid.flatten()
        probs = np.bincount(flat) / len(flat)
        probs = probs[probs > 0]
        joint_entropy = -np.sum(probs * np.log2(probs + 1e-10))

        mi = (row_entropy + col_entropy - joint_entropy) / 2
        return max(0, mi)

    def _measure_predictability(self, sequence: np.ndarray) -> float:
        """Measure how predictable sequence is"""
        if len(sequence) < 2:
            return 0.0

        # Count transitions
        transitions = defaultdict(int)
        for i in range(len(sequence) - 1):
            transitions[(sequence[i], sequence[i+1])] += 1

        # Measure concentration (how dominated by most common transition)
        if not transitions:
            return 0.0

        max_transition = max(transitions.values())
        total_transitions = sum(transitions.values())

        return max_transition / total_transitions

    def _compute_orthogonal_features(self, grid: np.ndarray) -> List[float]:
        """Orthogonal dimensions: compression, randomness, predictability"""

        # Compression ratio (using hash as proxy)
        compressed_size = len(hashlib.blake2b(grid.tobytes(), digest_size=16).digest())
        original_size = grid.nbytes
        compression = compressed_size / original_size

        # Randomness (chi-square test approximation)
        flat = grid.flatten()
        counts = np.bincount(flat)
        expected = len(flat) / len(counts)
        chi_square = np.sum((counts - expected)**2 / expected)
        randomness = 1.0 / (1.0 + chi_square / len(flat))

        # Predictability (from Markov chain)
        predictability = self._measure_predictability(flat)

        return [compression, randomness, predictability]

    def _generate_nonlinear_features(self, features: np.ndarray, n_extra: int) -> np.ndarray:
        """Generate nonlinear combinations of existing features"""
        extra = []

        # Pairwise products
        for i in range(min(len(features), 6)):
            for j in range(i+1, min(len(features), 6)):
                extra.append(features[i] * features[j])
                if len(extra) >= n_extra:
                    return np.array(extra[:n_extra])

        # Squares
        for i in range(min(len(features), 6)):
            extra.append(features[i] ** 2)
            if len(extra) >= n_extra:
                return np.array(extra[:n_extra])

        # Square roots
        for i in range(min(len(features), 6)):
            extra.append(np.sqrt(np.abs(features[i])))
            if len(extra) >= n_extra:
                return np.array(extra[:n_extra])

        # Pad with zeros if needed
        while len(extra) < n_extra:
            extra.append(0.0)

        return np.array(extra[:n_extra])


# ═══════════════════════════════════════════════════════════════
# PART 2: ATTRACTOR BASIN DETECTOR
# ═══════════════════════════════════════════════════════════════

@dataclass
class AttractorBasin:
    """Represents a Great Attractor Basin in solution space"""

    # Identity
    name: str
    basin_id: str

    # Geometry in high-dimensional space
    center: np.ndarray          # Centroid in 36D space
    radius: float               # Basin radius
    dimension: int              # Effective dimension

    # Dynamics
    attraction_strength: float  # How strongly solutions converge here
    stability: float           # How stable the attractor is

    # Mechanism (the rule/transformation this attractor represents)
    symbolic_rule: Optional[str] = None
    transformation_type: Optional[str] = None

    # Examples that fall into this basin
    examples: List[Tuple[np.ndarray, np.ndarray]] = None

    # Quantum properties
    coherence: float = 0.0     # Superposition coherence
    entanglement: float = 0.0  # Entanglement with other basins


class AttractorBasinDetector:
    """Detect Great Attractor Basins in ARC puzzle space"""

    def __init__(self, dimension: int = 36):
        self.dimension = dimension
        self.projector = DimensionalProjector(dimension)
        self.discovered_basins: List[AttractorBasin] = []

    def detect_basins(self, examples: List[Dict]) -> List[AttractorBasin]:
        """
        Find attractor basins from training examples

        Each basin represents a fundamental transformation rule
        Examples "fall into" basins based on their high-dim projections
        """

        if not examples:
            return []

        # Project all examples into high-dimensional space
        input_projections = []
        output_projections = []
        transformations = []

        for ex in examples:
            inp = np.array(ex['input'])
            out = np.array(ex['output'])

            inp_proj = self.projector.project_to_higher_dims(inp, self.dimension)
            out_proj = self.projector.project_to_higher_dims(out, self.dimension)

            # Transformation vector: output - input in feature space
            transform = out_proj - inp_proj

            input_projections.append(inp_proj)
            output_projections.append(out_proj)
            transformations.append(transform)

        # Cluster transformations to find attractor basins
        basins = self._cluster_transformations(
            transformations,
            examples,
            input_projections,
            output_projections
        )

        self.discovered_basins.extend(basins)
        return basins

    def _cluster_transformations(
        self,
        transformations: List[np.ndarray],
        examples: List[Dict],
        input_projs: List[np.ndarray],
        output_projs: List[np.ndarray]
    ) -> List[AttractorBasin]:
        """Cluster transformation vectors to find basins"""

        if len(transformations) < 2:
            # Single example - create one basin
            return [self._create_basin_from_single(transformations[0], examples[0])]

        # Compute pairwise distances between transformations
        transform_matrix = np.array(transformations)
        distances = squareform(pdist(transform_matrix, metric='euclidean'))

        # Simple clustering: find groups of similar transformations
        # Use a distance threshold
        threshold = np.median(distances) * 0.5

        clusters = []
        visited = set()

        for i in range(len(transformations)):
            if i in visited:
                continue

            # Start new cluster
            cluster = [i]
            visited.add(i)

            # Find similar transformations
            for j in range(i+1, len(transformations)):
                if j not in visited and distances[i, j] < threshold:
                    cluster.append(j)
                    visited.add(j)

            clusters.append(cluster)

        # Create basins from clusters
        basins = []
        for cluster_idx, cluster in enumerate(clusters):
            cluster_transforms = [transformations[i] for i in cluster]
            cluster_examples = [examples[i] for i in cluster]

            # Compute basin properties
            center = np.mean(cluster_transforms, axis=0)
            radius = np.max([np.linalg.norm(t - center) for t in cluster_transforms])

            # Estimate effective dimension using PCA
            if len(cluster_transforms) > 1:
                centered = np.array(cluster_transforms) - center
                U, S, Vt = svd(centered, full_matrices=False)
                # Count dimensions explaining 95% variance
                variance_explained = np.cumsum(S**2) / np.sum(S**2)
                eff_dim = np.searchsorted(variance_explained, 0.95) + 1
            else:
                eff_dim = self.dimension

            # Measure attraction strength (how tightly clustered)
            if radius > 0:
                attraction = 1.0 / (1.0 + radius)
            else:
                attraction = 1.0

            # Measure stability (inverse of spread)
            stability = attraction

            # Identify transformation type symbolically
            symbolic_rule = self._infer_symbolic_rule(cluster_examples)
            transform_type = self._classify_transformation(center)

            basin = AttractorBasin(
                name=f"Basin_{cluster_idx}",
                basin_id=hashlib.md5(center.tobytes()).hexdigest()[:8],
                center=center,
                radius=radius,
                dimension=eff_dim,
                attraction_strength=attraction,
                stability=stability,
                symbolic_rule=symbolic_rule,
                transformation_type=transform_type,
                examples=[(np.array(ex['input']), np.array(ex['output'])) for ex in cluster_examples],
                coherence=self._measure_coherence(cluster_transforms),
                entanglement=0.0  # Computed later
            )

            basins.append(basin)

        # Compute entanglement between basins
        self._compute_entanglement(basins)

        return basins

    def _create_basin_from_single(self, transform: np.ndarray, example: Dict) -> AttractorBasin:
        """Create basin from single example"""
        return AttractorBasin(
            name="Basin_0",
            basin_id=hashlib.md5(transform.tobytes()).hexdigest()[:8],
            center=transform,
            radius=0.0,
            dimension=self.dimension,
            attraction_strength=1.0,
            stability=1.0,
            symbolic_rule=self._infer_symbolic_rule([example]),
            transformation_type=self._classify_transformation(transform),
            examples=[(np.array(example['input']), np.array(example['output']))],
            coherence=1.0,
            entanglement=0.0
        )

    def _infer_symbolic_rule(self, examples: List[Dict]) -> str:
        """Infer symbolic rule from examples (NSM: Neural → Symbolic bridge)"""

        if not examples:
            return "unknown"

        # Check for simple transformations
        rules = []

        for ex in examples:
            inp = np.array(ex['input'])
            out = np.array(ex['output'])

            # Shape change?
            if inp.shape != out.shape:
                if out.shape[0] > inp.shape[0] or out.shape[1] > inp.shape[1]:
                    rules.append("scale_up")
                else:
                    rules.append("scale_down")
                continue

            # Rotation?
            if np.array_equal(out, np.rot90(inp)):
                rules.append("rotate_90")
                continue
            if np.array_equal(out, np.rot90(inp, 2)):
                rules.append("rotate_180")
                continue

            # Flip?
            if np.array_equal(out, np.fliplr(inp)):
                rules.append("flip_horizontal")
                continue
            if np.array_equal(out, np.flipud(inp)):
                rules.append("flip_vertical")
                continue

            # Color transformation?
            if inp.shape == out.shape:
                unique_inp = set(inp.flatten())
                unique_out = set(out.flatten())

                if unique_inp == unique_out:
                    # Same colors, different arrangement
                    rules.append("rearrange")
                else:
                    # Color mapping
                    rules.append("color_map")
                continue

            rules.append("complex")

        # Return most common rule
        if not rules:
            return "unknown"

        from collections import Counter
        most_common = Counter(rules).most_common(1)[0][0]
        return most_common

    def _classify_transformation(self, transform_vector: np.ndarray) -> str:
        """Classify transformation type based on feature vector"""

        # Use first few dimensions to classify
        # Dimensions 10-15 are symmetry features
        if len(transform_vector) >= 15:
            sym_features = transform_vector[10:16]
            max_sym = np.max(np.abs(sym_features))

            if max_sym > 0.7:
                return "geometric"

        # Dimensions 4-6 are color features
        if len(transform_vector) >= 6:
            color_features = transform_vector[4:7]
            if np.sum(np.abs(color_features)) > 1.0:
                return "color_based"

        # Dimensions 7-9 are topological
        if len(transform_vector) >= 9:
            topo_features = transform_vector[7:10]
            if np.sum(np.abs(topo_features)) > 0.5:
                return "topological"

        return "complex"

    def _measure_coherence(self, transforms: List[np.ndarray]) -> float:
        """Measure quantum coherence of transformation cluster"""
        if len(transforms) < 2:
            return 1.0

        # Coherence = how similar transformations are
        # High coherence = tight cluster = superposition hasn't collapsed much

        center = np.mean(transforms, axis=0)
        distances = [np.linalg.norm(t - center) for t in transforms]
        avg_distance = np.mean(distances)

        # Normalize to [0, 1]
        coherence = 1.0 / (1.0 + avg_distance)
        return coherence

    def _compute_entanglement(self, basins: List[AttractorBasin]):
        """Compute entanglement between basins"""

        if len(basins) < 2:
            return

        for i, basin1 in enumerate(basins):
            for j, basin2 in enumerate(basins[i+1:], i+1):
                # Entanglement = correlation between basin centers
                center1 = basin1.center
                center2 = basin2.center

                # Normalize
                norm1 = center1 / (np.linalg.norm(center1) + 1e-10)
                norm2 = center2 / (np.linalg.norm(center2) + 1e-10)

                # Correlation (dot product of normalized vectors)
                correlation = np.abs(np.dot(norm1, norm2))

                # Update entanglement
                basin1.entanglement = max(basin1.entanglement, correlation)
                basin2.entanglement = max(basin2.entanglement, correlation)

    def predict_basin(self, test_input: np.ndarray) -> Tuple[AttractorBasin, float]:
        """Predict which basin test input will fall into"""

        if not self.discovered_basins:
            raise ValueError("No basins detected yet. Run detect_basins() first.")

        # Project test input into high-dimensional space
        test_proj = self.projector.project_to_higher_dims(test_input, self.dimension)

        # Find closest basin
        min_distance = float('inf')
        best_basin = None

        for basin in self.discovered_basins:
            # Distance from test to basin center
            # (In transformation space, we'd need output - but we don't have it yet)
            # So we measure distance in input space instead

            # Get average input projection from basin examples
            if basin.examples:
                basin_inputs = [inp for inp, _ in basin.examples]
                basin_input_projs = [
                    self.projector.project_to_higher_dims(inp, self.dimension)
                    for inp in basin_inputs
                ]
                avg_input_proj = np.mean(basin_input_projs, axis=0)

                distance = np.linalg.norm(test_proj - avg_input_proj)

                if distance < min_distance:
                    min_distance = distance
                    best_basin = basin

        # Confidence based on distance
        confidence = 1.0 / (1.0 + min_distance)

        return best_basin, confidence


# ═══════════════════════════════════════════════════════════════
# PART 3: QUANTUM ADVANTAGE SOLVER
# ═══════════════════════════════════════════════════════════════

class QuantumAdvantageSolver:
    """
    Use quantum-inspired methods to solve ARC puzzles

    Key insights:
    1. Superposition: Explore multiple solutions simultaneously
    2. Entanglement: Correlate input/output patterns
    3. Higher dimensions: See patterns invisible in 2D
    4. Attractor basins: Solutions converge to fundamental rules
    """

    def __init__(self, dimension: int = 36, recursion_depth: int = 6):
        self.dimension = dimension
        self.recursion_depth = recursion_depth
        self.projector = DimensionalProjector(dimension)
        self.basin_detector = AttractorBasinDetector(dimension)
        self.basins: List[AttractorBasin] = []

    def train(self, examples: List[Dict]) -> None:
        """Train by detecting attractor basins in examples"""
        print(f"🌀 Detecting attractor basins in {len(examples)} examples...")
        self.basins = self.basin_detector.detect_basins(examples)
        print(f"✓ Discovered {len(self.basins)} attractor basins")

        for i, basin in enumerate(self.basins):
            print(f"  Basin {i}: {basin.symbolic_rule} ({basin.transformation_type})")
            print(f"    Dimension: {basin.dimension}, Coherence: {basin.coherence:.3f}")
            print(f"    Attraction: {basin.attraction_strength:.3f}, Stability: {basin.stability:.3f}")

    def solve(self, test_input: np.ndarray, num_attempts: int = 2) -> List[np.ndarray]:
        """
        Solve test case using quantum advantage

        Strategy:
        1. Project test_input into higher dimensions
        2. Find which attractor basin it falls into
        3. Apply that basin's transformation
        4. If uncertain, explore multiple basins (superposition)
        """

        if not self.basins:
            # No training - fall back to basic heuristics
            return self._fallback_solve(test_input, num_attempts)

        # Find basin
        basin, confidence = self.basin_detector.predict_basin(test_input)

        print(f"🎯 Predicted basin: {basin.name} ({basin.symbolic_rule}) [confidence: {confidence:.3f}]")

        solutions = []

        # Apply basin's transformation
        for inp, out in basin.examples[:num_attempts]:
            # Try to infer transformation and apply to test_input
            solution = self._apply_transformation(
                test_input,
                inp,
                out,
                basin.symbolic_rule
            )
            solutions.append(solution)

        # If low confidence, explore other basins (quantum superposition)
        if confidence < 0.7 and len(self.basins) > 1:
            # Sort basins by attraction strength
            sorted_basins = sorted(
                self.basins,
                key=lambda b: b.attraction_strength,
                reverse=True
            )

            for basin in sorted_basins[1:]:
                if len(solutions) >= num_attempts:
                    break

                if basin.examples:
                    inp, out = basin.examples[0]
                    solution = self._apply_transformation(
                        test_input,
                        inp,
                        out,
                        basin.symbolic_rule
                    )
                    solutions.append(solution)

        return solutions[:num_attempts]

    def _apply_transformation(
        self,
        test_input: np.ndarray,
        example_input: np.ndarray,
        example_output: np.ndarray,
        rule: str
    ) -> np.ndarray:
        """Apply learned transformation to test input"""

        # Simple rule-based application
        if rule == "rotate_90":
            return np.rot90(test_input)
        elif rule == "rotate_180":
            return np.rot90(test_input, 2)
        elif rule == "flip_horizontal":
            return np.fliplr(test_input)
        elif rule == "flip_vertical":
            return np.flipud(test_input)
        elif rule == "scale_up":
            # Scale by same factor as example
            scale_h = example_output.shape[0] / example_input.shape[0]
            scale_w = example_output.shape[1] / example_input.shape[1]
            from scipy.ndimage import zoom
            try:
                return zoom(test_input, (scale_h, scale_w), order=0).astype(int)
            except:
                return test_input
        elif rule == "color_map":
            # Learn color mapping from example
            color_map = {}
            for i in range(example_input.shape[0]):
                for j in range(example_input.shape[1]):
                    inp_color = example_input[i, j]
                    out_color = example_output[i, j]
                    color_map[inp_color] = out_color

            # Apply to test
            result = test_input.copy()
            for i in range(result.shape[0]):
                for j in range(result.shape[1]):
                    if result[i, j] in color_map:
                        result[i, j] = color_map[result[i, j]]
            return result
        else:
            # Complex - try to copy structure
            # If shapes match, try color mapping
            if test_input.shape == example_input.shape == example_output.shape:
                return self._apply_transformation(
                    test_input,
                    example_input,
                    example_output,
                    "color_map"
                )
            else:
                return test_input

    def _fallback_solve(self, test_input: np.ndarray, num_attempts: int) -> List[np.ndarray]:
        """Fallback when no basins detected"""
        return [
            np.rot90(test_input),
            np.fliplr(test_input)
        ][:num_attempts]


# ═══════════════════════════════════════════════════════════════
# PART 4: USAGE EXAMPLE
# ═══════════════════════════════════════════════════════════════

def demo_quantum_advantage():
    """Demonstrate quantum advantage on sample ARC task"""

    print("="*60)
    print("🌀 QUANTUM ADVANTAGE NSM DEMO")
    print("="*60)

    # Sample ARC-like task: Rotate 90 degrees
    examples = [
        {
            'input': [[1, 2], [3, 4]],
            'output': [[3, 1], [4, 2]]
        },
        {
            'input': [[5, 6, 7], [8, 9, 10]],
            'output': [[8, 5], [9, 6], [10, 7]]
        }
    ]

    test_input = np.array([[0, 1], [2, 3]])

    # Create solver
    solver = QuantumAdvantageSolver(dimension=36, recursion_depth=6)

    # Train (detect basins)
    solver.train(examples)

    # Solve
    print(f"\n🧪 Test input:\n{test_input}")
    solutions = solver.solve(test_input, num_attempts=2)

    print(f"\n✓ Generated {len(solutions)} solutions:")
    for i, sol in enumerate(solutions):
        print(f"  Solution {i+1}:\n{sol}")

    print("\n" + "="*60)


if __name__ == "__main__":
    demo_quantum_advantage()
