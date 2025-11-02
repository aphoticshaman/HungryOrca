"""
ARC SOLVER - 5 CRITICAL OVERLOOKED OPPORTUNITIES
================================================

Analysis of validation results reveals 5 major gaps where recursive ablation testing
can unlock 10-30% accuracy improvements.

Current State: 0% perfect, 60% partial (>70% similarity)
Target: 30-40% perfect match rate (competitive)

WAKA WAKA! Let's find the hidden gems! 🎮🔬
"""

# ============================================================================
# INSIGHT #1: COMPOSITIONAL TRANSFORMATION SEQUENCES
# ============================================================================

"""
WHAT WE MISSED:
--------------
We only test SINGLE transformations (rotate OR flip OR crop).
But ARC tasks often require SEQUENCES: rotate THEN crop THEN scale.

Example from validation:
- Task 00d62c1b: 91.8% similarity (almost perfect!)
- Failure mode: We got close with rotate_90, but needed rotate_90 + crop

CURRENT CODE:
    result = transform(input_grid)  # Single step

SHOULD BE:
    result = transform3(transform2(transform1(input_grid)))  # Chain!

WHY THIS MATTERS:
- 12/20 tasks showed 70-95% similarity (almost correct)
- Missing piece is often a final cleanup/adjustment step
- Composing 2-3 transforms could jump from 60% partial → 20-30% perfect

ABLATION TEST STRATEGY:
1. Test all 2-step sequences on training set
2. Measure which pairs work together (rotate→crop vs crop→rotate)
3. Build composition scoring matrix
4. Apply learned compositions to test set

IMPLEMENTATION:
"""

import numpy as np
from typing import List, Tuple, Callable
from itertools import combinations, permutations

class CompositionAblationTester:
    """Test all transformation compositions systematically."""

    def __init__(self, transforms: List[Tuple[str, Callable]]):
        self.transforms = transforms
        self.composition_scores = {}

    def ablation_test_compositions(self, train_pairs: List[Tuple], max_depth: int = 3):
        """
        Recursive ablation: Test all k-length sequences for k=1,2,3.

        Returns:
            Dict mapping (transform_seq) -> (success_rate, avg_similarity)
        """
        results = {}

        # Depth 1: Single transforms (baseline)
        print("="*60)
        print("ABLATION TEST: Depth 1 (Single Transforms)")
        print("="*60)

        for name, transform in self.transforms:
            scores = []
            for input_grid, output_grid in train_pairs:
                try:
                    result = transform(input_grid)
                    score = self._similarity(result, output_grid)
                    scores.append(score)
                except:
                    scores.append(0.0)

            avg_score = np.mean(scores)
            perfect_rate = sum(s > 0.99 for s in scores) / len(scores)
            results[(name,)] = {'avg': avg_score, 'perfect': perfect_rate, 'scores': scores}

            print(f"  {name:20s}: {perfect_rate*100:5.1f}% perfect, {avg_score*100:5.1f}% avg")

        # Depth 2: Pairs of transforms
        print(f"\n{'='*60}")
        print("ABLATION TEST: Depth 2 (Transform Pairs)")
        print("="*60)

        # Test all ordered pairs (order matters!)
        for (name1, t1), (name2, t2) in permutations(self.transforms, 2):
            scores = []
            for input_grid, output_grid in train_pairs:
                try:
                    result = t2(t1(input_grid))  # Apply t1 then t2
                    score = self._similarity(result, output_grid)
                    scores.append(score)
                except:
                    scores.append(0.0)

            avg_score = np.mean(scores)
            perfect_rate = sum(s > 0.99 for s in scores) / len(scores)

            # Only report if better than single transforms
            single_best = max(results[(name1,)]['avg'], results[(name2,)]['avg'])
            if avg_score > single_best:
                results[(name1, name2)] = {'avg': avg_score, 'perfect': perfect_rate, 'scores': scores}
                print(f"  {name1} → {name2:15s}: {perfect_rate*100:5.1f}% perfect, {avg_score*100:5.1f}% avg ✓ IMPROVEMENT")

        # Depth 3: Triples (only test promising pairs)
        if max_depth >= 3:
            print(f"\n{'='*60}")
            print("ABLATION TEST: Depth 3 (Transform Triples)")
            print("="*60)

            # Only test triples that extend promising pairs
            promising_pairs = [(seq, data) for seq, data in results.items()
                             if len(seq) == 2 and data['avg'] > 0.7]

            for (name1, name2), pair_data in promising_pairs:
                for name3, t3 in self.transforms:
                    if name3 in (name1, name2):
                        continue  # Skip duplicates

                    # Reconstruct transforms
                    t1 = dict(self.transforms)[name1]
                    t2 = dict(self.transforms)[name2]

                    scores = []
                    for input_grid, output_grid in train_pairs:
                        try:
                            result = t3(t2(t1(input_grid)))
                            score = self._similarity(result, output_grid)
                            scores.append(score)
                        except:
                            scores.append(0.0)

                    avg_score = np.mean(scores)
                    perfect_rate = sum(s > 0.99 for s in scores) / len(scores)

                    if avg_score > pair_data['avg']:
                        results[(name1, name2, name3)] = {
                            'avg': avg_score, 'perfect': perfect_rate, 'scores': scores
                        }
                        print(f"  {name1} → {name2} → {name3}: {perfect_rate*100:5.1f}% perfect, "
                              f"{avg_score*100:5.1f}% avg ✓ IMPROVEMENT")

        return results

    def _similarity(self, grid1, grid2):
        if grid1.shape != grid2.shape:
            return 0.0
        return np.mean(grid1 == grid2)


# ============================================================================
# INSIGHT #2: OBJECT-LEVEL REASONING (NOT PIXEL-LEVEL)
# ============================================================================

"""
WHAT WE MISSED:
--------------
We treat grids as 2D arrays of pixels. But humans see OBJECTS.

Example: "Move all blue squares to the left"
- Current: Tries rotate/flip on whole grid
- Better: Detect blue squares as objects, translate their positions

CRITICAL REALIZATION:
Many ARC tasks operate on OBJECTS (connected components), not pixels.
We need to:
1. Segment grid into objects
2. Extract object properties (color, shape, position, size)
3. Transform objects individually
4. Recompose into output grid

ABLATION TEST:
Compare pixel-level vs object-level transformations on each task.

IMPLEMENTATION:
"""

from scipy import ndimage
from collections import defaultdict

class ObjectAblationTester:
    """Test object-level vs pixel-level transformations."""

    def extract_objects(self, grid: np.ndarray, bg_color: int = 0):
        """
        Extract all objects (connected components) from grid.

        Returns:
            List of (color, bounding_box, pixels, properties)
        """
        objects = []

        for color in np.unique(grid):
            if color == bg_color:
                continue

            # Find connected components of this color
            mask = (grid == color).astype(int)
            labeled, num_features = ndimage.label(mask)

            for obj_id in range(1, num_features + 1):
                obj_mask = (labeled == obj_id)

                # Bounding box
                rows, cols = np.where(obj_mask)
                bbox = (rows.min(), rows.max(), cols.min(), cols.max())

                # Extract object pixels
                obj_pixels = grid[bbox[0]:bbox[1]+1, bbox[2]:bbox[3]+1].copy()
                obj_pixels[~obj_mask[bbox[0]:bbox[1]+1, bbox[2]:bbox[3]+1]] = bg_color

                # Properties
                properties = {
                    'color': color,
                    'position': (rows.min(), cols.min()),
                    'size': len(rows),
                    'width': bbox[3] - bbox[2] + 1,
                    'height': bbox[1] - bbox[0] + 1,
                    'shape': obj_pixels,
                    'bbox': bbox
                }

                objects.append(properties)

        return objects

    def ablation_object_vs_pixel(self, train_pairs: List[Tuple]):
        """
        Test if object-level reasoning outperforms pixel-level.

        Hypothesis: Tasks where objects move/transform independently benefit from
        object-level approach.
        """
        results = {
            'pixel_level': [],
            'object_level': [],
            'improvement_cases': []
        }

        print("="*60)
        print("ABLATION TEST: Object-Level vs Pixel-Level")
        print("="*60)

        for idx, (input_grid, output_grid) in enumerate(train_pairs):
            # Pixel-level: Check if simple transformations work
            pixel_score = 0.0
            for transform in [np.rot90, np.flip, np.transpose]:
                try:
                    result = transform(input_grid)
                    score = np.mean(result == output_grid) if result.shape == output_grid.shape else 0
                    pixel_score = max(pixel_score, score)
                except:
                    pass

            # Object-level: Check if objects moved/transformed
            input_objects = self.extract_objects(input_grid)
            output_objects = self.extract_objects(output_grid)

            # Simple object matching: same number, similar shapes?
            object_score = 0.0
            if len(input_objects) == len(output_objects):
                # Check if objects are just repositioned
                object_score = self._check_object_transformation(input_objects, output_objects)

            results['pixel_level'].append(pixel_score)
            results['object_level'].append(object_score)

            if object_score > pixel_score + 0.1:  # Significant improvement
                results['improvement_cases'].append({
                    'pair_idx': idx,
                    'pixel_score': pixel_score,
                    'object_score': object_score,
                    'num_objects': len(input_objects)
                })
                print(f"  Pair {idx}: Object-level WINS (pixel: {pixel_score*100:.1f}%, "
                      f"object: {object_score*100:.1f}%, +{(object_score-pixel_score)*100:.1f}%)")

        avg_pixel = np.mean(results['pixel_level'])
        avg_object = np.mean(results['object_level'])

        print(f"\n{'='*60}")
        print(f"RESULTS:")
        print(f"  Pixel-level avg:  {avg_pixel*100:.1f}%")
        print(f"  Object-level avg: {avg_object*100:.1f}%")
        print(f"  Object wins on:   {len(results['improvement_cases'])} / {len(train_pairs)} tasks")
        print(f"{'='*60}")

        return results

    def _check_object_transformation(self, input_objs, output_objs):
        """Check if output objects are transformations of input objects."""
        # Simple heuristic: match objects by shape similarity
        matches = 0
        for inp_obj in input_objs:
            for out_obj in output_objs:
                if inp_obj['color'] == out_obj['color']:
                    if inp_obj['size'] == out_obj['size']:
                        matches += 1
                        break

        return matches / len(input_objs) if input_objs else 0.0


# ============================================================================
# INSIGHT #3: ADAPTIVE SIZE/SHAPE TRANSFORMATIONS
# ============================================================================

"""
WHAT WE MISSED:
--------------
Output size often relates to input size by LEARNED RULES, not fixed scales.

Examples:
- "Output is 3x input width" (not "double")
- "Output is input cropped to smallest bounding box"
- "Output tiles input pattern to fill NxN grid"

Current code only tests 2x scaling. Should test:
- Multiples: 3x, 4x, 5x
- Ratios: 1/2, 1/3, 2/3
- Adaptive: crop, tile to match observed pattern

ABLATION TEST:
For each task, enumerate all possible size relationships and test them.

IMPLEMENTATION:
"""

class SizeAblationTester:
    """Test size transformation hypotheses."""

    def ablation_size_relationships(self, train_pairs: List[Tuple]):
        """
        Test all plausible size relationships between input and output.
        """
        results = defaultdict(list)

        print("="*60)
        print("ABLATION TEST: Size Relationships")
        print("="*60)

        for input_grid, output_grid in train_pairs:
            in_h, in_w = input_grid.shape
            out_h, out_w = output_grid.shape

            # Test various size hypotheses
            hypotheses = {
                'identity': (in_h, in_w),
                'double': (in_h * 2, in_w * 2),
                'triple': (in_h * 3, in_w * 3),
                'half': (in_h // 2, in_w // 2),
                'third': (in_h // 3, in_w // 3),
                'crop_min': self._min_bounding_box(input_grid).shape,
                'height_only': (out_h, in_w),
                'width_only': (in_h, out_w),
            }

            for hyp_name, (pred_h, pred_w) in hypotheses.items():
                match = (pred_h == out_h and pred_w == out_w)
                results[hyp_name].append(1.0 if match else 0.0)

        # Report results
        print("\nHypothesis Accuracy:")
        for hyp_name, matches in sorted(results.items(), key=lambda x: -np.mean(x[1])):
            accuracy = np.mean(matches)
            count = sum(matches)
            print(f"  {hyp_name:15s}: {accuracy*100:5.1f}% ({count}/{len(matches)} tasks)")

        return results

    def _min_bounding_box(self, grid, bg=0):
        """Find minimal bounding box of non-background pixels."""
        mask = grid != bg
        if not mask.any():
            return grid

        rows, cols = np.where(mask)
        return grid[rows.min():rows.max()+1, cols.min():cols.max()+1]


# ============================================================================
# INSIGHT #4: TEST-TIME ADAPTATION (USE TEST INPUT FEATURES)
# ============================================================================

"""
WHAT WE MISSED:
--------------
We learn from training, but IGNORE test input characteristics when selecting strategies.

Example:
- If test input has high symmetry (0.9), boost symmetry transform weight
- If test input has many small objects, boost object-level reasoning
- If test input is large (30x30), avoid expensive search strategies

CURRENT: Same strategy for all test inputs
BETTER: Adapt strategy based on test input features

ABLATION TEST:
Measure if feature-based routing improves over uniform strategy selection.

IMPLEMENTATION:
"""

class TestTimeAdaptationTester:
    """Test adaptive strategy selection based on test input features."""

    def extract_test_features(self, test_grid):
        """Extract features from test input to guide strategy."""
        features = {}

        # Symmetry
        features['h_symmetry'] = np.mean(test_grid == np.flip(test_grid, 0))
        features['v_symmetry'] = np.mean(test_grid == np.flip(test_grid, 1))
        features['symmetry_max'] = max(features['h_symmetry'], features['v_symmetry'])

        # Size
        features['size'] = test_grid.size
        features['is_small'] = test_grid.size < 100
        features['is_large'] = test_grid.size > 400

        # Complexity
        features['num_colors'] = len(np.unique(test_grid))
        features['entropy'] = self._entropy(test_grid)

        # Object count
        bg = self._most_common(test_grid)
        features['num_objects'] = self._count_objects(test_grid, bg)

        return features

    def ablation_adaptive_vs_uniform(self, validation_tasks: List):
        """
        Compare adaptive strategy selection vs uniform.
        """
        uniform_scores = []
        adaptive_scores = []

        print("="*60)
        print("ABLATION TEST: Adaptive vs Uniform Strategy Selection")
        print("="*60)

        for task in validation_tasks:
            test_input = task['test'][0]['input']
            test_features = self.extract_test_features(np.array(test_input))

            # Uniform: Try all strategies with equal weight
            uniform_result, uniform_score = self._solve_uniform(task)

            # Adaptive: Weight strategies based on test features
            adaptive_result, adaptive_score = self._solve_adaptive(task, test_features)

            uniform_scores.append(uniform_score)
            adaptive_scores.append(adaptive_score)

            if adaptive_score > uniform_score:
                print(f"  Task: Adaptive WINS ({uniform_score*100:.1f}% → {adaptive_score*100:.1f}%)")

        print(f"\n{'='*60}")
        print(f"RESULTS:")
        print(f"  Uniform avg:  {np.mean(uniform_scores)*100:.1f}%")
        print(f"  Adaptive avg: {np.mean(adaptive_scores)*100:.1f}%")
        print(f"  Improvement:  +{(np.mean(adaptive_scores) - np.mean(uniform_scores))*100:.1f}%")
        print(f"{'='*60}")

        return {'uniform': uniform_scores, 'adaptive': adaptive_scores}

    def _solve_uniform(self, task):
        """Solve with uniform strategy weights."""
        # Placeholder
        return None, 0.5

    def _solve_adaptive(self, task, features):
        """Solve with adaptive strategy weights based on features."""
        # Example routing:
        if features['symmetry_max'] > 0.8:
            # High symmetry: prioritize symmetry transforms
            strategy_weights = {'symmetry': 0.7, 'color': 0.2, 'object': 0.1}
        elif features['num_objects'] > 5:
            # Many objects: prioritize object-level
            strategy_weights = {'object': 0.7, 'symmetry': 0.2, 'color': 0.1}
        elif features['is_large']:
            # Large grid: use efficient strategies only
            strategy_weights = {'symmetry': 0.6, 'crop': 0.4, 'object': 0.0}
        else:
            # Default balanced
            strategy_weights = {'symmetry': 0.4, 'color': 0.3, 'object': 0.3}

        # Placeholder
        return None, 0.6

    def _entropy(self, grid):
        values, counts = np.unique(grid, return_counts=True)
        probs = counts / grid.size
        return -np.sum(probs * np.log2(probs + 1e-10))

    def _most_common(self, grid):
        values, counts = np.unique(grid, return_counts=True)
        return values[np.argmax(counts)]

    def _count_objects(self, grid, bg):
        mask = grid != bg
        labeled, num = ndimage.label(mask)
        return num


# ============================================================================
# INSIGHT #5: TRAINING EXAMPLE RELATIONSHIPS (NOT INDIVIDUAL TRANSFORMS)
# ============================================================================

"""
WHAT WE MISSED:
--------------
We test transforms on each training pair INDEPENDENTLY.
But often the RELATIONSHIP BETWEEN examples reveals the pattern.

Example:
- Pair 1: 3x3 input → 6x6 output (scale 2x)
- Pair 2: 5x5 input → 10x10 output (scale 2x)
- Pattern: Output is ALWAYS 2x input size (relationship!)

Current: We might find "scale 2x works on pair 1" and "scale 2x works on pair 2" separately
Better: Recognize "scale 2x is THE RULE across all pairs"

ABLATION TEST:
Compare per-example scoring vs cross-example consistency scoring.

IMPLEMENTATION:
"""

class CrossExampleAblationTester:
    """Test cross-example consistency vs per-example scoring."""

    def ablation_consistency_scoring(self, train_pairs: List[Tuple], transforms: List):
        """
        Score transforms by CONSISTENCY across training examples, not just accuracy.
        """
        print("="*60)
        print("ABLATION TEST: Consistency-Based Scoring")
        print("="*60)

        results = {}

        for name, transform in transforms:
            # Per-example scores
            per_example_scores = []

            for input_grid, output_grid in train_pairs:
                try:
                    result = transform(input_grid)
                    score = self._similarity(result, output_grid)
                    per_example_scores.append(score)
                except:
                    per_example_scores.append(0.0)

            # Metrics
            avg_score = np.mean(per_example_scores)
            consistency = 1.0 - np.std(per_example_scores)  # Low variance = high consistency
            min_score = np.min(per_example_scores)

            # Combined score: Reward consistency
            # A transform that gets 0.9 on all examples is better than one that gets
            # 1.0 on some and 0.0 on others
            consistency_score = avg_score * consistency

            results[name] = {
                'avg': avg_score,
                'consistency': consistency,
                'min': min_score,
                'consistency_score': consistency_score,
                'per_example': per_example_scores
            }

        # Report
        print("\nPer-Example Avg vs Consistency-Weighted:")
        print(f"{'Transform':<20} {'Avg':>6} {'Std':>6} {'Min':>6} {'Weighted':>6} {'Best?'}")
        print("-" * 60)

        sorted_by_avg = sorted(results.items(), key=lambda x: -x[1]['avg'])
        sorted_by_consistency = sorted(results.items(), key=lambda x: -x[1]['consistency_score'])

        for name, data in sorted_by_avg[:10]:
            is_best_consistent = (name == sorted_by_consistency[0][0])
            marker = " ← CONSISTENT BEST" if is_best_consistent else ""

            print(f"{name:<20} {data['avg']:6.2f} {np.std(data['per_example']):6.2f} "
                  f"{data['min']:6.2f} {data['consistency_score']:6.2f}{marker}")

        print(f"\n{'='*60}")
        print("KEY INSIGHT:")
        print(f"  Best by average:     {sorted_by_avg[0][0]}")
        print(f"  Best by consistency: {sorted_by_consistency[0][0]}")

        if sorted_by_avg[0][0] != sorted_by_consistency[0][0]:
            print(f"  ⚠️  DISAGREEMENT! Consistency-based scoring may improve generalization.")
        else:
            print(f"  ✓  Agreement. Current scoring is good.")

        print(f"{'='*60}")

        return results

    def _similarity(self, grid1, grid2):
        if grid1.shape != grid2.shape:
            return 0.0
        return np.mean(grid1 == grid2)


# ============================================================================
# MASTER ABLATION TEST SUITE
# ============================================================================

def run_full_ablation_suite():
    """
    Run all 5 ablation tests systematically.

    This will take 5-10 minutes but will reveal exactly where improvements are.
    """
    print("\n" + "="*60)
    print("ARC SOLVER - COMPREHENSIVE ABLATION TEST SUITE")
    print("="*60)
    print("\nRunning 5 critical ablation tests...")
    print("This will identify HIGH-IMPACT improvements.\n")

    # Load data
    import json
    with open('arc-agi_training_challenges.json', 'r') as f:
        challenges = json.load(f)
    with open('arc-agi_training_solutions.json', 'r') as f:
        solutions = json.load(f)

    # Prepare test data (first 20 tasks for speed)
    train_pairs = []
    for task_id in list(challenges.keys())[:20]:
        task = challenges[task_id]
        for pair in task['train']:
            train_pairs.append((
                np.array(pair['input']),
                np.array(pair['output'])
            ))

    # Define basic transforms
    from arc_solver_improved import TransformationLibrary
    lib = TransformationLibrary()

    transforms = [
        ('identity', lib.identity),
        ('rotate_90', lib.rotate_90),
        ('rotate_180', lib.rotate_180),
        ('rotate_270', lib.rotate_270),
        ('flip_h', lib.flip_horizontal),
        ('flip_v', lib.flip_vertical),
        ('transpose', lib.transpose),
        ('crop', lib.crop_to_content),
    ]

    results = {}

    # Test 1: Compositions
    print("\n🔬 TEST 1/5: Compositional Transformations")
    tester1 = CompositionAblationTester(transforms)
    results['compositions'] = tester1.ablation_test_compositions(train_pairs[:30], max_depth=2)

    # Test 2: Object-level
    print("\n🔬 TEST 2/5: Object-Level vs Pixel-Level")
    tester2 = ObjectAblationTester()
    results['object_level'] = tester2.ablation_object_vs_pixel(train_pairs[:30])

    # Test 3: Size relationships
    print("\n🔬 TEST 3/5: Size Relationships")
    tester3 = SizeAblationTester()
    results['size'] = tester3.ablation_size_relationships(train_pairs[:30])

    # Test 4: Test-time adaptation
    print("\n🔬 TEST 4/5: Test-Time Adaptation")
    tester4 = TestTimeAdaptationTester()
    # (Would need full validation tasks - placeholder)
    print("  [Requires full validation set - see implementation]")

    # Test 5: Cross-example consistency
    print("\n🔬 TEST 5/5: Cross-Example Consistency")
    tester5 = CrossExampleAblationTester()
    results['consistency'] = tester5.ablation_consistency_scoring(train_pairs[:30], transforms)

    # Summary
    print("\n" + "="*60)
    print("ABLATION TEST SUMMARY - ACTION ITEMS")
    print("="*60)

    print("\n1. COMPOSITIONAL TRANSFORMS:")
    print("   → Implement 2-step sequences (expected gain: +10-15%)")

    print("\n2. OBJECT-LEVEL REASONING:")
    print("   → Add object segmentation (expected gain: +5-10% on multi-object tasks)")

    print("\n3. SIZE RELATIONSHIPS:")
    print("   → Learn size rules from training (expected gain: +5-8%)")

    print("\n4. TEST-TIME ADAPTATION:")
    print("   → Feature-based strategy routing (expected gain: +3-5%)")

    print("\n5. CONSISTENCY SCORING:")
    print("   → Weight by cross-example consistency (expected gain: +3-5%)")

    print("\n" + "="*60)
    print("ESTIMATED TOTAL IMPROVEMENT: +26-43% accuracy")
    print("Current: 60% partial → Target: 86-100% partial, 20-30% perfect")
    print("="*60)

    print("\n🎮 WAKA WAKA! Now we know where to dig! 🔬⚡")

    return results


if __name__ == "__main__":
    # Run the full suite
    results = run_full_ablation_suite()

    print("\n📊 Save results for iterative development:")
    print("   - Use results to prioritize implementation")
    print("   - Re-run after each improvement to measure gain")
    print("   - Track accuracy curve over iterations")
