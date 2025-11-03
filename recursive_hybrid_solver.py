#!/usr/bin/env python3
"""
RECURSIVE HYBRID SOLVER

The missing piece: Recursive iteration with lesson-sharing between methods.

Key insight: Don't just try ONE approach. ITERATE:
1. Try method A
2. Learn why it failed
3. Pass lessons to method B
4. Try method B with lessons
5. Combine learnings
6. Try again

This is why we were getting 0% - no iteration!

Author: HungryOrca Phase 7 Week 2
Date: 2025-11-02
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from collections import Counter
from fuzzy_transformation_solver import TransformationLibrary, PatternMatcher, SimpleFeatureExtractor


class RecursiveHybridSolver:
    """
    Solver that ITERATES and LEARNS.

    Instead of: Try once → fail → done
    We do: Try → learn lesson → try different way → learn → combine → try again
    """

    def __init__(self, max_iterations: int = 5):
        self.transforms = TransformationLibrary()
        self.pattern_matcher = PatternMatcher()
        self.feature_extractor = SimpleFeatureExtractor()
        self.max_iterations = max_iterations

    def solve(self, train_pairs: List[Tuple[np.ndarray, np.ndarray]],
              test_input: np.ndarray) -> Optional[np.ndarray]:
        """
        Recursive hybrid solving with lesson-sharing.
        """
        # Extract features
        features = self.feature_extractor.extract(train_pairs, test_input)

        # Lesson storage (shared between iterations)
        lessons = {
            'tried_transforms': [],
            'failed_transforms': [],
            'partial_successes': [],
            'color_mappings': {},
            'size_patterns': {},
            'best_score': 0.0,
            'best_candidate': None
        }

        # Iteration 1: Try learned transformations
        candidate = self._try_learned_transforms(train_pairs, test_input, lessons)
        if candidate is not None:
            lessons['best_candidate'] = candidate

        # Iteration 2: Learn from failures, try compositions
        if lessons['failed_transforms']:
            candidate = self._try_compositions(train_pairs, test_input, lessons)
            if candidate is not None and self._score_vs_training(candidate, train_pairs, test_input) > lessons['best_score']:
                lessons['best_candidate'] = candidate

        # Iteration 3: Try pattern-based synthesis
        candidate = self._try_pattern_synthesis(train_pairs, test_input, lessons)
        if candidate is not None and self._score_vs_training(candidate, train_pairs, test_input) > lessons['best_score']:
            lessons['best_candidate'] = candidate

        # Iteration 4: Try color-focused approach
        candidate = self._try_color_focused(train_pairs, test_input, lessons)
        if candidate is not None and self._score_vs_training(candidate, train_pairs, test_input) > lessons['best_score']:
            lessons['best_candidate'] = candidate

        # Iteration 5: Try size-focused approach
        candidate = self._try_size_focused(train_pairs, test_input, lessons)
        if candidate is not None and self._score_vs_training(candidate, train_pairs, test_input) > lessons['best_score']:
            lessons['best_candidate'] = candidate

        return lessons['best_candidate']

    def _try_learned_transforms(self, train_pairs, test_input, lessons) -> Optional[np.ndarray]:
        """Iteration 1: Try learned transformations."""
        all_transforms = []

        for inp, out in train_pairs:
            transforms = self.pattern_matcher.find_best_transforms(inp, out)
            all_transforms.extend(transforms)

        # Aggregate
        transform_scores = {}
        transform_funcs = {}

        for name, func, score in all_transforms:
            if name not in transform_scores:
                transform_scores[name] = []
                transform_funcs[name] = func
            transform_scores[name].append(score)
            lessons['tried_transforms'].append(name)

        # Try best transform
        if transform_scores:
            best_name = max(transform_scores.items(), key=lambda x: np.mean(x[1]))[0]
            try:
                result = transform_funcs[best_name](test_input)
                score = self._score_vs_training(result, train_pairs, test_input)
                lessons['best_score'] = score

                if score < 1.0:
                    lessons['failed_transforms'].append(best_name)
                else:
                    return result  # Perfect!

                return result
            except:
                lessons['failed_transforms'].append(best_name)

        return None

    def _try_compositions(self, train_pairs, test_input, lessons) -> Optional[np.ndarray]:
        """Iteration 2: Learn from failures, try COMPOSING transforms."""
        # Lesson: If single transform failed, try combining them

        candidates = []

        # Try all pairwise compositions of transforms that partially succeeded
        for inp, out in train_pairs:
            transforms = self.pattern_matcher.find_best_transforms(inp, out)

            # Try composing top 2 transforms
            if len(transforms) >= 2:
                name1, func1, score1 = transforms[0]
                name2, func2, score2 = transforms[1]

                try:
                    # Compose: func2(func1(input))
                    intermediate = func1(test_input)
                    result = func2(intermediate)
                    score = self._score_vs_training(result, train_pairs, test_input)
                    candidates.append((result, score, f"{name1}+{name2}"))
                except:
                    pass

        if candidates:
            candidates.sort(key=lambda x: x[1], reverse=True)
            best_result, best_score, best_name = candidates[0]

            if best_score > lessons['best_score']:
                lessons['best_score'] = best_score
                lessons['partial_successes'].append(best_name)
                return best_result

        return None

    def _try_pattern_synthesis(self, train_pairs, test_input, lessons) -> Optional[np.ndarray]:
        """Iteration 3: Synthesize pattern from training examples."""
        # Lesson: Look for PATTERNS across multiple training pairs

        if len(train_pairs) < 2:
            return None

        # Check if all outputs are the same
        outputs = [out for inp, out in train_pairs]
        if all(np.array_equal(outputs[0], out) for out in outputs):
            # Pattern: Output is always the same!
            return outputs[0]

        # Check if output is always input with transformation
        # (already covered in iteration 1)

        # Check if output is combination/overlay
        for inp, out in train_pairs:
            if out.shape == test_input.shape:
                # Try using output as template
                return out

        return None

    def _try_color_focused(self, train_pairs, test_input, lessons) -> Optional[np.ndarray]:
        """Iteration 4: Focus on color transformations."""
        # Lesson: Maybe it's all about color mapping

        # Learn color mappings from training
        color_map = {}

        for inp, out in train_pairs:
            if inp.shape == out.shape:
                # Learn which input colors map to which output colors
                for i in range(inp.shape[0]):
                    for j in range(inp.shape[1]):
                        in_color = inp[i, j]
                        out_color = out[i, j]

                        if in_color not in color_map:
                            color_map[in_color] = []
                        color_map[in_color].append(out_color)

        # Apply most common mapping for each color
        if color_map:
            result = test_input.copy()
            for in_color, out_colors in color_map.items():
                most_common = Counter(out_colors).most_common(1)[0][0]
                result[test_input == in_color] = most_common

            lessons['color_mappings'] = color_map
            return result

        return None

    def _try_size_focused(self, train_pairs, test_input, lessons) -> Optional[np.ndarray]:
        """Iteration 5: Focus on size transformations."""
        # Lesson: Maybe it's about scaling/tiling

        # Learn size patterns
        size_ratios = []
        for inp, out in train_pairs:
            h_ratio = out.shape[0] / inp.shape[0] if inp.shape[0] > 0 else 1.0
            w_ratio = out.shape[1] / inp.shape[1] if inp.shape[1] > 0 else 1.0
            size_ratios.append((h_ratio, w_ratio))

        # Check if consistent
        if size_ratios:
            avg_h_ratio = np.mean([r[0] for r in size_ratios])
            avg_w_ratio = np.mean([r[1] for r in size_ratios])

            # If ratio is 2x2, try scale up
            if 1.8 < avg_h_ratio < 2.2 and 1.8 < avg_w_ratio < 2.2:
                try:
                    result = self.transforms.scale_up_2x(test_input)
                    return result
                except:
                    pass

            # If ratio is 0.5x0.5, try scale down
            if 0.4 < avg_h_ratio < 0.6 and 0.4 < avg_w_ratio < 0.6:
                try:
                    result = self.transforms.scale_down_2x(test_input)
                    return result
                except:
                    pass

        return None

    def _score_vs_training(self, candidate: np.ndarray, train_pairs, test_input) -> float:
        """Score candidate against training patterns."""
        if candidate is None:
            return 0.0

        # Score based on how well this would work on training
        scores = []

        for inp, out in train_pairs:
            # Try to apply same transformation to training input
            # If we get similar output, good sign
            # (simplified scoring for now)
            if candidate.shape == out.shape:
                similarity = np.sum(candidate == out) / candidate.size
                scores.append(similarity)

        return np.mean(scores) if scores else 0.0


# ============================================================================
# TESTING
# ============================================================================

def test_recursive_solver():
    """Test recursive hybrid solver."""
    import json

    print("="*80)
    print("RECURSIVE HYBRID SOLVER TEST")
    print("="*80)

    with open('arc-agi_training_challenges.json', 'r') as f:
        challenges = json.load(f)

    with open('arc-agi_training_solutions.json', 'r') as f:
        solutions = json.load(f)

    task_ids = list(challenges.keys())[:10]

    solver = RecursiveHybridSolver()

    exact_matches = 0
    partial_scores = []

    for task_id in task_ids:
        task = challenges[task_id]
        train_pairs = [(np.array(ex['input']), np.array(ex['output']))
                      for ex in task['train']]

        test_input = np.array(task['test'][0]['input'])
        expected = np.array(solutions[task_id][0]) if task_id in solutions else None

        try:
            predicted = solver.solve(train_pairs, test_input)

            if predicted is not None and expected is not None:
                if predicted.shape == expected.shape:
                    score = np.sum(predicted == expected) / predicted.size
                    partial_scores.append(score)

                    if np.array_equal(predicted, expected):
                        exact_matches += 1
                        print(f"✓ {task_id}: EXACT MATCH!")
                    elif score > 0.5:
                        print(f"≈ {task_id}: {score*100:.1f}% match")
                    else:
                        print(f"✗ {task_id}: {score*100:.1f}% match")
                else:
                    print(f"✗ {task_id}: Shape mismatch")
                    partial_scores.append(0.0)
            else:
                print(f"✗ {task_id}: No prediction")
                partial_scores.append(0.0)
        except Exception as e:
            print(f"✗ {task_id}: Error - {str(e)[:40]}")
            partial_scores.append(0.0)

    print("\n" + "="*80)
    print(f"Exact matches: {exact_matches}/10 ({exact_matches*10:.1f}%)")
    print(f"Avg partial match: {np.mean(partial_scores)*100:.1f}%")
    print(f"High similarity (>50%): {sum(1 for s in partial_scores if s > 0.5)}/10")
    print("="*80)


if __name__ == '__main__':
    test_recursive_solver()
