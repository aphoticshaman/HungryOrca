#!/usr/bin/env python3
"""
TEST INDIVIDUAL FUZZY RULES

Test each of the 10 rules independently against baseline.
Fine-tune thresholds if needed.
Keep only rules that improve performance.

Methodology: "bolt on piece at a time, testing til it works"
"""

import json
import numpy as np
from typing import Dict, List, Tuple
from fuzzy_transformation_solver import (
    FuzzyTransformationSolver,
    SimpleFeatureExtractor,
    PatternMatcher
)

# Load test data
with open('arc-agi_training_challenges.json', 'r') as f:
    challenges = json.load(f)

with open('arc-agi_training_solutions.json', 'r') as f:
    solutions = json.load(f)

task_ids = list(challenges.keys())[:10]


class SingleRuleTester:
    """Test individual fuzzy rules."""

    def __init__(self, rule_id: int):
        self.rule_id = rule_id
        self.feature_extractor = SimpleFeatureExtractor()
        self.pattern_matcher = PatternMatcher()

    def apply_rule(self, features: Dict[str, float]) -> Dict[str, float]:
        """Apply ONLY the specified rule."""
        # Base weights (neutral)
        weights = {
            'rotation': 0.5,
            'flip': 0.5,
            'color_mapping': 0.5,
            'pattern_learning': 0.5,
            'scaling': 0.5
        }

        symmetry = features['symmetry']
        consistency = features['consistency']
        size_ratio = features['size_ratio']
        complexity = features['complexity']

        # Apply ONLY the selected rule
        if self.rule_id == 1:
            # R1: High symmetry + low complexity → rotation/flip
            if symmetry > 0.7 and complexity < 0.4:
                weights['rotation'] = 0.9
                weights['flip'] = 0.9
                weights['pattern_learning'] = 0.6

        elif self.rule_id == 2:
            # R2: High symmetry + consistency → rotation + pattern
            if symmetry > 0.6 and consistency > 0.7:
                weights['rotation'] = 0.8
                weights['pattern_learning'] = 0.9

        elif self.rule_id == 3:
            # R3: Medium symmetry → both
            if 0.4 < symmetry < 0.7:
                weights['rotation'] = 0.6
                weights['flip'] = 0.6
                weights['pattern_learning'] = 0.7

        elif self.rule_id == 4:
            # R4: Low symmetry → pattern + colors
            if symmetry < 0.4:
                weights['rotation'] = 0.2
                weights['flip'] = 0.2
                weights['color_mapping'] = 0.7
                weights['pattern_learning'] = 0.9

        elif self.rule_id == 5:
            # R5: Consistency + same size → pattern learning
            if consistency > 0.8 and 0.9 < size_ratio < 1.1:
                weights['pattern_learning'] = 1.0
                weights['color_mapping'] = 0.8

        elif self.rule_id == 6:
            # R6: Consistency + size change → scaling
            if consistency > 0.7 and (size_ratio < 0.8 or size_ratio > 1.2):
                weights['scaling'] = 0.9
                weights['pattern_learning'] = 0.8

        elif self.rule_id == 7:
            # R7: Low consistency → try all
            if consistency < 0.3:
                weights['rotation'] = 0.7
                weights['flip'] = 0.7
                weights['color_mapping'] = 0.7
                weights['pattern_learning'] = 0.8
                weights['scaling'] = 0.7

        elif self.rule_id == 8:
            # R8: Complexity + consistency → color mapping
            if complexity > 0.6 and consistency > 0.6:
                weights['color_mapping'] = 0.9
                weights['pattern_learning'] = 0.9

        elif self.rule_id == 9:
            # R9: Low complexity + symmetry → simple transforms
            if complexity < 0.3 and symmetry > 0.6:
                weights['rotation'] = 0.9
                weights['flip'] = 0.9
                weights['color_mapping'] = 0.3

        elif self.rule_id == 10:
            # R10: Size doubling → scale up
            if 3.5 < size_ratio < 4.5:
                weights['scaling'] = 1.0

        return weights

    def solve(self, train_pairs: List[Tuple[np.ndarray, np.ndarray]],
              test_input: np.ndarray):
        """Solve using only this rule."""
        # Extract features
        features = self.feature_extractor.extract(train_pairs, test_input)

        # Apply rule
        weights = self.apply_rule(features)

        # Learn transformations
        learned_transforms = self._learn_from_training(train_pairs)

        # Apply with weights
        candidates = []
        for name, transform, score in learned_transforms:
            weighted_score = score

            if 'rotate' in name or 'flip' in name or 'transpose' in name:
                weighted_score *= weights['rotation']
            elif 'color' in name:
                weighted_score *= weights['color_mapping']
            elif 'scale' in name:
                weighted_score *= weights['scaling']
            else:
                weighted_score *= weights['pattern_learning']

            try:
                result = transform(test_input)
                candidates.append((result, weighted_score, name))
            except:
                pass

        if candidates:
            candidates.sort(key=lambda x: x[1], reverse=True)
            return candidates[0][0]

        return None

    def _learn_from_training(self, train_pairs):
        """Learn from training."""
        all_transforms = []
        for inp, out in train_pairs:
            transforms = self.pattern_matcher.find_best_transforms(inp, out)
            all_transforms.extend(transforms)

        transform_scores = {}
        transform_funcs = {}

        for name, func, score in all_transforms:
            if name not in transform_scores:
                transform_scores[name] = []
                transform_funcs[name] = func
            transform_scores[name].append(score)

        result = []
        for name, scores in transform_scores.items():
            avg_score = np.mean(scores)
            result.append((name, transform_funcs[name], avg_score))

        result.sort(key=lambda x: x[2], reverse=True)
        return result


def test_rule(rule_id: int, rule_description: str, num_runs: int = 3):
    """Test one rule vs baseline."""
    print(f"\n{'='*80}")
    print(f"RULE {rule_id}: {rule_description}")
    print(f"{'='*80}")

    # Baseline (no rules)
    baseline_scores = []
    print("\nBaseline (no rules):")
    for run in range(num_runs):
        solver_baseline = FuzzyTransformationSolver(num_rules=0)
        partial_scores = []

        for task_id in task_ids:
            task = challenges[task_id]
            train_pairs = [(np.array(ex['input']), np.array(ex['output']))
                          for ex in task['train']]
            test_input = np.array(task['test'][0]['input'])
            expected = np.array(solutions[task_id][0]) if task_id in solutions else None

            try:
                predicted = solver_baseline.solve(train_pairs, test_input)
                if predicted is not None and expected is not None and predicted.shape == expected.shape:
                    score = np.sum(predicted == expected) / predicted.size
                    partial_scores.append(score)
                else:
                    partial_scores.append(0.0)
            except:
                partial_scores.append(0.0)

        avg_score = np.mean(partial_scores) * 100
        baseline_scores.append(avg_score)
        print(f"  Run {run+1}: {avg_score:.1f}%")

    # Rule alone
    rule_scores = []
    print(f"\nRule {rule_id} applied:")
    for run in range(num_runs):
        solver_rule = SingleRuleTester(rule_id)
        partial_scores = []

        for task_id in task_ids:
            task = challenges[task_id]
            train_pairs = [(np.array(ex['input']), np.array(ex['output']))
                          for ex in task['train']]
            test_input = np.array(task['test'][0]['input'])
            expected = np.array(solutions[task_id][0]) if task_id in solutions else None

            try:
                predicted = solver_rule.solve(train_pairs, test_input)
                if predicted is not None and expected is not None and predicted.shape == expected.shape:
                    score = np.sum(predicted == expected) / predicted.size
                    partial_scores.append(score)
                else:
                    partial_scores.append(0.0)
            except:
                partial_scores.append(0.0)

        avg_score = np.mean(partial_scores) * 100
        rule_scores.append(avg_score)
        print(f"  Run {run+1}: {avg_score:.1f}%")

    # Analysis
    mean_baseline = np.mean(baseline_scores)
    mean_rule = np.mean(rule_scores)
    improvement = mean_rule - mean_baseline

    print(f"\nResults:")
    print(f"  Baseline: {mean_baseline:.1f}%")
    print(f"  Rule {rule_id}: {mean_rule:.1f}%")
    print(f"  Improvement: {improvement:+.1f} pp")

    # Decision
    if improvement >= 2.0:
        decision = "✅ GO"
    elif improvement > 0:
        decision = "⚠️ MARGINAL"
    else:
        decision = "❌ NO-GO"

    print(f"  Decision: {decision}")

    return {
        'rule_id': rule_id,
        'mean_baseline': mean_baseline,
        'mean_rule': mean_rule,
        'improvement': improvement,
        'decision': decision
    }


# Test all 10 rules
print("="*80)
print("INDIVIDUAL RULE TESTING (3 runs each for speed)")
print("="*80)

results = []

results.append(test_rule(1, "High symmetry + low complexity → rotation/flip"))
results.append(test_rule(2, "High symmetry + consistency → rotation + pattern"))
results.append(test_rule(3, "Medium symmetry → both rotation and pattern"))
results.append(test_rule(4, "Low symmetry → pattern + colors"))
results.append(test_rule(5, "Consistency + same size → pattern learning"))
results.append(test_rule(6, "Consistency + size change → scaling"))
results.append(test_rule(7, "Low consistency → try all strategies"))
results.append(test_rule(8, "Complexity + consistency → color mapping"))
results.append(test_rule(9, "Low complexity + symmetry → simple transforms"))
results.append(test_rule(10, "Size doubling → scale up"))

# Summary
print("\n" + "="*80)
print("SUMMARY: Individual Rule Performance")
print("="*80)

go_rules = [r for r in results if '✅' in r['decision']]
marginal_rules = [r for r in results if '⚠️' in r['decision']]
no_go_rules = [r for r in results if '❌' in r['decision']]

print(f"\n✅ GO Rules: {len(go_rules)}/10")
for r in go_rules:
    print(f"   R{r['rule_id']}: {r['improvement']:+.1f} pp")

print(f"\n⚠️ MARGINAL Rules: {len(marginal_rules)}/10")
for r in marginal_rules:
    print(f"   R{r['rule_id']}: {r['improvement']:+.1f} pp")

print(f"\n❌ NO-GO Rules: {len(no_go_rules)}/10")
for r in no_go_rules:
    print(f"   R{r['rule_id']}: {r['improvement']:+.1f} pp")

# Export
with open('individual_rules_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n✓ Results exported to: individual_rules_results.json")
print("="*80)
