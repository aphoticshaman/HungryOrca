#!/usr/bin/env python3
"""
COLLABORATIVE MULTI-AGENT SOLVER

Specialists that "think aloud" and share lessons with each other.

Agents:
1. SymmetrySpecialist - Tries rotations/flips
2. ColorSpecialist - Tries color mappings
3. SizeSpecialist - Tries scaling/tiling
4. PatternSpecialist - Tries pattern learning
5. CompositionSpecialist - Combines other agents' insights

Each agent:
- Tries their specialty
- Reports findings ("I tried X, got Y% match, noticed Z")
- Hears other agents' findings
- Adjusts strategy based on shared lessons

Like a team brainstorming together!

Author: HungryOrca Phase 7 Week 2
Date: 2025-11-02
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from collections import Counter
from fuzzy_transformation_solver import TransformationLibrary, PatternMatcher


@dataclass
class AgentReport:
    """What an agent learned and wants to share."""
    agent_name: str
    tried_approaches: List[str]
    best_score: float
    best_result: Optional[np.ndarray]
    insights: List[str]  # "Thinking aloud"


class SharedKnowledge:
    """Knowledge base shared between all agents."""

    def __init__(self):
        self.reports: List[AgentReport] = []
        self.global_insights: List[str] = []
        self.tried_transforms: set = set()
        self.color_patterns: Dict = {}
        self.size_patterns: Dict = {}

    def add_report(self, report: AgentReport):
        """Agent shares what they learned."""
        self.reports.append(report)
        self.global_insights.extend(report.insights)
        self.tried_transforms.update(report.tried_approaches)

    def get_best_candidate(self) -> Optional[np.ndarray]:
        """Get best result across all agents."""
        if not self.reports:
            return None

        best_report = max(self.reports, key=lambda r: r.best_score)
        return best_report.best_result if best_report.best_score > 0 else None


class SymmetrySpecialist:
    """Specialist in rotation/flip transformations."""

    def __init__(self):
        self.transforms = TransformationLibrary()
        self.name = "SymmetrySpecialist"

    def solve(self, train_pairs: List[Tuple[np.ndarray, np.ndarray]],
              test_input: np.ndarray,
              shared_knowledge: SharedKnowledge) -> AgentReport:
        """Try symmetry transformations."""
        insights = [f"{self.name} starting..."]

        # Check what others learned
        if shared_knowledge.global_insights:
            insights.append(f"Heard from team: {len(shared_knowledge.global_insights)} insights so far")

        tried = []
        best_score = 0.0
        best_result = None

        # Try each symmetry transform
        symmetry_transforms = [
            ('rotate_90', self.transforms.rotate_90),
            ('rotate_180', self.transforms.rotate_180),
            ('rotate_270', self.transforms.rotate_270),
            ('flip_h', self.transforms.flip_horizontal),
            ('flip_v', self.transforms.flip_vertical),
            ('transpose', self.transforms.transpose),
        ]

        for name, transform in symmetry_transforms:
            if name in shared_knowledge.tried_transforms:
                continue  # Someone already tried this

            tried.append(name)

            # Test on training pairs to score
            scores = []
            for inp, out in train_pairs:
                try:
                    result = transform(inp)
                    if result.shape == out.shape:
                        score = np.sum(result == out) / result.size
                        scores.append(score)
                except:
                    pass

            if scores:
                avg_score = np.mean(scores)
                insights.append(f"Tried {name}: {avg_score*100:.1f}% match on training")

                if avg_score > best_score:
                    best_score = avg_score
                    try:
                        best_result = transform(test_input)
                        insights.append(f"→ {name} is my best so far!")
                    except:
                        pass

        # Report symmetry level
        h_sym = np.mean(test_input == np.flip(test_input, axis=0))
        v_sym = np.mean(test_input == np.flip(test_input, axis=1))
        insights.append(f"Input symmetry: H={h_sym*100:.0f}%, V={v_sym*100:.0f}%")

        if max(h_sym, v_sym) > 0.7:
            insights.append("→ High symmetry detected! Likely rotation/flip task")

        return AgentReport(
            agent_name=self.name,
            tried_approaches=tried,
            best_score=best_score,
            best_result=best_result,
            insights=insights
        )


class ColorSpecialist:
    """Specialist in color mappings."""

    def __init__(self):
        self.name = "ColorSpecialist"

    def solve(self, train_pairs: List[Tuple[np.ndarray, np.ndarray]],
              test_input: np.ndarray,
              shared_knowledge: SharedKnowledge) -> AgentReport:
        """Try color transformations."""
        insights = [f"{self.name} starting..."]

        # Learn from team
        for report in shared_knowledge.reports:
            if "symmetry" in report.insights[0].lower():
                insights.append("Noted: SymmetrySpecialist found patterns, will adjust strategy")

        tried = []
        best_score = 0.0
        best_result = None

        # Learn color mapping from training
        color_map = {}

        for inp, out in train_pairs:
            if inp.shape == out.shape:
                for i in range(inp.shape[0]):
                    for j in range(inp.shape[1]):
                        in_c = inp[i, j]
                        out_c = out[i, j]

                        if in_c not in color_map:
                            color_map[in_c] = []
                        color_map[in_c].append(out_c)

        # Compute most common mapping
        final_map = {}
        for in_c, out_colors in color_map.items():
            most_common = Counter(out_colors).most_common(1)[0][0]
            final_map[in_c] = most_common
            if in_c != most_common:
                insights.append(f"Color {in_c} → {most_common}")

        # Apply mapping
        if final_map:
            tried.append('color_mapping')
            result = test_input.copy()

            for in_c, out_c in final_map.items():
                result[test_input == in_c] = out_c

            # Score against training
            scores = []
            for inp, out in train_pairs:
                if inp.shape == out.shape:
                    mapped = inp.copy()
                    for in_c, out_c in final_map.items():
                        mapped[inp == in_c] = out_c

                    if mapped.shape == out.shape:
                        score = np.sum(mapped == out) / mapped.size
                        scores.append(score)

            if scores:
                avg_score = np.mean(scores)
                best_score = avg_score
                best_result = result
                insights.append(f"Color mapping: {avg_score*100:.1f}% match on training")

                if avg_score > 0.8:
                    insights.append("→ Very high score! This is likely a color mapping task")

        shared_knowledge.color_patterns = final_map

        return AgentReport(
            agent_name=self.name,
            tried_approaches=tried,
            best_score=best_score,
            best_result=best_result,
            insights=insights
        )


class SizeSpecialist:
    """Specialist in scaling/tiling."""

    def __init__(self):
        self.transforms = TransformationLibrary()
        self.name = "SizeSpecialist"

    def solve(self, train_pairs: List[Tuple[np.ndarray, np.ndarray]],
              test_input: np.ndarray,
              shared_knowledge: SharedKnowledge) -> AgentReport:
        """Try size transformations."""
        insights = [f"{self.name} starting..."]

        tried = []
        best_score = 0.0
        best_result = None

        # Analyze size patterns
        size_ratios = []
        for inp, out in train_pairs:
            h_ratio = out.shape[0] / inp.shape[0]
            w_ratio = out.shape[1] / inp.shape[1]
            size_ratios.append((h_ratio, w_ratio))
            insights.append(f"Size change: {h_ratio:.1f}x{w_ratio:.1f}")

        avg_h = np.mean([r[0] for r in size_ratios])
        avg_w = np.mean([r[1] for r in size_ratios])

        # Try scale up 2x
        if 1.8 < avg_h < 2.2 and 1.8 < avg_w < 2.2:
            insights.append("→ Detected 2x scaling pattern!")
            tried.append('scale_up_2x')

            try:
                result = self.transforms.scale_up_2x(test_input)
                # Score
                scores = []
                for inp, out in train_pairs:
                    scaled = self.transforms.scale_up_2x(inp)
                    if scaled.shape == out.shape:
                        score = np.sum(scaled == out) / scaled.size
                        scores.append(score)

                if scores:
                    avg_score = np.mean(scores)
                    best_score = avg_score
                    best_result = result
                    insights.append(f"Scale up 2x: {avg_score*100:.1f}% match")
            except:
                pass

        # Try scale down
        elif 0.4 < avg_h < 0.6 and 0.4 < avg_w < 0.6:
            insights.append("→ Detected 0.5x scaling pattern!")
            tried.append('scale_down_2x')

            try:
                result = self.transforms.scale_down_2x(test_input)
                scores = []
                for inp, out in train_pairs:
                    scaled = self.transforms.scale_down_2x(inp)
                    if scaled.shape == out.shape:
                        score = np.sum(scaled == out) / scaled.size
                        scores.append(score)

                if scores:
                    avg_score = np.mean(scores)
                    best_score = avg_score
                    best_result = result
                    insights.append(f"Scale down 2x: {avg_score*100:.1f}% match")
            except:
                pass

        else:
            insights.append("No clear scaling pattern detected")

        shared_knowledge.size_patterns = {'avg_h_ratio': avg_h, 'avg_w_ratio': avg_w}

        return AgentReport(
            agent_name=self.name,
            tried_approaches=tried,
            best_score=best_score,
            best_result=best_result,
            insights=insights
        )


class PatternSpecialist:
    """Specialist in pattern learning from examples."""

    def __init__(self):
        self.pattern_matcher = PatternMatcher()
        self.name = "PatternSpecialist"

    def solve(self, train_pairs: List[Tuple[np.ndarray, np.ndarray]],
              test_input: np.ndarray,
              shared_knowledge: SharedKnowledge) -> AgentReport:
        """Try learned patterns."""
        insights = [f"{self.name} starting..."]

        # Learn from team
        for report in shared_knowledge.reports:
            if report.best_score > 0.7:
                insights.append(f"Noted: {report.agent_name} got {report.best_score*100:.0f}% - will incorporate that insight")

        tried = []
        best_score = 0.0
        best_result = None

        # Learn transformations
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

        # Find best
        if transform_scores:
            best_name = max(transform_scores.items(), key=lambda x: np.mean(x[1]))[0]
            avg_score = np.mean(transform_scores[best_name])

            insights.append(f"Best learned pattern: {best_name} ({avg_score*100:.1f}%)")
            tried.append(best_name)

            try:
                result = transform_funcs[best_name](test_input)
                best_score = avg_score
                best_result = result
            except:
                pass

        return AgentReport(
            agent_name=self.name,
            tried_approaches=tried,
            best_score=best_score,
            best_result=best_result,
            insights=insights
        )


class CompositionSpecialist:
    """Specialist in combining other agents' insights."""

    def __init__(self):
        self.name = "CompositionSpecialist"

    def solve(self, train_pairs: List[Tuple[np.ndarray, np.ndarray]],
              test_input: np.ndarray,
              shared_knowledge: SharedKnowledge) -> AgentReport:
        """Combine insights from all specialists."""
        insights = [f"{self.name} synthesizing team insights..."]

        tried = []
        best_score = 0.0
        best_result = None

        # Listen to all agents
        for report in shared_knowledge.reports:
            insights.append(f"From {report.agent_name}: {report.best_score*100:.0f}% success")

        # Find best result from team
        best_candidate = shared_knowledge.get_best_candidate()

        if best_candidate is not None:
            best_result = best_candidate
            best_score = max(r.best_score for r in shared_knowledge.reports)
            insights.append(f"→ Selecting team's best result ({best_score*100:.0f}%)")

        return AgentReport(
            agent_name=self.name,
            tried_approaches=['team_synthesis'],
            best_score=best_score,
            best_result=best_result,
            insights=insights
        )


class FillSpecialist:
    """Specialist in flood-fill operations (fill interior regions)."""

    def __init__(self):
        self.name = "FillSpecialist"

    def solve(self, train_pairs: List[Tuple[np.ndarray, np.ndarray]],
              test_input: np.ndarray,
              shared_knowledge: SharedKnowledge) -> AgentReport:
        """Try fill operations."""
        insights = [f"{self.name} starting..."]

        tried = []
        best_score = 0.0
        best_result = None

        # Detect if this is a fill task
        fill_detected = False
        fill_color = None

        for inp, out in train_pairs:
            if inp.shape == out.shape:
                # Check if output has MORE colors than input (filling added color)
                inp_colors = set(np.unique(inp))
                out_colors = set(np.unique(out))
                new_colors = out_colors - inp_colors

                if new_colors:
                    fill_color = list(new_colors)[0]
                    fill_detected = True
                    insights.append(f"→ Detected fill operation! New color: {fill_color}")
                    break

        if fill_detected and fill_color is not None:
            tried.append(f'flood_fill_{fill_color}')

            # Simple interior fill: cells surrounded by non-zero become fill_color
            result = test_input.copy()

            # Find cells that are 0 and completely surrounded by non-zero
            bg = 0
            for i in range(1, result.shape[0]-1):
                for j in range(1, result.shape[1]-1):
                    if result[i, j] == bg:
                        # Check if surrounded (4-way)
                        neighbors = [
                            result[i-1, j], result[i+1, j],
                            result[i, j-1], result[i, j+1]
                        ]
                        if all(n != bg for n in neighbors):
                            result[i, j] = fill_color

            # Score
            scores = []
            for inp, out in train_pairs:
                if inp.shape == out.shape:
                    test_fill = inp.copy()
                    for i in range(1, test_fill.shape[0]-1):
                        for j in range(1, test_fill.shape[1]-1):
                            if test_fill[i, j] == bg:
                                neighbors = [
                                    test_fill[i-1, j], test_fill[i+1, j],
                                    test_fill[i, j-1], test_fill[i, j+1]
                                ]
                                if all(n != bg for n in neighbors):
                                    test_fill[i, j] = fill_color

                    if test_fill.shape == out.shape:
                        score = np.sum(test_fill == out) / test_fill.size
                        scores.append(score)

            if scores:
                avg_score = np.mean(scores)
                best_score = avg_score
                best_result = result
                insights.append(f"Interior fill result: {avg_score*100:.1f}% match on training")

                if avg_score > 0.9:
                    insights.append("→ Very high score! This is likely a fill task!")

        return AgentReport(
            agent_name=self.name,
            tried_approaches=tried,
            best_score=best_score,
            best_result=best_result,
            insights=insights
        )


class CollaborativeSolver:
    """Main solver that coordinates specialist agents."""

    def __init__(self, verbose: bool = False):
        self.specialists = [
            PatternSpecialist(),      # Try first (fast, often works)
            SymmetrySpecialist(),
            ColorSpecialist(),
            SizeSpecialist(),
            FillSpecialist(),         # NEW: Fill interior regions
            CompositionSpecialist()  # Last (synthesizes others)
        ]
        self.verbose = verbose

    def solve(self, train_pairs: List[Tuple[np.ndarray, np.ndarray]],
              test_input: np.ndarray) -> Optional[np.ndarray]:
        """Solve using collaborative multi-agent approach."""
        shared_knowledge = SharedKnowledge()

        # Each specialist tries in turn, sharing lessons
        for specialist in self.specialists:
            report = specialist.solve(train_pairs, test_input, shared_knowledge)
            shared_knowledge.add_report(report)

            if self.verbose:
                print(f"\n--- {report.agent_name} ---")
                for insight in report.insights:
                    print(f"  {insight}")

        # Get best result from team
        return shared_knowledge.get_best_candidate()


# ============================================================================
# TESTING
# ============================================================================

def test_collaborative_solver():
    """Test collaborative multi-agent solver."""
    import json

    print("="*80)
    print("COLLABORATIVE MULTI-AGENT SOLVER TEST")
    print("="*80)

    with open('arc-agi_training_challenges.json', 'r') as f:
        challenges = json.load(f)

    with open('arc-agi_training_solutions.json', 'r') as f:
        solutions = json.load(f)

    task_ids = list(challenges.keys())[:10]

    solver = CollaborativeSolver(verbose=False)

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
    test_collaborative_solver()
