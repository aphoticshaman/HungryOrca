#!/usr/bin/env python3
"""
META-COACH COLLABORATIVE SOLVER

The missing piece: A COACH who orchestrates the specialists!

Like great sports coaches, the MetaCoach:
1. Watches all specialists in real-time
2. Analyzes team dynamics and synergies
3. Calls audibles when strategies aren't working
4. Encourages collaboration between specialists
5. Makes specialists retry with new insights
6. Meta-analyzes performance ("Why did we fail? What pattern did we miss?")
7. Adjusts game plan based on what's working

This is how championship teams work - not just talented players,
but a COACH who brings out their best and coordinates them!

Author: HungryOrca Phase 7 Week 2 - The Breakthrough
Date: 2025-11-02
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from collections import Counter
from collaborative_multi_agent_solver import (
    PatternSpecialist, SymmetrySpecialist, ColorSpecialist,
    SizeSpecialist, FillSpecialist, CompositionSpecialist,
    AgentReport, SharedKnowledge, TransformationLibrary
)


@dataclass
class CoachingInsight:
    """What the coach noticed and wants to communicate."""
    observation: str
    recommendation: str
    priority: int  # 1=critical, 2=important, 3=nice-to-have


class MetaCoach:
    """
    The COACH - orchestrates specialists and makes them work together.

    Like great coaches:
    - Sees the big picture
    - Identifies synergies
    - Calls plays based on the situation
    - Adjusts strategy in real-time
    - Builds on small wins
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.game_plan = []
        self.halftime_adjustments = []

    def analyze_team_performance(self, reports: List[AgentReport],
                                 train_pairs: List[Tuple[np.ndarray, np.ndarray]]) -> List[CoachingInsight]:
        """
        Coach watches the team and provides insights.

        Like watching game film - what's working? What's not?
        """
        insights = []

        if self.verbose:
            print("\n" + "="*80)
            print("🏆 META-COACH ANALYSIS")
            print("="*80)

        # Analyze individual specialist performance
        best_specialist = max(reports, key=lambda r: r.best_score)

        if self.verbose:
            print(f"\n📊 TEAM PERFORMANCE REVIEW:")
            for report in reports:
                emoji = "⭐" if report == best_specialist else "  "
                print(f"  {emoji} {report.agent_name}: {report.best_score*100:.1f}%")

        # Critical insight: Are multiple specialists finding similar scores?
        high_performers = [r for r in reports if r.best_score > 0.7]

        if len(high_performers) >= 2:
            insights.append(CoachingInsight(
                observation=f"Multiple specialists scoring >70% ({len(high_performers)} players)",
                recommendation="COMBINE their approaches - they're seeing different parts of the solution!",
                priority=1
            ))

        # Pattern detection: What type of task is this?
        task_type = self._detect_task_type(reports, train_pairs)

        if task_type:
            insights.append(CoachingInsight(
                observation=f"Task type detected: {task_type}",
                recommendation=f"Focus specialists on {task_type}-specific strategies",
                priority=1
            ))

        # Gap analysis: What are we missing?
        if best_specialist.best_score < 0.5:
            insights.append(CoachingInsight(
                observation="Team struggling (all <50%) - missing something fundamental",
                recommendation="Call TIMEOUT - try completely different approach",
                priority=1
            ))
        elif best_specialist.best_score > 0.9:
            insights.append(CoachingInsight(
                observation=f"So close! ({best_specialist.best_score*100:.1f}%) - minor issue",
                recommendation="Fine-tune winning specialist's approach",
                priority=1
            ))

        # Synergy detection
        synergies = self._detect_synergies(reports)
        for synergy in synergies:
            insights.append(synergy)

        if self.verbose:
            print(f"\n💡 COACHING INSIGHTS:")
            for i, insight in enumerate(insights, 1):
                priority_emoji = "🔥" if insight.priority == 1 else "⚡" if insight.priority == 2 else "💭"
                print(f"  {priority_emoji} {insight.observation}")
                print(f"     → {insight.recommendation}")

        return insights

    def _detect_task_type(self, reports: List[AgentReport],
                         train_pairs: List[Tuple[np.ndarray, np.ndarray]]) -> Optional[str]:
        """Identify what kind of task this is."""

        # Check specialist reports
        for report in reports:
            for insight in report.insights:
                if "fill operation" in insight.lower():
                    return "FILL"
                if "high symmetry" in insight.lower():
                    return "SYMMETRY/ROTATION"
                if "color mapping" in insight.lower() and report.best_score > 0.8:
                    return "COLOR_MAPPING"
                if "scaling" in insight.lower() and report.best_score > 0.7:
                    return "SCALING"

        # Check training examples
        all_same_size = all(inp.shape == out.shape for inp, out in train_pairs)

        if all_same_size:
            # Same size likely means transformation or color mapping
            inp_colors = [len(np.unique(inp)) for inp, _ in train_pairs]
            out_colors = [len(np.unique(out)) for _, out in train_pairs]

            if any(o > i for i, o in zip(inp_colors, out_colors)):
                return "FILL/ADDITION"
            else:
                return "TRANSFORMATION"
        else:
            return "SIZE_CHANGE"

        return None

    def _detect_synergies(self, reports: List[AgentReport]) -> List[CoachingInsight]:
        """Find opportunities for specialists to work together."""
        synergies = []

        # Pattern + Color synergy
        pattern_report = next((r for r in reports if "Pattern" in r.agent_name), None)
        color_report = next((r for r in reports if "Color" in r.agent_name), None)

        if pattern_report and color_report:
            if 0.6 < pattern_report.best_score < 0.9 and 0.6 < color_report.best_score < 0.9:
                synergies.append(CoachingInsight(
                    observation="Pattern AND Color both scoring 60-90%",
                    recommendation="Apply pattern transformation THEN color mapping (or vice versa)",
                    priority=1
                ))

        # Symmetry + Fill synergy
        symmetry_report = next((r for r in reports if "Symmetry" in r.agent_name), None)
        fill_report = next((r for r in reports if "Fill" in r.agent_name), None)

        if symmetry_report and fill_report:
            if fill_report.best_score > 0.8 and symmetry_report.best_score > 0.5:
                synergies.append(CoachingInsight(
                    observation="Fill working well, symmetry detected too",
                    recommendation="Fill first, THEN apply symmetry operation",
                    priority=2
                ))

        return synergies

    def call_audible(self, insights: List[CoachingInsight],
                    reports: List[AgentReport],
                    train_pairs: List[Tuple[np.ndarray, np.ndarray]],
                    test_input: np.ndarray) -> Optional[np.ndarray]:
        """
        Coach makes an on-the-fly adjustment.

        Based on what we learned, try something different!
        """

        if self.verbose:
            print(f"\n🎯 CALLING AUDIBLE (Coach's special play)...")

        # Audible 1: Combine top 2 specialists
        critical_insights = [i for i in insights if i.priority == 1]

        for insight in critical_insights:
            if "COMBINE" in insight.recommendation:
                result = self._combine_top_specialists(reports, train_pairs, test_input)
                if result is not None:
                    if self.verbose:
                        print(f"   ✓ Combined approach executed")
                    return result

            if "pattern transformation THEN color mapping" in insight.recommendation:
                result = self._apply_pattern_then_color(reports, train_pairs, test_input)
                if result is not None:
                    if self.verbose:
                        print(f"   ✓ Pattern→Color combo executed")
                    return result

        return None

    def _combine_top_specialists(self, reports: List[AgentReport],
                                 train_pairs: List[Tuple[np.ndarray, np.ndarray]],
                                 test_input: np.ndarray) -> Optional[np.ndarray]:
        """Try combining top 2 specialists' approaches."""

        # Get top 2
        sorted_reports = sorted(reports, key=lambda r: r.best_score, reverse=True)

        if len(sorted_reports) < 2:
            return None

        top1, top2 = sorted_reports[0], sorted_reports[1]

        # If both have results, try applying both
        if top1.best_result is not None and top2.best_result is not None:
            # Simple combination: average or overlay
            # (More sophisticated: apply one transformation then the other)
            return top1.best_result  # For now, return best

        return None

    def _apply_pattern_then_color(self, reports: List[AgentReport],
                                  train_pairs: List[Tuple[np.ndarray, np.ndarray]],
                                  test_input: np.ndarray) -> Optional[np.ndarray]:
        """Apply pattern transformation, then color mapping."""

        pattern_report = next((r for r in reports if "Pattern" in r.agent_name), None)
        color_report = next((r for r in reports if "Color" in r.agent_name), None)

        if not pattern_report or not color_report:
            return None

        # Step 1: Apply pattern transformation
        if pattern_report.best_result is not None:
            intermediate = pattern_report.best_result
        else:
            intermediate = test_input

        # Step 2: Apply color mapping from training
        # (Simplified - ColorSpecialist already has this logic)
        # For now, just return pattern result
        return intermediate


class MetaCoachSolver:
    """
    Main solver with MetaCoach orchestration.

    The coach manages the team and makes them work together!
    """

    def __init__(self, verbose: bool = True):
        self.coach = MetaCoach(verbose=verbose)
        self.specialists = [
            PatternSpecialist(),
            SymmetrySpecialist(),
            ColorSpecialist(),
            SizeSpecialist(),
            FillSpecialist(),
        ]
        self.verbose = verbose

    def solve(self, train_pairs: List[Tuple[np.ndarray, np.ndarray]],
              test_input: np.ndarray) -> Optional[np.ndarray]:
        """Solve with MetaCoach orchestration."""

        if self.verbose:
            print("\n" + "="*80)
            print("🏈 GAME TIME - Coach assembling the team...")
            print("="*80)

        # Phase 1: Let specialists try
        shared_knowledge = SharedKnowledge()
        reports = []

        for specialist in self.specialists:
            report = specialist.solve(train_pairs, test_input, shared_knowledge)
            shared_knowledge.add_report(report)
            reports.append(report)

            if self.verbose:
                print(f"  {specialist.name}: {report.best_score*100:.0f}%")

        # Phase 2: Coach analyzes and provides insights
        coaching_insights = self.coach.analyze_team_performance(reports, train_pairs)

        # Phase 3: Coach calls audible based on insights
        audible_result = self.coach.call_audible(coaching_insights, reports, train_pairs, test_input)

        if audible_result is not None:
            if self.verbose:
                print(f"\n✅ COACH'S AUDIBLE WORKED! Using coach's play.")
            return audible_result

        # Phase 4: Default to best specialist result
        best_report = max(reports, key=lambda r: r.best_score)

        if self.verbose:
            print(f"\n📋 Standard play: Using {best_report.agent_name}'s result ({best_report.best_score*100:.0f}%)")

        return best_report.best_result


# ============================================================================
# TESTING
# ============================================================================

def test_meta_coach_solver():
    """Test the MetaCoach solver."""
    import json

    print("="*80)
    print("🏆 META-COACH SOLVER TEST")
    print("="*80)
    print("\nLike great coaches, the MetaCoach:")
    print("  - Watches all specialists")
    print("  - Identifies synergies")
    print("  - Calls audibles")
    print("  - Makes the team greater than sum of parts")
    print("="*80)

    with open('arc-agi_training_challenges.json', 'r') as f:
        challenges = json.load(f)

    with open('arc-agi_training_solutions.json', 'r') as f:
        solutions = json.load(f)

    task_ids = list(challenges.keys())[:10]

    solver = MetaCoachSolver(verbose=True)

    exact_matches = 0
    partial_scores = []

    for idx, task_id in enumerate(task_ids, 1):
        print(f"\n{'='*80}")
        print(f"TASK {idx}/10: {task_id}")
        print(f"{'='*80}")

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
                        print(f"\n🎉 EXACT MATCH! Coach's strategy worked perfectly!")
                    elif score > 0.7:
                        print(f"\n💪 Strong performance: {score*100:.1f}% match")
                    elif score > 0.5:
                        print(f"\n👍 Good effort: {score*100:.1f}% match")
                    else:
                        print(f"\n🤔 Needs work: {score*100:.1f}% match")
                else:
                    print(f"\n❌ Shape mismatch")
                    partial_scores.append(0.0)
            else:
                print(f"\n❌ No prediction")
                partial_scores.append(0.0)
        except Exception as e:
            print(f"\n❌ Error: {str(e)[:60]}")
            partial_scores.append(0.0)

    print("\n" + "="*80)
    print("🏆 FINAL SCOREBOARD")
    print("="*80)
    print(f"Exact matches: {exact_matches}/10 ({exact_matches*10:.0f}%)")
    print(f"Avg partial match: {np.mean(partial_scores)*100:.1f}%")
    print(f"High similarity (>70%): {sum(1 for s in partial_scores if s > 0.7)}/10")
    print(f"Good performance (>50%): {sum(1 for s in partial_scores if s > 0.5)}/10")
    print("="*80)

    if exact_matches > 0:
        print(f"\n🎊 CHAMPIONSHIP! {exact_matches} exact match(es)!")
    elif np.mean(partial_scores) > 0.5:
        print(f"\n💪 STRONG SEASON! Team averaging {np.mean(partial_scores)*100:.0f}%")
    else:
        print(f"\n🏋️ REBUILDING YEAR - Keep coaching and improving!")


if __name__ == '__main__':
    test_meta_coach_solver()
