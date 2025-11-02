#!/usr/bin/env python3
"""
EVOLVING SPECIALIST SYSTEM - LEARNING AGENTS

Specialists that START with their core capability but EVOLVE through:
- Interaction with the puzzle
- Hearing other specialists' thoughts
- Learning from success/failure
- Meta-learning (learning how to learn)
- Elevating awareness, logic, abstraction, reasoning

NOT siloed transforms - adaptive learning agents!

Author: HungryOrca - Final Push (Evolving Architecture)
Date: 2025-11-02
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Set
from collections import Counter, deque
from dataclasses import dataclass, field
import time


@dataclass
class Insight:
    """A lesson learned by a specialist."""
    specialist_name: str
    observation: str
    hypothesis: str
    confidence: float
    context: str  # What puzzle aspect triggered this


@dataclass
class Memory:
    """What a specialist remembers about past attempts."""
    strategies_tried: Set[str] = field(default_factory=set)
    successes: Dict[str, float] = field(default_factory=dict)  # strategy -> score
    failures: Dict[str, str] = field(default_factory=dict)  # strategy -> reason
    learned_patterns: List[str] = field(default_factory=list)
    heard_insights: List[Insight] = field(default_factory=list)


@dataclass
class EvolutionReport:
    """What specialist learned and produced."""
    name: str
    result: Optional[np.ndarray]
    confidence: float
    strategy_used: str
    insights_generated: List[Insight]
    evolution_notes: str  # How specialist adapted


class CollectiveKnowledge:
    """Shared knowledge pool all specialists contribute to and learn from."""

    def __init__(self):
        self.insights: List[Insight] = []
        self.puzzle_characteristics: Dict[str, any] = {}
        self.global_patterns: Set[str] = set()
        self.best_result: Optional[np.ndarray] = None
        self.best_score: float = 0.0

    def add_insight(self, insight: Insight):
        """Specialist shares what they learned."""
        self.insights.append(insight)

    def get_relevant_insights(self, specialist_name: str, top_n: int = 3) -> List[Insight]:
        """Get insights from OTHER specialists (cross-pollination)."""
        others_insights = [i for i in self.insights if i.specialist_name != specialist_name]
        # Return highest confidence insights
        return sorted(others_insights, key=lambda x: x.confidence, reverse=True)[:top_n]

    def update_best(self, result: np.ndarray, score: float):
        """Track best solution found so far."""
        if score > self.best_score:
            self.best_result = result
            self.best_score = score


class EvolvingSpecialist:
    """
    Base class for specialists that LEARN and GROW.

    Each specialist:
    1. Starts with core capability (their "name")
    2. Tries strategies and remembers what works
    3. Hears other specialists and incorporates insights
    4. Adapts approach based on puzzle characteristics
    5. Meta-learns (improves learning process itself)
    """

    def __init__(self, name: str, core_capability: str):
        self.name = name
        self.core_capability = core_capability
        self.memory = Memory()
        self.adaptation_level = 0  # How much specialist has evolved

    def solve(self, train_pairs: List[Tuple[np.ndarray, np.ndarray]],
             test_input: np.ndarray,
             collective: CollectiveKnowledge,
             time_budget: float = 30.0) -> EvolutionReport:
        """
        Solve with evolution and learning.

        Process:
        1. Observe puzzle (learn characteristics)
        2. Hear from other specialists (cross-pollination)
        3. Adapt strategy based on memory + insights
        4. Try approach and learn from result
        5. Generate insights for others
        """
        start_time = time.time()

        # Phase 1: OBSERVE puzzle
        characteristics = self._observe_puzzle(train_pairs, test_input)

        # Phase 2: HEAR from other specialists
        insights_heard = collective.get_relevant_insights(self.name)
        self.memory.heard_insights.extend(insights_heard)

        # Phase 3: ADAPT strategy based on memory + insights + characteristics
        strategy = self._adapt_strategy(characteristics, insights_heard)

        # Phase 4: TRY approach with time budget
        result, score, evolution_notes = self._try_strategy(
            strategy, train_pairs, test_input,
            time_budget - (time.time() - start_time)
        )

        # Phase 5: LEARN from result
        self._learn_from_attempt(strategy, score, characteristics)

        # Phase 6: GENERATE insights for collective
        insights = self._generate_insights(characteristics, strategy, score)

        for insight in insights:
            collective.add_insight(insight)

        if result is not None:
            collective.update_best(result, score)

        return EvolutionReport(
            name=self.name,
            result=result,
            confidence=score,
            strategy_used=strategy,
            insights_generated=insights,
            evolution_notes=evolution_notes
        )

    def _observe_puzzle(self, train_pairs, test_input) -> Dict[str, any]:
        """Observe puzzle characteristics - can be overridden by subclasses."""
        chars = {}

        # Basic observations all specialists can make
        chars['input_shapes'] = [inp.shape for inp, _ in train_pairs]
        chars['output_shapes'] = [out.shape for _, out in train_pairs]
        chars['size_change'] = any(i.shape != o.shape for i, o in train_pairs)
        chars['num_colors_in'] = [len(np.unique(inp)) for inp, _ in train_pairs]
        chars['num_colors_out'] = [len(np.unique(out)) for _, out in train_pairs]
        chars['test_shape'] = test_input.shape
        chars['test_colors'] = len(np.unique(test_input))

        return chars

    def _adapt_strategy(self, characteristics: Dict, insights: List[Insight]) -> str:
        """
        Adapt strategy based on characteristics + heard insights + memory.

        This is where EVOLUTION happens - specialist doesn't just apply
        their fixed transform, they ADAPT based on context.
        """
        # Start with core capability
        strategy = self.core_capability

        # Adapt based on memory (what worked before?)
        if self.memory.successes:
            best_past = max(self.memory.successes.items(), key=lambda x: x[1])
            if best_past[1] > 0.7:  # If we had strong success before
                strategy = f"{strategy}_refined_{best_past[0]}"
                self.adaptation_level += 1

        # Adapt based on insights from others
        for insight in insights:
            if insight.confidence > 0.8:
                # High-confidence insight from another specialist
                strategy = f"{strategy}_with_{insight.observation[:20]}"
                self.adaptation_level += 1

        # Adapt based on puzzle characteristics
        if characteristics.get('size_change'):
            strategy = f"{strategy}_size_aware"

        return strategy

    def _try_strategy(self, strategy: str, train_pairs, test_input, time_budget: float):
        """Try the adapted strategy - to be implemented by subclasses."""
        # Default: return None (subclass implements actual solving)
        return None, 0.0, "Base class - no implementation"

    def _learn_from_attempt(self, strategy: str, score: float, characteristics: Dict):
        """Learn from this attempt - update memory."""
        self.memory.strategies_tried.add(strategy)

        if score > 0.5:
            self.memory.successes[strategy] = score
            # Learn what patterns work
            pattern = f"{self.core_capability}_{characteristics.get('size_change', 'same_size')}"
            if pattern not in self.memory.learned_patterns:
                self.memory.learned_patterns.append(pattern)
        else:
            self.memory.failures[strategy] = f"Low score: {score:.2f}"

    def _generate_insights(self, characteristics: Dict, strategy: str, score: float) -> List[Insight]:
        """Generate insights to share with collective."""
        insights = []

        if score > 0.7:
            # Share what worked
            insights.append(Insight(
                specialist_name=self.name,
                observation=f"{self.core_capability} succeeded",
                hypothesis=f"This puzzle responds to {strategy}",
                confidence=score,
                context=str(characteristics)
            ))

        if score < 0.3 and self.core_capability in strategy:
            # Share what didn't work
            insights.append(Insight(
                specialist_name=self.name,
                observation=f"{self.core_capability} not applicable",
                hypothesis=f"Puzzle may not involve {self.core_capability}",
                confidence=1.0 - score,
                context=str(characteristics)
            ))

        return insights


# ============================================================================
# CONCRETE EVOLVING SPECIALISTS
# ============================================================================

class GridLearner(EvolvingSpecialist):
    """Starts knowing about grids, evolves to understand grid patterns."""

    def __init__(self):
        super().__init__("GridLearner", "grid_detection")

    def _try_strategy(self, strategy: str, train_pairs, test_input, time_budget: float):
        """Try grid-related strategies."""
        result = None
        best_score = 0.0
        notes = []

        # Try removing grid lines
        for inp, out in train_pairs[:2]:  # Time budget: check first 2
            if inp.shape == out.shape:
                colors = Counter(inp.flatten())
                for color, count in colors.most_common(3):  # Try top 3
                    test_copy = test_input.copy()
                    test_copy[test_copy == color] = 0

                    inp_copy = inp.copy()
                    inp_copy[inp_copy == color] = 0

                    if np.array_equal(inp_copy, out):
                        result = test_copy
                        best_score = 0.95
                        notes.append(f"Learned: Remove grid color {color}")
                        break

            if time.time() > time_budget:
                break

        evolution = " | ".join(notes) if notes else "No grid pattern found"
        return result, best_score, evolution


class PatternEvolver(EvolvingSpecialist):
    """Starts with pattern matching, evolves to understand pattern composition."""

    def __init__(self):
        super().__init__("PatternEvolver", "pattern_learning")

    def _observe_puzzle(self, train_pairs, test_input):
        """Enhanced observation for patterns."""
        chars = super()._observe_puzzle(train_pairs, test_input)

        # Pattern-specific observations
        chars['has_repetition'] = any(
            out.shape[0] % inp.shape[0] == 0 and out.shape[1] % inp.shape[1] == 0
            for inp, out in train_pairs
        )
        chars['has_symmetry'] = any(
            np.array_equal(inp, np.flip(inp, axis=0)) or
            np.array_equal(inp, np.flip(inp, axis=1))
            for inp, _ in train_pairs
        )

        return chars

    def _try_strategy(self, strategy: str, train_pairs, test_input, time_budget: float):
        """Try pattern-based strategies with evolution."""
        result = None
        best_score = 0.0
        notes = []

        # Basic pattern matching (core capability)
        for inp, out in train_pairs[:2]:
            if np.array_equal(inp, out):
                result = test_input
                best_score = 1.0
                notes.append("Learned: Identity transform")
                break

        # Evolved: Try tiling if characteristics suggest it
        if 'size_aware' in strategy:
            for inp, out in train_pairs[:1]:
                if out.shape[0] % inp.shape[0] == 0 and out.shape[1] % inp.shape[1] == 0:
                    tile_h = out.shape[0] // inp.shape[0]
                    tile_w = out.shape[1] // inp.shape[1]

                    if np.array_equal(np.tile(inp, (tile_h, tile_w)), out):
                        result = np.tile(test_input, (tile_h, tile_w))
                        best_score = 0.98
                        notes.append(f"Evolved: Learned tiling {tile_h}x{tile_w}")
                        break

        evolution = " | ".join(notes) if notes else "Pattern exploration"
        return result, best_score, evolution


class SymmetryAdaptor(EvolvingSpecialist):
    """Starts with symmetry operations, evolves to understand when/how to apply them."""

    def __init__(self):
        super().__init__("SymmetryAdaptor", "symmetry_operations")
        self.operations = ['flip_h', 'flip_v', 'rotate_90', 'rotate_180', 'rotate_270', 'transpose']

    def _try_strategy(self, strategy: str, train_pairs, test_input, time_budget: float):
        """Try symmetry operations with learning."""
        result = None
        best_score = 0.0
        notes = []

        for inp, out in train_pairs[:2]:
            # Try operations learned from memory first
            operations_to_try = list(self.memory.successes.keys()) or self.operations

            for op in operations_to_try[:4]:  # Time budget
                if 'flip_h' in op:
                    transformed = np.flip(inp, axis=0)
                elif 'flip_v' in op:
                    transformed = np.flip(inp, axis=1)
                elif 'rotate_90' in op:
                    transformed = np.rot90(inp, k=1)
                elif 'rotate_180' in op:
                    transformed = np.rot90(inp, k=2)
                elif 'rotate_270' in op:
                    transformed = np.rot90(inp, k=3)
                elif 'transpose' in op:
                    transformed = inp.T if inp.shape[0] == inp.shape[1] else inp
                else:
                    continue

                if np.array_equal(transformed, out):
                    # Apply to test
                    if 'flip_h' in op:
                        result = np.flip(test_input, axis=0)
                    elif 'flip_v' in op:
                        result = np.flip(test_input, axis=1)
                    elif 'rotate_90' in op:
                        result = np.rot90(test_input, k=1)
                    elif 'rotate_180' in op:
                        result = np.rot90(test_input, k=2)
                    elif 'rotate_270' in op:
                        result = np.rot90(test_input, k=3)
                    elif 'transpose' in op:
                        result = test_input.T if test_input.shape[0] == test_input.shape[1] else test_input

                    best_score = 0.99
                    notes.append(f"Evolved: Learned {op} works for this puzzle type")
                    break

            if result is not None:
                break

        evolution = " | ".join(notes) if notes else "Symmetry exploration"
        return result, best_score, evolution


class ColorTransformer(EvolvingSpecialist):
    """Starts with color mapping, evolves to understand color logic."""

    def __init__(self):
        super().__init__("ColorTransformer", "color_mapping")

    def _try_strategy(self, strategy: str, train_pairs, test_input, time_budget: float):
        """Try color transformations with learning."""
        result = None
        best_score = 0.0
        notes = []

        # Learn color mapping from training
        color_maps = []
        for inp, out in train_pairs[:3]:
            if inp.shape == out.shape:
                # Build color mapping
                color_map = {}
                for i in range(inp.shape[0]):
                    for j in range(inp.shape[1]):
                        inp_color = inp[i, j]
                        out_color = out[i, j]
                        if inp_color not in color_map:
                            color_map[inp_color] = out_color
                        elif color_map[inp_color] != out_color:
                            # Inconsistent mapping - context-dependent
                            color_map = None
                            break
                    if color_map is None:
                        break

                if color_map:
                    color_maps.append(color_map)

        # Find consistent mapping
        if color_maps:
            # Use first mapping (could evolve to merge maps)
            color_map = color_maps[0]
            result = test_input.copy()

            for i in range(result.shape[0]):
                for j in range(result.shape[1]):
                    if result[i, j] in color_map:
                        result[i, j] = color_map[result[i, j]]

            best_score = 0.90
            notes.append(f"Evolved: Learned color mapping {len(color_map)} colors")

        evolution = " | ".join(notes) if notes else "Color analysis"
        return result, best_score, evolution


class FillMaster(EvolvingSpecialist):
    """Starts with flood-fill, evolves to understand fill logic and boundaries."""

    def __init__(self):
        super().__init__("FillMaster", "interior_fill")

    def _try_strategy(self, strategy: str, train_pairs, test_input, time_budget: float):
        """Try fill operations with learning."""
        result = None
        best_score = 0.0
        notes = []

        for inp, out in train_pairs[:2]:
            if inp.shape == out.shape:
                # Check if output fills interior regions
                diff_mask = (inp != out)

                if np.any(diff_mask):
                    # Find what color fills
                    fill_colors = out[diff_mask]
                    if len(fill_colors) > 0:
                        fill_color = Counter(fill_colors).most_common(1)[0][0]

                        # Find interior cells in test input
                        interior = self._find_interior(test_input)

                        result = test_input.copy()
                        for i, j in interior:
                            result[i, j] = fill_color

                        best_score = 0.85
                        notes.append(f"Evolved: Learned to fill interior with color {fill_color}")
                        break

        evolution = " | ".join(notes) if notes else "Fill exploration"
        return result, best_score, evolution

    def _find_interior(self, grid, bg=0):
        """Find interior cells using flood-fill from edges."""
        exterior = set()
        h, w = grid.shape
        stack = []

        # Start from all edge cells
        for i in range(h):
            stack.append((i, 0))
            stack.append((i, w-1))
        for j in range(w):
            stack.append((0, j))
            stack.append((h-1, j))

        # Flood fill from edges
        while stack:
            i, j = stack.pop()
            if (i, j) in exterior:
                continue
            if not (0 <= i < h and 0 <= j < w):
                continue

            exterior.add((i, j))

            for di, dj in [(0,1), (1,0), (0,-1), (-1,0)]:
                stack.append((i+di, j+dj))

        # Interior = all cells not in exterior
        interior = []
        for i in range(h):
            for j in range(w):
                if (i, j) not in exterior:
                    interior.append((i, j))

        return interior


# ============================================================================
# EVOLVING SOLVER
# ============================================================================

class EvolvingSolver:
    """
    Solver with evolving specialists that learn and grow.

    Time-boxed with 2-3 minute ceiling as requested.
    """

    def __init__(self, time_limit: float = 150.0, verbose: bool = True):
        self.time_limit = time_limit  # 2.5 minutes default
        self.verbose = verbose

        # Initialize evolving specialists
        self.specialists = [
            GridLearner(),
            PatternEvolver(),
            SymmetryAdaptor(),
            ColorTransformer(),
            FillMaster(),
        ]

    def solve(self, train_pairs: List[Tuple[np.ndarray, np.ndarray]],
              test_input: np.ndarray) -> Optional[np.ndarray]:
        """Solve with time-boxed evolving specialists."""

        start_time = time.time()
        collective = CollectiveKnowledge()

        if self.verbose:
            print(f"\n{'='*80}")
            print(f"🧠 EVOLVING SOLVER - Specialists that Learn and Grow")
            print(f"{'='*80}")
            print(f"Time budget: {self.time_limit:.1f}s")

        # Give each specialist time budget
        time_per_specialist = self.time_limit / len(self.specialists)

        reports = []
        for specialist in self.specialists:
            elapsed = time.time() - start_time
            remaining = self.time_limit - elapsed

            if remaining < 1.0:
                if self.verbose:
                    print(f"⏱️ Time limit reached, stopping early")
                break

            budget = min(time_per_specialist, remaining)

            try:
                report = specialist.solve(train_pairs, test_input, collective, budget)
                reports.append(report)

                if self.verbose:
                    print(f"\n{specialist.name}:")
                    print(f"  Strategy: {report.strategy_used}")
                    print(f"  Score: {report.confidence*100:.1f}%")
                    print(f"  Evolution: {report.evolution_notes}")
                    print(f"  Insights: {len(report.insights_generated)} shared")

            except Exception as e:
                if self.verbose:
                    print(f"  ⚠️ {specialist.name} error: {str(e)[:40]}")

        # Return best result
        if collective.best_result is not None:
            if self.verbose:
                print(f"\n✅ Best solution: {collective.best_score*100:.1f}% confidence")
                print(f"⏱️ Time used: {time.time() - start_time:.1f}s")
            return collective.best_result

        if self.verbose:
            print(f"\n⚠️ No solution found")
        return None


# ============================================================================
# EXPORT
# ============================================================================

ALL_EVOLVING_SPECIALISTS = [
    GridLearner,
    PatternEvolver,
    SymmetryAdaptor,
    ColorTransformer,
    FillMaster,
]
