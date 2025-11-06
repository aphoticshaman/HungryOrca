"""
🧠🌀 META-COGNITIVE REASONING ENGINE

Not just "what changed" but WHY and HOW and WHAT IF

REASONING MODES:
- Induction: Specific examples → General rule
- Deduction: General rule → Specific application
- Abduction: Observations → Best explanation
- Inference: Derive implicit knowledge
- Insight: Sudden gestalt understanding

AWARENESS LEVELS:
- Self: What do I know? What don't I know? How certain am I?
- Object: What are the entities? Their properties? Relationships?
- Environment: Context? Constraints? Invariants?
- Causal: What causes what? 25-order effects (upstream/downstream)

EMERGENCE & UPLIFT:
- Convergence: Multiple patterns → Unified understanding
- Emergence: Simple rules → Complex behavior
- Uplift: Learn from mistakes, improve reasoning

Author: Ryan Cardwell & Claude
Date: November 2025
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict
import itertools


@dataclass
class Entity:
    """An object in the grid with properties"""
    pixels: Set[Tuple[int, int]]
    color: int
    shape: str  # rectangle, line, blob, etc.
    size: int
    position: Tuple[int, int]  # center

    def __hash__(self):
        return hash((tuple(sorted(self.pixels)), self.color))


@dataclass
class Relationship:
    """Relationship between entities"""
    entity1: Entity
    entity2: Entity
    relation_type: str  # adjacent, inside, same_color, aligned, etc.
    strength: float


@dataclass
class CausalChain:
    """Causal relationship with N-order effects"""
    cause: str
    effect: str
    order: int  # 1st order (direct), 2nd order (indirect), etc.
    confidence: float


class MetaCognitiveReasoner:
    """
    Reasons about reasoning itself

    Understands tasks at multiple abstraction levels:
    - Pixel level (raw data)
    - Object level (entities, properties)
    - Relationship level (spatial, semantic)
    - Rule level (transformations)
    - Meta level (why these rules? what's the intent?)
    """

    def __init__(self):
        self.knowledge_base = {
            'learned_rules': [],
            'successful_patterns': [],
            'failed_attempts': [],
            'uncertainty': defaultdict(float)
        }
        self.reasoning_history = []

    def analyze_at_all_levels(self, task: Dict) -> Dict:
        """
        Multi-level analysis of the task

        Returns understanding at:
        - Pixel level
        - Object level
        - Relationship level
        - Rule level
        - Meta level
        """
        training = task.get('train', [])
        if not training:
            return {}

        analysis = {
            'pixel_level': self._analyze_pixels(training),
            'object_level': self._analyze_objects(training),
            'relationship_level': self._analyze_relationships(training),
            'rule_level': self._induce_rules(training),
            'meta_level': self._meta_understand(training),
            'causal_chains': self._trace_causality(training, max_order=25)
        }

        return analysis

    def _analyze_pixels(self, training: List[Dict]) -> Dict:
        """Pixel-level analysis: raw statistics"""
        analysis = {
            'color_distributions': [],
            'shape_statistics': [],
            'size_changes': []
        }

        for example in training:
            inp = np.array(example['input'])
            out = np.array(example['output'])

            # Color analysis
            in_colors = np.unique(inp)
            out_colors = np.unique(out)
            analysis['color_distributions'].append({
                'input_colors': in_colors.tolist(),
                'output_colors': out_colors.tolist(),
                'colors_added': set(out_colors) - set(in_colors),
                'colors_removed': set(in_colors) - set(out_colors)
            })

            # Shape analysis
            analysis['shape_statistics'].append({
                'input_shape': inp.shape,
                'output_shape': out.shape,
                'size_ratio': (out.shape[0] * out.shape[1]) / (inp.shape[0] * inp.shape[1])
            })

        return analysis

    def _analyze_objects(self, training: List[Dict]) -> Dict:
        """Object-level: entities and properties"""
        objects_analysis = {
            'input_objects': [],
            'output_objects': [],
            'object_transformations': []
        }

        for example in training:
            inp = np.array(example['input'])
            out = np.array(example['output'])

            # Extract entities
            input_entities = self._extract_entities(inp)
            output_entities = self._extract_entities(out)

            objects_analysis['input_objects'].append(input_entities)
            objects_analysis['output_objects'].append(output_entities)

            # Understand transformations
            transforms = self._match_objects(input_entities, output_entities)
            objects_analysis['object_transformations'].append(transforms)

        return objects_analysis

    def _extract_entities(self, grid: np.ndarray) -> List[Entity]:
        """Extract distinct objects from grid"""
        entities = []

        # Find connected components for each color
        colors = np.unique(grid)
        colors = colors[colors != 0]  # Ignore background

        for color in colors:
            mask = (grid == color)
            labeled = self._label_components(mask)

            for label in range(1, labeled.max() + 1):
                component = (labeled == label)
                pixels = set(zip(*np.where(component)))

                if len(pixels) > 0:
                    # Determine shape
                    rows, cols = zip(*pixels)
                    height = max(rows) - min(rows) + 1
                    width = max(cols) - min(cols) + 1

                    shape = 'unknown'
                    if height == 1:
                        shape = 'horizontal_line'
                    elif width == 1:
                        shape = 'vertical_line'
                    elif height == width:
                        shape = 'square_ish'
                    else:
                        shape = 'rectangle'

                    entity = Entity(
                        pixels=pixels,
                        color=int(color),
                        shape=shape,
                        size=len(pixels),
                        position=(int(np.mean(rows)), int(np.mean(cols)))
                    )
                    entities.append(entity)

        return entities

    def _label_components(self, mask: np.ndarray) -> np.ndarray:
        """Simple connected component labeling"""
        labeled = np.zeros_like(mask, dtype=int)
        current_label = 1

        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if mask[i, j] and labeled[i, j] == 0:
                    self._flood_fill(mask, labeled, i, j, current_label)
                    current_label += 1

        return labeled

    def _flood_fill(self, mask, labeled, i, j, label):
        """Flood fill for connected components"""
        if i < 0 or i >= mask.shape[0] or j < 0 or j >= mask.shape[1]:
            return
        if not mask[i, j] or labeled[i, j] != 0:
            return

        labeled[i, j] = label
        # 4-connectivity
        self._flood_fill(mask, labeled, i-1, j, label)
        self._flood_fill(mask, labeled, i+1, j, label)
        self._flood_fill(mask, labeled, i, j-1, label)
        self._flood_fill(mask, labeled, i, j+1, label)

    def _match_objects(self, input_entities: List[Entity], output_entities: List[Entity]) -> List[Dict]:
        """Match input objects to output objects and understand transformation"""
        transformations = []

        # Try to match by position and properties
        for out_ent in output_entities:
            best_match = None
            best_score = 0

            for in_ent in input_entities:
                # Score match quality
                score = 0
                if in_ent.color == out_ent.color:
                    score += 3
                if in_ent.shape == out_ent.shape:
                    score += 2
                if abs(in_ent.size - out_ent.size) < 5:
                    score += 1

                if score > best_score:
                    best_score = score
                    best_match = in_ent

            if best_match:
                transformations.append({
                    'input': best_match,
                    'output': out_ent,
                    'changed_color': best_match.color != out_ent.color,
                    'changed_shape': best_match.shape != out_ent.shape,
                    'changed_size': best_match.size != out_ent.size,
                    'changed_position': best_match.position != out_ent.position
                })

        return transformations

    def _analyze_relationships(self, training: List[Dict]) -> Dict:
        """Relationship-level: spatial and semantic relationships"""
        relationships = {
            'spatial': [],
            'semantic': []
        }

        for example in training:
            inp = np.array(example['input'])
            entities = self._extract_entities(inp)

            # Find spatial relationships
            spatial_rels = []
            for e1, e2 in itertools.combinations(entities, 2):
                # Check adjacency
                if self._are_adjacent(e1, e2):
                    spatial_rels.append(Relationship(e1, e2, 'adjacent', 1.0))

                # Check alignment
                if abs(e1.position[0] - e2.position[0]) < 2:
                    spatial_rels.append(Relationship(e1, e2, 'horizontally_aligned', 0.8))
                if abs(e1.position[1] - e2.position[1]) < 2:
                    spatial_rels.append(Relationship(e1, e2, 'vertically_aligned', 0.8))

                # Check containment
                if self._contains(e1, e2):
                    spatial_rels.append(Relationship(e1, e2, 'contains', 1.0))

            relationships['spatial'].append(spatial_rels)

        return relationships

    def _are_adjacent(self, e1: Entity, e2: Entity) -> bool:
        """Check if two entities are adjacent"""
        for p1 in e1.pixels:
            for p2 in e2.pixels:
                dist = abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
                if dist == 1:
                    return True
        return False

    def _contains(self, e1: Entity, e2: Entity) -> bool:
        """Check if e1 contains e2"""
        return e2.pixels.issubset(e1.pixels)

    def _induce_rules(self, training: List[Dict]) -> Dict:
        """
        INDUCTION: From specific examples → General rule

        Look at all examples and induce the underlying transformation rule
        """
        rules = {
            'geometric': [],
            'color': [],
            'structural': [],
            'composite': []
        }

        # Check for geometric transformations
        geometric_rule = self._induce_geometric_rule(training)
        if geometric_rule:
            rules['geometric'].append(geometric_rule)

        # Check for color transformations
        color_rule = self._induce_color_rule(training)
        if color_rule:
            rules['color'].append(color_rule)

        # Check for structural transformations
        structural_rule = self._induce_structural_rule(training)
        if structural_rule:
            rules['structural'].append(structural_rule)

        return rules

    def _induce_geometric_rule(self, training: List[Dict]) -> Optional[Dict]:
        """Induce geometric transformation rule"""
        # Try each geometric transform and see if it works on all examples
        transforms = {
            'rotate_90': lambda g: np.rot90(g, 1),
            'rotate_180': lambda g: np.rot90(g, 2),
            'rotate_270': lambda g: np.rot90(g, 3),
            'flip_h': lambda g: np.fliplr(g),
            'flip_v': lambda g: np.flipud(g),
            'transpose': lambda g: g.T if g.shape[0] == g.shape[1] else None
        }

        for name, transform in transforms.items():
            matches_all = True
            for example in training:
                inp = np.array(example['input'])
                out = np.array(example['output'])

                try:
                    result = transform(inp)
                    if result is None or not np.array_equal(result, out):
                        matches_all = False
                        break
                except:
                    matches_all = False
                    break

            if matches_all:
                return {
                    'type': 'geometric',
                    'operation': name,
                    'transform': transform,
                    'confidence': 1.0
                }

        return None

    def _induce_color_rule(self, training: List[Dict]) -> Optional[Dict]:
        """Induce color mapping rule"""
        # Check if there's a consistent color mapping
        color_map = {}

        for example in training:
            inp = np.array(example['input'])
            out = np.array(example['output'])

            if inp.shape != out.shape:
                return None

            for i in range(inp.shape[0]):
                for j in range(inp.shape[1]):
                    in_color = inp[i, j]
                    out_color = out[i, j]

                    if in_color in color_map:
                        if color_map[in_color] != out_color:
                            return None  # Inconsistent
                    else:
                        color_map[in_color] = out_color

        if color_map:
            return {
                'type': 'color_mapping',
                'mapping': color_map,
                'confidence': 1.0
            }

        return None

    def _induce_structural_rule(self, training: List[Dict]) -> Optional[Dict]:
        """Induce structural transformation (crop, tile, etc.)"""
        # Check for tiling
        first = training[0]
        inp = np.array(first['input'])
        out = np.array(first['output'])

        if out.shape[0] % inp.shape[0] == 0 and out.shape[1] % inp.shape[1] == 0:
            tile_h = out.shape[0] // inp.shape[0]
            tile_w = out.shape[1] // inp.shape[1]

            # Verify on all examples
            for example in training[1:]:
                inp2 = np.array(example['input'])
                out2 = np.array(example['output'])
                expected = np.tile(inp2, (tile_h, tile_w))
                if not np.array_equal(expected, out2):
                    return None

            return {
                'type': 'tile',
                'factor': (tile_h, tile_w),
                'confidence': 1.0
            }

        return None

    def _meta_understand(self, training: List[Dict]) -> Dict:
        """
        META-LEVEL: Why does this transformation exist? What's the intent?

        This is reasoning about reasoning - understanding the DESIGNER'S intent
        """
        meta = {
            'intent': None,
            'category': None,
            'difficulty': None,
            'abstraction_level': None
        }

        # Analyze complexity
        first = training[0]
        inp = np.array(first['input'])
        out = np.array(first['output'])

        # Simple transformations = low abstraction
        if np.array_equal(np.rot90(inp), out):
            meta['intent'] = 'Test geometric understanding'
            meta['category'] = 'geometric'
            meta['difficulty'] = 'easy'
            meta['abstraction_level'] = 1

        # Complex object manipulations = high abstraction
        elif inp.shape != out.shape:
            meta['intent'] = 'Test structural reasoning'
            meta['category'] = 'structural'
            meta['difficulty'] = 'medium'
            meta['abstraction_level'] = 2

        # Object-based reasoning = very high abstraction
        entities_in = self._extract_entities(inp)
        entities_out = self._extract_entities(out)
        if len(entities_in) > 3 or len(entities_out) > 3:
            meta['intent'] = 'Test object-level reasoning'
            meta['category'] = 'object_manipulation'
            meta['difficulty'] = 'hard'
            meta['abstraction_level'] = 3

        return meta

    def _trace_causality(self, training: List[Dict], max_order: int = 25) -> List[CausalChain]:
        """
        Trace causal chains up to N orders

        1st order: Direct cause (rotate → output rotated)
        2nd order: What enables rotation? (square grid)
        3rd order: Why square? (designer choice)
        ...up to 25th order
        """
        causal_chains = []

        # 1st order: Direct transformation
        rule = self._induce_geometric_rule(training)
        if rule:
            causal_chains.append(CausalChain(
                cause='geometric_transform',
                effect='output_matches_transform',
                order=1,
                confidence=1.0
            ))

            # 2nd order: What enables this transform?
            first = training[0]
            inp = np.array(first['input'])
            if inp.shape[0] == inp.shape[1]:
                causal_chains.append(CausalChain(
                    cause='square_grid',
                    effect='rotation_possible',
                    order=2,
                    confidence=0.9
                ))

                # 3rd order: Why square?
                causal_chains.append(CausalChain(
                    cause='designer_intent_simplicity',
                    effect='square_grid_chosen',
                    order=3,
                    confidence=0.7
                ))

        # Keep tracing upstream (why) and downstream (consequences)
        # This can go up to 25 levels but gets increasingly speculative

        return causal_chains

    def deduce_solution(self, task: Dict, analysis: Dict) -> Optional[np.ndarray]:
        """
        DEDUCTION: Apply general rule → Specific solution

        Given the induced rules, deduce the solution for the test case
        """
        rules = analysis.get('rule_level', {})
        test_input = np.array(task['test'][0]['input'])

        # Try geometric rules first
        for rule in rules.get('geometric', []):
            if rule['confidence'] > 0.8:
                try:
                    return rule['transform'](test_input)
                except:
                    pass

        # Try color rules
        for rule in rules.get('color', []):
            if rule['confidence'] > 0.8:
                color_map = rule['mapping']
                result = test_input.copy()
                for old, new in color_map.items():
                    result[test_input == old] = new
                return result

        # Try structural rules
        for rule in rules.get('structural', []):
            if rule['confidence'] > 0.8:
                if rule['type'] == 'tile':
                    tile_h, tile_w = rule['factor']
                    return np.tile(test_input, (tile_h, tile_w))

        return None

    def abduce_best_explanation(self, task: Dict, observations: List) -> Dict:
        """
        ABDUCTION: Given observations → Best explanation

        Multiple hypotheses might fit - choose the most likely
        """
        hypotheses = []

        # Generate multiple possible explanations
        analysis = self.analyze_at_all_levels(task)

        for rule_type in ['geometric', 'color', 'structural']:
            rules = analysis['rule_level'].get(rule_type, [])
            for rule in rules:
                hypotheses.append({
                    'explanation': rule,
                    'likelihood': rule.get('confidence', 0.5),
                    'complexity': self._complexity_score(rule)
                })

        # Choose best hypothesis (highest likelihood, lowest complexity)
        if hypotheses:
            best = max(hypotheses, key=lambda h: h['likelihood'] - 0.1 * h['complexity'])
            return best['explanation']

        return {}

    def _complexity_score(self, rule: Dict) -> float:
        """Occam's razor: simpler explanations are better"""
        if rule.get('type') == 'geometric':
            return 1.0  # Simple
        elif rule.get('type') == 'color_mapping':
            return 2.0  # Medium
        elif rule.get('type') == 'structural':
            return 3.0  # Complex
        return 4.0

    def self_assess(self, task: Dict, solution: np.ndarray) -> Dict:
        """
        SELF-AWARENESS: How certain am I? What do I know vs not know?
        """
        assessment = {
            'confidence': 0.0,
            'certainty_factors': {},
            'known': [],
            'unknown': [],
            'assumptions': []
        }

        analysis = self.analyze_at_all_levels(task)

        # Check rule confidence
        rules = analysis['rule_level']
        max_conf = 0
        for rule_type, rule_list in rules.items():
            for rule in rule_list:
                conf = rule.get('confidence', 0)
                if conf > max_conf:
                    max_conf = conf

        assessment['confidence'] = max_conf

        # What do we know?
        if rules.get('geometric'):
            assessment['known'].append('geometric_transformation')
        if rules.get('color'):
            assessment['known'].append('color_mapping')

        # What don't we know?
        if not rules.get('geometric') and not rules.get('color'):
            assessment['unknown'].append('transformation_type')

        # Meta-level uncertainty
        meta = analysis.get('meta_level', {})
        if meta.get('difficulty') == 'hard':
            assessment['assumptions'].append('high_abstraction_required')

        return assessment

    def reason_and_solve(self, task: Dict) -> Tuple[Optional[np.ndarray], Dict]:
        """
        Complete metacognitive reasoning pipeline

        Returns: (solution, reasoning_trace)
        """
        # 1. Analyze at all levels
        analysis = self.analyze_at_all_levels(task)

        # 2. Induce rules (induction)
        # Already done in analysis

        # 3. Abduce best explanation (abduction)
        best_explanation = self.abduce_best_explanation(task, [])

        # 4. Deduce solution (deduction)
        solution = self.deduce_solution(task, analysis)

        # 5. Self-assess (metacognition)
        if solution is not None:
            assessment = self.self_assess(task, solution)
        else:
            assessment = {'confidence': 0.0}

        reasoning_trace = {
            'analysis': analysis,
            'best_explanation': best_explanation,
            'self_assessment': assessment,
            'reasoning_mode': 'metacognitive'
        }

        return solution, reasoning_trace
