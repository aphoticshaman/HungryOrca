# Multi-Stage Reasoning Architecture for ARC

## The Problem with Previous Solvers

### GatORCA (3.3% accuracy)
- **Issue**: Random mutation of operation sequences
- **Level**: L1 only (pixel transforms)
- **Approach**: Evolutionary search through 65 operations hoping to find the right sequence
- **Why it fails**: Most ARC tasks require object-level reasoning, not pixel-level transforms

### LucidOrca v2.0 Beta
- **Issue**: Has object detection primitives but doesn't use them properly
- **Level**: Still mostly L1 with some L2 components
- **Approach**: EvolutionaryBeamSearch randomly evolves operation sequences
- **Why it may fail**: No structured multi-stage reasoning - still hoping random combinations work

## The Solution: True Multi-Stage Reasoning

### Architecture: `arc_multi_stage_reasoner.py`

Implements **4 levels of reasoning** as described in ARC best practices:

```
L1 (Pixel):     Direct grid transforms (rotate, flip, crop)
                ↓
L2 (Object):    Transform individual objects (move, recolor, resize)
                ↓
L3 (Pattern):   Infer rules between input/output object sets
                ↓
L4 (Constraint): Filter hypotheses using properties
```

### Key Components

#### 1. Object Decomposition (Foundation)

```python
class ARCObject:
    """
    Represents a single object with properties:
    - mask, color, bounding_box, centroid
    - size, width, height
    - is_square, is_rectangle, is_line
    - neighbors, relative_position
    """

class ObjectDecomposer:
    """
    Decomposes grid into objects using connected components (4 or 8-connectivity).
    Analyzes spatial relationships between objects.
    """
```

**What this enables**: Instead of treating the grid as a blob of pixels, we understand it as a scene containing objects with properties and relationships.

#### 2. L1: Pixel-Level Transforms

```python
class L1_PixelTransforms:
    - rotate_90/180/270
    - flip_h/flip_v
    - transpose
    - crop_to_objects
```

**When used**: Simple tasks where the entire grid is transformed uniformly.

**Example**: "Rotate the input 90 degrees"

#### 3. L2: Object-Level Transforms

```python
class L2_ObjectTransforms:
    - move_object(delta_r, delta_c)
    - recolor_object(new_color)
    - scale_object(scale_factor)
    - delete_object()
    - duplicate_object(offset)
```

**When used**: Tasks where individual objects are manipulated independently.

**Example**: "Move the red square to the center" or "Delete all small objects"

#### 4. L3: Pattern-Based Rule Inference

```python
class L3_PatternReasoner:
    def infer_rule(input_grid, output_grid) -> Rule

    Detects patterns like:
    - "select_largest": Keep only the largest object
    - "select_by_color": Keep objects of specific color
    - "color_map": Map colors {1→3, 2→5}
    - "move_objects": Move each object by (Δr, Δc)
    - "duplicate_objects": Replicate objects N times
    - "rotate_90"/"flip_h": Grid-level transforms
```

**When used**: This is the **core reasoning engine**. It analyzes training examples to understand "what changed" and infers the transformation rule.

**Example Process**:
1. Input has 5 objects, output has 1 object
2. The output object is the largest input object
3. **Inferred rule**: `{"type": "select_largest", "criterion": "size", "confidence": 0.9}`

#### 5. L4: Constraint-Based Filtering

```python
class L4_ConstraintFilter:
    - validate_size_constraint(max_size)
    - validate_color_count(expected_colors)
    - validate_object_count(min, max)
```

**When used**: Filter out invalid hypotheses that violate known constraints.

**Example**: If all training outputs have 3 objects, reject solutions with 5 objects.

### Solver Pipeline

```python
class MultiStageSolver:
    def solve_task(task):
        # Step 1: Learn from training examples (L3)
        for train_example in task['train']:
            rule = l3.infer_rule(train_example['input'], train_example['output'])
            rules.append(rule)

        # Step 2: Select most confident consistent rule
        best_rule = select_best_rule(rules)  # Votes across examples

        # Step 3: Apply rule to test inputs
        for test_input in task['test']:
            prediction = apply_rule(best_rule, test_input)

            # Uses L1, L2 transforms as needed based on rule type
            if best_rule.type == "select_largest":
                # Decompose → Find largest object → Extract
                prediction = select_largest_object(test_input)

            predictions.append(prediction)

        return predictions
```

## Why This is Better

### Comparison Table

| Feature | GatORCA | LucidOrca | Multi-Stage |
|---------|---------|-----------|-------------|
| **Object Awareness** | ❌ No | ✅ Has primitives | ✅ Core foundation |
| **Structured Reasoning** | ❌ Random | ❌ Random evolution | ✅ 4-level hierarchy |
| **Rule Inference** | ❌ No | ❌ Weak | ✅ Explicit L3 layer |
| **Learns from Training** | ⚠️ Fitness only | ⚠️ Fitness only | ✅ Explicit rules |
| **Complexity** | L1 only | L1 + partial L2 | L1 + L2 + L3 + L4 |

### Expected Performance Improvement

**GatORCA**: 3.3% (1/30 tasks)
- Why: Only handles trivial L1 tasks

**LucidOrca**: 10-15% estimated
- Why: Has better primitives but still random search

**Multi-Stage**: 20-30% estimated
- Why: Proper object-centric reasoning
- Handles L2 tasks (object manipulation)
- Handles L3 tasks (pattern recognition)
- Learns explicit rules instead of fitness-guided search

### Key Insight

> **Most ARC tasks are NOT solvable by random combinations of operations!**
>
> They require understanding:
> 1. **What are the objects?** (Decomposition)
> 2. **What relationships exist?** (Spatial analysis)
> 3. **What transformation occurred?** (Rule inference)
> 4. **How do I apply that rule?** (Structured execution)

Random evolution of operation sequences is like trying to solve a math word problem by randomly trying arithmetic operations until something works. It might solve "2 + 2 = ?" but will fail on "If Alice has 3 apples and gives 1 to Bob..."

## Deployment

### For Kaggle

The multi-stage solver can be used standalone:

```python
from arc_multi_stage_reasoner import MultiStageSolver, generate_submission

# Generate submission
generate_submission(
    'arc-agi_test_challenges.json',
    'submission.json'
)
```

### Integration with LucidOrca

LucidOrca's infrastructure (checkpointing, metrics, time management) can be combined with the Multi-Stage reasoning engine:

1. Replace `EvolutionaryBeamSearch` with `MultiStageSolver`
2. Keep the task classification and routing
3. Use multi-stage solver instead of random genome evolution

## Next Steps

1. **Test on training set**: Measure actual accuracy improvement
2. **Add more L3 rules**: Current implementation handles ~10 rule types, can expand to 50+
3. **Implement L4 filtering**: Add constraint validation
4. **Hybrid approach**: Use multi-stage for L2/L3 tasks, fall back to evolution for complex tasks
5. **Deploy to Kaggle**: Test on actual competition

## References

- ARC Prize 2025: https://arcprize.org/
- ARC Dataset: https://github.com/fchollet/ARC-AGI
- Object-Centric Learning: Key to solving ARC (Chollet, 2019)
