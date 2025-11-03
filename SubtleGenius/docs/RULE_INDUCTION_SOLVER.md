# Rule Induction Solver
## Learn Rules from Examples, Apply to Test Cases

**Priority**: HIGH (common in ARC tasks)
**Expected Coverage**: 15-25% of tasks
**Difficulty**: Medium
**Integration**: Add as new solver layer above pattern matching

---

## Concept

**Rule Induction**: Learn transformation rules by analyzing input→output pairs in training examples, then apply those rules to test inputs.

**Example:**
```
Training pair 1:
  Input:  [[1,1,2], [1,2,2], [2,2,2]]
  Output: [[3,3,4], [3,4,4], [4,4,4]]
  Rule: All 1s → 3, All 2s → 4

Training pair 2:
  Input:  [[1,0,1], [0,1,0], [1,0,1]]
  Output: [[3,0,3], [0,3,0], [3,0,3]]
  Rule: Confirms - 1→3 mapping

Test input: [[2,1,2], [1,1,1], [2,2,1]]
Apply rule: [[4,3,4], [3,3,3], [4,4,3]]
```

---

## Rule Types to Detect

### 1. **Color Mapping Rules** (Most Common)
```python
def detect_color_mapping_rule(train_pairs):
    """
    Detect if transformation is: color X → color Y

    Returns: dict {old_color: new_color} or None
    """
    # Extract mapping from first pair
    inp = train_pairs[0]['input']
    out = train_pairs[0]['output']

    # Must be same shape
    if shape(inp) != shape(out):
        return None

    # Build mapping
    mapping = {}
    for (in_val, out_val) in zip(flatten(inp), flatten(out)):
        if in_val in mapping:
            if mapping[in_val] != out_val:
                return None  # Inconsistent
        else:
            mapping[in_val] = out_val

    # Verify on all other training pairs
    for pair in train_pairs[1:]:
        if not verify_mapping(pair, mapping):
            return None

    return mapping
```

**Coverage**: ~10-15% of ARC tasks

### 2. **Conditional Rules** (If-Then)
```python
def detect_conditional_rule(train_pairs):
    """
    Detect rules like:
    - If color X appears → do Y
    - If count(X) > N → do Z
    - If X is in corner → do W
    """

    rules = []

    # Rule: "If rare color appears, keep only that color"
    if all_have_rare_color_isolation(train_pairs):
        rules.append({
            'type': 'rare_color_isolation',
            'apply': lambda grid: isolate_rare_colors(grid)
        })

    # Rule: "If majority color, fill entire grid with it"
    if all_have_majority_fill(train_pairs):
        rules.append({
            'type': 'majority_fill',
            'apply': lambda grid: fill_with_majority(grid)
        })

    # Rule: "If background is X, change to Y"
    background_rule = detect_background_change(train_pairs)
    if background_rule:
        rules.append(background_rule)

    return rules if rules else None
```

**Coverage**: ~5-8% of ARC tasks

### 3. **Size/Shape Rules**
```python
def detect_size_rules(train_pairs):
    """
    Detect rules about grid size changes:
    - Output size = input size × N
    - Output size = count(objects) × M
    - Output size = fixed size regardless of input
    """

    # Check if output size is related to input size
    ratios = []
    for pair in train_pairs:
        inp_h, inp_w = shape(pair['input'])
        out_h, out_w = shape(pair['output'])

        h_ratio = out_h / inp_h if inp_h > 0 else 0
        w_ratio = out_w / inp_w if inp_w > 0 else 0

        ratios.append((h_ratio, w_ratio))

    # Consistent ratio?
    if all(r == ratios[0] for r in ratios):
        return {
            'type': 'size_scale',
            'h_ratio': ratios[0][0],
            'w_ratio': ratios[0][1]
        }

    # Check if output size = count(something)
    # [etc.]

    return None
```

**Coverage**: ~3-5% of ARC tasks

### 4. **Spatial Rules**
```python
def detect_spatial_rules(train_pairs):
    """
    Detect spatial transformation rules:
    - Mirror across axis
    - Rotation by consistent angle
    - Crop to bounding box
    - Tile/repeat pattern
    """

    # Already covered by pattern matching
    # But can be LEARNED from examples vs hardcoded

    for pair in train_pairs:
        inp = pair['input']
        out = pair['output']

        # Is output a crop of input?
        if is_crop(inp, out):
            # Learn crop rule
            bbox = find_crop_bbox(inp, out)
            # Verify on other pairs
            # Return rule

    return None
```

**Coverage**: ~5-10% of ARC tasks (overlap with pattern matching)

### 5. **Counting Rules**
```python
def detect_counting_rules(train_pairs):
    """
    Detect rules based on counting:
    - Output has N×M grid where N = count(objects)
    - Color of pixel = count of neighbors
    - Grid size = number of unique colors
    """

    # Rule: Output size based on object count
    for pair in train_pairs:
        inp = pair['input']
        out = pair['output']

        objects = find_connected_components(inp)
        num_objects = len(objects)

        if shape(out) == (num_objects, num_objects):
            # Verify on all pairs
            if all_have_size_equal_count(train_pairs):
                return {
                    'type': 'count_to_size',
                    'apply': lambda grid: create_grid_from_count(grid)
                }

    return None
```

**Coverage**: ~2-4% of ARC tasks

---

## Implementation Architecture

```python
# rule_induction_solver.py

class RuleInductionSolver:
    """
    Learn rules from training examples, apply to test.
    """

    def __init__(self):
        self.rule_detectors = [
            detect_color_mapping_rule,
            detect_conditional_rule,
            detect_size_rules,
            detect_spatial_rules,
            detect_counting_rules,
        ]

    def detect_rules(self, train_pairs):
        """Try all rule detectors, return best matches"""
        detected_rules = []

        for detector in self.rule_detectors:
            try:
                rule = detector(train_pairs)
                if rule is not None:
                    detected_rules.append(rule)
            except Exception as e:
                logger.debug(f"Rule detector {detector.__name__} failed: {e}")

        return detected_rules

    def apply_rules(self, test_input, rules):
        """Apply detected rules to test input"""
        if not rules:
            return None

        # Try each rule
        for rule in rules:
            try:
                if callable(rule):
                    result = rule(test_input)
                elif isinstance(rule, dict):
                    # Color mapping
                    if 'type' in rule and rule['type'] == 'color_mapping':
                        result = apply_color_mapping(test_input, rule['mapping'])
                    # Size scaling
                    elif rule.get('type') == 'size_scale':
                        result = scale_grid(test_input, rule['h_ratio'], rule['w_ratio'])
                    # Custom apply function
                    elif 'apply' in rule:
                        result = rule['apply'](test_input)

                # Validate result
                if is_valid_grid(result):
                    return result

            except Exception as e:
                logger.debug(f"Rule application failed: {e}")
                continue

        return None

    def solve(self, test_input, task_data, attempt=1):
        """Main solving function"""
        train_pairs = task_data.get('train', [])

        if not train_pairs:
            return None

        # Detect rules from training examples
        rules = self.detect_rules(train_pairs)

        if not rules:
            return None

        # Apply rules to test input
        result = self.apply_rules(test_input, rules)

        if result is not None:
            confidence = 0.85  # High confidence for learned rules
            return (result, confidence, "rule_induction")

        return None
```

---

## Integration into Ensemble

```python
# In cell5_iteration4_enhanced_ensemble.py

from rule_induction_solver import RuleInductionSolver

def collect_predictions(test_input, task_data):
    predictions = []

    # NEW: Rule induction (try FIRST - most specific)
    rule_solver = RuleInductionSolver()
    result = rule_solver.solve(test_input, task_data)
    if result:
        pred, conf, name = result
        predictions.append(SolverPrediction(pred, conf, name))

    # Object detection
    result = apply_object_detection(test_input, task_data)
    if result:
        predictions.append(SolverPrediction(*result))

    # Pattern matching
    result = apply_pattern_matching(test_input, task_data)
    if result:
        predictions.append(SolverPrediction(*result))

    # [other solvers...]

    return predictions
```

---

## Priority Order (Cascading Layers)

```
1. Rule Induction     (most specific - learned from THIS task)
   ↓
2. Object Detection   (specific - object-level)
   ↓
3. Pattern Matching   (medium - geometric/color)
   ↓
4. Symmetry          (general - spatial)
   ↓
5. Identity          (fallback)
```

**Why rule induction first?**
- Rules are learned from THIS SPECIFIC task's training examples
- Most likely to be correct for this task
- Higher specificity = higher priority

---

## Test Cases

### Test 1: Color Mapping Rule
```python
def test_color_mapping_rule():
    task_data = {
        'train': [
            {
                'input': [[1,1,2], [1,2,2]],
                'output': [[3,3,4], [3,4,4]]
            },
            {
                'input': [[2,1,1], [2,2,1]],
                'output': [[4,3,3], [4,4,3]]
            }
        ]
    }

    solver = RuleInductionSolver()
    rules = solver.detect_rules(task_data['train'])

    assert rules is not None
    assert rules['type'] == 'color_mapping'
    assert rules['mapping'] == {1: 3, 2: 4}

    # Apply to test
    test_input = [[1,2,1], [2,2,2]]
    result = solver.apply_rules(test_input, [rules])

    expected = [[3,4,3], [4,4,4]]
    assert result == expected
```

### Test 2: Size Scaling Rule
```python
def test_size_scaling_rule():
    task_data = {
        'train': [
            {
                'input': [[1,2]],
                'output': [[1,2], [1,2]]  # 2× height
            },
            {
                'input': [[3,4]],
                'output': [[3,4], [3,4]]  # 2× height
            }
        ]
    }

    solver = RuleInductionSolver()
    rules = solver.detect_rules(task_data['train'])

    assert rules['type'] == 'size_scale'
    assert rules['h_ratio'] == 2.0
    assert rules['w_ratio'] == 1.0

    test_input = [[5,6]]
    result = solver.apply_rules(test_input, [rules])

    expected = [[5,6], [5,6]]
    assert result == expected
```

### Test 3: Rare Color Isolation
```python
def test_rare_color_isolation():
    task_data = {
        'train': [
            {
                'input': [[0,0,0,1], [0,0,0,0], [0,0,0,0]],
                'output': [[0,0,0,1], [0,0,0,0], [0,0,0,0]]
                # Only color 1 (rare) is kept
            }
        ]
    }

    solver = RuleInductionSolver()
    rules = solver.detect_rules(task_data['train'])

    # [test rule detection and application]
```

---

## Expected Performance

**Coverage**: 15-25% of tasks
- Color mapping: 10-15%
- Conditional rules: 5-8%
- Size rules: 3-5%
- Counting rules: 2-4%
- **Some overlap** with other solvers

**Accuracy**: 70-85% (when rules are detected)
- High confidence because learned from task's own examples
- Rules validated across multiple training pairs

**Total Contribution**: 15-25% × 70-85% = **10-20% overall accuracy**

**Combined with Object Detection Fix**: 10-20% (rules) + 10-15% (objects) = **20-35% total**

---

## Implementation Priority

**Phase 1** (2 hours):
- Color mapping rules (highest coverage)
- Size scaling rules (common)

**Phase 2** (2 hours):
- Conditional rules (rare color, majority)
- Counting rules (object count → size)

**Phase 3** (1 hour):
- Integration into ensemble
- Testing on training data
- Validation

**Total**: 5 hours for 10-20% gain

---

## Success Criteria

- [ ] Detects color mapping in >80% of applicable tasks
- [ ] Detects size rules in >60% of applicable tasks
- [ ] Accuracy >70% when rules detected
- [ ] Overall contribution >10%
- [ ] Integrates cleanly with existing solvers
- [ ] Tests pass
- [ ] Validated on training data

---

## Next Steps

1. **Implement color mapping** (most common, highest ROI)
2. **Test on training data** (measure actual coverage)
3. **Add to ensemble** (as highest priority layer)
4. **Validate improvement** (should see +10-15% gain)
5. **Iterate** (add more rule types based on data)

---

**Status**: Documented, ready for implementation
**Priority**: HIGH (after object detection fix)
**Expected Impact**: +10-20% accuracy
**Integration**: New layer above object detection

---

**Rule induction is a GAME CHANGER for ARC** - it learns from the task's own examples instead of relying on pre-coded patterns. This is exactly how humans solve ARC tasks! 🧠
