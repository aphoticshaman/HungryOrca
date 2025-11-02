# Iteration 3: Ensemble Methods & Voting

**Status**: 🚧 In Development
**Target Accuracy**: 40-50%
**Expected Improvement**: +15-20% over Iteration 2
**Estimated Development Time**: 3-4 hours
**Lines of Code**: ~600 lines

## Overview

Iteration 3 implements ensemble methods that combine multiple solving strategies through confidence-weighted voting. Instead of cascading (Iteration 2's approach), we generate multiple predictions from different solvers and vote on the best answer.

## Core Principle: Wisdom of Crowds

The insight: **No single solver is best for all tasks, but agreement between diverse solvers indicates correctness.**

Key innovation: Each solver returns both a prediction AND a confidence score. We use weighted voting to combine predictions.

## Architecture

```python
# Solver types (from most to least specific)
1. Object transformation solver (Iteration 2)
2. Pattern matching solver (Iteration 1)
3. Grid arithmetic solver (NEW)
4. Symmetry completion solver (NEW)
5. Color frequency solver (NEW)

# Voting mechanism
def ensemble_solver(test_input, task_data, attempt=1):
    predictions = []

    # Generate predictions from all solvers
    for solver in ALL_SOLVERS:
        try:
            pred, confidence = solver(test_input, task_data)
            predictions.append((pred, confidence, solver.name))
        except:
            continue

    # Weighted voting
    return vote_on_predictions(predictions, attempt)
```

## New Solvers (Beyond Iterations 1-2)

### 3. Grid Arithmetic Solver

Detects when output = f(input) where f is arithmetic operation:
- Addition of constant to all cells
- Multiplication by constant
- Modulo operation
- Bitwise operations

```python
def detect_grid_arithmetic(task_data) -> Optional[Tuple[str, Callable]]:
    """
    Analyze training pairs to find arithmetic pattern.

    Returns:
        ("add_5", lambda grid: grid + 5) if all cells increase by 5
        ("multiply_2", lambda grid: grid * 2) if all cells double
        ("mod_3", lambda grid: grid % 3) if modulo pattern
        None if no arithmetic pattern found
    """
    # Test multiple arithmetic operations
    # Return operation with 100% consistency across training pairs
```

**Expected coverage**: 5-8% of tasks

### 4. Symmetry Completion Solver

Detects incomplete symmetric patterns and completes them:
- Horizontal symmetry (left/right mirror)
- Vertical symmetry (top/bottom mirror)
- Diagonal symmetry
- Rotational symmetry (180°, 90°)

```python
def detect_incomplete_symmetry(grid) -> Optional[str]:
    """
    Check if grid has partial symmetry that should be completed.

    Returns:
        "h_symmetry" if horizontally symmetric (complete the reflection)
        "v_symmetry" if vertically symmetric
        "d_symmetry" if diagonally symmetric
        "r180_symmetry" if 180° rotationally symmetric
        None if fully symmetric or no symmetry pattern
    """
    # Measure symmetry scores for each type
    # If score is 60-95%, it's incomplete symmetry
    # Return symmetry type with highest score
```

**Expected coverage**: 3-5% of tasks

### 5. Color Frequency Solver

Detects patterns in color frequency distribution:
- Most common color becomes background
- Rare colors get promoted
- Frequency-based filtering

```python
def detect_color_frequency_pattern(task_data) -> Optional[Callable]:
    """
    Analyze if transformation is based on color frequency.

    Examples:
        - Input: 70% color 0, 20% color 1, 10% color 2
        - Output: Only color 2 remains (rare color promotion)

        - Input: Multiple colors scattered
        - Output: Only most common color (frequency filtering)
    """
    # Count color frequencies across training pairs
    # Detect if rare colors are promoted/filtered
```

**Expected coverage**: 2-4% of tasks

## Voting Mechanism

### Confidence Scoring

Each solver returns confidence 0.0-1.0 based on:

```python
def calculate_confidence(solver_type, task_data) -> float:
    """
    Confidence = pattern_match_rate * consistency_score * solver_specificity

    pattern_match_rate: % of training pairs where pattern detected
    consistency_score: How consistent the pattern is (0.8-1.0)
    solver_specificity:
        - Object transform: 1.0 (very specific)
        - Pattern matching: 0.8 (medium specific)
        - Grid arithmetic: 0.9 (specific)
        - Symmetry: 0.7 (less specific)
        - Color frequency: 0.6 (least specific)
    """
```

### Weighted Voting Algorithm

```python
def vote_on_predictions(predictions: List[Tuple[grid, float, str]],
                        attempt: int) -> List[List[int]]:
    """
    Vote on predictions using weighted confidence.

    Algorithm:
    1. Group identical predictions
    2. Sum confidence scores for each unique prediction
    3. Sort by total confidence (descending)
    4. Return top prediction for attempt 1, second for attempt 2

    Special case: If top confidence > 0.9, use same for both attempts
    """
    # Group by grid equality
    vote_groups = {}
    for pred, conf, name in predictions:
        key = grid_to_tuple(pred)
        if key not in vote_groups:
            vote_groups[key] = {"grid": pred, "total_conf": 0, "solvers": []}
        vote_groups[key]["total_conf"] += conf
        vote_groups[key]["solvers"].append(name)

    # Sort by confidence
    ranked = sorted(vote_groups.values(),
                   key=lambda x: x["total_conf"],
                   reverse=True)

    # Return based on attempt
    if attempt == 1:
        return ranked[0]["grid"] if ranked else test_input
    else:
        # For attempt 2, return second-best (or first if very confident)
        if ranked and ranked[0]["total_conf"] > 0.9:
            return ranked[0]["grid"]
        return ranked[1]["grid"] if len(ranked) > 1 else ranked[0]["grid"]
```

## Integration with Previous Iterations

Iteration 3 **includes** Iterations 1 and 2 as ensemble members:

```python
ENSEMBLE_SOLVERS = [
    # From Iteration 2
    ObjectTransformSolver(),

    # From Iteration 1
    PatternMatchingSolver(),

    # New in Iteration 3
    GridArithmeticSolver(),
    SymmetryCompletionSolver(),
    ColorFrequencySolver(),
]
```

## Expected Performance

### Individual Solver Coverage (Estimated)
- Object transform (Iter 2): 15-20%
- Pattern matching (Iter 1): 10-15%
- Grid arithmetic: 5-8%
- Symmetry completion: 3-5%
- Color frequency: 2-4%

### Ensemble Effect
- **Pure addition**: 35-52% (sum of individual)
- **Expected overlap**: ~5% (some tasks solved by multiple solvers)
- **Voting boost**: +5-10% (voting catches errors, improves confidence)
- **Target**: 40-50%

## Implementation Checklist

### Phase 1: New Solvers (2 hours)
- [ ] Grid arithmetic solver (45 min)
- [ ] Symmetry completion solver (45 min)
- [ ] Color frequency solver (30 min)

### Phase 2: Voting System (1 hour)
- [ ] Confidence scoring for all solvers (30 min)
- [ ] Weighted voting algorithm (30 min)

### Phase 3: Integration & Testing (1 hour)
- [ ] Integrate with Iterations 1-2 (20 min)
- [ ] Test suite for new solvers (20 min)
- [ ] End-to-end ensemble test (20 min)

## Testing Strategy

### Unit Tests
```python
def test_grid_arithmetic():
    # Test: all cells +5
    task = create_task([
        {"input": [[1,2],[3,4]], "output": [[6,7],[8,9]]}
    ])
    pattern, func = detect_grid_arithmetic(task)
    assert pattern == "add_5"

def test_symmetry_completion():
    # Test: half-symmetric grid
    grid = [[1,2,0,0],
            [3,4,0,0]]
    result = complete_symmetry(grid, "h_symmetry")
    expected = [[1,2,2,1],
                [3,4,4,3]]
    assert np.array_equal(result, expected)

def test_voting_mechanism():
    # Test: 3 solvers, 2 agree
    predictions = [
        ([[1,2]], 0.8, "solver_a"),
        ([[1,2]], 0.7, "solver_b"),
        ([[3,4]], 0.6, "solver_c"),
    ]
    result = vote_on_predictions(predictions, attempt=1)
    assert result == [[1,2]]  # Winner: total confidence 1.5 vs 0.6
```

### Integration Test
```python
def test_ensemble_beats_cascade():
    # Test on 50 ARC tasks
    # Measure: ensemble accuracy vs cascade (Iteration 2)
    # Expected: +10-15% improvement
```

## Success Criteria

### Minimum (Gate for Ratcheting)
- [ ] 40%+ accuracy on local test set (50 tasks)
- [ ] Outperforms Iteration 2 by +10%
- [ ] All 5 solvers contribute (no dead code)

### Target
- [ ] 45%+ accuracy on local test set
- [ ] At least 3 solvers agree on 20%+ of tasks (voting working)
- [ ] Less than 5% overlap between solvers (good diversity)

### Stretch
- [ ] 50%+ accuracy
- [ ] Voting mechanism catches and corrects errors (evidence in logs)

## Lessons from Iterations 1-2 Applied

1. **Documentation-as-Specification**: This document written FIRST, code follows mechanically
2. **Production Constraints**: No scipy, only numpy (grid_to_tuple uses native Python)
3. **Cascading Fallbacks**: If voting fails, fall back to Iteration 2 cascade
4. **Token Efficiency**: Modular design, only Cell 5 changes

## Risks & Mitigations

### Risk 1: Solvers too similar (high overlap)
**Mitigation**: Each solver targets different task type (arithmetic vs symmetry vs objects)

### Risk 2: Voting dilutes confidence
**Mitigation**: Only vote if 2+ solvers produce predictions; otherwise cascade

### Risk 3: Performance overhead
**Mitigation**: Short-circuit on first high-confidence (>0.95) prediction

## Next Steps After Completion

1. **Test locally** on 50-task subset
2. **Measure overlap** between solvers (want <5%)
3. **Analyze failures** to identify Iteration 4 target
4. **Deploy to Kaggle** if validated
5. **Extract meta-insights** from ensemble approach

## Integration Instructions

Add to `subtlegeniusv1.ipynb` Cell 5:

```python
# Replace cell5_iteration2_objects import with:
from cell5_iteration3_ensemble import ensemble_solver

# Update submission generator to use:
prediction = ensemble_solver(test_input, task_data, attempt=attempt_num)
```

---

**Development Start**: 2025-11-02
**Expected Completion**: 2025-11-02 (3-4 hours from start)
**Next Iteration**: Iteration 4 (Meta-Cognition & Task Classification)
