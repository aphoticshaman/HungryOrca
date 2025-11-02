# Living and Evolving Lessons Learned Doc (LAELD)
## HungryOrca / SubtleGenius Development
**Last Updated**: 2025-11-02
**Status**: Active - Update after every iteration

---

# 🎯 5x What We Did GOOD (Keep Pushing!)

## 1. ✅ Documentation-First Development (Novel Insight #11)

**What we did:**
- Wrote `ITERATION_2_OBJECTS.md` BEFORE coding
- Included test cases in documentation
- Defined success criteria up-front
- Integration instructions pre-written

**Evidence of success:**
- Iteration 2: 2.8 hours (30% faster than Iteration 1's 4 hours)
- Code passed tests on first run
- No ambiguity, no rework loops

**Why it worked:**
- Spec removed uncertainty
- Tests validated spec itself
- Implementation = mechanical translation, not exploration

**Where to push harder:**
```markdown
BEFORE every iteration:
1. Write ITERATION_N_*.md with:
   - Objective (1 clear sentence)
   - Components (precise list)
   - Test cases (with expected outputs)
   - Integration steps (copy-paste ready)
   - Success criteria (boolean checks)
2. Write tests BEFORE implementation
3. THEN code to satisfy spec
```

**Metric to track:** Time from start to completion (should decrease each iteration)

---

## 2. ✅ Production-First Architecture (Insight #6)

**What we did:**
- Built validator BEFORE solver (Cell 2 before Cell 5)
- 4-tier cascading fallbacks (object → pattern → identity → [[0]])
- Never-crash design (try-except at every level)
- Comprehensive error handling

**Evidence of success:**
- 100% completion rate (0 crashes in Kaggle run)
- Valid submission on first try
- 0.85 seconds for 240 tasks (blazing fast)
- All validation checks passed

**Why it worked:**
- Defensive programming prevented exceptions
- Fallbacks guaranteed completion
- Validator caught format errors before submission

**Where to push harder:**
```python
# Apply to EVERY new solver:
def new_solver_pattern(input, task_data):
    try:
        # Primary logic
        result = sophisticated_approach(input, task_data)

        # Validate result
        if not is_valid_grid(result):
            raise ValueError("Invalid grid")

        return result

    except Exception as e:
        logger.warning(f"Solver failed: {e}, using fallback")
        try:
            return simple_fallback(input, task_data)
        except:
            return input  # Ultimate fallback
```

**Metric to track:** Completion rate (should always be 100%)

---

## 3. ✅ Modular Cell Architecture (Insight #5)

**What we did:**
- 6-cell design: Config, Validator, Fallbacks, Generator, **Solver**, Execution
- Edit Cell 5 (solver) without touching Cells 1-4, 6
- Extract iterations to separate .py files
- Compile back when needed

**Evidence of success:**
- Iteration 1: 350 lines added (not 1,350 regenerated)
- Iteration 2: 490 lines added (not 1,840 regenerated)
- Iteration 3: 620 lines added (not 2,460 regenerated)
- **Token savings**: 72% reduction (2,810 vs ~10,000 monolithic)

**Why it worked:**
- Edit only what changes
- Infrastructure stable and reusable
- 3.5× more iteration cycles possible

**Where to push harder:**
```python
# Each iteration should be:
# - In its own file: cell5_iterationN_feature.py
# - Import-able: from cell5_iterationN import solver_function
# - Tested separately: test_iterationN.py
# - Documented separately: ITERATION_N_FEATURE.md
# - Integrated via simple import change in Cell 5

# Cell 5 should look like:
from cell5_iteration3_ensemble import ensemble_solver as main_solver

def solve(test_input, task_data, attempt):
    return main_solver(test_input, task_data, attempt)
```

**Metric to track:** Lines edited per iteration (should stay <800 lines)

---

## 4. ✅ Cascading Layer Architecture (Novel Insight #9)

**What we did:**
- Organized solvers as LAYERS not competitors
- Specificity ordering: object (most) → pattern (medium) → identity (general)
- Each layer PRESERVES previous layers
- Coverage is ADDITIVE (15% + 10% = 25%)

**Evidence of success:**
- Iteration 2 improved without modifying Iteration 1 code
- Pure addition, no replacement
- Each layer handles different task types independently

**Why it worked:**
- Layers don't conflict (independence)
- Higher specificity = higher priority (efficiency)
- New iterations build on, don't replace (ratcheting)

**Where to push harder:**
```python
# TEMPLATE for every new iteration:
def iterationN_solver(test_input, task_data, attempt):
    # NEW LAYER (highest priority)
    if detect_iterationN_pattern(task_data):
        return apply_iterationN_transform(test_input, task_data)

    # PRESERVE Iteration N-1 (fall through)
    return iterationN_minus_1_solver(test_input, task_data, attempt)

# NEVER:
# - Replace previous solvers
# - Merge into one function
# - Make them compete via voting at this level

# ALWAYS:
# - Add new layer on top
# - Preserve all previous layers
# - Measure ADDITIVE coverage
```

**Metric to track:** Coverage addition per iteration (should be +5-15%)

---

## 5. ✅ Real Validation on Kaggle (Not Just Theory)

**What we did:**
- Built UberOrcaSubtleGenius_v3.ipynb (single file)
- Uploaded to Kaggle and RAN IT
- Generated actual submission.json (1.7MB)
- Got REAL performance data (0.85s, log.txt, solver triggers)

**Evidence of success:**
- Discovered object detection is broken (100% detection, 0% solving)
- Confirmed pattern rotate_180 works perfectly (task 3c9b0459)
- Found symmetry runs on 35% of tasks
- Measured 41.3% attempt diversity
- **Stopped building in a vacuum**

**Why it worked:**
- Reality > theory
- Real data reveals real problems
- Can't improve what you don't measure

**Where to push harder:**
```markdown
VALIDATION CHECKLIST (before every iteration):

□ Test on training data FIRST
  - Know ground truth
  - Measure accuracy on known answers
  - Debug before Kaggle

□ Run full pipeline locally
  - All 240 tasks (or subset)
  - Measure time, memory, errors
  - Validate submission.json format

□ Log everything
  - Which solvers trigger
  - Which patterns match
  - Success/failure rates

□ Submit to Kaggle
  - Get real score
  - Compare to prediction
  - Iterate based on ACTUAL feedback

NEVER submit to Kaggle without local validation first!
```

**Metric to track:** Prediction accuracy (predicted score vs actual score, should converge)

---

# 🔄 5x Where We Can Apply Good to Bad Spots (Refactor NOW)

## 1. 🔧 Apply Production-First to Object Detection

**Bad spot:**
```python
# Current (BROKEN - returns input unchanged):
def apply_object_detection(test_input, task_data):
    try:
        if detect_object_pattern(task_data):
            # BUG: Just flags detection, doesn't transform!
            pred = test_input  # Returns input unchanged
            return (pred, 0.60, "object_detected")
    except ImportError:
        return None
```

**Good approach to apply:** Production-first with actual solving

**Refactor:**
```python
def apply_object_detection(test_input, task_data):
    try:
        # Detect pattern
        pattern = detect_object_transformation_pattern(task_data)
        if pattern is None:
            return None

        # ACTUALLY APPLY TRANSFORMATION (not just detect!)
        try:
            transformed = apply_object_transformation(test_input, task_data, pattern)

            # Validate result
            is_valid, _ = SubmissionValidator.validate_grid(transformed)
            if not is_valid:
                return None

            return (transformed, 0.90, f"object_{pattern['type']}")

        except Exception as e:
            logger.warning(f"Object transformation failed: {e}")
            return None  # Fall through to next solver

    except Exception as e:
        logger.warning(f"Object detection failed: {e}")
        return None
```

**ROI:** +10-15% accuracy (100% detection → 50-70% solving)

---

## 2. 🔧 Apply Documentation-First to Pattern Matching Expansion

**Bad spot:**
- Only 1/240 tasks matched (rotate_180)
- Missing obvious patterns (crop, tile, transpose)
- Built ad-hoc without systematic planning

**Good approach to apply:** Write spec first, then code

**Refactor process:**
```markdown
1. BEFORE coding new patterns, write:

# PATTERN_EXPANSION.md

## Objective
Add 10 common ARC patterns based on training data analysis

## Patterns to Add (Priority Order)
1. Crop to bounding box (observed in 15% of training tasks)
2. Tile/repeat pattern (observed in 12%)
3. Extract unique colors (observed in 8%)
4. Transpose (observed in 6%)
5. Diagonal flip (observed in 5%)
6. Color majority (observed in 4%)
7. Grid resize proportional (observed in 3%)
8. Extract largest object (observed in 3%)
9. Color filtering (observed in 2%)
10. Border addition/removal (observed in 2%)

## Test Cases (Pre-written)
### Test 1: Crop to bounding box
Input: [[0,0,0],[0,1,0],[0,0,0]]
Expected: [[1]]

### Test 2: Tile 2x2
Input: [[1,2],[3,4]]
Expected: [[1,2,1,2],[3,4,3,4],[1,2,1,2],[3,4,3,4]]

[etc for all 10 patterns]

## Success Criteria
- All 10 patterns implemented
- All tests pass
- Coverage increases by +10-15%
- Integration takes <30 min

2. THEN implement to spec
3. THEN test against pre-written cases
4. THEN deploy
```

**ROI:** +10-15% accuracy (systematic vs ad-hoc)

---

## 3. 🔧 Apply Modular Architecture to Test Suites

**Bad spot:**
- Tests don't run locally (ModuleNotFoundError: numpy)
- Can't validate before Kaggle
- Tests only work in Kaggle environment

**Good approach to apply:** Modular, environment-aware testing

**Refactor:**
```python
# tests/test_helpers.py
import sys
import os

def setup_test_environment():
    """Auto-detect and setup test environment"""
    try:
        import numpy
        NUMPY_AVAILABLE = True
    except ImportError:
        NUMPY_AVAILABLE = False

    if not NUMPY_AVAILABLE:
        print("⚠️  Numpy not available - using mock implementations")
        # Provide minimal mocks for testing
        sys.modules['numpy'] = MockNumpy()

    return NUMPY_AVAILABLE

# tests/test_pattern_solver.py
from test_helpers import setup_test_environment

NUMPY_AVAILABLE = setup_test_environment()

def test_rotate_90():
    if not NUMPY_AVAILABLE:
        pytest.skip("Numpy not available")

    # Test logic here
    pass

# OR: Pure Python test data (no numpy needed)
def test_rotate_90_pure_python():
    input_grid = [[1,2],[3,4]]
    expected = [[3,1],[4,2]]

    result = rotate_90_cw(input_grid)
    assert result == expected  # No numpy needed!
```

**ROI:** Can validate locally before Kaggle (faster iteration)

---

## 4. 🔧 Apply Constraint-Driven Design to Ensemble Voting

**Bad spot:**
- Voting happens even when only 1 solver matches
- No minimum confidence threshold
- Complexity without clear benefit

**Good approach to apply:** Let constraints simplify

**Refactor:**
```python
# Add constraints to simplify voting logic:

def vote_on_predictions(predictions, attempt, test_input):
    """Constraint-driven voting logic"""

    # Constraint 1: Need at least 2 predictions to vote
    if len(predictions) < 2:
        if len(predictions) == 1:
            return predictions[0].grid  # Single solver wins
        return test_input  # No solvers → identity

    # Constraint 2: High confidence (>0.95) wins immediately
    for pred in predictions:
        if pred.confidence > 0.95:
            return pred.grid  # Skip voting, just use it

    # Constraint 3: Only vote if there's actual disagreement
    if all(grids_equal(predictions[0].grid, p.grid) for p in predictions):
        return predictions[0].grid  # All agree, skip voting

    # NOW do voting (only when needed)
    # [voting logic here]
```

**ROI:** Simpler code, faster execution, same accuracy

---

## 5. 🔧 Apply Validation to Training Data BEFORE Kaggle

**Bad spot:**
- Never tested on training data
- Don't know which solvers actually work
- Flying blind into Kaggle submissions

**Good approach to apply:** Validate locally first

**Refactor approach:**
```python
# NEW: validation_harness.py

def validate_on_training_data(solver_func, training_data_path):
    """Validate solver on training data before Kaggle"""

    # Load training data (has ground truth)
    with open(training_data_path, 'r') as f:
        training_data = json.load(f)

    results = {
        'total_tasks': 0,
        'solvers_triggered': defaultdict(int),
        'solvers_correct': defaultdict(int),
        'coverage': {},
        'accuracy': {}
    }

    # Test each training task
    for task_id, task_data in training_data.items():
        results['total_tasks'] += 1

        for test_idx, test_pair in enumerate(task_data['test']):
            test_input = test_pair['input']
            expected_output = test_pair['output']

            # Run solver
            prediction = solver_func(test_input, task_data, attempt=1)

            # Check if correct
            correct = grids_equal(prediction, expected_output)

            # Track which solver triggered (from logs)
            # Track if it was correct

    # Calculate metrics
    for solver_name in results['solvers_triggered'].keys():
        results['coverage'][solver_name] = (
            results['solvers_triggered'][solver_name] / results['total_tasks']
        )
        results['accuracy'][solver_name] = (
            results['solvers_correct'][solver_name] /
            max(1, results['solvers_triggered'][solver_name])
        )

    return results

# USAGE:
results = validate_on_training_data(
    ensemble_solver,
    'data/arc-agi_training_challenges.json'
)

print(f"Pattern matching: {results['coverage']['pattern']:.1%} coverage, "
      f"{results['accuracy']['pattern']:.1%} accuracy")
print(f"Object detection: {results['coverage']['object']:.1%} coverage, "
      f"{results['accuracy']['object']:.1%} accuracy")

# THEN decide what to build next based on REAL data
```

**ROI:** Build based on data, not guesses (10× better decisions)

---

# ❌ 5x Where We Went Off Track (Learn & Avoid)

## 1. ❌ Built Object Detection That Doesn't Solve

**What went wrong:**
```python
# Iteration 2: detect_object_pattern() works
# BUT: apply_object_transformation() just returns input!
def apply_object_detection(test_input, task_data):
    if detect_object_pattern(task_data):
        # BUG: Returns input unchanged
        return (test_input, 0.60, "object_detected")
```

**Why it's bad:**
- 100% detection coverage wasted
- Spent time building detection, forgot transformation
- Would have caught this with validation on training data

**Root cause:**
- Built detection and transformation separately
- Never tested end-to-end before Kaggle
- Assumed "detect" = "solve"

**Lesson:**
```markdown
DETECTION ≠ SOLVING

Always build and test BOTH together:
1. detect_pattern() → returns pattern details
2. apply_transformation() → uses pattern to transform
3. Test on training data → verify correct output
4. THEN deploy

Never separate detection from transformation!
```

**How to avoid:**
- Write test cases with EXPECTED OUTPUT first
- Test detect + apply together, not separately
- Validate on training data before Kaggle

---

## 2. ❌ Built Grid Arithmetic Solver (0% Coverage)

**What went wrong:**
- Added addition, multiplication, modulo solvers
- 0 tasks matched out of 240
- Wasted development time
- Added code complexity for no benefit

**Why it's bad:**
- ARC is about visual/spatial reasoning, not arithmetic
- Should have analyzed training data FIRST
- Built based on intuition, not data

**Root cause:**
- Didn't validate hypothesis before coding
- Assumed ARC would have arithmetic patterns
- No data-driven prioritization

**Lesson:**
```markdown
VALIDATE BEFORE BUILDING

Before adding any solver:
1. Analyze training data → Does this pattern exist?
2. Estimate coverage → How many tasks would match?
3. If coverage < 3%, DON'T BUILD IT
4. Focus on high-coverage patterns first

Build based on DATA, not intuition!
```

**How to avoid:**
- Analyze training data for pattern frequency
- Priority queue: coverage × expected_accuracy
- Only build if coverage > 5%

---

## 3. ❌ Pattern Matching Too Narrow (1/240 Tasks)

**What went wrong:**
- Only implemented 7 patterns (rotate, flip, color map)
- Missed obvious patterns (crop, tile, transpose)
- Ad-hoc selection without systematic analysis

**Why it's bad:**
- 99.6% of tasks not covered
- Low ROI on development time
- Missing low-hanging fruit

**Root cause:**
- Didn't analyze training data for common patterns
- Built what seemed intuitive
- No systematic prioritization

**Lesson:**
```markdown
ANALYZE FIRST, BUILD SECOND

Pattern development process:
1. Study 50-100 training tasks manually
2. List patterns that appear (with frequency)
3. Sort by frequency × simplicity
4. Build top 10-15 patterns
5. Validate coverage on training data
6. THEN deploy

Don't guess what patterns exist - MEASURE IT!
```

**How to avoid:**
- Spend 2 hours analyzing training data
- Create frequency table of patterns
- Build most common patterns first

---

## 4. ❌ Never Validated on Training Data Before Kaggle

**What went wrong:**
- Built 3 iterations (1,700 lines of solver code)
- Never tested on tasks with known answers
- First validation was live Kaggle submission
- Discovered critical bugs AFTER submission

**Why it's bad:**
- Wasted daily submission discovering bugs
- Could have found issues in 30 min locally
- Built in a vacuum without feedback

**Root cause:**
- Eager to submit and see score
- Skipped local validation step
- No validation harness built

**Lesson:**
```markdown
LOCAL VALIDATION IS MANDATORY

NEVER submit to Kaggle without:
1. Testing on training data (has ground truth)
2. Measuring actual accuracy (not guesses)
3. Confirming solvers work as expected
4. Validating submission format

Training data → Know what works
Kaggle → Confirm it generalizes

ALWAYS validate locally FIRST!
```

**How to avoid:**
- Build validation_harness.py (see Refactor #5)
- Make it part of development workflow
- No Kaggle submission without local validation

---

## 5. ❌ Symmetry Completion Without Accuracy Verification

**What went wrong:**
- Built symmetry completion solver
- Detected 35% of tasks (75-90 tasks)
- Assumed completions are correct
- DON'T KNOW if they actually solve tasks

**Why it's bad:**
- Might be 35% false positives (wasted coverage)
- Or might be 35% correct solutions (huge win)
- Can't improve what we can't measure

**Root cause:**
- Didn't test against ground truth
- Assumed detection = correct solving
- No accuracy measurement

**Lesson:**
```markdown
MEASURE ACCURACY, NOT JUST COVERAGE

For every solver:
1. Coverage = % of tasks it triggers on
2. Accuracy = % correct when it triggers
3. Contribution = Coverage × Accuracy

Example:
- Symmetry: 35% coverage, unknown accuracy
- Could be: 35% × 80% = 28% contribution (GREAT!)
- Or: 35% × 20% = 7% contribution (meh)

ALWAYS measure BOTH coverage AND accuracy!
```

**How to avoid:**
- Test on training data with ground truth
- Report: "Pattern X: 15% coverage, 73% accuracy = 11% contribution"
- Track accuracy, not just detection

---

# 🛡️ 5x Ways to Avoid Bad Work (Proactive Planning)

## 1. 🛡️ RULE: Write Validation Harness FIRST (Before Iteration 1)

**Proactive plan:**
```markdown
SESSION 0: VALIDATION INFRASTRUCTURE (1-2 hours)

Before building ANY solvers:

□ Build validation_harness.py
  - Load training data
  - Run solver on all training tasks
  - Compare to ground truth
  - Report coverage & accuracy per solver

□ Build test data fixtures
  - 10-task subset for quick testing
  - Known difficult tasks
  - Known easy tasks

□ Build submission validator
  - Check format
  - Check all tasks covered
  - Validate grid structure

ONLY AFTER validation is ready → Start building solvers

This prevents:
- Building in a vacuum
- Discovering bugs in Kaggle
- Guessing accuracy vs measuring it
```

**When to apply:** Before ANY development starts

**Expected ROI:** Catch bugs in minutes (not after Kaggle submission)

---

## 2. 🛡️ RULE: Analyze Training Data Before Building Patterns

**Proactive plan:**
```markdown
SESSION 0.5: PATTERN FREQUENCY ANALYSIS (2-3 hours)

Before coding pattern matchers:

□ Manually study 50-100 training tasks
  - What transformations appear?
  - How often does each appear?
  - How difficult to implement?

□ Create pattern frequency table:
  Pattern | Frequency | Difficulty | Priority
  Crop to bbox | 18% | Easy | 1
  Tile 2x | 15% | Easy | 2
  Extract color | 12% | Easy | 3
  Rotate 90 | 8% | Easy | 4
  [etc]

□ Sort by: Frequency × (1/Difficulty)

□ Build top 10-15 patterns ONLY

This prevents:
- Building patterns that don't exist (grid arithmetic)
- Missing obvious patterns (crop, tile)
- Ad-hoc pattern selection
```

**When to apply:** Before Iteration 1 (pattern matching)

**Expected ROI:** Build patterns that actually exist (2× higher coverage)

---

## 3. 🛡️ RULE: Test Detection + Transformation Together

**Proactive plan:**
```markdown
FOR EVERY NEW SOLVER TYPE:

□ BEFORE coding implementation:
  1. Write test case with EXPECTED OUTPUT

  Example:
  def test_object_color_change():
      task_data = {...}  # Training example
      test_input = [[1,1],[1,1]]
      expected_output = [[2,2],[2,2]]  # Known answer

      # Test BOTH detect AND apply
      pattern = detect_object_pattern(task_data)
      assert pattern is not None

      result = apply_object_transformation(test_input, task_data, pattern)
      assert result == expected_output  # MUST MATCH!

□ DURING coding:
  - Implement detect_pattern()
  - Implement apply_transformation()
  - Test TOGETHER on training data
  - VERIFY correct output

□ NEVER separate detect from apply!

This prevents:
- Detection without transformation (object detection bug)
- Assuming detection = solving
- Finding bugs in Kaggle instead of locally
```

**When to apply:** Every new solver type

**Expected ROI:** Catch bugs immediately (not after submission)

---

## 4. 🛡️ RULE: Document Expected Behavior BEFORE Coding

**Proactive plan:**
```markdown
FOR EVERY ITERATION:

BEFORE writing ANY code:

□ Create ITERATION_N_FEATURE.md with:
  1. Objective (1 sentence)
  2. Expected behavior (detailed)
  3. Test cases (with expected outputs)
  4. Success criteria (boolean checks)
  5. Integration steps (copy-paste ready)

□ Write test_iterationN.py with:
  - All test cases from docs
  - Expected outputs defined
  - Test detect + apply together

□ Get approval/feedback on spec

□ THEN implement to match spec

□ Verify against pre-written tests

This prevents:
- Ambiguity leading to rework
- Building wrong thing
- Missing edge cases
- Integration headaches
```

**When to apply:** Start of every iteration

**Expected ROI:** 30% faster iterations (see Iteration 2 vs 1)

---

## 5. 🛡️ RULE: Coverage + Accuracy Reporting Built-In

**Proactive plan:**
```markdown
ADD TO EVERY SOLVER:

Built-in performance tracking:

class SolverStats:
    def __init__(self):
        self.triggers = 0
        self.attempts = 0
        self.successes = 0  # Only if we have ground truth

    def record_trigger(self, task_id):
        self.triggers += 1

    def record_attempt(self, task_id, correct=None):
        self.attempts += 1
        if correct is not None:
            self.successes += correct

    def report(self):
        coverage = self.triggers / total_tasks
        accuracy = self.successes / max(1, self.attempts)
        contribution = coverage * accuracy

        print(f"Coverage: {coverage:.1%}")
        print(f"Accuracy: {accuracy:.1%}")
        print(f"Contribution: {contribution:.1%}")

# USE IT:
pattern_stats = SolverStats()
object_stats = SolverStats()

# During solving:
if pattern_detected:
    pattern_stats.record_trigger(task_id)
    result = apply_pattern()
    correct = (result == ground_truth) if has_ground_truth else None
    pattern_stats.record_attempt(task_id, correct)

# After run:
pattern_stats.report()
object_stats.report()

This prevents:
- Unknown solver accuracy (symmetry issue)
- Can't identify underperforming solvers
- No data for prioritizing improvements
```

**When to apply:** Add to base infrastructure (Cell 1)

**Expected ROI:** Always know which solvers work (data-driven iteration)

---

# 📋 ACTIONABLE CHECKLIST: Apply Before Next Iteration

## Pre-Development (2-3 hours)
- [ ] Build validation harness (test on training data)
- [ ] Analyze training data for pattern frequency
- [ ] Create pattern priority list (frequency × simplicity)
- [ ] Write ITERATION_N.md spec (before coding)
- [ ] Write tests with expected outputs

## During Development (4-6 hours)
- [ ] Implement to spec (detection + transformation together)
- [ ] Test against pre-written tests
- [ ] Validate on training data (coverage + accuracy)
- [ ] Apply production-first (error handling, fallbacks)
- [ ] Log everything (solver triggers, success rates)

## Post-Development (1 hour)
- [ ] Run full validation harness
- [ ] Generate coverage + accuracy report
- [ ] Compare to predictions
- [ ] Update ITERATION_LOG.md
- [ ] Submit to Kaggle ONLY if local validation passes

## Continuous
- [ ] Update LAELD after every iteration
- [ ] Track metrics (time, coverage, accuracy)
- [ ] Identify new patterns (good & bad)
- [ ] Refine process

---

# 🎯 SPECIFIC ACTIONS FOR NEXT ITERATION (Iteration 4)

## MUST DO (Prevents Past Mistakes)

### 1. Build Validation Harness FIRST
**Time**: 1 hour
**File**: `validation_harness.py`
**Purpose**: Test on training data with ground truth

### 2. Analyze Training Data for Object Patterns
**Time**: 1 hour
**Purpose**: Know which object transformations actually exist
**Output**: Frequency table (color change: 12%, movement: 8%, etc.)

### 3. Write ITERATION_4_OBJECTS_FIXED.md FIRST
**Time**: 30 minutes
**Include**:
- Objective: Fix object detection to actually solve
- Patterns to implement: [from frequency analysis]
- Test cases: [with expected outputs]
- Success: >50% accuracy on object tasks

### 4. Implement Detection + Transformation TOGETHER
**Time**: 2-3 hours
**Test continuously on training data**

### 5. Validate Before Kaggle
**Time**: 30 minutes
**Measure**: Coverage + Accuracy on training data
**Submit ONLY if: Accuracy >60% on triggered tasks**

---

# 📊 METRICS TO TRACK (Living Document)

## Per Iteration

| Iteration | Time (hrs) | Coverage (%) | Accuracy (%) | Contribution (%) | Prediction Error |
|-----------|-----------|--------------|--------------|------------------|------------------|
| 0 (Baseline) | 6 | 0 | 0 | 0 | N/A |
| 1 (Patterns) | 4 | 0.4 | TBD | TBD | TBD |
| 2 (Objects) | 2.8 | 100 | 0 | 0 | HUGE (thought 20-30%) |
| 3 (Ensemble) | 3 | TBD | TBD | TBD | TBD |
| 4 (Objects Fixed) | ? | ? | >60 target | >10 target | ? |

**Update after each Kaggle submission!**

## Process Improvement

| Metric | Iteration 1 | Iteration 2 | Iteration 3 | Iteration 4 | Target |
|--------|------------|------------|------------|------------|--------|
| Spec before code | No | Yes | Yes | MUST | Always |
| Local validation | No | No | No | MUST | Always |
| Training data analysis | No | No | No | MUST | Always |
| Prediction error | N/A | N/A | TBD | <20% | <10% |
| Rework loops | 1 | 1 | 0 | 0 | 0 |

---

# 🔄 LAELD UPDATE PROCESS

**After every iteration:**

1. **What went well?** (Add to "Good" section if new pattern)
2. **What went poorly?** (Add to "Off Track" section)
3. **What should we refactor?** (Add to "Apply Good to Bad" section)
4. **How do we prevent this?** (Add to "Avoid Bad Work" section)
5. **Update metrics** (Track progress)

**This document should GROW with every iteration!**

---

**Status**: Living document - update after Iteration 4
**Next Review**: After next Kaggle submission
**Owner**: Ryan + Claude collaborative

---

# 🚀 ONE RULE TO RULE THEM ALL

```
VALIDATE ON TRAINING DATA BEFORE KAGGLE

Everything else follows from this.
```

If we had done this from the start:
- Would have caught object detection bug immediately
- Would have known grid arithmetic has 0 coverage
- Would have measured symmetry accuracy
- Would have known pattern matching too narrow

**This one rule prevents 80% of our mistakes.**

**Make it non-negotiable.** ✅
