# SubtleGenius Integration Guide
## How to Add Iterations to Main Notebook

**Quick Reference**: Update Cell 5 without touching infrastructure cells

---

## 🔄 Integration Workflow

### Step 1: Develop Iteration in Separate File

```bash
# Create new iteration file
SubtleGenius/notebooks/cell5_iterationN_feature.py

# Develop and test independently
python tests/test_feature.py
```

### Step 2: Validate Locally

```python
# Import new solver
from cell5_iterationN_feature import enhanced_solver

# Test with small subset
small_test = dict(list(test_data.items())[:10])
gen = SubmissionGenerator(solver_func=enhanced_solver)
submission = gen.generate_submission(small_test)

# Validate
is_valid, msg = SubmissionValidator.validate_submission(submission, small_test)
```

### Step 3: Check for Improvement

```python
# Run baseline
baseline_gen = SubmissionGenerator(solver_func=simple_solver)  # Old solver
baseline_sub = baseline_gen.generate_submission(small_test)

# Run new iteration
new_gen = SubmissionGenerator(solver_func=enhanced_solver)  # New solver
new_sub = new_gen.generate_submission(small_test)

# Compare stats
baseline_gen.print_stats()
new_gen.print_stats()

# Asymmetric ratcheting decision:
# IF new_solver better THEN commit and integrate
# ELSE reject and debug
```

### Step 4: Integrate into Main Notebook

**Option A: Replace Cell 5** (for major changes)
```python
# In subtlegeniusv1.ipynb Cell 5:
# Delete old code
# Paste new code from cell5_iterationN_feature.py
```

**Option B: Add Cell 5b** (for additive changes)
```python
# Keep Cell 5 as-is
# Add new Cell 5b with additional functions
# Update solver call in Cell 6 to use new solver
```

### Step 5: Test Full Pipeline

```python
# Run all 6 cells in sequence
# Verify no errors
# Check submission.json validates
# Compare stats to previous iteration
```

### Step 6: Commit if Improvement

```bash
git add SubtleGenius/
git commit -m "Iteration N: [feature] - [X%] accuracy improvement"
git push origin <branch>
```

---

## 📝 Integration Checklist

Before integrating new iteration:

- [ ] Iteration code runs without errors
- [ ] Tests pass (if test suite exists)
- [ ] Validates submission.json format
- [ ] Shows improvement over previous iteration
- [ ] Documented in ITERATION_N_*.md
- [ ] Fallbacks handle edge cases
- [ ] No regression on baseline tasks

---

## 🔧 Current Iterations

### Iteration 0: Baseline
- **File**: Built into Cell 5
- **Solver**: `simple_solver` (identity transform)
- **Accuracy**: ~0-5%
- **Status**: ✅ Deployed

### Iteration 1: Pattern Matching
- **File**: `cell5_iteration1_patterns.py`
- **Solver**: `enhanced_pattern_solver`
- **Target**: 10-15%
- **Status**: ✅ Ready for integration
- **Integration**: Replace Cell 5 contents

### Iteration 2: Object Detection (Planned)
- **File**: `cell5_iteration2_objects.py`
- **Solver**: `object_detection_solver`
- **Target**: 20-30%
- **Status**: ⏳ Not started

---

## 🚀 Quick Integration: Iteration 1

### Copy-Paste Integration

1. Open `subtlegeniusv1.ipynb`
2. Go to Cell 5
3. Delete everything after the header comment
4. Copy ALL code from `cell5_iteration1_patterns.py`
5. Paste into Cell 5
6. Change the final solver function:

```python
# At end of Cell 5, update:
def simple_solver(test_input, task_data, attempt=1):
    return enhanced_pattern_solver(test_input, task_data, attempt, track_stats=True)
```

7. Run all cells
8. Check `pattern_stats.print_stats()` output

### Verify Integration

```python
# After running all cells, you should see:
# - Pattern detection statistics
# - No errors in any cell
# - submission.json generated and validated
# - Patterns found: X% (should be >10%)
```

---

## ⚠️ Common Integration Issues

### Issue: Import Errors
**Cause**: Cell 5 expects numpy
**Fix**: Make sure numpy is imported in Cell 1

### Issue: Solver Not Called
**Cause**: Cell 4 still using old solver
**Fix**: Update `SubmissionGenerator(solver_func=simple_solver)` call

### Issue: Stats Not Printing
**Cause**: `track_stats=False` or not called
**Fix**: Set `track_stats=True` and call `pattern_stats.print_stats()` in Cell 6

### Issue: Validation Fails
**Cause**: Solver returning invalid grids
**Fix**: Check fallback logic, ensure all code paths return valid grids

---

## 📊 Tracking Improvements

### Iteration Log Template

```markdown
## Iteration N (YYYY-MM-DD)
- **Feature**: [What was added]
- **Lines Added**: [Approximate]
- **Tests**: [Pass/Total]
- **Local Accuracy**: [X%]
- **Kaggle Accuracy**: [Y%]
- **Improvement**: [+Z%]
- **Status**: [Committed/Deployed]
```

### Performance Tracking

| Iteration | Feature | Local Acc | Kaggle Acc | Improvement |
|-----------|---------|-----------|------------|-------------|
| 0 | Baseline (identity) | 0% | TBD | - |
| 1 | Pattern matching | TBD | TBD | TBD |
| 2 | Object detection | TBD | TBD | TBD |

---

## 🎯 Best Practices

1. **One feature per iteration** - Don't mix patterns + objects in same iteration
2. **Test before integrating** - Validate improvement locally
3. **Document changes** - Create ITERATION_N_*.md file
4. **Commit if better** - Asymmetric ratcheting: only commit improvements
5. **Track stats** - Use stats classes to measure improvement
6. **Keep fallbacks** - Always return valid grid, even if solver fails

---

## 🔄 Rollback Procedure

If iteration causes regression:

```bash
# Revert to previous iteration
git checkout HEAD~1 SubtleGenius/notebooks/subtlegeniusv1.ipynb

# Or manually restore Cell 5 from previous commit
git show HEAD~1:SubtleGenius/notebooks/subtlegeniusv1.ipynb
```

---

## 💡 Pro Tips

1. **Keep iteration files** - Don't delete cell5_iterationN files, they're your history
2. **Test with subset first** - 10 tasks catches most issues
3. **Compare stats** - New vs old solver_success and fallback_uses
4. **Document failures** - Track what didn't work to avoid repeating
5. **Celebrate wins** - Even +2% is progress worth celebrating!

---

**Remember**: Token-efficient iterations = more shots at championship! 🏆
