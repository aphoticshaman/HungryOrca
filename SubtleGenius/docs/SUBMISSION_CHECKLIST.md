# SubtleGenius v1 - Pre-Submission Checklist

**DO NOT SUBMIT TO KAGGLE WITHOUT COMPLETING THIS CHECKLIST!**

Daily submissions are limited. Don't waste them on preventable errors.

---

## ✅ LOCAL TESTING (Required)

### Quick Test (10 tasks, <1 minute)
- [ ] Created small test subset (10 tasks)
- [ ] All 6 cells run without errors
- [ ] submission.json generated
- [ ] Validation passes
- [ ] No import errors, no path errors

### Full Test (240 tasks, realistic timing)
- [ ] Tested with complete test dataset
- [ ] All 240 tasks completed
- [ ] No timeouts or memory errors
- [ ] submission.json generated successfully
- [ ] File size reasonable (not 0 bytes, not >100MB)

---

## ✅ VALIDATION (Critical)

### Format Validation
- [ ] submission.json is a **DICT** (not a list!)
- [ ] All task IDs present (240 tasks)
- [ ] Each task has list of prediction dicts
- [ ] Each prediction has **both** attempt_1 and attempt_2
- [ ] No missing keys, no extra keys

### Grid Validation
- [ ] All grids are 2D lists
- [ ] All values are integers 0-9
- [ ] No ragged arrays (equal row lengths)
- [ ] No empty grids

### Test Output Matching
- [ ] Tasks with multiple test outputs handled correctly
- [ ] Number of predictions matches number of test inputs per task

---

## ✅ CODE QUALITY

### Cell Execution
- [ ] Cell 1 (Config) runs without errors
- [ ] Cell 2 (Validator) runs without errors
- [ ] Cell 3 (Fallbacks) runs without errors
- [ ] Cell 4 (Generator) runs without errors
- [ ] Cell 5 (Solver) runs without errors
- [ ] Cell 6 (Pipeline) runs without errors

### No Errors
- [ ] No syntax errors
- [ ] No import errors (all libraries available in Kaggle)
- [ ] No path errors (using /kaggle/working/ for output)
- [ ] No type errors
- [ ] No exceptions uncaught

---

## ✅ KAGGLE ENVIRONMENT

### Paths
- [ ] Input: /kaggle/input/arc-prize-2025/arc-agi_test_challenges.json
- [ ] Output: /kaggle/working/submission.json
- [ ] No hardcoded local paths

### Resources
- [ ] Time budget reasonable (<11.4 hours with 95% rule)
- [ ] Memory usage acceptable (no out-of-memory errors)
- [ ] Internet will be DISABLED during run

### Dataset
- [ ] ARC Prize 2025 dataset attached to notebook
- [ ] Dataset version is correct/latest

---

## ✅ SUBMISSION FILE QUALITY

### File Checks
- [ ] submission.json exists in /kaggle/working/
- [ ] File size >0 bytes
- [ ] File size <100MB (reasonable)
- [ ] File is valid JSON (not corrupted)

### Manual Spot Check
- [ ] Opened submission.json and verified structure looks correct
- [ ] First task has correct format: `{"task_id": [{"attempt_1": ..., "attempt_2": ...}]}`
- [ ] Random middle task checked
- [ ] Last task checked

---

## ✅ LOGGING & DEBUGGING

### Logs
- [ ] Reviewed solver_log.jsonl (if logging enabled)
- [ ] No unexpected errors in logs
- [ ] Solver success rate reasonable
- [ ] Fallback usage rate acceptable

### Statistics
- [ ] Checked generation statistics output
- [ ] Total tasks matches expected (240)
- [ ] Total predictions makes sense
- [ ] Error count low or zero

---

## ✅ COMPETITION COMPLIANCE

### Rules
- [ ] Reviewed latest ARC Prize 2025 rules
- [ ] Checked Kaggle discussion for known issues
- [ ] No rule violations in approach
- [ ] Open source plan ready (required before official scores)

### Submission Limits
- [ ] Confirmed I have submissions remaining today (1/day limit)
- [ ] Not wasting this submission on untested code

---

## ✅ DOUBLE-CHECK CRITICAL ERRORS

### The 5 Fatal Errors (from failed submission)
- [ ] ✅ Using DICT, not LIST for submission
- [ ] ✅ Both attempt_1 and attempt_2 present
- [ ] ✅ task_id as TOP-LEVEL key (not internal)
- [ ] ✅ Two attempts per test output
- [ ] ✅ Multiple test outputs per task handled

### Common Pitfalls
- [ ] No ragged arrays (all rows equal length)
- [ ] All values 0-9 (no -1, no 10+)
- [ ] Correct number of tasks (not 239, not 241)
- [ ] Correct paths (/kaggle/working/, not /tmp/ or ~/)

---

## ✅ FINAL VERIFICATION

### Before Clicking "Submit"
- [ ] Ran full checklist (this entire page)
- [ ] Local testing passed completely
- [ ] Validation passed completely
- [ ] All cells run successfully in sequence
- [ ] submission.json validated and saved
- [ ] Confident this won't be a wasted submission

### Post-Submission Plan
- [ ] Will save submission.json locally
- [ ] Will save solver_log.jsonl locally
- [ ] Will record public leaderboard score
- [ ] Will analyze failures for next iteration
- [ ] Will document learnings

---

## 🎯 Checklist Complete?

**If ALL boxes are checked:**
- ✅ You're ready to submit to Kaggle
- ✅ Minimal risk of format errors
- ✅ Valid submission guaranteed

**If ANY box is unchecked:**
- ❌ DO NOT SUBMIT YET
- ❌ Fix the issue first
- ❌ Re-run the checklist

---

## 📊 Submission History Template

Track your submissions to learn and improve:

```markdown
## Submission 1 (YYYY-MM-DD)
- Score: X%
- Strategy: Basic patterns + safe defaults
- Issues: None (or list issues)
- Next: Add object detection

## Submission 2 (YYYY-MM-DD)
- Score: Y%
- Strategy: + Object detection
- Issues: Some timeouts on complex tasks
- Next: Optimize time allocation
```

---

**Remember**: Valid 5% > Crashing 95%

**Daily submission limit**: Don't waste it!

---

**Last updated**: 2025-11-02
**Version**: SubtleGenius v1
