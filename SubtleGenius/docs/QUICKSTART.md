# SubtleGenius v1 - Quick Start Guide

**Get from zero to valid submission in 30 minutes.**

---

## 🚀 Fastest Path to Valid Submission

### Step 1: Copy Notebook (5 min)

1. Go to Kaggle
2. Create new notebook
3. Add ARC Prize 2025 dataset
4. Copy all 6 cells from `notebooks/subtlegeniusv1.ipynb`
5. Paste sequentially into Kaggle notebook

### Step 2: Configure (2 min)

Cell 1 auto-detects Kaggle environment. Just verify:
```python
print(f"Environment: {config.IS_KAGGLE}")  # Should print True
print(f"Output path: {config.OUTPUT_PATH}")  # Should be /kaggle/working/
```

### Step 3: Test Locally First (10 min)

**IMPORTANT**: Test locally before wasting Kaggle submission!

Download ARC data to `SubtleGenius/data/`:
- arc-agi_test_challenges.json
- arc-agi_training_challenges.json

Run notebook locally:
```python
# Quick test with 10 tasks
with open('data/arc-agi_test_challenges.json', 'r') as f:
    test_data = json.load(f)

small_test = dict(list(test_data.items())[:10])

gen = SubmissionGenerator(solver_func=simple_solver)
submission = gen.generate_submission(small_test)

# Validate
is_valid, msg = SubmissionValidator.validate_submission(submission, small_test)
print(f"Result: {msg}")
```

**Expected output**: `✅ SUBMISSION VALIDATION PASSED!`

### Step 4: Run in Kaggle (10 min)

1. Disable internet (competition requirement)
2. Run all cells sequentially
3. Wait for completion
4. Check output: `/kaggle/working/submission.json`

### Step 5: Validate Before Submitting (3 min)

The notebook auto-validates. Look for:
```
🎉 SUBMISSION VALIDATION PASSED!
✅ Saved to: /kaggle/working/submission.json
```

**If you see this**, you're good to submit!

---

## 🔧 Common Issues & Fixes

### Issue: "File not found"
**Fix**: Check paths in Cell 1. Make sure dataset is attached.

### Issue: "Validation failed"
**Fix**: Check error message. Most common:
- Ragged arrays → Cell 3 fallbacks should prevent this
- Invalid values → Cell 2 validator should catch this
- Missing tasks → Cell 4 generator should handle this

### Issue: "Out of memory"
**Fix**: Add `gc.collect()` more frequently in Cell 6

### Issue: "Time limit exceeded"
**Fix**: Adjust `config.EFFECTIVE_TIME` in Cell 1 for testing

---

## 📈 Iteration Workflow

Once you have a valid baseline:

### Iteration 1: Pattern Matching
1. Expand Cell 5 with flip/rotate logic
2. Test locally (10 tasks)
3. If improved: commit and test full dataset
4. Submit to Kaggle

### Iteration 2: Object Detection
1. Add connected component analysis to Cell 5
2. Test locally
3. If improved: commit and submit
4. Track score improvement

### Iteration 3+: Ensemble & Meta-Cognition
Follow Phase 3-5 from build plan

---

## ✅ 30-Minute Checklist

- [ ] **Min 0-5**: Copy 6 cells to Kaggle notebook
- [ ] **Min 5-7**: Verify configuration
- [ ] **Min 7-17**: Test locally with 10 tasks
- [ ] **Min 17-27**: Run in Kaggle (full 240 tasks)
- [ ] **Min 27-30**: Validate and submit

---

## 🎯 Success Criteria

**First submission should have**:
- ✅ Valid submission.json
- ✅ 100% task coverage (240 tasks)
- ✅ No format errors
- ✅ Non-zero score on leaderboard

**Accuracy target**: 5-10% on first submission (identity baseline)

**Then iterate**: Each submission should improve on previous

---

## 🚨 Before Your First Submission

Run through [SUBMISSION_CHECKLIST.md](SUBMISSION_CHECKLIST.md):
- Local testing complete
- Validation passed
- All cells run without errors
- submission.json looks correct
- Have daily submission remaining

---

## 💡 Pro Tips

1. **Start simple**: Don't expand solver until baseline works
2. **Test locally**: Catch 90% of errors before Kaggle
3. **Use validator**: Cell 2 catches format errors
4. **Trust fallbacks**: Cell 3 ensures no crashes
5. **Track iterations**: Log each submission's score and strategy

---

## 📞 Need Help?

Check:
1. [Build Plan](SUBTLEGENIUS_BUILD_PLAN.md) - Full technical details
2. [Checklist](SUBMISSION_CHECKLIST.md) - Pre-submission verification
3. [README](../README.md) - Architecture overview

---

**Remember**: Valid 5% > Crashing 95%

Get that first valid submission, THEN iterate to championship performance! 🏆
