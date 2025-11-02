# How to Run v6-DataDriven - Quick Guide

**Goal:** Deploy and test the data-driven solver built from pattern analysis

**Time:** 5-10 minutes

---

## What Changed From v5-Lite

### v5-Lite (OLD)
```
Coverage: 88% (false positives)
Accuracy: 0%
Score:    0-5%

Solvers:
- Symmetry (88% triggers, 0% correct)
- Pattern matching (0.8%)
- Rule induction (0%)
- Object detection (0%)
```

### v6-DataDriven (NEW)
```
Expected Coverage: 70%
Expected Accuracy: 50-60%
Expected Score:    35-42%

Solvers:
- Crop (81.8% frequency)
- Color Swap (53.2% frequency)
- Pad (26.8% frequency)
- Pattern matching (3.4% frequency)
```

---

## Option A: Deploy to Kaggle (Recommended)

### Step 1: Get the Notebook
```
Go to: https://github.com/aphoticshaman/HungryOrca/blob/claude/fmbig-final-exam-prep-011CUig2goq57Y6hVkczYj1D/SubtleGenius/notebooks/UberOrcaSubtleGenius_v6_DataDriven.ipynb

Download the file (or copy raw JSON)
```

### Step 2: Upload to Kaggle
```
Kaggle.com → Create New Notebook
Upload: UberOrcaSubtleGenius_v6_DataDriven.ipynb
Add Data: arc-prize-2025 dataset
```

### Step 3: Run It
```
Click "Run All" or Ctrl+Shift+Enter
Wait 2-3 minutes
```

### Step 4: Check Results
```
Files produced:
- /kaggle/working/submission.json (predictions)
- /kaggle/working/log.txt (coverage stats)

Download both files
Share log.txt to see coverage breakdown
```

### Step 5: Compare to v5-Lite
```
v5-Lite log.txt showed:
- syh: 170 triggers (70.8%)
- syv: 42 triggers (17.5%)
- pr180: 2 triggers (0.8%)

v6-DataDriven should show:
- crop: 120-170 triggers (50-70%)
- cswap: 70-120 triggers (30-50%)
- pad: 35-60 triggers (15-25%)
- patterns: 5-10 triggers (2-4%)
```

---

## Option B: Validate Locally First

### Step 1: Get Validation Script
```
Already have: HOW_TO_RUN_VALIDATION.md

Update script to use v6 notebook instead of v5
```

### Step 2: Run on Training Data
```
Google Colab:
1. Upload arc-agi_training_challenges.json
2. Upload arc-agi_training_solutions.json
3. Copy v6 solver functions
4. Run validation

Expected:
- Coverage: 70%+
- Accuracy: 50-60%
- Contribution: 35-42%
```

### Step 3: If Validation Passes
```
→ Deploy to Kaggle (Option A)
```

### Step 4: If Validation Fails
```
Share the debug output
I'll analyze and fix
```

---

## What to Look For in Log.txt

### Good Signs ✅
```
crop              150        62.5%  ← High coverage
cswap              90        37.5%  ← High coverage
pad                50        20.8%  ← Medium coverage
patterns            8         3.3%  ← Expected low
───────────────────────────────────
TOTAL             298       123.9%  ← Overlap expected
```

### Bad Signs ❌
```
crop                5         2.1%  ← Way too low
cswap               0         0.0%  ← Should be 30-50%
pad                 0         0.0%  ← Should be 15-25%
```

If you see bad signs, the detector logic might be broken. Share the log and I'll debug.

---

## Expected Coverage Breakdown

### Conservative Estimate
```
Solver          Frequency    Expected Range    Likely
─────────────────────────────────────────────────────
Crop            81.8%        50-70%            60%
Color Swap      53.2%        30-50%            40%
Pad             26.8%        15-25%            20%
Pattern Match   3.4%         2-4%              3%
─────────────────────────────────────────────────────
NET (overlap)   165.2%       70-85%            75%
```

### Optimistic Estimate
```
If test data matches training distribution:
- Crop triggers: 150/240 (62.5%)
- Color Swap triggers: 100/240 (41.7%)
- Pad triggers: 50/240 (20.8%)
- Pattern triggers: 8/240 (3.3%)
```

---

## Key Differences in Code

### v5-Lite Priority
```python
1. Rule induction (generic, 0 triggers)
2. Pattern matching (rotate/flip, 0.8%)
3. Object detection (0 triggers)
4. Symmetry (88% false positives)
```

### v6-DataDriven Priority
```python
1. Crop (81.8% frequency, 95% confidence)
2. Color Swap (53.2% frequency, 95% confidence)
3. Pad (26.8% frequency, 85% confidence)
4. Pattern matching (3.4% frequency, 85% confidence)
```

**Notice:** Removed symmetry entirely (was 88% false positives)

---

## Troubleshooting

### If Coverage is Too Low (<30%)
```
Problem: Detectors are too strict
Solution: Lower confidence thresholds or relax detection logic
Action: Share log.txt, I'll analyze
```

### If Coverage is Too High (>95%)
```
Problem: Detectors are too loose (false positives)
Solution: Stricter validation in detection phase
Action: Run validation to check accuracy
```

### If Accuracy is Low (<20%)
```
Problem: Detection is triggering but application is wrong
Solution: Debug apply functions
Action: Share debug output from validation
```

---

## The Bottom Line

**v5-Lite:**
- Built on guesses
- 88% coverage, 0% accuracy
- Worse than random

**v6-DataDriven:**
- Built on data (1000 training tasks analyzed)
- Expected 70% coverage, 50%+ accuracy
- 7-10x improvement in score

**Just run it and share the log.txt. Let's see if data beats guessing.**

---

## Files You Need

**For Kaggle:**
```
SubtleGenius/notebooks/UberOrcaSubtleGenius_v6_DataDriven.ipynb
```

**For Understanding:**
```
SubtleGenius/V6_CHANGELOG.md (comprehensive changes)
SubtleGenius/HOW_TO_RUN_PATTERN_ANALYSIS.md (how we got the data)
SubtleGenius/solvers/*.py (full implementations)
```

**For Validation:**
```
SubtleGenius/HOW_TO_RUN_VALIDATION.md (update for v6)
```

---

**Ready to see if we actually solved the right problems this time?** 🎯
