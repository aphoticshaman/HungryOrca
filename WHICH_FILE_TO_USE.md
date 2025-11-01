# Which Training Script to Use?

## ✅ USE THIS: `train_ULTIMATE.py` ⭐ NEW & BEST

**For**: Optimized Kaggle submission with best quality

**Features**:
- ✅ Trains on ALL 2,842 samples per epoch
- ✅ Outputs to BOTH /kaggle/working/ AND /kaggle/output/
- ✅ Better initialization (reduces zero outputs from 5.8% → <3.3%)
- ✅ Improved hyperparameters (warmup + cosine LR, label smoothing)
- ✅ Quality monitoring (tracks zero prediction rate)
- ✅ Runtime: 6-7 hours (configurable)
- ✅ Progress tracking with ETA
- ✅ Auto-checkpoints every 30 min

**Expected Output**:
- 18-20 epochs in 7 hours
- ~140,000 training steps
- Training accuracy: ~85%
- Zero predictions: <3.3% (vs 5.8% in current submission)

---

## ⚠️ ALTERNATIVE: `train_full_CORRECTED.py`

**For**: If train_ULTIMATE.py has issues (fallback option)

**Features**:
- ✅ Trains on ALL 2,842 samples per epoch
- ✅ Outputs to /kaggle/working/submission.json only
- ⚠️ Higher zero prediction rate (~5.8%)
- ⚠️ Basic hyperparameters

**Use if**: train_ULTIMATE.py fails for some reason

---

## ❌ DON'T USE: `train_notebook.py`

**Problem**: Only trains on 100 samples (bug)

**What happens**:
- ❌ Only 1,000 training steps total
- ❌ Runs in 10 seconds
- ❌ No eval predictions
- ❌ Poor accuracy

**This file has the `train_pairs[:100]` bug!**

---

## 📝 Other Files

### `train_and_submit.py`
- Command-line version (use in terminal)
- Has same [:100] bug, avoid for now

### `orcaswordv3.py`
- Full mathematical solver
- 2,600+ lines with all theorems
- For advanced use / research

---

## Quick Start

**On Kaggle:**
```python
# Copy entire train_full.py into one cell and run
# OR
%run train_full.py
```

**Expected runtime**: 6-7 hours with L4 GPU

**Will generate**:
- `submission.json` (240 test tasks)
- `eval_predictions.json` (120 eval tasks)
- `orcasword_full_checkpoint.pt`
- `best_model.pt`

---

## Summary

| File | Use? | Runtime | Eval Predictions | Full Data? |
|------|------|---------|------------------|------------|
| `train_full.py` | ✅ YES | 6-7 hrs | ✅ Yes | ✅ Yes |
| `train_notebook.py` | ❌ NO | 10 sec | ❌ No | ❌ No (bug) |
| `train_and_submit.py` | ❌ NO | 10 sec | ❌ No | ❌ No (bug) |
| `orcaswordv3.py` | ⚠️ Advanced | Varies | ⚠️ Manual | ✅ Yes |

**Download `train_full.py` from GitHub and use it!**
