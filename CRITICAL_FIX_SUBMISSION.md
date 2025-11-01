# CRITICAL FIX: Kaggle Submission File

## Problem

Kaggle error: "Competition requires submission.json but notebook doesn't output this file"

## Root Cause

The training script saves:
- ❌ `eval_predictions.json` (not needed for competition)
- ✅ `submission.json` (needed, but may be in wrong location)

## Solution

### Quick Fix (In Your Running Notebook)

**Delete this line** from `train_full.py`:
```python
'eval_predictions_path': 'eval_predictions.json',  # ← DELETE THIS
'save_eval_predictions': True,                     # ← CHANGE TO False
```

**Change this**:
```python
CONFIG = {
    # ...
    'save_eval_predictions': True,   # ← Change to False
    # ...
}
```

**And ensure this**:
```python
# At end of script, make sure submission.json is in /kaggle/working/
import shutil
shutil.copy('submission.json', '/kaggle/working/submission.json')
print("✓ submission.json saved to /kaggle/working/")
```

### Proper Fix (Updated Script)

The issue is in `train_full.py` lines where it saves eval predictions.

**Remove these sections:**
1. Eval predictions file path from CONFIG
2. The `save_predictions=True` parameter
3. The entire eval_predictions.json save operation

**Keep only:**
- submission.json generation for test set
- Save to `/kaggle/working/submission.json`

## What Kaggle Needs

```
/kaggle/working/
└── submission.json  ← ONLY THIS FILE

Format:
[
  {
    "task_id": "00576224",
    "attempt_1": [[3, 2], [1, 4]],
    "attempt_2": [[3, 2], [1, 4]]
  },
  ... (240 tasks total)
]
```

## Immediate Action

Run this in console RIGHT NOW:

```python
# Generate ONLY submission.json
import torch
import json
import numpy as np

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load best model
if os.path.exists('best_model.pt'):
    model.load_state_dict(torch.load('best_model.pt', map_location=DEVICE))
elif os.path.exists('orcasword_full_checkpoint.pt'):
    ckpt = torch.load('orcasword_full_checkpoint.pt', map_location=DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])

model.eval()
submission = []

with torch.no_grad():
    for task_id, task_data in test_tasks.items():
        test_input = task_data['test'][0]['input']
        x = torch.from_numpy(pad_grid(test_input)).unsqueeze(0).to(DEVICE)
        logits = model(x)
        pred = logits.argmax(dim=-1).squeeze(0).cpu().numpy()
        
        H, W = len(test_input), len(test_input[0]) if test_input else 1
        pred_grid = pred[:H, :W].tolist()
        
        submission.append({
            "task_id": task_id,
            "attempt_1": pred_grid,
            "attempt_2": pred_grid
        })

# Save to /kaggle/working/ (REQUIRED!)
with open('/kaggle/working/submission.json', 'w') as f:
    json.dump(submission, f)

print(f"✓ submission.json saved to /kaggle/working/")
print(f"✓ Contains {len(submission)} tasks")

# Verify it's there
import os
if os.path.exists('/kaggle/working/submission.json'):
    size = os.path.getsize('/kaggle/working/submission.json') / 1024
    print(f"✓ File size: {size:.1f} KB")
else:
    print("✗ ERROR: File not created!")
```

## Validation

After running above, check:
```python
!ls -lh /kaggle/working/submission.json
```

Should show: `submission.json` with size ~100-500 KB

## Then in Kaggle

1. Click "Save Version"
2. Select "Save & Run All"
3. Wait for completion
4. In "Output" tab, you should see submission.json
5. Now you can submit!

## DO NOT

❌ Create `eval_predictions.json`
❌ Save to any other location besides `/kaggle/working/`
❌ Use any other filename besides `submission.json`
❌ Create multiple output files

## DO

✅ Create ONLY `submission.json`
✅ Save to `/kaggle/working/submission.json`
✅ Include exactly 240 tasks (test set)
✅ Each task has: task_id, attempt_1, attempt_2
✅ Verify file exists before finishing notebook

