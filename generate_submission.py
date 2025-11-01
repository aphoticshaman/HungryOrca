"""
CRITICAL FIX: Proper submission.json generation
================================================

This generates ONLY submission.json in the correct locations.
No eval_predictions.json - that was for diagnostics only.

For Kaggle competition, you only need:
  /kaggle/working/submission.json
  /kaggle/output/submission.json (Kaggle auto-copies from working)
"""

import torch
import json
import numpy as np
from pathlib import Path

print("=" * 80)
print("GENERATING SUBMISSION.JSON (Competition Format)")
print("=" * 80)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =============================================================================
# LOAD BEST MODEL
# =============================================================================

print("\n[1] Loading model...")

# Try to load checkpoint if available
checkpoint_loaded = False
if hasattr(locals(), 'model') or 'model' in globals():
    print("  ✓ Model already in memory")
else:
    print("  ✗ Model not found - need to define it first")
    print("  Run the model definition cells first!")

# Load checkpoint if exists
for ckpt_file in ['best_model.pt', 'orcasword_full_checkpoint.pt', 'orcasword_checkpoint.pt']:
    try:
        if Path(ckpt_file).exists():
            checkpoint = torch.load(ckpt_file, map_location=DEVICE)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                epoch = checkpoint.get('epoch', '?')
                print(f"  ✓ Loaded {ckpt_file} (epoch {epoch})")
            else:
                model.load_state_dict(checkpoint)
                print(f"  ✓ Loaded {ckpt_file}")
            checkpoint_loaded = True
            break
    except Exception as e:
        print(f"  ✗ Could not load {ckpt_file}: {e}")

if not checkpoint_loaded:
    print("  ⚠️  Using current model state (no checkpoint loaded)")

# =============================================================================
# GENERATE SUBMISSION.JSON
# =============================================================================

print("\n[2] Generating predictions for test set...")

model.eval()
submission = []

# Assuming test_tasks is already loaded
if 'test_tasks' not in globals():
    print("  ✗ test_tasks not found - loading from file...")
    import json
    data_dir = Path('/kaggle/input/arc-prize-2025')
    with open(data_dir / 'arc-agi_test_challenges.json', 'r') as f:
        test_tasks = json.load(f)
    print(f"  ✓ Loaded {len(test_tasks)} test tasks")

total = len(test_tasks)
with torch.no_grad():
    for i, (task_id, task_data) in enumerate(test_tasks.items(), 1):
        test_examples = task_data.get('test', [])
        if not test_examples:
            # Fallback: empty grid
            submission.append({
                "task_id": task_id,
                "attempt_1": [[0]],
                "attempt_2": [[0]]
            })
            continue

        # Get test input
        test_input = test_examples[0]['input']

        # Pad to model size
        padded_input = pad_grid(test_input)

        # Predict
        x = torch.from_numpy(padded_input).unsqueeze(0).to(DEVICE)
        logits = model(x)
        pred = logits.argmax(dim=-1).squeeze(0).cpu().numpy()

        # Unpad to original size
        H, W = len(test_input), len(test_input[0]) if test_input else 1
        pred_grid = pred[:H, :W].tolist()

        # Add to submission (2 attempts with same prediction)
        submission.append({
            "task_id": task_id,
            "attempt_1": pred_grid,
            "attempt_2": pred_grid
        })

        if i % 50 == 0:
            print(f"  Progress: {i}/{total} tasks")

print(f"  ✓ Generated {len(submission)} predictions")

# =============================================================================
# SAVE TO CORRECT LOCATIONS
# =============================================================================

print("\n[3] Saving submission.json...")

# Save to /kaggle/working/ (required for download)
working_path = Path('/kaggle/working/submission.json')
with open(working_path, 'w') as f:
    json.dump(submission, f)
print(f"  ✓ Saved to {working_path}")

# Also save to current directory if not /kaggle/working
if Path.cwd() != Path('/kaggle/working'):
    local_path = Path('submission.json')
    with open(local_path, 'w') as f:
        json.dump(submission, f)
    print(f"  ✓ Saved to {local_path}")

# Kaggle auto-copies from /kaggle/working to /kaggle/output
print("  ✓ Kaggle will auto-copy to /kaggle/output/")

# =============================================================================
# VALIDATE
# =============================================================================

print("\n[4] Validating submission.json...")

# Reload and validate
with open(working_path, 'r') as f:
    loaded_submission = json.load(f)

errors = []

# Check structure
if not isinstance(loaded_submission, list):
    errors.append("Submission is not a list")

# Check each task
for i, task in enumerate(loaded_submission):
    if not isinstance(task, dict):
        errors.append(f"Task {i} is not a dict")
        continue

    # Check required fields
    if 'task_id' not in task:
        errors.append(f"Task {i} missing task_id")
    if 'attempt_1' not in task:
        errors.append(f"Task {i} missing attempt_1")
    if 'attempt_2' not in task:
        errors.append(f"Task {i} missing attempt_2")

    # Check grid is valid
    for attempt_key in ['attempt_1', 'attempt_2']:
        if attempt_key in task:
            grid = task[attempt_key]
            if not isinstance(grid, list):
                errors.append(f"Task {i} {attempt_key} is not a list")
                continue

            for row_idx, row in enumerate(grid):
                if not isinstance(row, list):
                    errors.append(f"Task {i} {attempt_key} row {row_idx} is not a list")
                    continue

                for col_idx, val in enumerate(row):
                    if not isinstance(val, int) or val < 0 or val > 9:
                        errors.append(f"Task {i} {attempt_key} [{row_idx}][{col_idx}] = {val} (must be 0-9)")

# Report validation results
if errors:
    print(f"  ✗ Found {len(errors)} errors:")
    for error in errors[:10]:  # Show first 10
        print(f"    - {error}")
    if len(errors) > 10:
        print(f"    ... and {len(errors) - 10} more")
else:
    print("  ✓ Validation passed!")
    print(f"  ✓ {len(loaded_submission)} tasks")
    print(f"  ✓ All tasks have task_id, attempt_1, attempt_2")
    print(f"  ✓ All grids are valid (values 0-9)")

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 80)
print("SUBMISSION GENERATION COMPLETE")
print("=" * 80)

print(f"\n📁 File location:")
print(f"  /kaggle/working/submission.json")
print(f"\n📊 Statistics:")
print(f"  Total tasks: {len(loaded_submission)}")

# Sample task
if loaded_submission:
    sample = loaded_submission[0]
    print(f"\n📝 Sample task:")
    print(f"  Task ID: {sample['task_id']}")
    grid = sample['attempt_1']
    print(f"  Grid size: {len(grid)}x{len(grid[0]) if grid else 0}")
    print(f"  First row: {grid[0] if grid else []}")

print(f"\n✅ Ready to download and submit!")
print(f"\n📥 Download: submission.json from Kaggle output files")
print(f"📤 Submit: Upload to ARC Prize 2025 competition")

print("\n" + "=" * 80)
