"""
Diagnostic & Recovery Script for Stopped Training
==================================================

Run this to diagnose why training stopped and resume from checkpoint.
"""

import torch
import os
import json
import subprocess
from pathlib import Path

print("=" * 80)
print("TRAINING DIAGNOSTIC & RECOVERY")
print("=" * 80)

# =============================================================================
# 1. CHECK SYSTEM RESOURCES
# =============================================================================

print("\n" + "=" * 80)
print("1. SYSTEM RESOURCE CHECK")
print("=" * 80)

# Check GPU memory
if torch.cuda.is_available():
    print("\n[GPU Memory]")
    for i in range(torch.cuda.device_count()):
        total = torch.cuda.get_device_properties(i).total_memory / 1e9
        allocated = torch.cuda.memory_allocated(i) / 1e9
        reserved = torch.cuda.memory_reserved(i) / 1e9
        free = total - reserved

        print(f"  GPU {i} ({torch.cuda.get_device_name(i)}):")
        print(f"    Total: {total:.2f} GB")
        print(f"    Allocated: {allocated:.2f} GB")
        print(f"    Reserved: {reserved:.2f} GB")
        print(f"    Free: {free:.2f} GB")

        if free < 0.5:
            print(f"    ⚠️  WARNING: Low GPU memory!")

# Check disk space
print("\n[Disk Space]")
try:
    result = subprocess.run(['df', '-h', '.'], capture_output=True, text=True)
    lines = result.stdout.strip().split('\n')
    if len(lines) >= 2:
        print("  " + lines[0])
        print("  " + lines[1])

        # Parse usage
        parts = lines[1].split()
        if len(parts) >= 5:
            usage_pct = parts[4].rstrip('%')
            if int(usage_pct) > 90:
                print(f"    ⚠️  WARNING: Disk {usage_pct}% full!")
except:
    print("  Could not check disk space")

# Check RAM
print("\n[System Memory]")
try:
    result = subprocess.run(['free', '-h'], capture_output=True, text=True)
    lines = result.stdout.strip().split('\n')
    for line in lines[:2]:
        print("  " + line)
except:
    print("  Could not check RAM")

# =============================================================================
# 2. CHECK CHECKPOINT FILES
# =============================================================================

print("\n" + "=" * 80)
print("2. CHECKPOINT FILE CHECK")
print("=" * 80)

checkpoint_files = [
    'orcasword_full_checkpoint.pt',
    'best_model.pt',
    'orcasword_checkpoint.pt'
]

print("\n[Checkpoint Files]")
checkpoint_found = None
checkpoint_info = None

for ckpt_file in checkpoint_files:
    if os.path.exists(ckpt_file):
        size = os.path.getsize(ckpt_file) / 1e6
        mtime = os.path.getmtime(ckpt_file)
        from datetime import datetime
        mod_time = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')

        print(f"\n  ✓ Found: {ckpt_file}")
        print(f"    Size: {size:.2f} MB")
        print(f"    Modified: {mod_time}")

        # Try to load it
        try:
            checkpoint = torch.load(ckpt_file, map_location='cpu')
            if isinstance(checkpoint, dict):
                print(f"    Contents: {list(checkpoint.keys())}")
                if 'epoch' in checkpoint:
                    print(f"    Epoch: {checkpoint['epoch']}")
                    if checkpoint_found is None:
                        checkpoint_found = ckpt_file
                        checkpoint_info = checkpoint
                if 'loss' in checkpoint:
                    print(f"    Loss: {checkpoint['loss']:.4f}")
                if 'accuracy' in checkpoint:
                    print(f"    Accuracy: {checkpoint['accuracy']:.4f}")
            print(f"    ✓ Checkpoint is valid")
        except Exception as e:
            print(f"    ✗ Error loading: {e}")
    else:
        print(f"  ✗ Not found: {ckpt_file}")

# =============================================================================
# 3. CHECK OUTPUT FILES
# =============================================================================

print("\n" + "=" * 80)
print("3. OUTPUT FILE CHECK")
print("=" * 80)

output_files = [
    'submission.json',
    'eval_predictions.json'
]

print("\n[Output Files]")
for output_file in output_files:
    if os.path.exists(output_file):
        size = os.path.getsize(output_file) / 1024
        print(f"  ✓ Found: {output_file} ({size:.1f} KB)")

        # Check if valid JSON
        try:
            with open(output_file, 'r') as f:
                data = json.load(f)
            print(f"    Contains {len(data)} entries")
        except:
            print(f"    ⚠️  WARNING: May be corrupted")
    else:
        print(f"  ✗ Not found: {output_file}")

# =============================================================================
# 4. DIAGNOSIS
# =============================================================================

print("\n" + "=" * 80)
print("4. LIKELY CAUSE ANALYSIS")
print("=" * 80)

causes = []

# Check GPU memory
if torch.cuda.is_available():
    free_mem = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved(0)
    free_gb = free_mem / 1e9
    if free_gb < 0.5:
        causes.append(("GPU Out of Memory", "HIGH",
                      f"Only {free_gb:.2f} GB free. Clear cache with torch.cuda.empty_cache()"))

# Check disk space
try:
    result = subprocess.run(['df', '.'], capture_output=True, text=True)
    lines = result.stdout.strip().split('\n')
    if len(lines) >= 2:
        parts = lines[1].split()
        if len(parts) >= 5:
            usage_pct = int(parts[4].rstrip('%'))
            if usage_pct > 95:
                causes.append(("Disk Full", "HIGH",
                              f"Disk is {usage_pct}% full. Delete old checkpoints."))
            elif usage_pct > 85:
                causes.append(("Low Disk Space", "MEDIUM",
                              f"Disk is {usage_pct}% full."))
except:
    pass

# Check if checkpoint exists
if checkpoint_found:
    if checkpoint_info and 'epoch' in checkpoint_info:
        epoch = checkpoint_info['epoch']
        if epoch >= 29:
            causes.append(("Training Progressed Normally", "INFO",
                          f"Checkpoint shows epoch {epoch}. May have hit runtime limit."))
else:
    causes.append(("No Checkpoint Found", "HIGH",
                  "Training may have crashed before saving."))

# Kaggle-specific
if os.path.exists('/kaggle'):
    causes.append(("Kaggle Session Timeout", "MEDIUM",
                  "Kaggle notebooks have 9-12 hour limits. Check session time."))

if not causes:
    causes.append(("Unknown", "LOW", "No obvious issues detected. Check Kaggle logs."))

print("\nMost likely causes (ordered by probability):\n")
for i, (cause, severity, detail) in enumerate(causes, 1):
    print(f"{i}. [{severity}] {cause}")
    print(f"   {detail}\n")

# =============================================================================
# 5. RECOVERY OPTIONS
# =============================================================================

print("=" * 80)
print("5. RECOVERY OPTIONS")
print("=" * 80)

if checkpoint_found:
    print(f"\n✓ Can resume from: {checkpoint_found}")
    print(f"  Last saved: Epoch {checkpoint_info.get('epoch', '?')}")

    print("\n[Option 1] Resume Training (Recommended)")
    print("  Copy and run the code below to continue training:\n")

    print("```python")
    print("# Resume training from checkpoint")
    print(f"checkpoint = torch.load('{checkpoint_found}', map_location=DEVICE)")
    print("model.load_state_dict(checkpoint['model_state_dict'])")
    print("optimizer.load_state_dict(checkpoint['optimizer_state_dict'])")
    print(f"start_epoch = checkpoint['epoch'] + 1")
    print("print(f'Resuming from epoch {start_epoch}')")
    print("# Then continue training loop...")
    print("```")

    print("\n[Option 2] Use Best Model and Generate Submission")
    if os.path.exists('best_model.pt'):
        print("  ✓ best_model.pt exists")
        print("  Run submission generation with best model")
    else:
        print("  Load checkpoint and generate submission now")

    print("\n[Option 3] Start Fresh with More Memory")
    print("  Reduce model size: embed_dim=128, num_layers=4")
    print("  Add: torch.cuda.empty_cache() before training")

else:
    print("\n✗ No checkpoint found - cannot resume")
    print("\n[Option] Start Fresh")
    print("  1. Clear GPU memory: torch.cuda.empty_cache()")
    print("  2. Reduce model size")
    print("  3. Add frequent checkpointing")

# =============================================================================
# 6. QUICK FIXES
# =============================================================================

print("\n" + "=" * 80)
print("6. QUICK FIXES TO TRY NOW")
print("=" * 80)

print("\n[Clear GPU Memory]")
print("```python")
print("import torch")
print("import gc")
print("gc.collect()")
print("torch.cuda.empty_cache()")
print("print('GPU memory cleared')")
print("```")

print("\n[Check Kaggle Session Time]")
print("```python")
print("# In Kaggle, check how long you've been running")
print("import time")
print("# Session started at: [check notebook start time]")
print("# Kaggle limit: 9 hours (GPU) or 12 hours (CPU)")
print("```")

print("\n[Generate Submission Now]")
if checkpoint_found:
    print("```python")
    print(f"# Load checkpoint and generate submission")
    print(f"checkpoint = torch.load('{checkpoint_found}', map_location=DEVICE)")
    print("model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))")
    print("# Then run submission generation code...")
    print("```")

print("\n" + "=" * 80)
print("DIAGNOSTIC COMPLETE")
print("=" * 80)

print(f"\n📊 Summary:")
print(f"  - Checkpoint found: {'✓' if checkpoint_found else '✗'}")
print(f"  - Can resume: {'✓' if checkpoint_found else '✗'}")
print(f"  - Issues detected: {len([c for c in causes if c[1] in ['HIGH', 'MEDIUM']])}")
print(f"\nRecommendation: {causes[0][2] if causes else 'Check manually'}")
