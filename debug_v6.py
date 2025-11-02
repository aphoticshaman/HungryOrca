#!/usr/bin/env python3
"""
Debug why v6 detectors aren't triggering
"""

import json
import numpy as np

# Load training data
with open('arc-agi_training_challenges.json', 'r') as f:
    challenges = json.load(f)

# Take first 10 tasks and debug detection
print("Debugging detection logic on first 10 tasks...\n")

def dcr_debug(td, task_id):
    """Detect crop with debug output"""
    if not td.get('train'):
        print(f"  {task_id}: No training data")
        return None

    bg = None
    for idx, p in enumerate(td['train']):
        i, o = np.array(p['input']), np.array(p['output'])

        # Check size
        if o.shape[0] > i.shape[0] or o.shape[1] > i.shape[1]:
            print(f"  {task_id}: Output LARGER than input ({o.shape} > {i.shape}) - NOT crop")
            return None

        # Detect background
        b = np.argmax(np.bincount(i.flatten()))
        if bg is None:
            bg = b
        elif bg != b:
            print(f"  {task_id}: Inconsistent background color (pair {idx}: {b} != {bg})")
            return None

        # Find bounding box
        m = i != bg
        if not m.any():
            print(f"  {task_id}: No non-background pixels")
            return None

        r, c = np.where(m)
        cr = i[r.min():r.max()+1, c.min():c.max()+1]

        # Check if output matches crop
        if not np.array_equal(o, cr):
            print(f"  {task_id}: Pair {idx} - Output != cropped input")
            print(f"    Input shape: {i.shape}, Output shape: {o.shape}, Crop shape: {cr.shape}")
            print(f"    Background: {bg}")
            return None

    print(f"  {task_id}: ✅ CROP DETECTED (bg={bg})")
    return {'bg': int(bg)}

def dcs_debug(td, task_id):
    """Detect color swap with debug output"""
    if not td.get('train'):
        return None

    gm = None
    for idx, p in enumerate(td['train']):
        i, o = np.array(p['input']), np.array(p['output'])

        if i.shape != o.shape:
            print(f"  {task_id}: Shape mismatch (pair {idx}) - NOT color swap")
            return None

        m = {}
        for iv, ov in zip(i.flatten(), o.flatten()):
            if iv in m:
                if m[iv] != ov:
                    print(f"  {task_id}: Inconsistent color mapping in pair {idx}")
                    return None
            else:
                m[iv] = ov

        if all(k == v for k, v in m.items()):
            print(f"  {task_id}: Identity mapping (pair {idx}) - NOT color swap")
            return None

        if gm is None:
            gm = m
        elif m != gm:
            print(f"  {task_id}: Inconsistent mapping across pairs")
            return None

    print(f"  {task_id}: ✅ COLOR SWAP DETECTED (mapping={gm})")
    return {'m': {int(k): int(v) for k, v in gm.items()}}

# Test on first 10 tasks
print("="*70)
print("CROP DETECTION DEBUG")
print("="*70)
for i, (task_id, task_data) in enumerate(list(challenges.items())[:10]):
    result = dcr_debug(task_data, task_id)

print("\n" + "="*70)
print("COLOR SWAP DETECTION DEBUG")
print("="*70)
for i, (task_id, task_data) in enumerate(list(challenges.items())[:10]):
    result = dcs_debug(task_data, task_id)

# Count how many training pairs (not tasks) show each pattern
print("\n" + "="*70)
print("PATTERN FREQUENCY IN TRAINING PAIRS")
print("="*70)

crop_pairs = 0
cswap_pairs = 0
total_pairs = 0

for task_id, task_data in challenges.items():
    for pair in task_data['train']:
        total_pairs += 1
        inp = np.array(pair['input'])
        out = np.array(pair['output'])

        # Check if this PAIR shows crop pattern
        if out.shape[0] <= inp.shape[0] and out.shape[1] <= inp.shape[1]:
            bg = np.argmax(np.bincount(inp.flatten()))
            m = inp != bg
            if m.any():
                r, c = np.where(m)
                cr = inp[r.min():r.max()+1, c.min():c.max()+1]
                if np.array_equal(out, cr):
                    crop_pairs += 1

        # Check if this PAIR shows color swap pattern
        if inp.shape == out.shape:
            m = {}
            is_swap = True
            for iv, ov in zip(inp.flatten(), out.flatten()):
                if iv in m:
                    if m[iv] != ov:
                        is_swap = False
                        break
                else:
                    m[iv] = ov
            if is_swap and not all(k == v for k, v in m.items()):
                cswap_pairs += 1

print(f"Total training pairs: {total_pairs}")
print(f"Crop pairs: {crop_pairs} ({crop_pairs/total_pairs*100:.1f}%)")
print(f"Color swap pairs: {cswap_pairs} ({cswap_pairs/total_pairs*100:.1f}%)")
print("\nNOTE: Pattern analyzer counted PAIRS, but detector requires ALL pairs in a task to match!")
