#!/usr/bin/env python3
"""
Understand what the "crop_pattern" pairs actually do
"""

import json
import numpy as np
from collections import defaultdict

with open('arc-agi_training_challenges.json', 'r') as f:
    challenges = json.load(f)

# Analyze all pairs where output is smaller
crop_pairs = []

for task_id, task_data in challenges.items():
    for pair in task_data['train']:
        inp = np.array(pair['input'])
        out = np.array(pair['output'])

        # Pattern analyzer's "crop_pattern" definition
        if out.shape[0] <= inp.shape[0] and out.shape[1] <= inp.shape[1]:
            if inp.shape != out.shape:  # Actually smaller
                crop_pairs.append({
                    'task_id': task_id,
                    'inp_shape': inp.shape,
                    'out_shape': out.shape,
                    'inp': inp,
                    'out': out
                })

print(f"Found {len(crop_pairs)} pairs where output is smaller than input\n")

# Categorize what type of crop
categories = defaultdict(int)

for p in crop_pairs[:50]:  # Analyze first 50
    inp, out = p['inp'], p['out']

    # Is it bounding box crop?
    bg = np.argmax(np.bincount(inp.flatten()))
    m = inp != bg
    if m.any():
        r, c = np.where(m)
        bbox_crop = inp[r.min():r.max()+1, c.min():c.max()+1]
        if np.array_equal(out, bbox_crop):
            categories['bounding_box_crop'] += 1
            continue

    # Is it extracting a specific object/color?
    unique_colors = np.unique(out)
    if len(unique_colors) <= 3:
        # Check if output is subset of input (extracted region)
        categories['color_extraction'] += 1
        continue

    # Is it center crop?
    h_start = (inp.shape[0] - out.shape[0]) // 2
    w_start = (inp.shape[1] - out.shape[1]) // 2
    center_crop = inp[h_start:h_start+out.shape[0], w_start:w_start+out.shape[1]]
    if np.array_equal(out, center_crop):
        categories['center_crop'] += 1
        continue

    categories['other'] += 1

print("Crop Pattern Categories (first 50):")
for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
    print(f"  {cat}: {count} ({count/50*100:.1f}%)")

# Show examples
print("\n" + "="*70)
print("EXAMPLES")
print("="*70)

print("\nExample 'crop' that IS bounding box:")
for p in crop_pairs:
    inp, out = p['inp'], p['out']
    bg = np.argmax(np.bincount(inp.flatten()))
    m = inp != bg
    if m.any():
        r, c = np.where(m)
        bbox_crop = inp[r.min():r.max()+1, c.min():c.max()+1]
        if np.array_equal(out, bbox_crop):
            print(f"  Task: {p['task_id']}")
            print(f"  Input shape: {p['inp_shape']}, Output shape: {p['out_shape']}")
            print(f"  Background: {bg}")
            break

print("\nExample 'crop' that is NOT bounding box:")
for p in crop_pairs:
    inp, out = p['inp'], p['out']
    bg = np.argmax(np.bincount(inp.flatten()))
    m = inp != bg
    if m.any():
        r, c = np.where(m)
        bbox_crop = inp[r.min():r.max()+1, c.min():c.max()+1]
        if not np.array_equal(out, bbox_crop):
            print(f"  Task: {p['task_id']}")
            print(f"  Input shape: {p['inp_shape']}, Output shape: {p['out_shape']}")
            print(f"  Input:\n{inp}")
            print(f"  Output:\n{out}")
            print(f"  Bounding box crop would be:\n{bbox_crop}")
            break
