"""
PATTERN FREQUENCY ANALYSIS - Google Colab Script
Copy this entire script into Google Colab to discover what patterns actually exist

This will answer:
- Why did rule induction get 0 triggers?
- Are geometric patterns rare?
- What should we build next?

INSTRUCTIONS:
1. Go to https://colab.research.google.com/
2. Click "New Notebook"
3. Upload arc-agi_training_challenges.json (from Kaggle)
4. Paste this entire script into a cell
5. Run it (Shift+Enter)
6. Read the report!
"""

import json
import numpy as np
from collections import defaultdict

print("🔍 Pattern Frequency Analyzer - v5-Lite Edition")
print("="*70)

# ===== STEP 1: Load Training Data =====
print("\n📂 Loading training data...")

with open('arc-agi_training_challenges.json', 'r') as f:
    training_data = json.load(f)

print(f"✅ Loaded {len(training_data)} training tasks")

# ===== STEP 2: Pattern Detection Functions =====

def analyze_shape_change(inp, out):
    """Analyze if shapes match or change"""
    if inp.shape == out.shape:
        return "same_shape"
    elif out.shape[0] < inp.shape[0] or out.shape[1] < inp.shape[1]:
        return "crop_smaller"
    elif out.shape[0] > inp.shape[0] or out.shape[1] > inp.shape[1]:
        return "expand_larger"
    return "shape_change"

def analyze_geometric(inp, out):
    """Detect geometric transformations"""
    if inp.shape != out.shape:
        return None

    # Rotate 90
    if np.array_equal(out, np.rot90(inp, k=-1)):
        return "rotate_90_cw"

    # Rotate 180
    if np.array_equal(out, np.rot90(inp, k=2)):
        return "rotate_180"

    # Rotate 270
    if np.array_equal(out, np.rot90(inp, k=1)):
        return "rotate_270_cw"

    # Flip horizontal
    if np.array_equal(out, np.fliplr(inp)):
        return "flip_horizontal"

    # Flip vertical
    if np.array_equal(out, np.flipud(inp)):
        return "flip_vertical"

    # Transpose
    if inp.shape[0] == inp.shape[1] and np.array_equal(out, inp.T):
        return "transpose"

    return None

def analyze_color_change(inp, out):
    """Analyze color transformations"""
    if inp.shape != out.shape:
        return None

    # Check if it's a color mapping
    mapping = {}
    is_mapping = True
    for i_val, o_val in zip(inp.flatten(), out.flatten()):
        if i_val in mapping:
            if mapping[i_val] != o_val:
                is_mapping = False
                break
        else:
            mapping[i_val] = o_val

    if is_mapping and mapping and not all(k == v for k, v in mapping.items()):
        return "color_mapping"

    # Check if all colors changed by same amount
    diff = out - inp
    if len(np.unique(diff)) == 1 and np.unique(diff)[0] != 0:
        return "color_shift"

    # Check if colors swapped
    inp_colors = set(inp.flatten())
    out_colors = set(out.flatten())
    if inp_colors == out_colors and len(inp_colors) <= 3:
        return "color_swap"

    return None

def analyze_size_change(inp, out):
    """Analyze size changes"""
    inp_h, inp_w = inp.shape
    out_h, out_w = out.shape

    if inp_h == out_h and inp_w == out_w:
        return None

    # Check if it's a tiling pattern
    if out_h % inp_h == 0 and out_w % inp_w == 0:
        h_factor = out_h // inp_h
        w_factor = out_w // inp_w
        if h_factor > 1 or w_factor > 1:
            expected = np.tile(inp, (h_factor, w_factor))
            if np.array_equal(out, expected):
                return f"tile_{h_factor}x{w_factor}"

    # Check if it's cropping
    if out_h <= inp_h and out_w <= inp_w:
        return "crop_pattern"

    # Check if it's padding
    if out_h >= inp_h and out_w >= inp_w:
        return "pad_pattern"

    return "size_change"

def analyze_symmetry_in_output(out):
    """Check if OUTPUT has symmetry (different from input symmetry!)"""
    arr = np.array(out)
    h, w = arr.shape

    # Perfect horizontal symmetry
    if w >= 2:
        left = arr[:, :w//2]
        right = np.fliplr(arr[:, w//2:])
        if left.shape == right.shape and np.array_equal(left, right):
            return "output_symmetric_h"

    # Perfect vertical symmetry
    if h >= 2:
        top = arr[:h//2, :]
        bottom = np.flipud(arr[h//2:, :])
        if top.shape == bottom.shape and np.array_equal(top, bottom):
            return "output_symmetric_v"

    return None

# ===== STEP 3: Analyze All Training Tasks =====
print("\n🔍 Analyzing patterns...")

results = {
    'total_tasks': len(training_data),
    'shape_patterns': defaultdict(int),
    'geometric_patterns': defaultdict(int),
    'color_patterns': defaultdict(int),
    'size_patterns': defaultdict(int),
    'symmetry_patterns': defaultdict(int),
    'examples': defaultdict(list)
}

for task_id, task_data in training_data.items():
    # Analyze each training pair
    for pair in task_data['train']:
        inp = np.array(pair['input'])
        out = np.array(pair['output'])

        # Shape patterns
        shape_pattern = analyze_shape_change(inp, out)
        if shape_pattern:
            results['shape_patterns'][shape_pattern] += 1
            if len(results['examples'][shape_pattern]) < 3:
                results['examples'][shape_pattern].append(task_id)

        # Geometric patterns
        geom_pattern = analyze_geometric(inp, out)
        if geom_pattern:
            results['geometric_patterns'][geom_pattern] += 1
            if len(results['examples'][geom_pattern]) < 3:
                results['examples'][geom_pattern].append(task_id)

        # Color patterns
        color_pattern = analyze_color_change(inp, out)
        if color_pattern:
            results['color_patterns'][color_pattern] += 1
            if len(results['examples'][color_pattern]) < 3:
                results['examples'][color_pattern].append(task_id)

        # Size patterns
        size_pattern = analyze_size_change(inp, out)
        if size_pattern:
            results['size_patterns'][size_pattern] += 1
            if len(results['examples'][size_pattern]) < 3:
                results['examples'][size_pattern].append(task_id)

        # Symmetry in OUTPUT
        sym_pattern = analyze_symmetry_in_output(out)
        if sym_pattern:
            results['symmetry_patterns'][sym_pattern] += 1
            if len(results['examples'][sym_pattern]) < 3:
                results['examples'][sym_pattern].append(task_id)

# ===== STEP 4: Print Report =====
print("\n" + "="*70)
print("PATTERN FREQUENCY REPORT")
print("="*70)

print(f"\nTotal tasks analyzed: {results['total_tasks']}")

# Shape patterns
print(f"\n📐 SHAPE PATTERNS")
print(f"  {'Pattern':<25} {'Count':>8} {'Frequency':>12} {'Examples'}")
print(f"  {'-'*25} {'-'*8} {'-'*12} {'-'*20}")
for pattern, count in sorted(results['shape_patterns'].items(),
                             key=lambda x: x[1],
                             reverse=True)[:15]:
    freq = count / results['total_tasks']
    examples = ', '.join(results['examples'][pattern][:2])
    print(f"  {pattern:<25} {count:>8} {freq*100:>11.1f}% {examples}")

# Geometric patterns
if results['geometric_patterns']:
    print(f"\n🔄 GEOMETRIC PATTERNS")
    print(f"  {'Pattern':<25} {'Count':>8} {'Frequency':>12} {'Examples'}")
    print(f"  {'-'*25} {'-'*8} {'-'*12} {'-'*20}")
    for pattern, count in sorted(results['geometric_patterns'].items(),
                                 key=lambda x: x[1],
                                 reverse=True)[:15]:
        freq = count / results['total_tasks']
        examples = ', '.join(results['examples'][pattern][:2])
        print(f"  {pattern:<25} {count:>8} {freq*100:>11.1f}% {examples}")
else:
    print(f"\n🔄 GEOMETRIC PATTERNS: None found")

# Color patterns
if results['color_patterns']:
    print(f"\n🎨 COLOR PATTERNS")
    print(f"  {'Pattern':<25} {'Count':>8} {'Frequency':>12} {'Examples'}")
    print(f"  {'-'*25} {'-'*8} {'-'*12} {'-'*20}")
    for pattern, count in sorted(results['color_patterns'].items(),
                                 key=lambda x: x[1],
                                 reverse=True)[:15]:
        freq = count / results['total_tasks']
        examples = ', '.join(results['examples'][pattern][:2])
        print(f"  {pattern:<25} {count:>8} {freq*100:>11.1f}% {examples}")
else:
    print(f"\n🎨 COLOR PATTERNS: None found")

# Size patterns
if results['size_patterns']:
    print(f"\n📏 SIZE PATTERNS")
    print(f"  {'Pattern':<25} {'Count':>8} {'Frequency':>12} {'Examples'}")
    print(f"  {'-'*25} {'-'*8} {'-'*12} {'-'*20}")
    for pattern, count in sorted(results['size_patterns'].items(),
                                 key=lambda x: x[1],
                                 reverse=True)[:15]:
        freq = count / results['total_tasks']
        examples = ', '.join(results['examples'][pattern][:2])
        print(f"  {pattern:<25} {count:>8} {freq*100:>11.1f}% {examples}")
else:
    print(f"\n📏 SIZE PATTERNS: None found")

# Symmetry patterns (OUTPUT symmetry)
if results['symmetry_patterns']:
    print(f"\n🪞 OUTPUT SYMMETRY PATTERNS")
    print(f"  {'Pattern':<25} {'Count':>8} {'Frequency':>12} {'Examples'}")
    print(f"  {'-'*25} {'-'*8} {'-'*12} {'-'*20}")
    for pattern, count in sorted(results['symmetry_patterns'].items(),
                                 key=lambda x: x[1],
                                 reverse=True)[:15]:
        freq = count / results['total_tasks']
        examples = ', '.join(results['examples'][pattern][:2])
        print(f"  {pattern:<25} {count:>8} {freq*100:>11.1f}% {examples}")
else:
    print(f"\n🪞 OUTPUT SYMMETRY PATTERNS: None found")

# Priority recommendations
print(f"\n🎯 TOP PRIORITIES (Build These First)")
print(f"  {'Rank':>4} {'Pattern':<25} {'Frequency':>12} {'Category':<20}")
print(f"  {'-'*4} {'-'*25} {'-'*12} {'-'*20}")

all_patterns = []
for category in ['geometric_patterns', 'color_patterns', 'size_patterns']:
    for pattern, count in results[category].items():
        freq = count / results['total_tasks']
        all_patterns.append((pattern, freq, category))

all_patterns.sort(key=lambda x: x[1], reverse=True)

for rank, (pattern, freq, category) in enumerate(all_patterns[:15], 1):
    print(f"  {rank:>4} {pattern:<25} {freq*100:>11.1f}% {category:<20}")

# Critical insights
print(f"\n💡 CRITICAL INSIGHTS")
print(f"  {'Metric':<30} {'Value':>15}")
print(f"  {'-'*30} {'-'*15}")

total_geom = sum(results['geometric_patterns'].values())
total_color = sum(results['color_patterns'].values())
total_size = sum(results['size_patterns'].values())

print(f"  {'Geometric transforms':<30} {total_geom:>15}")
print(f"  {'Color transforms':<30} {total_color:>15}")
print(f"  {'Size transforms':<30} {total_size:>15}")

if total_geom > 0:
    print(f"\n  ✅ Geometric patterns EXIST ({total_geom} instances)")
    print(f"     → Pattern matching should work!")
else:
    print(f"\n  ❌ Geometric patterns RARE/NONE ({total_geom} instances)")
    print(f"     → Explains why pattern matching got 0.8% coverage")

if total_color > 0:
    print(f"\n  ✅ Color patterns EXIST ({total_color} instances)")
    print(f"     → Color mapping/rule induction should work!")
else:
    print(f"\n  ❌ Color patterns RARE/NONE ({total_color} instances)")
    print(f"     → Explains why rule induction got 0% coverage")

print("\n" + "="*70)
print("✅ ANALYSIS COMPLETE")
print("="*70)

print("\n🎯 NEXT STEPS:")
print("  1. Build patterns with >5% frequency")
print("  2. Test on example tasks listed above")
print("  3. Abandon patterns with 0% frequency")
print("  4. Focus on what ACTUALLY EXISTS in data")
print("\n")
