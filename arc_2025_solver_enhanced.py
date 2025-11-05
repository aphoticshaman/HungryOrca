#!/usr/bin/env python3
"""
ARC-AGI 2025 Enhanced Solver - "HungryOrca v2"
Implements ctf.txt 5-Axiom Framework with enhanced pattern recognition
Target: <1MB .ipynb

Key Patterns Detected:
1. Tiling/Repetition (with variations)
2. Color replacement based on shapes/patterns
3. Object extraction and manipulation
4. Grid filling based on rules
"""

import json
from typing import List, Dict, Any

# =====================================================
# KERNEL-MODE OPERATIONS (No Dependencies)
# =====================================================

def k_rot90(g):
    """Rotate 90° clockwise"""
    try:
        h, w = len(g), len(g[0])
        return [[g[h-1-y][x] for y in range(h)] for x in range(w)]
    except:
        return g

def k_copy(g):
    """Deep copy grid"""
    return [r[:] for r in g]

def k_find_objs(g):
    """Find connected components"""
    h, w = len(g), len(g[0])
    v = [[0]*w for _ in range(h)]
    objs = []

    def dfs(y, x, obj, c):
        if y<0 or y>=h or x<0 or x>=w or v[y][x] or g[y][x]!=c: return
        v[y][x]=1; obj.append((y,x))
        dfs(y+1,x,obj,c); dfs(y-1,x,obj,c); dfs(y,x+1,obj,c); dfs(y,x-1,obj,c)

    for y in range(h):
        for x in range(w):
            if not v[y][x] and g[y][x]!=0:
                obj=[]
                dfs(y,x,obj,g[y][x])
                if obj: objs.append(obj)
    return objs

def k_get_bbox(obj):
    """Get bounding box of object"""
    if not obj: return None
    ys = [y for y,x in obj]
    xs = [x for y,x in obj]
    return (min(ys), min(xs), max(ys), max(xs))

# =====================================================
# ADVANCED PAYLOADS
# =====================================================

def p_tile_nxm(g, n, m):
    """Tile grid n times vertically, m times horizontally"""
    try:
        h, w = len(g), len(g[0])
        result = [[0]*(w*m) for _ in range(h*n)]
        for ty in range(n):
            for tx in range(m):
                for y in range(h):
                    for x in range(w):
                        result[ty*h+y][tx*w+x] = g[y][x]
        return result
    except:
        return g

def p_tile_with_pattern(g, n, m, pattern='normal'):
    """Tile with variations (flip, rotate, etc)"""
    try:
        h, w = len(g), len(g[0])
        result = [[0]*(w*m) for _ in range(h*n)]

        for ty in range(n):
            for tx in range(m):
                # Select pattern variation
                if pattern == 'checkerboard':
                    tile = g if (ty+tx)%2==0 else p_refl_x(g)
                elif pattern == 'mirror_h':
                    tile = g if tx%2==0 else p_refl_x(g)
                elif pattern == 'mirror_v':
                    tile = g if ty%2==0 else p_refl_y(g)
                else:
                    tile = g

                # Copy tile
                for y in range(h):
                    for x in range(w):
                        result[ty*h+y][tx*w+x] = tile[y][x]
        return result
    except:
        return g

def p_refl_y(g):
    """Y-axis reflection"""
    return g[::-1]

def p_refl_x(g):
    """X-axis reflection"""
    return [r[::-1] for r in g]

def p_color_map(g, color_mapping):
    """Apply color mapping"""
    try:
        h, w = len(g), len(g[0])
        result = [[0]*w for _ in range(h)]
        for y in range(h):
            for x in range(w):
                c = g[y][x]
                result[y][x] = color_mapping.get(c, c)
        return result
    except:
        return g

def p_extract_obj(g, obj_index=0):
    """Extract specific object from grid"""
    try:
        objs = k_find_objs(g)
        if obj_index >= len(objs): return g

        obj = objs[obj_index]
        h, w = len(g), len(g[0])
        result = [[0]*w for _ in range(h)]

        for y, x in obj:
            result[y][x] = g[y][x]

        return result
    except:
        return g

def p_largest_obj(g):
    """Extract largest object"""
    try:
        objs = k_find_objs(g)
        if not objs: return g

        largest = max(objs, key=len)
        h, w = len(g), len(g[0])
        result = [[0]*w for _ in range(h)]

        for y, x in largest:
            result[y][x] = g[y][x]

        return result
    except:
        return g

def p_crop_to_content(g):
    """Crop grid to non-zero content"""
    try:
        h, w = len(g), len(g[0])

        # Find bounds
        min_y, max_y, min_x, max_x = h, -1, w, -1
        for y in range(h):
            for x in range(w):
                if g[y][x] != 0:
                    min_y = min(min_y, y)
                    max_y = max(max_y, y)
                    min_x = min(min_x, x)
                    max_x = max(max_x, x)

        if max_y < 0: return [[0]]

        # Crop
        result = []
        for y in range(min_y, max_y+1):
            result.append(g[y][min_x:max_x+1])

        return result
    except:
        return g

# =====================================================
# PATTERN ANALYSIS
# =====================================================

def analyze_pattern(task):
    """Deep analysis of transformation pattern"""
    try:
        if not task.get('train') or len(task['train'])==0:
            return {}

        inp = task['train'][0]['input']
        out = task['train'][0]['output']

        h_in, w_in = len(inp), len(inp[0]) if inp else 0
        h_out, w_out = len(out), len(out[0]) if out else 0

        pattern = {
            'h_in': h_in, 'w_in': w_in,
            'h_out': h_out, 'w_out': w_out,
            'size_mult_h': h_out / h_in if h_in > 0 else 0,
            'size_mult_w': w_out / w_in if w_in > 0 else 0
        }

        # Detect tiling
        if h_out == h_in * 3 and w_out == w_in * 3:
            pattern['tiling'] = (3, 3)
        elif h_out == h_in * 2 and w_out == w_in * 2:
            pattern['tiling'] = (2, 2)
        elif h_out % h_in == 0 and w_out % w_in == 0:
            pattern['tiling'] = (h_out // h_in, w_out // w_in)

        # Detect color changes
        colors_in = set()
        for r in inp: colors_in.update(r)
        colors_out = set()
        for r in out: colors_out.update(r)

        pattern['colors_in'] = sorted(colors_in)
        pattern['colors_out'] = sorted(colors_out)
        pattern['color_mapping'] = {}

        # Try to detect color mapping
        if len(colors_in) == len(colors_out) and h_in == h_out and w_in == w_out:
            # Same size - likely color replacement
            mapping = {}
            for y in range(h_in):
                for x in range(w_in):
                    c_in = inp[y][x]
                    c_out = out[y][x]
                    if c_in not in mapping:
                        mapping[c_in] = c_out
                    elif mapping[c_in] != c_out:
                        # Inconsistent mapping - not simple color replacement
                        mapping = {}
                        break
            if mapping:
                pattern['color_mapping'] = mapping

        return pattern
    except:
        return {}

# =====================================================
# ENHANCED ROUTER
# =====================================================

def solve(task):
    """Main solver with enhanced pattern recognition"""
    try:
        if 'test' not in task or len(task['test']) == 0:
            return [[0]]

        test_input = task['test'][0]['input']

        # Analyze pattern from training data
        pattern = analyze_pattern(task)

        # Route to appropriate solver

        # Tiling patterns
        if 'tiling' in pattern:
            n, m = pattern['tiling']

            # Try different tiling patterns
            if n == m:  # Square tiling
                # Try normal tiling
                result = p_tile_nxm(test_input, n, m)

                # Check if there's a checkerboard pattern
                if len(task['train']) > 0:
                    inp = task['train'][0]['input']
                    expected = task['train'][0]['output']
                    test_normal = p_tile_nxm(inp, n, m)

                    if test_normal != expected:
                        # Try checkerboard
                        result = p_tile_with_pattern(test_input, n, m, 'checkerboard')
                        test_check = p_tile_with_pattern(inp, n, m, 'checkerboard')
                        if test_check != expected:
                            # Try mirror patterns
                            result = p_tile_with_pattern(test_input, n, m, 'mirror_h')
                            test_mirror = p_tile_with_pattern(inp, n, m, 'mirror_h')
                            if test_mirror != expected:
                                result = p_tile_with_pattern(test_input, n, m, 'mirror_v')

                return result
            else:
                return p_tile_nxm(test_input, n, m)

        # Color mapping
        if pattern.get('color_mapping'):
            return p_color_map(test_input, pattern['color_mapping'])

        # Same size transformations
        if pattern.get('h_in') == pattern.get('h_out') and pattern.get('w_in') == pattern.get('w_out'):
            # Try reflections
            if len(task['train']) > 0:
                inp = task['train'][0]['input']
                out = task['train'][0]['output']

                # Test Y-reflection
                if p_refl_y(inp) == out:
                    return p_refl_y(test_input)

                # Test X-reflection
                if p_refl_x(inp) == out:
                    return p_refl_x(test_input)

                # Test rotation
                if k_rot90(inp) == out:
                    return k_rot90(test_input)

        # Object extraction
        objs_in = k_find_objs(task['train'][0]['input']) if task.get('train') else []
        if len(objs_in) > 1:
            # Multiple objects - try extracting largest
            return p_largest_obj(test_input)

        # Default: return input unchanged
        return k_copy(test_input)

    except Exception as e:
        # Fail gracefully
        if 'test' in task and len(task['test']) > 0:
            return k_copy(task['test'][0]['input'])
        return [[0]]

# =====================================================
# SUBMISSION GENERATOR
# =====================================================

def generate_submission(challenges_path, output_path):
    """Generate Kaggle submission"""
    with open(challenges_path, 'r') as f:
        challenges = json.load(f)

    submission = {}
    solved = 0

    for i, (task_id, task) in enumerate(challenges.items()):
        try:
            solution = solve(task)
            submission[task_id] = {
                'attempt_1': solution,
                'attempt_2': solution
            }
            solved += 1
            if (i+1) % 10 == 0:
                print(f"Processed {i+1}/{len(challenges)} tasks...")
        except Exception as e:
            submission[task_id] = {
                'attempt_1': [[0]],
                'attempt_2': [[0]]
            }

    with open(output_path, 'w') as f:
        json.dump(submission, f)

    print(f"\nSubmission complete: {output_path}")
    print(f"Tasks processed: {solved}/{len(challenges)}")

# =====================================================
# EVALUATION
# =====================================================

def evaluate(training_path, solutions_path, max_tasks=50):
    """Evaluate on training data"""
    with open(training_path, 'r') as f:
        training = json.load(f)
    with open(solutions_path, 'r') as f:
        solutions = json.load(f)

    correct = 0
    total = 0

    for task_id, task in list(training.items())[:max_tasks]:
        try:
            predicted = solve(task)
            expected = solutions[task_id][0] if task_id in solutions else None

            if predicted == expected:
                correct += 1
                print(f"✓ {task_id}")
            else:
                print(f"✗ {task_id}")
            total += 1
        except Exception as e:
            print(f"✗ {task_id} - Error: {e}")
            total += 1

    acc = correct/total if total > 0 else 0
    print(f"\n{'='*60}")
    print(f"Accuracy: {correct}/{total} = {acc:.1%}")
    print(f"{'='*60}")
    return acc

# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":
    print("="*60)
    print("ARC-AGI 2025 Enhanced Solver - HungryOrca v2")
    print("="*60)

    print("\n[1] Evaluating on training data...")
    try:
        evaluate(
            'arc-agi_training_challenges.json',
            'arc-agi_training_solutions.json',
            max_tasks=30
        )
    except FileNotFoundError as e:
        print(f"Training files not found: {e}")

    print("\n[2] Generating submission...")
    try:
        generate_submission(
            'arc-agi_evaluation_challenges.json',
            'arc_2025_submission_v2.json'
        )
    except FileNotFoundError as e:
        print(f"Evaluation file not found: {e}")

    print("\n" + "="*60)
    print("✅ Enhanced solver ready!")
    print("="*60)
