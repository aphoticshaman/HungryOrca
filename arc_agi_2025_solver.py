#!/usr/bin/env python3
"""
ARC-AGI 2025 Solver - "HungryOrca"
Based on ctf.txt 5-Axiom Framework
Target: <1MB .ipynb for Kaggle submission

Strategy: Cryptographic Keystore + Exploit Chain + Kernel-Mode Operations
"""

import json
import zlib
import base64
from typing import Dict, List, Any, Tuple, Optional

# =====================================================
# AXIOM 1: CRYPTOGRAPHIC KEYSTORE - COMPRESSED LOGIC
# =====================================================

# Compressed solvers (zlib + base64) - "decrypt" at runtime
COMPRESSED_SOLVERS = {
    # These will store compressed solver functions
    # Format: 'solver_name': b'compressed_bytes'
}

def load_solver(solver_name: str):
    """Decrypt a compressed solver and inject into globals"""
    if solver_name in COMPRESSED_SOLVERS:
        code = zlib.decompress(base64.b64decode(COMPRESSED_SOLVERS[solver_name])).decode('utf-8')
        exec(code, globals())

# RLE Compression for grids (saves massive space)
def rle_decode(rle_row):
    """Decode RLE-compressed row"""
    return [val for val, count in rle_row for _ in range(count)]

def rle_encode(row):
    """Encode row to RLE"""
    if not row: return []
    encoded, val, count = [], row[0], 1
    for i in range(1, len(row)):
        if row[i] == val:
            count += 1
        else:
            encoded.append((val, count))
            val, count = row[i], 1
    encoded.append((val, count))
    return encoded

# =====================================================
# AXIOM 4: KERNEL-MODE OPERATIONS (No Dependencies)
# =====================================================

# Syscall-level grid operations (byte-optimized, no numpy)
K_H = lambda g: len(g)
K_W = lambda g: len(g[0]) if g else 0
K_SIZE = lambda g: (len(g), len(g[0]) if g else 0)
K_NEW = lambda h, w, v=0: [[v] * w for _ in range(h)]

def k_rot90(g):
    """Kernel: Rotate 90° clockwise"""
    try:
        h, w = K_H(g), K_W(g)
        n = K_NEW(w, h)
        for y in range(h):
            for x in range(w):
                n[x][h - 1 - y] = g[y][x]
        return n
    except:
        return g

def k_find_objects(g):
    """Kernel: Find connected components via DFS flood fill"""
    h, w = K_H(g), K_W(g)
    v = K_NEW(h, w, 0)
    objs = []

    def dfs(y, x, obj, c):
        if y < 0 or y >= h or x < 0 or x >= w or v[y][x] or g[y][x] != c:
            return
        v[y][x] = 1
        obj.append((y, x))
        dfs(y+1, x, obj, c)
        dfs(y-1, x, obj, c)
        dfs(y, x+1, obj, c)
        dfs(y, x-1, obj, c)

    for y in range(h):
        for x in range(w):
            c = g[y][x]
            if not v[y][x] and c != 0:
                obj = []
                dfs(y, x, obj, c)
                if obj:
                    objs.append(obj)
    return objs

# =====================================================
# AXIOM 2: EXPLOIT PAYLOADS (Byte-Optimized Solvers)
# =====================================================

def p_refl_y(g):
    """Payload: Y-axis reflection"""
    return g[::-1]

def p_refl_x(g):
    """Payload: X-axis reflection"""
    return [r[::-1] for r in g]

def p_rot_90(g):
    """Payload: 90° rotation"""
    return k_rot90(g)

def p_rot_180(g):
    """Payload: 180° rotation"""
    return p_refl_y(p_refl_x(g))

def p_rot_270(g):
    """Payload: 270° rotation"""
    return k_rot90(k_rot90(k_rot90(g)))

def p_transpose(g):
    """Payload: Transpose (main diagonal flip)"""
    try:
        h, w = K_H(g), K_W(g)
        n = K_NEW(w, h)
        for y in range(h):
            for x in range(w):
                n[x][y] = g[y][x]
        return n
    except:
        return g

def p_color_swap(g, c_from, c_to):
    """Payload: Swap two colors"""
    h, w = K_H(g), K_W(g)
    n = K_NEW(h, w)
    for y in range(h):
        for x in range(w):
            if g[y][x] == c_from:
                n[y][x] = c_to
            elif g[y][x] == c_to:
                n[y][x] = c_from
            else:
                n[y][x] = g[y][x]
    return n

def p_scale_up(g, factor):
    """Payload: Scale grid up by factor"""
    h, w = K_H(g), K_W(g)
    n = K_NEW(h * factor, w * factor)
    for y in range(h):
        for x in range(w):
            for dy in range(factor):
                for dx in range(factor):
                    n[y * factor + dy][x * factor + dx] = g[y][x]
    return n

def p_find_largest_obj(g):
    """Payload: Extract largest connected object"""
    try:
        objs = k_find_objects(g)
        if not objs:
            return g
        largest = max(objs, key=len)
        h, w = K_H(g), K_W(g)
        n = K_NEW(h, w)
        for y, x in largest:
            n[y][x] = g[y][x]
        return n
    except:
        return g

def p_remove_color(g, color):
    """Payload: Remove specific color (set to 0)"""
    h, w = K_H(g), K_W(g)
    n = K_NEW(h, w)
    for y in range(h):
        for x in range(w):
            n[y][x] = 0 if g[y][x] == color else g[y][x]
    return n

# =====================================================
# AXIOM 5: PACKET DISSECTOR - PROBLEM FINGERPRINTING
# =====================================================

def nmap_fingerprint(g):
    """
    'Nmap scan' - Extract grid metadata for routing
    Returns dict of properties for exploit selection
    """
    try:
        h, w = K_H(g), K_W(g)

        # Collect colors
        colors = set()
        for r in g:
            colors.update(r)
        colors = sorted(list(colors))

        # Basic symmetry checks (byte-cheap heuristics)
        sym_y = len(g) > 1 and g[0] == g[-1]
        sym_x = len(g) > 0 and [r[0] for r in g] == [r[-1] for r in g]

        # Hash for routing
        props_hash = (h * 31 + w) * 17 + len(colors)

        return {
            'h': h,
            'w': w,
            'size': f'{h}x{w}',
            'is_square': h == w,
            'colors': colors,
            'color_count': len(colors),
            'sym_y': sym_y,
            'sym_x': sym_x,
            'hash': props_hash & 0xFFFF
        }
    except:
        return {'error': True}

def analyze_delta(task):
    """
    Metadata delta analysis - infer transformation type
    by comparing input/output properties
    """
    try:
        if not task.get('train') or len(task['train']) == 0:
            return {}

        i_grid = task['train'][0]['input']
        o_grid = task['train'][0]['output']

        i_props = nmap_fingerprint(i_grid)
        o_props = nmap_fingerprint(o_grid)

        delta = {
            'size_changed': i_props['size'] != o_props['size'],
            'colors_changed': i_props['colors'] != o_props['colors'],
            'color_count_delta': o_props['color_count'] - i_props['color_count']
        }

        # Check for specific transformations
        if i_props['h'] == o_props['w'] and i_props['w'] == o_props['h']:
            delta['transpose_likely'] = True

        if i_props['size'] == o_props['size']:
            # Check if it's a simple reflection
            i_first_row = i_grid[0] if i_grid else []
            o_first_row = o_grid[0] if o_grid else []
            o_last_row = o_grid[-1] if o_grid else []

            if i_first_row == o_last_row:
                delta['y_reflection_likely'] = True
            if i_first_row == o_first_row[::-1]:
                delta['x_reflection_likely'] = True

        return delta
    except:
        return {}

# =====================================================
# AXIOM 2: EXPLOIT ROUTER - MAIN CLASSIFIER
# =====================================================

def fingerprint_problem(task):
    """
    'OS Fingerprinting' - Identify vulnerability (problem class)
    Returns exploit ID for payload routing
    """
    try:
        if not task.get('train') or len(task['train']) == 0:
            return 'unknown'

        i = task['train'][0]['input']
        o = task['train'][0]['output']

        i_fp = nmap_fingerprint(i)
        o_fp = nmap_fingerprint(o)
        delta = analyze_delta(task)

        # Exploit ID selection based on fingerprint

        # Reflection exploits
        if delta.get('y_reflection_likely'):
            return 'refl_y'
        if delta.get('x_reflection_likely'):
            return 'refl_x'

        # Rotation exploits
        if delta.get('transpose_likely'):
            return 'transpose'

        # Scaling exploits
        if i_fp['h'] * 2 == o_fp['h'] and i_fp['w'] * 2 == o_fp['w']:
            return 'scale_2x'
        if i_fp['h'] * 3 == o_fp['h'] and i_fp['w'] * 3 == o_fp['w']:
            return 'scale_3x'

        # Object finding exploits
        if o_fp['color_count'] < i_fp['color_count']:
            return 'extract_largest'

        # Default: copy input
        return 'copy'

    except:
        return 'copy'

# Payload Router - maps exploit IDs to solver functions
PAYLOAD_ROUTER = {
    'refl_y': p_refl_y,
    'refl_x': p_refl_x,
    'rot_90': p_rot_90,
    'rot_180': p_rot_180,
    'rot_270': p_rot_270,
    'transpose': p_transpose,
    'scale_2x': lambda g: p_scale_up(g, 2),
    'scale_3x': lambda g: p_scale_up(g, 3),
    'extract_largest': p_find_largest_obj,
    'copy': lambda g: [r[:] for r in g]
}

# =====================================================
# MAIN SOLVER - THE EXPLOIT HANDLER
# =====================================================

def solve(task):
    """
    Main exploit handler
    1. Fingerprint (Enumerate)
    2. Route to payload (Exploit)
    3. Execute solver
    """
    try:
        # 1. Fingerprint the problem
        exploit_id = fingerprint_problem(task)

        # 2. Get the payload
        payload = PAYLOAD_ROUTER.get(exploit_id, PAYLOAD_ROUTER['copy'])

        # 3. Execute on test input
        if 'test' in task and len(task['test']) > 0:
            test_input = task['test'][0]['input']
            result = payload(test_input)
            return result

        return task['test'][0]['input'] if 'test' in task else []

    except Exception as e:
        # Fail gracefully
        if 'test' in task and len(task['test']) > 0:
            return task['test'][0]['input']
        return []

# =====================================================
# KAGGLE SUBMISSION FORMAT
# =====================================================

def generate_submission(challenges_path: str, output_path: str):
    """
    Generate Kaggle submission JSON
    """
    with open(challenges_path, 'r') as f:
        challenges = json.load(f)

    submission = {}

    for task_id, task in challenges.items():
        try:
            # Solve the task
            solution = solve(task)

            # Convert to submission format
            submission[task_id] = {
                'attempt_1': solution,
                'attempt_2': solution  # Same solution for both attempts
            }
        except:
            # Failed - submit empty grid
            submission[task_id] = {
                'attempt_1': [[0]],
                'attempt_2': [[0]]
            }

    # Write submission
    with open(output_path, 'w') as f:
        json.dump(submission, f)

    print(f"Submission generated: {output_path}")
    return submission

# =====================================================
# TESTING & EVALUATION
# =====================================================

def evaluate_on_training(training_path: str, solutions_path: str, max_tasks: int = 10):
    """
    Test solver on training data with known solutions
    Returns accuracy
    """
    with open(training_path, 'r') as f:
        training = json.load(f)

    with open(solutions_path, 'r') as f:
        solutions = json.load(f)

    correct = 0
    total = 0

    for task_id, task in list(training.items())[:max_tasks]:
        try:
            # Get our solution
            predicted = solve(task)

            # Get correct solution
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

    accuracy = correct / total if total > 0 else 0
    print(f"\nAccuracy: {correct}/{total} = {accuracy:.1%}")
    return accuracy

# =====================================================
# MAIN EXECUTION
# =====================================================

if __name__ == "__main__":
    print("="*60)
    print("ARC-AGI 2025 Solver - HungryOrca")
    print("Based on ctf.txt 5-Axiom Framework")
    print("="*60)

    # Test on training data
    print("\n[1] Testing on training data...")
    try:
        evaluate_on_training(
            'arc-agi_training_challenges.json',
            'arc-agi_training_solutions.json',
            max_tasks=20
        )
    except FileNotFoundError as e:
        print(f"Training files not found: {e}")

    # Generate submission for evaluation set
    print("\n[2] Generating submission for evaluation set...")
    try:
        generate_submission(
            'arc-agi_evaluation_challenges.json',
            'arc_2025_submission.json'
        )
    except FileNotFoundError as e:
        print(f"Evaluation file not found: {e}")

    print("\n" + "="*60)
    print("Solver ready for deployment!")
    print("Next step: Convert to .ipynb and compress to <1MB")
    print("="*60)
