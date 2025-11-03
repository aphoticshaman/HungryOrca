#!/usr/bin/env python3
"""
TurboOrcav8.py - ARC Prize 2025 Submission Generator
Fixed to output correct Kaggle submission format
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import defaultdict, Counter
import sys

print("🐋 TurboOrca v8 - Kaggle Format Compliant Submission Generator")

# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """Configuration for TurboOrca v8"""
    # File paths
    test_challenges = 'arc-agi_test_challenges.json'
    eval_challenges = 'arc-agi_evaluation_challenges.json'
    train_challenges = 'arc-agi_training_challenges.json'
    train_solutions = 'arc-agi_training_solutions.json'

    output_submission = 'submission_v8.json'

    # Kaggle paths (auto-detect)
    is_kaggle = Path('/kaggle/input').exists()
    if is_kaggle:
        test_challenges = '/kaggle/input/arc-prize-2025/arc-agi_test_challenges.json'
        train_challenges = '/kaggle/input/arc-prize-2025/arc-agi_training_challenges.json'
        train_solutions = '/kaggle/input/arc-prize-2025/arc-agi_training_solutions.json'
        output_submission = '/kaggle/working/submission.json'

# =============================================================================
# SOLVER FUNCTIONS (v6 Data-Driven Big Three)
# =============================================================================

def detect_crop(task_data):
    """Detect if task is a crop pattern"""
    if not task_data.get('train'):
        return None

    background_color = None
    for pair in task_data['train']:
        input_grid = np.array(pair['input'])
        output_grid = np.array(pair['output'])

        # Output should be smaller or equal
        if output_grid.shape[0] > input_grid.shape[0] or output_grid.shape[1] > input_grid.shape[1]:
            return None

        # Find background color
        bg = np.argmax(np.bincount(input_grid.flatten()))
        if background_color is None:
            background_color = bg
        elif background_color != bg:
            return None

        # Check if output is cropped region
        mask = input_grid != bg
        if not mask.any():
            return None

        rows, cols = np.where(mask)
        cropped = input_grid[rows.min():rows.max()+1, cols.min():cols.max()+1]

        if not np.array_equal(output_grid, cropped):
            return None

    return {'type': 'crop', 'background': int(background_color)}

def apply_crop(test_input, params):
    """Apply crop transformation"""
    grid = np.array(test_input)
    mask = grid != params['background']

    if not mask.any():
        return test_input

    rows, cols = np.where(mask)
    cropped = grid[rows.min():rows.max()+1, cols.min():cols.max()+1]
    return cropped.tolist()

def detect_color_swap(task_data):
    """Detect if task is a color swap pattern"""
    if not task_data.get('train'):
        return None

    global_mapping = None
    for pair in task_data['train']:
        input_grid = np.array(pair['input'])
        output_grid = np.array(pair['output'])

        # Grids must be same shape
        if input_grid.shape != output_grid.shape:
            return None

        # Build color mapping
        mapping = {}
        for in_val, out_val in zip(input_grid.flatten(), output_grid.flatten()):
            if in_val in mapping:
                if mapping[in_val] != out_val:
                    return None
            else:
                mapping[in_val] = out_val

        # Check if mapping is not identity
        if all(k == v for k, v in mapping.items()):
            return None

        # Check consistency across examples
        if global_mapping is None:
            global_mapping = mapping
        elif mapping != global_mapping:
            return None

    return {'type': 'color_swap', 'mapping': {int(k): int(v) for k, v in global_mapping.items()}} if global_mapping else None

def apply_color_swap(test_input, params):
    """Apply color swap transformation"""
    grid = np.array(test_input)
    output = np.copy(grid)

    for from_color, to_color in params['mapping'].items():
        output[grid == from_color] = to_color

    return output.tolist()

def detect_pad(task_data):
    """Detect if task is a padding pattern"""
    if not task_data.get('train'):
        return None

    pad_params = []
    for pair in task_data['train']:
        input_grid = np.array(pair['input'])
        output_grid = np.array(pair['output'])

        # Output should be larger or equal
        if output_grid.shape[0] < input_grid.shape[0] or output_grid.shape[1] < input_grid.shape[1]:
            return None

        bg = np.argmax(np.bincount(output_grid.flatten()))

        # Find where input is placed in output
        found = False
        for r in range(output_grid.shape[0] - input_grid.shape[0] + 1):
            for c in range(output_grid.shape[1] - input_grid.shape[1] + 1):
                if np.array_equal(output_grid[r:r+input_grid.shape[0], c:c+input_grid.shape[1]], input_grid):
                    top, left = r, c
                    bottom = output_grid.shape[0] - input_grid.shape[0] - top
                    right = output_grid.shape[1] - input_grid.shape[1] - left
                    pad_params.append((top, bottom, left, right, bg))
                    found = True
                    break
            if found:
                break

        if not found:
            return None

    # Check consistency
    if not all(p == pad_params[0] for p in pad_params):
        return None

    top, bottom, left, right, bg = pad_params[0]
    return {'type': 'pad', 'top': top, 'bottom': bottom, 'left': left, 'right': right, 'background': int(bg)}

def apply_pad(test_input, params):
    """Apply padding transformation"""
    grid = np.array(test_input)
    output = np.pad(
        grid,
        ((params['top'], params['bottom']), (params['left'], params['right'])),
        mode='constant',
        constant_values=params['background']
    )
    return output.tolist()

# =============================================================================
# FALLBACK SOLVERS
# =============================================================================

def identity_solver(test_input):
    """Return input as-is"""
    return test_input

def most_common_output(task_data):
    """Return the most common output from training examples"""
    if not task_data.get('train'):
        return None

    outputs = [pair['output'] for pair in task_data['train'] if 'output' in pair]
    if outputs:
        # Return first output as fallback
        return outputs[0]
    return None

# =============================================================================
# MAIN SOLVER PIPELINE
# =============================================================================

def solve_task(task_id, task_data):
    """
    Solve a single ARC task
    Returns: (attempt_1_grid, attempt_2_grid)
    """

    # Try detection functions in order
    detectors = [
        (detect_crop, apply_crop),
        (detect_color_swap, apply_color_swap),
        (detect_pad, apply_pad)
    ]

    for detect_fn, apply_fn in detectors:
        params = detect_fn(task_data)
        if params is not None:
            # Found a pattern!
            try:
                test_input = task_data['test'][0]['input']
                attempt_1 = apply_fn(test_input, params)

                # For attempt 2, try a slight variation or same
                # TODO: implement alternative strategies
                attempt_2 = apply_fn(test_input, params)

                return attempt_1, attempt_2
            except Exception as e:
                print(f"  Error applying {params['type']}: {e}")
                continue

    # No pattern detected - use fallbacks
    test_input = task_data['test'][0]['input']
    fallback_output = most_common_output(task_data)

    if fallback_output:
        return fallback_output, fallback_output
    else:
        # Ultimate fallback: return input
        return test_input, test_input

# =============================================================================
# SUBMISSION GENERATOR
# =============================================================================

def generate_submission(test_challenges_path, output_path):
    """
    Generate submission file in correct Kaggle format

    Correct format:
    {
      "task_id": [{
        "attempt_1": [[grid]],
        "attempt_2": [[grid]]
      }]
    }
    """

    print(f"Loading test challenges from: {test_challenges_path}")
    with open(test_challenges_path, 'r') as f:
        test_challenges = json.load(f)

    print(f"Found {len(test_challenges)} test tasks")

    submission = {}
    stats = defaultdict(int)

    for task_id, task_data in test_challenges.items():
        try:
            attempt_1, attempt_2 = solve_task(task_id, task_data)

            # CORRECT KAGGLE FORMAT
            submission[task_id] = [{
                "attempt_1": attempt_1,
                "attempt_2": attempt_2
            }]

            # Track which solver was used
            if attempt_1 == attempt_2:
                stats['identical_attempts'] += 1
            else:
                stats['different_attempts'] += 1

            stats['total_tasks'] += 1

            if stats['total_tasks'] % 50 == 0:
                print(f"  Processed {stats['total_tasks']}/{len(test_challenges)} tasks...")

        except Exception as e:
            print(f"ERROR on task {task_id}: {e}")
            # Create fallback entry
            submission[task_id] = [{
                "attempt_1": [[0]],
                "attempt_2": [[0]]
            }]
            stats['errors'] += 1

    # Save submission
    print(f"\nSaving submission to: {output_path}")
    with open(output_path, 'w') as f:
        json.dump(submission, f)

    # Print statistics
    print("\n" + "="*80)
    print("SUBMISSION GENERATION COMPLETE")
    print("="*80)
    print(f"Total tasks:          {stats['total_tasks']}")
    print(f"Identical attempts:   {stats['identical_attempts']}")
    print(f"Different attempts:   {stats['different_attempts']}")
    print(f"Errors:               {stats['errors']}")
    print(f"\nOutput file:          {output_path}")
    print(f"File size:            {Path(output_path).stat().st_size / 1024:.2f} KB")
    print("="*80)

    # Validate format
    print("\nValidating format...")
    validate_submission_format(submission)

    return submission

def validate_submission_format(submission):
    """Validate submission matches Kaggle format"""
    print("\nFormat Validation:")
    print("-"*80)

    errors = []

    for task_id, task_data in list(submission.items())[:3]:  # Check first 3
        if not isinstance(task_data, list):
            errors.append(f"{task_id}: task_data is not a list")
            continue

        if len(task_data) != 1:
            errors.append(f"{task_id}: task_data should have exactly 1 item")
            continue

        attempts_dict = task_data[0]
        if not isinstance(attempts_dict, dict):
            errors.append(f"{task_id}: attempts should be a dict")
            continue

        if 'attempt_1' not in attempts_dict or 'attempt_2' not in attempts_dict:
            errors.append(f"{task_id}: missing attempt_1 or attempt_2 keys")
            continue

        print(f"✓ {task_id}: Format correct")

    if errors:
        print("\n❌ Format errors found:")
        for error in errors:
            print(f"  - {error}")
        return False
    else:
        print("\n✓ Format validation passed!")
        return True

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    config = Config()

    print("\nTurboOrca v8 Configuration:")
    print(f"  Running on: {'Kaggle' if config.is_kaggle else 'Local'}")
    print(f"  Test challenges: {config.test_challenges}")
    print(f"  Output: {config.output_submission}")
    print()

    # Generate submission
    submission = generate_submission(config.test_challenges, config.output_submission)

    print("\n✅ TurboOrca v8 complete!")
    print(f"✅ Submission ready at: {config.output_submission}")
    print(f"✅ Ready for Kaggle upload!")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
