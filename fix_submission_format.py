#!/usr/bin/env python3
"""
Fix submission.json to match Kaggle ARC Prize 2025 format requirements
"""

import json
import sys

def fix_submission_format(input_file, output_file):
    """
    Convert submission from simple list format to Kaggle required dict format.

    Input format:  {"task_id": [grid1, grid2]}
    Output format: {"task_id": [{"attempt_1": grid1, "attempt_2": grid2}]}
    """

    print("="*80)
    print("FIXING SUBMISSION FORMAT FOR KAGGLE")
    print("="*80)
    print()

    # Load current submission
    print(f"Loading: {input_file}")
    with open(input_file, 'r') as f:
        submission = json.load(f)

    print(f"Tasks found: {len(submission)}")

    # Convert format
    fixed_submission = {}
    errors = []

    for task_id, attempts in submission.items():
        if not isinstance(attempts, list):
            errors.append(f"Task {task_id}: attempts is not a list")
            continue

        if len(attempts) < 2:
            errors.append(f"Task {task_id}: only {len(attempts)} attempts (expected 2)")
            # Pad with empty grid if needed
            while len(attempts) < 2:
                attempts.append([[0, 0], [0, 0]])

        # Check if already in correct format
        if isinstance(attempts[0], dict) and 'attempt_1' in attempts[0]:
            # Already in correct format
            fixed_submission[task_id] = attempts
        else:
            # Convert to correct format
            fixed_submission[task_id] = [{
                "attempt_1": attempts[0],
                "attempt_2": attempts[1] if len(attempts) > 1 else attempts[0]
            }]

    print()
    print("CONVERSION RESULTS:")
    print("-"*80)
    print(f"Tasks processed: {len(fixed_submission)}")
    print(f"Errors: {len(errors)}")

    if errors:
        print("\nErrors encountered:")
        for error in errors[:10]:  # Show first 10 errors
            print(f"  - {error}")

    # Save fixed submission
    print()
    print(f"Saving fixed submission to: {output_file}")
    with open(output_file, 'w') as f:
        json.dump(fixed_submission, f)

    print()
    print("✓ Format fixed successfully!")
    print()

    # Verify against sample format
    print("VERIFICATION:")
    print("-"*80)

    sample_path = input_file.replace('submission.json', 'sample_submission.json')
    try:
        with open(sample_path, 'r') as f:
            sample = json.load(f)

        sample_task_id = list(sample.keys())[0]
        sample_format = sample[sample_task_id]

        fixed_task_id = list(fixed_submission.keys())[0]
        fixed_format = fixed_submission[fixed_task_id]

        print(f"Sample format structure: {type(sample_format)} with {len(sample_format)} items")
        if sample_format:
            print(f"  First item type: {type(sample_format[0])}")
            if isinstance(sample_format[0], dict):
                print(f"  Keys: {list(sample_format[0].keys())}")

        print()
        print(f"Fixed format structure: {type(fixed_format)} with {len(fixed_format)} items")
        if fixed_format:
            print(f"  First item type: {type(fixed_format[0])}")
            if isinstance(fixed_format[0], dict):
                print(f"  Keys: {list(fixed_format[0].keys())}")

        print()
        if (isinstance(sample_format, list) and isinstance(fixed_format, list) and
            len(sample_format) > 0 and len(fixed_format) > 0 and
            isinstance(sample_format[0], dict) and isinstance(fixed_format[0], dict) and
            set(sample_format[0].keys()) == set(fixed_format[0].keys())):
            print("✓ Format matches sample_submission.json!")
        else:
            print("⚠ Format may not match sample_submission.json exactly")

    except Exception as e:
        print(f"Could not verify against sample: {e}")

    print()
    print("="*80)
    print("DONE")
    print("="*80)

if __name__ == "__main__":
    input_file = "/home/user/HungryOrca/submission.json"
    output_file = "/home/user/HungryOrca/submission_fixed.json"

    fix_submission_format(input_file, output_file)
