#!/usr/bin/env python3
"""
Fix ARC Prize 2025 submission format: LIST → DICT

WRONG (list format):
[
  {"task_id": "00576224", "attempt_1": [[3,2]], "attempt_2": [[7,8]]},
  {"task_id": "009d5c81", "attempt_1": [[1,1]], "attempt_2": [[2,2]]}
]

CORRECT (dict format):
{
  "00576224": [{"attempt_1": [[3,2]], "attempt_2": [[7,8]]}],
  "009d5c81": [{"attempt_1": [[1,1]], "attempt_2": [[2,2]]}]
}
"""

import json
import sys
from pathlib import Path

def fix_submission_format(input_path, output_path=None):
    """Convert list format submission to dict format"""

    # Read input
    print(f"Reading: {input_path}")
    with open(input_path, 'r') as f:
        data = json.load(f)

    # Check current format
    if isinstance(data, dict):
        print("✓ Already in DICT format - no conversion needed")
        if output_path and output_path != input_path:
            with open(output_path, 'w') as f:
                json.dump(data, f, separators=(',', ':'))
            print(f"✓ Copied to: {output_path}")
        return

    if not isinstance(data, list):
        print(f"✗ ERROR: Unknown format type: {type(data)}")
        sys.exit(1)

    # Convert LIST → DICT
    print(f"Converting LIST → DICT ({len(data)} tasks)")
    submission = {}

    for item in data:
        if 'task_id' not in item:
            print(f"✗ ERROR: Missing task_id in entry: {item}")
            continue

        task_id = item['task_id']

        # Create dict entry with list of attempts
        submission[task_id] = [{
            'attempt_1': item.get('attempt_1', [[0]]),
            'attempt_2': item.get('attempt_2', [[0]])
        }]

    # Save output
    output_path = output_path or input_path.replace('.json', '_fixed.json')
    with open(output_path, 'w') as f:
        json.dump(submission, f, separators=(',', ':'))

    size_kb = Path(output_path).stat().st_size / 1024
    print(f"✓ Fixed format saved to: {output_path}")
    print(f"  Tasks: {len(submission)}")
    print(f"  Size: {size_kb:.1f} KB")
    print(f"  Format: DICT (correct for ARC Prize 2025)")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python fix_submission_format.py <input.json> [output.json]")
        print("\nExample:")
        print("  python fix_submission_format.py 'submission (1).json' submission.json")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None

    fix_submission_format(input_file, output_file)
