#!/usr/bin/env python3
"""
Convert submission.json from LIST format to DICT format (Kaggle requirement)

OLD (list format):
[
  {"task_id": "00576224", "attempt_1": [[...]], "attempt_2": [[...]]},
  {"task_id": "007bbfb7", "attempt_1": [[...]], "attempt_2": [[...]]}
]

NEW (dict format - Kaggle compliant):
{
  "00576224": [{"attempt_1": [[...]], "attempt_2": [[...]]}],
  "007bbfb7": [{"attempt_1": [[...]], "attempt_2": [[...]]}]
}
"""

import json
import sys

def convert_list_to_dict_format(input_path, output_path):
    """Convert submission from list to dict format"""

    print(f"Loading: {input_path}")
    with open(input_path, 'r') as f:
        old_submission = json.load(f)

    # Check format
    if isinstance(old_submission, dict):
        print("✓ Already in dict format!")
        return

    if not isinstance(old_submission, list):
        print(f"ERROR: Expected list or dict, got {type(old_submission)}")
        return

    print(f"Converting {len(old_submission)} tasks from LIST to DICT format...")

    # Convert
    new_submission = {}

    for item in old_submission:
        task_id = item.get('task_id')
        attempt_1 = item.get('attempt_1')
        attempt_2 = item.get('attempt_2')

        if not task_id:
            print(f"Warning: Skipping item without task_id: {item}")
            continue

        # Each task maps to a list of attempts (one per test item)
        # Since old format had one prediction per task, we create a list with one element
        new_submission[task_id] = [{
            "attempt_1": attempt_1,
            "attempt_2": attempt_2
        }]

    print(f"✓ Converted {len(new_submission)} tasks")

    # Save
    print(f"Saving to: {output_path}")
    with open(output_path, 'w') as f:
        json.dump(new_submission, f, indent=2)

    # Verify
    with open(output_path, 'r') as f:
        verify = json.load(f)

    print(f"\n✓ Verification:")
    print(f"  Format: {'DICT ✓' if isinstance(verify, dict) else 'LIST ✗'}")
    print(f"  Tasks: {len(verify)}")

    # Show sample
    sample_task_id = list(verify.keys())[0]
    sample_task = verify[sample_task_id]
    print(f"\n  Sample task '{sample_task_id}':")
    print(f"    Type: {type(sample_task)}")
    print(f"    Length: {len(sample_task)} test item(s)")
    print(f"    First attempt shape: {len(sample_task[0]['attempt_1'])}x{len(sample_task[0]['attempt_1'][0]) if sample_task[0]['attempt_1'] else 0}")

    print(f"\n✅ Conversion complete!")

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python convert_submission_format.py <input.json> <output.json>")
        print("\nExample:")
        print("  python convert_submission_format.py ultv3_submission.json ultv3_submission_FIXED.json")
        sys.exit(1)

    convert_list_to_dict_format(sys.argv[1], sys.argv[2])
