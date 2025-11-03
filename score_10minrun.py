#!/usr/bin/env python3
"""Score submission_10minrun.json against evaluation solutions"""

import json

# Load submission (test set)
with open('submission_10minrun.json') as f:
    submission = json.load(f)

print("Submission file: submission_10minrun.json")
print(f"Total tasks in submission: {len(submission)}")
print()

# Check if we have evaluation solutions
try:
    with open('arc-agi_evaluation_challenges.json') as f:
        eval_challenges = json.load(f)
    with open('arc-agi_evaluation_solutions.json') as f:
        eval_solutions = json.load(f)

    print(f"Evaluation tasks: {len(eval_challenges)}")

    # Find overlap
    submission_ids = set(submission.keys())
    eval_ids = set(eval_solutions.keys())
    overlap = submission_ids & eval_ids

    print(f"Overlap between submission and evaluation: {len(overlap)} tasks")
    print()

    if len(overlap) > 0:
        print("SCORING ON EVALUATION SET:")
        print("="*80)

        correct_attempt_1 = 0
        correct_attempt_2 = 0
        correct_either = 0
        total = 0

        for task_id in overlap:
            ground_truth = eval_solutions[task_id]
            attempts = submission[task_id]

            # Handle different formats
            if isinstance(attempts, list):
                if len(attempts) > 0 and isinstance(attempts[0], dict) and 'attempt_1' in attempts[0]:
                    # Dict format
                    attempt_1 = attempts[0]['attempt_1']
                    attempt_2 = attempts[0]['attempt_2']
                else:
                    # Simple list format
                    attempt_1 = attempts[0] if len(attempts) > 0 else []
                    attempt_2 = attempts[1] if len(attempts) > 1 else []

                # Compare with first test output
                if len(ground_truth) > 0:
                    expected = ground_truth[0]
                    match_1 = (attempt_1 == expected)
                    match_2 = (attempt_2 == expected)

                    if match_1:
                        correct_attempt_1 += 1
                        correct_either += 1
                    elif match_2:
                        correct_attempt_2 += 1
                        correct_either += 1

                    total += 1

        if total > 0:
            print(f"Total tasks scored: {total}")
            print(f"Correct on attempt 1: {correct_attempt_1} ({correct_attempt_1/total*100:.2f}%)")
            print(f"Correct on attempt 2: {correct_attempt_2} ({correct_attempt_2/total*100:.2f}%)")
            print(f"Correct on either:    {correct_either} ({correct_either/total*100:.2f}%)")
            print()
            print(f"FINAL SCORE: {correct_either/total*100:.2f}%")
    else:
        print("❌ No overlap - submission is for TEST set only (no ground truth available)")

except FileNotFoundError as e:
    print(f"❌ Error: {e}")
    print("Cannot score - evaluation solutions not available")
