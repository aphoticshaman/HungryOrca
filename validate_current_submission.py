#!/usr/bin/env python3
"""Validate current submission.json to find near-perfect tasks."""

import json
import numpy as np
from collections import Counter

# Load data
with open('arc-agi_evaluation_challenges.json') as f:
    eval_challenges = json.load(f)

with open('arc-agi_evaluation_solutions.json') as f:
    eval_solutions = json.load(f)

with open('submission.json') as f:
    submission = json.load(f)

print("="*80)
print("SUBMISSION VALIDATION - Finding Near-Perfect Tasks")
print("="*80)

scores = []
near_perfect = []  # 90-99%
perfect = []  # 100%

for task_id in eval_solutions.keys():
    if task_id not in submission:
        continue
    
    # Get ground truth
    solution = eval_solutions[task_id]
    
    # Get our predictions (2 attempts)
    predictions = submission[task_id]
    
    # Check each test case
    for test_idx, truth in enumerate(solution):
        if test_idx >= len(predictions):
            continue
        
        pred = predictions[test_idx]
        
        # Calculate similarity
        truth_arr = np.array(truth)
        pred_arr = np.array(pred)
        
        if truth_arr.shape != pred_arr.shape:
            similarity = 0.0
        else:
            matching = np.sum(truth_arr == pred_arr)
            total = truth_arr.size
            similarity = matching / total if total > 0 else 0.0
        
        scores.append({
            'task_id': task_id,
            'test_idx': test_idx,
            'similarity': similarity
        })
        
        # Categorize
        if similarity == 1.0:
            perfect.append((task_id, test_idx))
        elif 0.90 <= similarity < 1.0:
            near_perfect.append((task_id, test_idx, similarity))

# Sort near-perfect by similarity (descending)
near_perfect.sort(key=lambda x: x[2], reverse=True)

print(f"\n✅ Perfect (100%): {len(perfect)}")
print(f"🎯 Near-Perfect (90-99%): {len(near_perfect)}")
print(f"📊 Total evaluated: {len(scores)}")

if perfect:
    print(f"\nPerfect matches:")
    for task_id, idx in perfect[:5]:
        print(f"  - {task_id} (test {idx})")

if near_perfect:
    print(f"\nNear-perfect tasks (top 10):")
    for task_id, idx, sim in near_perfect[:10]:
        print(f"  - {task_id} (test {idx}): {sim*100:.1f}%")
    
    # Save for analysis
    with open('near_perfect_tasks.json', 'w') as f:
        json.dump({
            'near_perfect': [
                {'task_id': t[0], 'test_idx': t[1], 'similarity': t[2]}
                for t in near_perfect
            ]
        }, f, indent=2)
    
    print(f"\n💾 Saved {len(near_perfect)} near-perfect tasks to near_perfect_tasks.json")

# Overall stats
similarities = [s['similarity'] for s in scores]
print(f"\n📈 Overall Statistics:")
print(f"   Mean: {np.mean(similarities)*100:.1f}%")
print(f"   Median: {np.median(similarities)*100:.1f}%")
print(f"   Std: {np.std(similarities)*100:.1f}%")
