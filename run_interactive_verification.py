#!/usr/bin/env python3
"""
Use Interactive Verification Framework to push 90-99% solutions to 100%

Pipeline:
1. Load existing submission.json (baseline predictions)
2. Identify near-perfect tasks (90-99% similarity on training)
3. Apply InteractiveVerificationSystem to refine each
4. Generate verified_submission.json with improved results
"""

import json
import numpy as np
import sys
from pathlib import Path

# Import the interactive verification framework
sys.path.insert(0, str(Path(__file__).parent))
from interactive_verification_framework import (
    InteractiveVerificationSystem,
    VerificationMode
)

print("="*80)
print("INTERACTIVE VERIFICATION: 90% → 100% REFINEMENT")
print("="*80)

# Load training data for constraint extraction
print("\n📂 Loading training data...")
with open('arc-agi_training_challenges.json') as f:
    train_challenges = json.load(f)

with open('arc-agi_training_solutions.json') as f:
    train_solutions = json.load(f)

print(f"✅ Loaded {len(train_challenges)} training tasks")

# Load current submission
print("\n📂 Loading current submission...")
with open('submission.json') as f:
    submission = json.load(f)

print(f"✅ Loaded submission with {len(submission)} tasks")

# Initialize verification system
print("\n🔬 Initializing Interactive Verification System...")
verifier = InteractiveVerificationSystem(mode=VerificationMode.ACTIVE)

# Test on a few training tasks first to validate the approach
print("\n🎯 Testing verification on sample training tasks...")

test_task_ids = [
    '00d62c1b',  # Mentioned in reality check (91.8% baseline)
    '009d5c81',  # Shape-preserving
    '00576224', # Non-shape-preserving
]

improved_count = 0
total_tested = 0

for task_id in test_task_ids:
    if task_id not in train_challenges or task_id not in train_solutions:
        continue
    
    total_tested += 1
    task = train_challenges[task_id]
    solutions = train_solutions[task_id]
    
    print(f"\n{'='*70}")
    print(f"Task: {task_id}")
    print(f"{'='*70}")
    
    # Extract training pairs
    train_pairs = [
        (np.array(pair['input']), np.array(pair['output']))
        for pair in task['train']
    ]
    
    # For each test case
    for test_idx, test_pair in enumerate(task['test']):
        test_input = np.array(test_pair['input'])
        ground_truth = np.array(solutions[test_idx])
        
        # Get our prediction from submission (if exists)
        if task_id in submission and test_idx < len(submission[task_id]):
            hypothesis = np.array(submission[task_id][test_idx])
        else:
            # Skip if no prediction
            print(f"  Test {test_idx}: No prediction available")
            continue
        
        # Calculate baseline similarity
        if hypothesis.shape != ground_truth.shape:
            baseline_sim = 0.0
        else:
            baseline_sim = np.sum(hypothesis == ground_truth) / ground_truth.size
        
        print(f"\n  Test {test_idx}:")
        print(f"    Baseline similarity: {baseline_sim*100:.1f}%")
        
        # Skip if already perfect
        if baseline_sim == 1.0:
            print(f"    ✅ Already perfect!")
            continue
        
        # Skip if too low (< 80%)
        if baseline_sim < 0.80:
            print(f"    ⏭️  Too low for interactive refinement")
            continue
        
        # Apply interactive verification
        try:
            verified_solution, confidence, proof_trace = verifier.verify_solution(
                hypothesis=hypothesis,
                training_pairs=train_pairs,
                test_input=test_input
            )
            
            # Calculate improved similarity
            if verified_solution.shape != ground_truth.shape:
                improved_sim = 0.0
            else:
                improved_sim = np.sum(verified_solution == ground_truth) / ground_truth.size
            
            print(f"    After verification: {improved_sim*100:.1f}%")
            print(f"    Confidence: {confidence*100:.1f}%")
            print(f"    Proof steps: {len(proof_trace)}")
            
            if improved_sim > baseline_sim:
                print(f"    🎉 IMPROVED: +{(improved_sim-baseline_sim)*100:.1f}%")
                improved_count += 1
            elif improved_sim == baseline_sim and confidence > 0.95:
                print(f"    ✅ VERIFIED: High confidence")
            else:
                print(f"    ⚠️  No improvement")
        
        except Exception as e:
            print(f"    ❌ Verification failed: {e}")

print(f"\n{'='*80}")
print(f"SUMMARY")
print(f"{'='*80}")
print(f"Tasks tested: {total_tested}")
print(f"Solutions improved: {improved_count}")

if improved_count > 0:
    print(f"\n✅ Interactive verification is working!")
    print(f"📊 Ready to apply to full submission")
else:
    print(f"\n⚠️  No improvements found on test set")
    print(f"💡 May need to tune verification parameters")

