#!/usr/bin/env python3
"""
Quick baseline test on 10 ARC tasks to validate system works.
"""

import json
import numpy as np
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'SubtleGenius' / 'primitives'))

from symbolic_solver import SymbolicProgramSynthesizer, SearchStrategy, GridState

# Load first 10 tasks
with open('arc-agi_training_challenges.json', 'r') as f:
    challenges = json.load(f)

with open('arc-agi_training_solutions.json', 'r') as f:
    solutions = json.load(f)

task_ids = list(challenges.keys())[:10]

print("="*80)
print("BASELINE TEST: NSPSA on 10 ARC Tasks")
print("="*80)
print(f"\nTesting {len(task_ids)} tasks\n")

synthesizer = SymbolicProgramSynthesizer(max_program_length=4, beam_width=5)

results = []
for i, task_id in enumerate(task_ids):
    print(f"[{i+1}/10] Task {task_id}")

    task = challenges[task_id]
    solution = solutions.get(task_id, {})

    train_pairs = [(np.array(ex['input']), np.array(ex['output']))
                   for ex in task['train']]

    # Try to find program on first training example
    if train_pairs:
        inp, out = train_pairs[0]

        start = time.time()
        program = synthesizer.synthesize(inp, out, timeout=5.0)
        elapsed = time.time() - start

        if program:
            # Validate on test
            correct = 0
            total = 0

            for test_ex in task['test']:
                test_inp = np.array(test_ex['input'])
                expected = solution.get('test', [{}])[total].get('output')

                if expected is not None:
                    state = GridState.from_array(test_inp)
                    for prim in program:
                        state = synthesizer.executor.execute(state, prim)
                        if state is None:
                            break

                    if state and np.array_equal(state.to_array(), np.array(expected)):
                        correct += 1

                total += 1

            accuracy = correct / total if total > 0 else 0
            print(f"  Program: {program}")
            print(f"  Test accuracy: {correct}/{total} = {100*accuracy:.0f}%")
            print(f"  Time: {elapsed:.2f}s")

            results.append({
                'task_id': task_id,
                'solved': accuracy > 0,
                'accuracy': accuracy,
                'program': program,
                'time': elapsed
            })
        else:
            print(f"  ❌ No program found")
            results.append({
                'task_id': task_id,
                'solved': False,
                'accuracy': 0,
                'program': None,
                'time': elapsed
            })

print("\n" + "="*80)
print("SUMMARY")
print("="*80)

solved = sum(1 for r in results if r['solved'])
total_tasks = len(results)

print(f"Tasks solved: {solved}/{total_tasks} = {100*solved/total_tasks:.0f}%")
print(f"Avg accuracy: {100*np.mean([r['accuracy'] for r in results]):.1f}%")
print(f"Avg time: {np.mean([r['time'] for r in results]):.2f}s")

print("\nBASELINE VALIDATION:", "✅ PASS" if solved >= 3 else "⚠️ NEEDS WORK")
