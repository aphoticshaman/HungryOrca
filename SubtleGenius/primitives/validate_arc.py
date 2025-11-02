"""
Validate NSPSA on real ARC-AGI tasks

Load actual ARC training tasks and test NSPSA's performance.
"""

import json
import numpy as np
from nspsa import NSPSA
from orcawhiskey_nspsa_bridge import ExtendedEpistemicOrchestrator

def load_arc_tasks(filepath: str, max_tasks: int = 10):
    """Load ARC tasks from JSON"""
    with open(filepath, 'r') as f:
        data = json.load(f)

    tasks = []
    for task_id, task_data in list(data.items())[:max_tasks]:
        tasks.append({
            'id': task_id,
            'train': task_data['train'],
            'test': task_data['test']
        })

    return tasks


def evaluate_nspsa_on_arc(tasks_filepath: str, num_tasks: int = 20):
    """Evaluate NSPSA on ARC tasks"""

    print("="*70)
    print("NSPSA VALIDATION ON ARC-AGI TASKS")
    print("="*70)

    # Load tasks
    print(f"\nLoading tasks from: {tasks_filepath}")
    tasks = load_arc_tasks(tasks_filepath, max_tasks=num_tasks)
    print(f"Loaded {len(tasks)} tasks")

    # Initialize NSPSA
    nspsa = NSPSA(latent_dim=128)

    # Statistics
    num_solved = 0
    num_attempted = 0
    num_with_programs = 0

    results = []

    print(f"\n{'='*70}")
    print("EVALUATING TASKS...")
    print(f"{'='*70}\n")

    for i, task in enumerate(tasks):
        task_id = task['id']
        train_data = task['train']
        test_data = task['test']

        print(f"Task {i+1}/{len(tasks)}: {task_id}")

        # Prepare training data
        train_inputs = [np.array(pair['input']) for pair in train_data]
        train_outputs = [np.array(pair['output']) for pair in train_data]

        print(f"  Training examples: {len(train_inputs)}")

        # Evaluate on each test case
        task_results = []

        for test_idx, test_pair in enumerate(test_data):
            test_input = np.array(test_pair['input'])

            num_attempted += 1

            # Solve with NSPSA
            pred, trace = nspsa.solve(
                train_inputs,
                train_outputs,
                test_input,
                return_trace=True
            )

            # Check if program found
            if pred is not None:
                num_with_programs += 1
                print(f"  Test {test_idx}: ✅ PROGRAM FOUND - {trace['selected_program']}")

                task_results.append({
                    'test_idx': test_idx,
                    'program': trace['selected_program'],
                    'has_program': True
                })
            else:
                print(f"  Test {test_idx}: ⚠️  NO PROGRAM")
                task_results.append({
                    'test_idx': test_idx,
                    'program': None,
                    'has_program': False
                })

        results.append({
            'task_id': task_id,
            'test_results': task_results
        })

        print()

    # Summary statistics
    print("="*70)
    print("VALIDATION RESULTS")
    print("="*70)

    coverage = num_with_programs / max(1, num_attempted)

    print(f"\nTotal test cases: {num_attempted}")
    print(f"Programs synthesized: {num_with_programs}")
    print(f"\nCoverage (program synthesis rate): {coverage:.1%}")

    # NSPSA statistics
    nspsa_stats = nspsa.get_stats()
    print(f"\nNSPSA Statistics:")
    print(f"  Average search time: {nspsa_stats['avg_search_time']:.3f}s")

    print("="*70)

    return {
        'num_attempted': num_attempted,
        'num_with_programs': num_with_programs,
        'coverage': coverage,
        'results': results
    }


if __name__ == '__main__':
    # Validate on ARC training tasks
    results = evaluate_nspsa_on_arc(
        '/home/user/HungryOrca/arc-agi_training_challenges.json',
        num_tasks=20
    )

    print("\n" + "="*70)
    print("KEY INSIGHTS:")
    print("="*70)
    print(f"NSPSA is designed for tasks solvable by primitive transformations.")
    print(f"Coverage of {results['coverage']:.1%} shows which ARC tasks match this category.")
    print("\nFor complex multi-step transformations:")
    print("  → Combine with OrcaWhiskey's neural agents (A, B, VAE)")
    print("  → NSPSA handles symbolic, others handle complex patterns")
    print("="*70)
