"""
ROUND 3.1: Collect Baseline Search Statistics

Purpose: Understand current search behavior before optimization
Goal: Identify bottlenecks and opportunities for pruning
"""

import numpy as np
import json
import sys
import time
from typing import List, Dict, Tuple
sys.path.insert(0, '/home/user/HungryOrca/SubtleGenius/primitives')

from symbolic_solver import SymbolicProgramSynthesizer, SearchStrategy


def generate_test_tasks(n: int = 100) -> List[Tuple[str, np.ndarray, np.ndarray]]:
    """
    Generate diverse test tasks across different transform types

    Returns:
        List of (task_name, input_grid, output_grid) tuples
    """
    tasks = []

    # 1-step transforms (40 tasks)
    for _ in range(10):
        size = np.random.choice([2, 3, 4])
        inp = np.random.randint(0, 10, (size, size))

        # Rotate
        out = np.rot90(inp, k=-1)
        tasks.append(('1-step-rotate', inp, out))

        # Reflect
        out = np.flip(inp, axis=0)
        tasks.append(('1-step-reflect', inp, out))

        # Transpose
        out = np.transpose(inp)
        tasks.append(('1-step-transpose', inp, out))

        # Scale
        out = np.kron(inp, np.ones((2, 2), dtype=int))
        tasks.append(('1-step-scale', inp, out))

    # 2-step transforms (30 tasks)
    for _ in range(10):
        size = np.random.choice([2, 3, 4])
        inp = np.random.randint(0, 10, (size, size))

        # Rotate + reflect
        out = np.flip(np.rot90(inp, k=-1), axis=1)
        tasks.append(('2-step-rotate-reflect', inp, out))

        # Reflect + transpose
        out = np.transpose(np.flip(inp, axis=0))
        tasks.append(('2-step-reflect-transpose', inp, out))

        # Rotate + rotate
        out = np.rot90(inp, k=2)
        tasks.append(('2-step-rotate-rotate', inp, out))

    # 3-step transforms (20 tasks)
    for _ in range(20):
        size = np.random.choice([2, 3])
        inp = np.random.randint(0, 10, (size, size))

        # Rotate + reflect + transpose
        out = np.transpose(np.flip(np.rot90(inp, k=-1), axis=0))
        tasks.append(('3-step-complex', inp, out))

    # Identity (10 tasks) - should be instant
    for _ in range(10):
        size = np.random.choice([2, 3, 4])
        inp = np.random.randint(0, 10, (size, size))
        tasks.append(('identity', inp, inp))

    return tasks


def collect_statistics(synthesizer: SymbolicProgramSynthesizer,
                       tasks: List[Tuple[str, np.ndarray, np.ndarray]]) -> Dict:
    """
    Run synthesizer on tasks and collect detailed statistics

    Args:
        synthesizer: The program synthesizer
        tasks: List of (name, input, output) tuples

    Returns:
        Dictionary of statistics
    """
    stats = {
        'total_tasks': len(tasks),
        'solved': 0,
        'unsolved': 0,
        'timeouts': 0,
        'states_explored': [],
        'search_times': [],
        'program_lengths': [],
        'by_task_type': {}
    }

    print(f"\nRunning {len(tasks)} tasks...")
    print("=" * 70)

    for i, (task_type, inp, out) in enumerate(tasks):
        # Reset synthesizer stats
        synthesizer.stats = {'states_explored': 0, 'programs_found': 0, 'search_time': 0.0}

        # Run synthesis
        start = time.time()
        program = synthesizer.synthesize(inp, out, strategy=SearchStrategy.BIDIRECTIONAL, timeout=2.0)
        elapsed = time.time() - start

        # Collect results
        if program is not None:
            stats['solved'] += 1
            stats['states_explored'].append(synthesizer.stats['states_explored'])
            stats['search_times'].append(elapsed)
            stats['program_lengths'].append(len(program))

            # By task type
            if task_type not in stats['by_task_type']:
                stats['by_task_type'][task_type] = {
                    'count': 0,
                    'solved': 0,
                    'avg_states': 0,
                    'avg_time': 0,
                    'states': []
                }

            stats['by_task_type'][task_type]['count'] += 1
            stats['by_task_type'][task_type]['solved'] += 1
            stats['by_task_type'][task_type]['states'].append(synthesizer.stats['states_explored'])

            if (i + 1) % 10 == 0:
                print(f"  Task {i+1:3d}/{len(tasks)}: {task_type:20s} | "
                      f"States: {synthesizer.stats['states_explored']:4d} | "
                      f"Time: {elapsed:.3f}s | "
                      f"Length: {len(program)}")
        else:
            stats['unsolved'] += 1
            if elapsed >= 1.9:  # Close to timeout
                stats['timeouts'] += 1

    # Compute aggregates
    if stats['states_explored']:
        stats['avg_states'] = np.mean(stats['states_explored'])
        stats['max_states'] = np.max(stats['states_explored'])
        stats['min_states'] = np.min(stats['states_explored'])
        stats['median_states'] = np.median(stats['states_explored'])
        stats['p95_states'] = np.percentile(stats['states_explored'], 95)

    if stats['search_times']:
        stats['avg_time'] = np.mean(stats['search_times'])
        stats['max_time'] = np.max(stats['search_times'])

    if stats['program_lengths']:
        stats['avg_program_length'] = np.mean(stats['program_lengths'])

    # Compute task type averages
    for task_type, data in stats['by_task_type'].items():
        if data['states']:
            data['avg_states'] = np.mean(data['states'])
            data['avg_time'] = np.mean(stats['search_times'])  # Approximate

    return stats


def print_statistics(stats: Dict):
    """Pretty print statistics"""
    print("\n" + "=" * 70)
    print("ROUND 3.1: BASELINE SEARCH STATISTICS")
    print("=" * 70)

    print(f"\n{'OVERALL RESULTS':>30s}")
    print("-" * 70)
    print(f"  Total tasks:        {stats['total_tasks']:6d}")
    print(f"  Solved:             {stats['solved']:6d} ({100*stats['solved']/stats['total_tasks']:.1f}%)")
    print(f"  Unsolved:           {stats['unsolved']:6d}")
    print(f"  Timeouts:           {stats['timeouts']:6d}")

    if 'avg_states' in stats:
        print(f"\n{'STATES EXPLORED':>30s}")
        print("-" * 70)
        print(f"  Average:            {stats['avg_states']:6.1f}")
        print(f"  Median:             {stats['median_states']:6.1f}")
        print(f"  Min:                {stats['min_states']:6d}")
        print(f"  Max:                {stats['max_states']:6d}")
        print(f"  95th percentile:    {stats['p95_states']:6.1f}")

    if 'avg_time' in stats:
        print(f"\n{'SEARCH TIME':>30s}")
        print("-" * 70)
        print(f"  Average:            {stats['avg_time']:6.3f}s")
        print(f"  Max:                {stats['max_time']:6.3f}s")

    if 'avg_program_length' in stats:
        print(f"\n{'PROGRAM LENGTH':>30s}")
        print("-" * 70)
        print(f"  Average:            {stats['avg_program_length']:6.2f} steps")

    print(f"\n{'BY TASK TYPE':>30s}")
    print("-" * 70)
    print(f"  {'Type':20s} {'Count':>8s} {'Solved':>8s} {'Avg States':>12s}")
    print("-" * 70)
    for task_type in sorted(stats['by_task_type'].keys()):
        data = stats['by_task_type'][task_type]
        avg_states = data['avg_states'] if 'avg_states' in data else 0
        print(f"  {task_type:20s} {data['count']:8d} {data['solved']:8d} {avg_states:12.1f}")


def save_baseline(stats: Dict, filepath: str = 'R3_baseline_stats.json'):
    """Save statistics to JSON file"""
    # Convert numpy types to native Python types
    stats_copy = {}
    for key, value in stats.items():
        if isinstance(value, (np.integer, np.floating)):
            stats_copy[key] = float(value)
        elif isinstance(value, np.ndarray):
            stats_copy[key] = value.tolist()
        elif isinstance(value, list):
            stats_copy[key] = [float(x) if isinstance(x, (np.integer, np.floating)) else x for x in value]
        elif isinstance(value, dict):
            stats_copy[key] = {
                k: float(v) if isinstance(v, (np.integer, np.floating)) else v
                for k, v in value.items()
            }
        else:
            stats_copy[key] = value

    with open(filepath, 'w') as f:
        json.dump(stats_copy, f, indent=2)

    print(f"\n✅ Baseline statistics saved to {filepath}")


if __name__ == '__main__':
    print("=" * 70)
    print("ROUND 3.1: BASELINE STATISTICS COLLECTION")
    print("=" * 70)
    print("\nGoal: Understand current search behavior before optimization")
    print("Tasks: 100 diverse transforms (1-step, 2-step, 3-step, identity)")
    print()

    # Create synthesizer
    synthesizer = SymbolicProgramSynthesizer(max_program_length=3)

    # Generate tasks
    print("Generating test tasks...")
    tasks = generate_test_tasks(n=100)
    print(f"✅ Generated {len(tasks)} tasks")

    # Collect statistics
    stats = collect_statistics(synthesizer, tasks)

    # Print results
    print_statistics(stats)

    # Save baseline
    save_baseline(stats, 'R3_baseline_stats.json')

    print("\n" + "=" * 70)
    print("BASELINE COLLECTION COMPLETE")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Analyze baseline: identify bottlenecks (high state count tasks)")
    print("  2. Implement pruning: A*, beam search, visited-state")
    print("  3. Re-run: measure improvement (target: 50-70% reduction)")
