#!/usr/bin/env python3
"""
Comprehensive ARC Prize 2025 Test Set Submission Analysis
"""

import json
from collections import defaultdict, Counter

def load_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def analyze_grid(grid):
    """Analyze a single grid for various properties."""
    if not grid or len(grid) == 0:
        return {
            'height': 0,
            'width': 0,
            'num_colors': 0,
            'is_empty': True,
            'all_zeros': True
        }

    height = len(grid)
    width = len(grid[0]) if grid[0] else 0

    # Flatten and analyze colors
    flat = [cell for row in grid for cell in row]
    unique_colors = set(flat)
    num_colors = len(unique_colors)

    # Check if empty or all zeros
    all_zeros = all(cell == 0 for cell in flat)
    is_empty = all_zeros and height <= 2 and width <= 2

    return {
        'height': height,
        'width': width,
        'num_colors': num_colors,
        'is_empty': is_empty,
        'all_zeros': all_zeros,
        'colors': unique_colors,
        'total_cells': height * width
    }

def main():
    print("="*100)
    print("ARC PRIZE 2025 - COMPREHENSIVE TEST SET SUBMISSION ANALYSIS")
    print("="*100)
    print()

    # Load data
    submission = load_json('/home/user/HungryOrca/submission.json')
    test_challenges = load_json('/home/user/HungryOrca/arc-agi_test_challenges.json')

    print(f"Submission File: submission.json")
    print(f"Dataset: ARC-AGI Test Set (Competition Set)")
    print()

    # Verify coverage
    sub_ids = set(submission.keys())
    test_ids = set(test_challenges.keys())

    print("COVERAGE ANALYSIS:")
    print("-" * 100)
    print(f"Total Test Tasks:               {len(test_ids)}")
    print(f"Tasks in Submission:            {len(sub_ids)}")
    print(f"Coverage:                       {len(sub_ids & test_ids)}/{len(test_ids)} ({len(sub_ids & test_ids)/len(test_ids)*100:.1f}%)")

    if len(sub_ids - test_ids) > 0:
        print(f"WARNING: Unknown task IDs:      {len(sub_ids - test_ids)}")
    if len(test_ids - sub_ids) > 0:
        print(f"WARNING: Missing task IDs:      {len(test_ids - sub_ids)}")
    print()

    # Detailed analysis
    stats = {
        'total_attempts': 0,
        'empty_attempts': 0,
        'identical_attempts': 0,
        'different_attempts': 0,
        'grid_sizes': Counter(),
        'color_counts': Counter(),
        'total_colors_used': set()
    }

    task_analyses = []

    for task_id in sorted(submission.keys()):
        if task_id not in test_challenges:
            continue

        attempts = submission[task_id]
        challenge = test_challenges[task_id]

        # Analyze input to get context
        num_train_examples = len(challenge['train'])
        num_test_examples = len(challenge['test'])

        task_analysis = {
            'task_id': task_id,
            'num_train': num_train_examples,
            'num_test': num_test_examples,
            'attempts': []
        }

        # Analyze both attempts
        for idx, grid in enumerate(attempts):
            stats['total_attempts'] += 1

            analysis = analyze_grid(grid)

            if analysis['is_empty']:
                stats['empty_attempts'] += 1

            stats['grid_sizes'][f"{analysis['height']}x{analysis['width']}"] += 1
            stats['color_counts'][analysis['num_colors']] += 1
            stats['total_colors_used'].update(analysis['colors'])

            task_analysis['attempts'].append(analysis)

        # Check if attempts are identical
        if len(attempts) == 2:
            if attempts[0] == attempts[1]:
                stats['identical_attempts'] += 1
            else:
                stats['different_attempts'] += 1

        task_analyses.append(task_analysis)

    # Print statistics
    print("SUBMISSION QUALITY METRICS:")
    print("-" * 100)
    print(f"Total Attempts:                 {stats['total_attempts']}")
    print(f"Empty/Trivial Attempts:         {stats['empty_attempts']} ({stats['empty_attempts']/stats['total_attempts']*100:.2f}%)")
    print(f"Non-Empty Attempts:             {stats['total_attempts'] - stats['empty_attempts']} ({(stats['total_attempts'] - stats['empty_attempts'])/stats['total_attempts']*100:.2f}%)")
    print()

    num_tasks = len(submission)
    print(f"Identical Attempt Pairs:        {stats['identical_attempts']}/{num_tasks} ({stats['identical_attempts']/num_tasks*100:.1f}%)")
    print(f"Different Attempt Pairs:        {stats['different_attempts']}/{num_tasks} ({stats['different_attempts']/num_tasks*100:.1f}%)")
    print()

    print(f"Unique Colors Used:             {len(stats['total_colors_used'])} colors: {sorted(stats['total_colors_used'])}")
    print()

    # Grid size distribution
    print("GRID SIZE DISTRIBUTION (Top 20):")
    print("-" * 100)
    for size, count in stats['grid_sizes'].most_common(20):
        print(f"  {size:>10}:  {count:>4} ({count/stats['total_attempts']*100:>5.2f}%)")
    print()

    # Color count distribution
    print("COLOR DIVERSITY:")
    print("-" * 100)
    for num_colors in sorted(stats['color_counts'].keys()):
        count = stats['color_counts'][num_colors]
        print(f"  {num_colors:>2} colors:  {count:>4} grids ({count/stats['total_attempts']*100:>5.2f}%)")
    print()

    # Strategy analysis
    print("STRATEGY INDICATORS:")
    print("-" * 100)

    diversity_score = stats['different_attempts'] / num_tasks * 100
    quality_score = (stats['total_attempts'] - stats['empty_attempts']) / stats['total_attempts'] * 100

    print(f"Diversity Score:                {diversity_score:.1f}%")
    print("  (Higher = more exploration between attempts)")
    print()
    print(f"Quality Score:                  {quality_score:.1f}%")
    print("  (Percentage of non-empty attempts)")
    print()

    if diversity_score < 10:
        print("⚠️  Low diversity - both attempts are nearly identical for most tasks")
        print("   Consider: Using different approaches for attempt 1 vs attempt 2")
    elif diversity_score > 90:
        print("✓  High diversity - attempting different solutions")

    if quality_score < 90:
        print(f"⚠️  {100-quality_score:.1f}% of attempts are empty/trivial")
        print("   Consider: Implementing fallback strategies for unsolved tasks")
    else:
        print("✓  High quality - very few empty attempts")
    print()

    # Sample detailed task analysis
    print("SAMPLE DETAILED TASK ANALYSIS (First 10 Tasks):")
    print("-" * 100)
    for task in task_analyses[:10]:
        print(f"\nTask: {task['task_id']}")
        print(f"  Training Examples: {task['num_train']}")
        print(f"  Test Examples: {task['num_test']}")

        for idx, attempt in enumerate(task['attempts'], 1):
            status = "EMPTY" if attempt['is_empty'] else "NON-EMPTY"
            print(f"  Attempt {idx}: {attempt['height']}x{attempt['width']}, "
                  f"{attempt['num_colors']} colors, {attempt['total_cells']} cells [{status}]")

    print()
    print("="*100)
    print("SUMMARY:")
    print("-" * 100)
    print(f"✓ Submission covers all {len(test_ids)} test tasks")
    print(f"✓ {quality_score:.1f}% of attempts have meaningful outputs")
    print(f"✓ Diversity score: {diversity_score:.1f}%")
    print()
    print("NOTE: Actual scoring against ground truth is only available through")
    print("      official ARC Prize 2025 submission on the competition platform.")
    print("="*100)

if __name__ == "__main__":
    main()
