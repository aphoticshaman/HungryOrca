#!/usr/bin/env python3
"""
META-ANALYSIS: What's Missing for the Last 10% Accuracy?

Current status:
- Baseline (collaborative): 51.8% partial, 0 exact
- Evolving specialists: 51.9% partial, 1 exact

Gap to bridge: ~10-15% to reach 60-65%+ range

This analysis identifies:
1. What types of puzzles we're failing on
2. What capabilities are missing
3. What improvements would have highest impact
4. Concrete next steps ranked by ROI
"""

import json
import numpy as np
from collections import Counter, defaultdict


def load_training_data():
    """Load training challenges and solutions."""
    with open('arc-agi_training_challenges.json') as f:
        challenges = json.load(f)

    with open('arc-agi_training_solutions.json') as f:
        solutions = json.load(f)

    return challenges, solutions


def analyze_puzzle_characteristics(challenges):
    """Categorize puzzles by characteristics."""

    categories = defaultdict(list)

    for task_id, task in challenges.items():
        train_pairs = [(np.array(p['input']), np.array(p['output'])) for p in task['train']]

        # Size characteristics
        size_changes = [inp.shape != out.shape for inp, out in train_pairs]
        if any(size_changes):
            categories['size_change'].append(task_id)
        else:
            categories['same_size'].append(task_id)

        # Color characteristics
        for inp, out in train_pairs:
            inp_colors = len(np.unique(inp))
            out_colors = len(np.unique(out))

            if out_colors > inp_colors:
                categories['adds_colors'].append(task_id)
                break
            elif out_colors < inp_colors:
                categories['removes_colors'].append(task_id)
                break
            else:
                categories['same_colors'].append(task_id)
                break

        # Complexity - grid size
        avg_size = np.mean([inp.size + out.size for inp, out in train_pairs])
        if avg_size < 200:
            categories['small_grids'].append(task_id)
        elif avg_size < 500:
            categories['medium_grids'].append(task_id)
        else:
            categories['large_grids'].append(task_id)

        # Check for symmetry in inputs
        for inp, _ in train_pairs:
            if np.array_equal(inp, np.flip(inp, 0)) or np.array_equal(inp, np.flip(inp, 1)):
                categories['symmetric_input'].append(task_id)
                break

        # Check for repetition/tiling
        for inp, out in train_pairs:
            if out.shape[0] > inp.shape[0] or out.shape[1] > inp.shape[1]:
                if out.shape[0] % inp.shape[0] == 0 and out.shape[1] % inp.shape[1] == 0:
                    categories['tiling_pattern'].append(task_id)
                    break

    return categories


def analyze_failure_patterns():
    """Analyze what puzzles we're failing on."""

    # Load results
    try:
        with open('evolving_solver_proportional_results.json') as f:
            results = json.load(f)
    except FileNotFoundError:
        print("⚠️ No results file found. Run evolving solver first.")
        return

    # Categorize results
    exact_matches = []
    high_scores = []  # 70%+
    medium_scores = []  # 30-70%
    failures = []  # <30%

    for r in results['results']:
        task_id = r['task_id']
        score = r['score']

        if r['exact']:
            exact_matches.append(task_id)
        elif score >= 0.7:
            high_scores.append(task_id)
        elif score >= 0.3:
            medium_scores.append(task_id)
        else:
            failures.append(task_id)

    print(f"{'='*80}")
    print(f"📊 PERFORMANCE BREAKDOWN")
    print(f"{'='*80}\n")

    print(f"✅ Exact matches (100%): {len(exact_matches)}")
    for tid in exact_matches:
        print(f"   - {tid}")

    print(f"\n🎯 High scores (70-99%): {len(high_scores)}")
    for tid in high_scores:
        score = next(r['score'] for r in results['results'] if r['task_id'] == tid)
        print(f"   - {tid}: {score*100:.1f}%")

    print(f"\n⚠️ Medium scores (30-69%): {len(medium_scores)}")
    for tid in medium_scores:
        score = next(r['score'] for r in results['results'] if r['task_id'] == tid)
        print(f"   - {tid}: {score*100:.1f}%")

    print(f"\n❌ Failures (<30%): {len(failures)}")
    for tid in failures:
        print(f"   - {tid}")

    return exact_matches, high_scores, medium_scores, failures


def identify_missing_capabilities(challenges, failures, medium_scores):
    """What capabilities would help most?"""

    print(f"\n{'='*80}")
    print(f"🔍 MISSING CAPABILITIES ANALYSIS")
    print(f"{'='*80}\n")

    # Analyze failure patterns
    problem_tasks = failures + medium_scores

    failure_characteristics = defaultdict(int)

    for task_id in problem_tasks:
        if task_id not in challenges:
            continue

        task = challenges[task_id]
        train_pairs = [(np.array(p['input']), np.array(p['output'])) for p in task['train']]

        for inp, out in train_pairs:
            # Size changes
            if inp.shape != out.shape:
                failure_characteristics['size_transformation'] += 1

            # Complex color mapping
            if len(np.unique(out)) > 5:
                failure_characteristics['many_colors'] += 1

            # Large grids
            if inp.size > 400 or out.size > 400:
                failure_characteristics['large_grid'] += 1

            # Output larger than input (composition/tiling)
            if out.shape[0] > inp.shape[0] or out.shape[1] > inp.shape[1]:
                failure_characteristics['expansion'] += 1

            # Output smaller (cropping/selection)
            if out.shape[0] < inp.shape[0] or out.shape[1] < inp.shape[1]:
                failure_characteristics['reduction'] += 1

    print("Characteristics of problematic tasks:")
    for char, count in sorted(failure_characteristics.items(), key=lambda x: -x[1]):
        print(f"  {char}: {count} occurrences")

    return failure_characteristics


def recommend_improvements(failure_characteristics):
    """Recommend highest-impact improvements."""

    print(f"\n{'='*80}")
    print(f"💡 RECOMMENDED IMPROVEMENTS (Ranked by Impact)")
    print(f"{'='*80}\n")

    recommendations = []

    # Rank by frequency in failures
    if failure_characteristics.get('expansion', 0) > 0:
        recommendations.append({
            'name': 'Composition/Tiling Detector',
            'impact': failure_characteristics['expansion'],
            'description': 'Detect when output is composed by repeating/tiling input',
            'implementation': 'Check if output is np.tile(input, (h, w)) or concatenation',
            'priority': 'HIGH'
        })

    if failure_characteristics.get('reduction', 0) > 0:
        recommendations.append({
            'name': 'Object Selection/Cropping',
            'impact': failure_characteristics['reduction'],
            'description': 'Extract specific objects or crop to relevant region',
            'implementation': 'Connected component analysis + bounding box extraction',
            'priority': 'HIGH'
        })

    if failure_characteristics.get('size_transformation', 0) > 0:
        recommendations.append({
            'name': 'Size Relationship Learner',
            'impact': failure_characteristics['size_transformation'],
            'description': 'Learn size relationships between input/output',
            'implementation': 'Detect scale factors, ratios, patterns in size changes',
            'priority': 'MEDIUM'
        })

    if failure_characteristics.get('many_colors', 0) > 0:
        recommendations.append({
            'name': 'Advanced Color Logic',
            'impact': failure_characteristics['many_colors'],
            'description': 'Handle complex color rules beyond simple mapping',
            'implementation': 'Color counting, sorting, conditional coloring',
            'priority': 'MEDIUM'
        })

    if failure_characteristics.get('large_grid', 0) > 0:
        recommendations.append({
            'name': 'Pattern Recognition at Scale',
            'impact': failure_characteristics['large_grid'],
            'description': 'Handle larger grids with repeated patterns',
            'implementation': 'Sliding window pattern matching, hierarchical analysis',
            'priority': 'LOW'
        })

    # Additional capabilities not directly measured
    recommendations.append({
        'name': 'Multi-Step Transformation',
        'impact': 999,  # High estimated impact
        'description': 'Chain multiple transformations (e.g., rotate THEN color map)',
        'implementation': 'Try compositions of successful individual transforms',
        'priority': 'HIGH'
    })

    recommendations.append({
        'name': 'Negative Space Analysis',
        'impact': 500,  # Medium estimated
        'description': 'Understand background vs foreground, holes vs solid',
        'implementation': 'Invert thinking - what is NOT there vs what IS there',
        'priority': 'MEDIUM'
    })

    recommendations.append({
        'name': 'Iterative Refinement',
        'impact': 800,  # High estimated
        'description': 'Improve solution through multiple attempts',
        'implementation': 'Score intermediate results, adjust based on partial match',
        'priority': 'HIGH'
    })

    # Sort by impact
    recommendations.sort(key=lambda x: x['impact'], reverse=True)

    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec['name']} [{rec['priority']}]")
        print(f"   Impact: {rec['impact']} tasks")
        print(f"   What: {rec['description']}")
        print(f"   How: {rec['implementation']}")
        print()

    return recommendations


def summarize_meta_insights():
    """High-level insights on what's working and what's not."""

    print(f"{'='*80}")
    print(f"🧠 META-INSIGHTS: The Big Picture")
    print(f"{'='*80}\n")

    insights = [
        ("✅ What's Working", [
            "Simple transformations: flip, rotate, color map",
            "Interior fill on bounded regions",
            "Grid line removal when color is consistent",
            "Identity transforms (output = input)",
            "Basic symmetry operations"
        ]),
        ("❌ What's NOT Working", [
            "Multi-step transformations (composition)",
            "Object selection and manipulation",
            "Size relationship learning",
            "Complex pattern repetition/tiling",
            "Contextual color rules (color depends on position/neighbors)"
        ]),
        ("🎯 Quick Wins (High ROI)", [
            "Add multi-step transformation chaining",
            "Implement object extraction with bounding boxes",
            "Try composition of existing successful transforms",
            "Add iterative refinement (score + adjust)",
        ]),
        ("🔬 Research Needed (Lower ROI)", [
            "Advanced pattern recognition",
            "Hierarchical analysis for large grids",
            "Meta-learning across task families",
            "Negative space reasoning"
        ]),
        ("⚡ Implementation Priority", [
            "1. Multi-step chaining (try transform A, then B, then C)",
            "2. Object extraction (connected components + crop)",
            "3. Iterative refinement (improve solution via feedback)",
            "4. Tiling detector (check if np.tile works)",
            "5. Advanced color logic (conditional rules)"
        ])
    ]

    for category, items in insights:
        print(f"{category}:")
        for item in items:
            print(f"  • {item}")
        print()


def main():
    """Run complete meta-analysis."""

    print(f"\n{'='*80}")
    print(f"🔬 META-ANALYSIS: Finding the Missing 10%")
    print(f"{'='*80}\n")

    print("Loading data...")
    challenges, solutions = load_training_data()

    print(f"Analyzing {len(challenges)} training tasks...")

    # Analyze what we're failing on
    exact, high, medium, failures = analyze_failure_patterns()

    # Identify missing capabilities
    failure_chars = identify_missing_capabilities(challenges, failures, medium)

    # Recommend improvements
    recommendations = recommend_improvements(failure_chars)

    # High-level insights
    summarize_meta_insights()

    # Save analysis
    analysis = {
        'performance': {
            'exact_matches': len(exact),
            'high_scores': len(high),
            'medium_scores': len(medium),
            'failures': len(failures),
        },
        'failure_characteristics': dict(failure_chars),
        'recommendations': recommendations,
    }

    with open('meta_analysis_results.json', 'w') as f:
        json.dump(analysis, f, indent=2)

    print(f"{'='*80}")
    print(f"💾 Analysis saved to: meta_analysis_results.json")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
