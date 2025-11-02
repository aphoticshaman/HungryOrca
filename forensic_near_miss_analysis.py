#!/usr/bin/env python3
"""
FORENSIC ANALYSIS: Why 91.8% But Not 100%?

R&D + NSM x10 on UNDERFIT problem:
Why are we getting high partial matches but missing exact?

Analyze:
1. Which specific cells are wrong?
2. What patterns in the errors?
3. Generate 10 hypotheses
4. Test each hypothesis
5. Identify root causes
"""

import json
import numpy as np
from collections import Counter, defaultdict
from evolving_specialist_system import EvolvingSolver


def load_task(task_id):
    """Load specific training task."""
    with open('arc-agi_training_challenges.json') as f:
        challenges = json.load(f)

    with open('arc-agi_training_solutions.json') as f:
        solutions = json.load(f)

    task = challenges[task_id]
    train_pairs = [(np.array(p['input']), np.array(p['output'])) for p in task['train']]
    test_input = np.array(task['test'][0]['input'])
    test_output = np.array(solutions[task_id][0])

    return train_pairs, test_input, test_output


def visualize_grid(grid, label=""):
    """ASCII visualization of grid."""
    print(f"\n{label}:")
    h, w = grid.shape
    print(f"Shape: {h}x{w}")

    for i in range(h):
        row = ""
        for j in range(w):
            row += f"{grid[i,j]:2d} "
        print(row)


def analyze_differences(predicted, actual):
    """Analyze exactly what's different."""

    if predicted.shape != actual.shape:
        return {
            'type': 'SHAPE_MISMATCH',
            'pred_shape': predicted.shape,
            'actual_shape': actual.shape,
        }

    diff_mask = (predicted != actual)
    num_diff = np.sum(diff_mask)
    total = actual.size

    if num_diff == 0:
        return {'type': 'EXACT_MATCH'}

    # Find different cells
    diff_positions = np.argwhere(diff_mask)

    # Analyze spatial patterns
    spatial_analysis = {}

    # Are errors clustered or scattered?
    if len(diff_positions) > 1:
        distances = []
        for i in range(len(diff_positions)):
            for j in range(i+1, len(diff_positions)):
                dist = np.linalg.norm(diff_positions[i] - diff_positions[j])
                distances.append(dist)

        spatial_analysis['avg_distance'] = np.mean(distances) if distances else 0
        spatial_analysis['clustered'] = np.mean(distances) < 3.0 if distances else False
    else:
        spatial_analysis['single_cell'] = True

    # Edge vs interior errors?
    h, w = actual.shape
    edge_errors = 0
    interior_errors = 0

    for pos in diff_positions:
        i, j = pos
        if i == 0 or i == h-1 or j == 0 or j == w-1:
            edge_errors += 1
        else:
            interior_errors += 1

    spatial_analysis['edge_errors'] = edge_errors
    spatial_analysis['interior_errors'] = interior_errors

    # Color analysis
    pred_colors = [predicted[i, j] for i, j in diff_positions]
    actual_colors = [actual[i, j] for i, j in diff_positions]

    color_analysis = {
        'predicted_colors': Counter(pred_colors),
        'actual_colors': Counter(actual_colors),
    }

    # Pattern analysis: what should have been changed?
    pattern_analysis = {}

    # Check if errors are background (0) that should be filled
    zeros_wrong = sum(1 for pc in pred_colors if pc == 0)
    pattern_analysis['missed_fills'] = zeros_wrong

    # Check if errors are wrong color choice
    nonzeros_wrong = sum(1 for pc in pred_colors if pc != 0)
    pattern_analysis['wrong_color'] = nonzeros_wrong

    return {
        'type': 'PARTIAL_MATCH',
        'num_different': int(num_diff),
        'total_cells': int(total),
        'accuracy': float((total - num_diff) / total),
        'diff_positions': diff_positions.tolist(),
        'spatial': spatial_analysis,
        'colors': color_analysis,
        'patterns': pattern_analysis,
    }


def forensic_analysis_task(task_id):
    """Deep forensic analysis of one near-miss task."""

    print(f"{'='*80}")
    print(f"🔬 FORENSIC ANALYSIS: {task_id}")
    print(f"{'='*80}")

    train_pairs, test_input, test_output = load_task(task_id)

    print(f"\nTraining examples: {len(train_pairs)}")

    # Show training examples
    for idx, (inp, out) in enumerate(train_pairs):
        print(f"\n--- Training Example {idx+1} ---")
        print(f"Input: {inp.shape}, Output: {out.shape}")
        print(f"Input colors: {np.unique(inp).tolist()}")
        print(f"Output colors: {np.unique(out).tolist()}")

        visualize_grid(inp, f"Train {idx+1} Input")
        visualize_grid(out, f"Train {idx+1} Output")

    # Solve with our best solver
    print(f"\n{'='*80}")
    print(f"SOLVING WITH EVOLVING SPECIALIST SYSTEM")
    print(f"{'='*80}")

    solver = EvolvingSolver(time_limit=30.0, verbose=True)
    predicted = solver.solve(train_pairs, test_input)

    # Analyze
    print(f"\n{'='*80}")
    print(f"COMPARISON: PREDICTED vs ACTUAL")
    print(f"{'='*80}")

    if predicted is None:
        print("❌ Solver returned None")
        return

    visualize_grid(test_input, "Test Input")
    visualize_grid(predicted, "Predicted Output")
    visualize_grid(test_output, "Actual Output")

    # Detailed difference analysis
    print(f"\n{'='*80}")
    print(f"DIFFERENCE ANALYSIS")
    print(f"{'='*80}")

    diff_analysis = analyze_differences(predicted, actual=test_output)

    if diff_analysis['type'] == 'EXACT_MATCH':
        print("✅ EXACT MATCH!")
        return diff_analysis

    if diff_analysis['type'] == 'SHAPE_MISMATCH':
        print(f"❌ Shape mismatch:")
        print(f"   Predicted: {diff_analysis['pred_shape']}")
        print(f"   Actual: {diff_analysis['actual_shape']}")
        return diff_analysis

    # Partial match - deep dive
    print(f"\n📊 Accuracy: {diff_analysis['accuracy']*100:.1f}%")
    print(f"Different cells: {diff_analysis['num_different']}/{diff_analysis['total_cells']}")

    print(f"\n🗺️ Spatial Pattern:")
    for key, val in diff_analysis['spatial'].items():
        print(f"   {key}: {val}")

    print(f"\n🎨 Color Analysis:")
    print(f"   Predicted wrong colors: {dict(diff_analysis['colors']['predicted_colors'])}")
    print(f"   Should have been: {dict(diff_analysis['colors']['actual_colors'])}")

    print(f"\n🔍 Pattern Analysis:")
    for key, val in diff_analysis['patterns'].items():
        print(f"   {key}: {val}")

    print(f"\n📍 Wrong cell positions:")
    for pos in diff_analysis['diff_positions'][:10]:  # First 10
        i, j = pos
        print(f"   ({i},{j}): predicted={predicted[i,j]}, actual={test_output[i,j]}")

    return diff_analysis


def generate_hypotheses(analyses):
    """Generate 10 hypotheses for why we're underfitting."""

    print(f"\n\n{'='*80}")
    print(f"🧠 NSM x10: HYPOTHESIS GENERATION")
    print(f"{'='*80}\n")

    hypotheses = []

    # Analyze aggregate patterns
    total_edge_errors = sum(a.get('spatial', {}).get('edge_errors', 0) for a in analyses if isinstance(a, dict))
    total_interior_errors = sum(a.get('spatial', {}).get('interior_errors', 0) for a in analyses if isinstance(a, dict))
    total_missed_fills = sum(a.get('patterns', {}).get('missed_fills', 0) for a in analyses if isinstance(a, dict))
    total_wrong_color = sum(a.get('patterns', {}).get('wrong_color', 0) for a in analyses if isinstance(a, dict))

    # Hypothesis 1: Incomplete fill (background should be colored)
    if total_missed_fills > 0:
        hypotheses.append({
            'id': 1,
            'name': 'Incomplete Fill',
            'description': 'Flood-fill not reaching all interior cells that should be filled',
            'evidence': f'{total_missed_fills} cells are 0 (background) but should be colored',
            'root_cause': 'Flood-fill algorithm starts from edges, may miss enclosed regions',
            'fix': 'Improve interior detection - try from all bg cells, not just edges',
            'priority': 'HIGH',
        })

    # Hypothesis 2: Wrong fill color chosen
    if total_wrong_color > 0:
        hypotheses.append({
            'id': 2,
            'name': 'Wrong Fill Color',
            'description': 'Filling with wrong color (not 0, but still wrong)',
            'evidence': f'{total_wrong_color} non-zero cells have wrong color',
            'root_cause': 'Color learning from training examples is incomplete',
            'fix': 'Improve color mapping - look at ALL training examples, not just first',
            'priority': 'HIGH',
        })

    # Hypothesis 3: Edge handling
    if total_edge_errors > total_interior_errors:
        hypotheses.append({
            'id': 3,
            'name': 'Edge Boundary Errors',
            'description': 'Errors concentrated on grid edges',
            'evidence': f'{total_edge_errors} edge errors vs {total_interior_errors} interior',
            'root_cause': 'Edge cells handled differently in flood-fill logic',
            'fix': 'Special handling for edge cells in fill algorithms',
            'priority': 'MEDIUM',
        })

    # Hypothesis 4: Stopping too early
    hypotheses.append({
        'id': 4,
        'name': 'Stopping at First Solution',
        'description': 'Specialists stop after first success instead of iterating',
        'evidence': 'High scores (90%) but not perfect - close but not exact',
        'root_cause': 'Specialists return first result with high confidence',
        'fix': 'Add iterative refinement - compare to training outputs and adjust',
        'priority': 'HIGH',
    })

    # Hypothesis 5: Single-strategy per specialist
    hypotheses.append({
        'id': 5,
        'name': 'Single Strategy Limitation',
        'description': 'Each specialist tries ONE approach and stops',
        'evidence': 'Evolution notes show single strategy per specialist',
        'root_cause': 'Specialists not trying variations of their core capability',
        'fix': 'Each specialist should try multiple variations (e.g., fill color 1, 2, 3...)',
        'priority': 'HIGH',
    })

    # Hypothesis 6: Training example sampling
    hypotheses.append({
        'id': 6,
        'name': 'Insufficient Training Analysis',
        'description': 'Only analyzing first 1-2 training examples',
        'evidence': 'Code shows [:2] limiting to first 2 examples',
        'root_cause': 'Time budget causes early stopping in training analysis',
        'fix': 'Analyze ALL training examples before deciding strategy',
        'priority': 'MEDIUM',
    })

    # Hypothesis 7: No feedback loop
    hypotheses.append({
        'id': 7,
        'name': 'No Self-Correction',
        'description': 'No mechanism to check result against training patterns',
        'evidence': 'Specialists generate result and return immediately',
        'root_cause': 'No validation step comparing result to expected pattern',
        'fix': 'Add validation: apply same transform to training inputs, check if matches training outputs',
            'priority': 'HIGH',
    })

    # Hypothesis 8: Boundary conditions
    hypotheses.append({
        'id': 8,
        'name': 'Off-by-One Errors',
        'description': 'Boundary conditions in algorithms (e.g., range(h) vs range(h-1))',
        'evidence': 'Edge errors common in analysis',
        'root_cause': 'Classic off-by-one in loop boundaries',
        'fix': 'Audit all range() calls, especially in flood-fill',
        'priority': 'MEDIUM',
    })

    # Hypothesis 9: Partial transformation
    hypotheses.append({
        'id': 9,
        'name': 'Incomplete Transformation',
        'description': 'Transform applied to some cells but not all that need it',
        'evidence': 'High accuracy suggests right approach, wrong coverage',
        'root_cause': 'Conditional logic may skip some cells',
        'fix': 'Ensure transforms apply to ALL relevant cells',
        'priority': 'HIGH',
    })

    # Hypothesis 10: Confidence threshold
    hypotheses.append({
        'id': 10,
        'name': 'Premature Confidence',
        'description': 'Specialists report 0.85-0.95 confidence and stop trying',
        'evidence': 'FillMaster reports 0.85 confidence on 91.8% match',
        'root_cause': 'Confidence calibration is wrong - 0.85 should mean 85%, not 91%',
        'fix': 'Recalibrate confidence OR keep trying even with high confidence',
        'priority': 'MEDIUM',
    })

    # Print hypotheses
    for h in hypotheses:
        print(f"{h['id']}. {h['name']} [{h['priority']}]")
        print(f"   What: {h['description']}")
        print(f"   Evidence: {h['evidence']}")
        print(f"   Root cause: {h['root_cause']}")
        print(f"   Fix: {h['fix']}")
        print()

    return hypotheses


def test_hypothesis(hypothesis_id, task_id='00d62c1b'):
    """Test specific hypothesis with fix."""

    print(f"\n{'='*80}")
    print(f"🧪 TESTING HYPOTHESIS #{hypothesis_id}")
    print(f"{'='*80}\n")

    train_pairs, test_input, test_output = load_task(task_id)

    # Hypothesis-specific tests
    if hypothesis_id == 7:
        # Test: Add validation step
        print("Testing: Add self-correction via validation")

        # Original approach
        solver = EvolvingSolver(time_limit=30.0, verbose=False)
        pred_original = solver.solve(train_pairs, test_input)

        # With validation: apply same transform to training, check if correct
        # Then iterate if not exact

        from evolving_specialist_system import FillMaster, CollectiveKnowledge

        specialist = FillMaster()
        collective = CollectiveKnowledge()

        report = specialist.solve(train_pairs, test_input, collective, 30.0)

        # Validate: apply to training inputs
        if report.result is not None:
            print(f"Original result: {report.confidence*100:.1f}% confidence")

            # Check on training
            for idx, (train_in, train_out) in enumerate(train_pairs[:2]):
                # Apply same strategy to training input
                test_result = specialist.solve([(train_in, train_out)], train_in, collective, 5.0)

                if test_result.result is not None:
                    match = np.array_equal(test_result.result, train_out)
                    print(f"Training {idx+1} validation: {'✅ MATCH' if match else '❌ MISMATCH'}")

                    if not match:
                        score = np.sum(test_result.result == train_out) / train_out.size
                        print(f"   Score: {score*100:.1f}%")

    elif hypothesis_id == 5:
        # Test: Try multiple variations
        print("Testing: Multiple fill color variations")

        from evolving_specialist_system import FillMaster, CollectiveKnowledge

        specialist = FillMaster()
        collective = CollectiveKnowledge()

        # Try different fill colors
        for fill_color in range(1, 10):
            # Modified specialist that tries specific color
            result = test_input.copy()

            # Find interior
            interior = specialist._find_interior(test_input)

            # Fill with this color
            for i, j in interior:
                result[i, j] = fill_color

            # Check score
            if result.shape == test_output.shape:
                score = np.sum(result == test_output) / test_output.size

                if score > 0.9:
                    print(f"Fill color {fill_color}: {score*100:.1f}%")

                    if score == 1.0:
                        print(f"✅ EXACT MATCH with fill color {fill_color}!")
                        return True

    return False


def main():
    """Run complete forensic analysis."""

    print(f"{'='*80}")
    print(f"🔬 FORENSIC R&D: Why Underfit on Exact Matches?")
    print(f"{'='*80}\n")

    # Near-miss tasks from our results
    near_miss_tasks = [
        ('00d62c1b', 91.8),
        ('045e512c', 90.2),
        ('025d127b', 88.0),
    ]

    analyses = []

    for task_id, reported_score in near_miss_tasks:
        print(f"\n\n{'='*80}")
        print(f"Analyzing: {task_id} (reported: {reported_score}%)")
        print(f"{'='*80}")

        analysis = forensic_analysis_task(task_id)
        analyses.append(analysis)

    # Generate hypotheses
    hypotheses = generate_hypotheses(analyses)

    # Test top hypotheses
    print(f"\n{'='*80}")
    print(f"🧪 TESTING TOP HYPOTHESES")
    print(f"{'='*80}")

    # Test hypothesis 5 (multiple variations)
    test_hypothesis(5, '00d62c1b')

    # Test hypothesis 7 (validation)
    test_hypothesis(7, '00d62c1b')

    # Save analysis
    output = {
        'near_miss_analyses': analyses,
        'hypotheses': hypotheses,
    }

    with open('forensic_analysis_results.json', 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n💾 Analysis saved: forensic_analysis_results.json")


if __name__ == '__main__':
    main()
