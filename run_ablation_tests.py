#!/usr/bin/env python3
"""
ABLATION TEST RUNNER - Automated Parameter Optimization
========================================================

Tests all cortex configurations and finds optimal parameters.
"""

import numpy as np
import json
import time
from unified_cortex import UnifiedCortex, CorticalSolver, ABLATION_SCORES
from itertools import product

# Load ARC training data
print("Loading ARC training data...")
with open('arc-agi_training_challenges.json') as f:
    training_tasks = json.load(f)

# Sample 10 tasks for quick testing
task_ids = list(training_tasks.keys())[:10]
test_tasks = [training_tasks[tid] for tid in task_ids]

print(f"Loaded {len(test_tasks)} tasks for ablation testing\n")

# ============================================================================
# ABLATION TEST 1: Cortex Size
# ============================================================================

print("="*80)
print("ABLATION TEST 1: Cortex Size")
print("="*80)

cortex_sizes = [5_000, 10_000, 50_000, 100_000]
results_size = {}

for size in cortex_sizes:
    print(f"\nTesting size={size}...")

    total_confidence = 0
    total_time = 0

    for task in test_tasks[:5]:  # Quick test on 5 tasks
        solver = CorticalSolver()
        solver.cortex = UnifiedCortex(size=size, connection_density=0.01)

        train_pairs = [(np.array(p['input']), np.array(p['output']))
                      for p in task['train']]
        test_input = np.array(task['test'][0]['input'])

        start = time.time()
        try:
            solution, confidence = solver.solve_task(train_pairs, test_input, time_limit=2.0)
            total_confidence += confidence
            total_time += time.time() - start
        except Exception as e:
            print(f"  Error: {e}")
            continue

    avg_confidence = total_confidence / 5
    avg_time = total_time / 5

    results_size[size] = {
        'confidence': avg_confidence,
        'time': avg_time,
        'score': avg_confidence / avg_time if avg_time > 0 else 0  # Efficiency score
    }

    print(f"  Confidence: {avg_confidence:.3f}")
    print(f"  Time: {avg_time:.2f}s")
    print(f"  Efficiency: {results_size[size]['score']:.4f}")

print("\n📊 RESULTS - Cortex Size:")
best_size = max(results_size.items(), key=lambda x: x[1]['score'])
print(f"✅ BEST: size={best_size[0]} (score={best_size[1]['score']:.4f})")

# ============================================================================
# ABLATION TEST 2: Connection Density
# ============================================================================

print("\n" + "="*80)
print("ABLATION TEST 2: Connection Density")
print("="*80)

densities = [0.001, 0.005, 0.01, 0.02]
results_density = {}
best_size_val = best_size[0]

for density in densities:
    print(f"\nTesting density={density}...")

    total_confidence = 0
    total_time = 0

    for task in test_tasks[:5]:
        solver = CorticalSolver()
        solver.cortex = UnifiedCortex(size=best_size_val, connection_density=density)

        train_pairs = [(np.array(p['input']), np.array(p['output']))
                      for p in task['train']]
        test_input = np.array(task['test'][0]['input'])

        start = time.time()
        try:
            solution, confidence = solver.solve_task(train_pairs, test_input, time_limit=2.0)
            total_confidence += confidence
            total_time += time.time() - start
        except:
            continue

    avg_confidence = total_confidence / 5
    avg_time = total_time / 5

    results_density[density] = {
        'confidence': avg_confidence,
        'time': avg_time,
        'score': avg_confidence / avg_time if avg_time > 0 else 0
    }

    print(f"  Confidence: {avg_confidence:.3f}")
    print(f"  Time: {avg_time:.2f}s")
    print(f"  Efficiency: {results_density[density]['score']:.4f}")

print("\n📊 RESULTS - Connection Density:")
best_density = max(results_density.items(), key=lambda x: x[1]['score'])
print(f"✅ BEST: density={best_density[0]} (score={best_density[1]['score']:.4f})")

# ============================================================================
# ABLATION TEST 3: Propagation Iterations
# ============================================================================

print("\n" + "="*80)
print("ABLATION TEST 3: Propagation Iterations")
print("="*80)

iterations_list = [5, 10, 20, 50]
results_iterations = {}
best_density_val = best_density[0]

for iterations in iterations_list:
    print(f"\nTesting iterations={iterations}...")

    total_confidence = 0
    total_time = 0

    for task in test_tasks[:5]:
        solver = CorticalSolver()
        cortex = UnifiedCortex(size=best_size_val, connection_density=best_density_val)
        solver.cortex = cortex

        train_pairs = [(np.array(p['input']), np.array(p['output']))
                      for p in task['train']]
        test_input = np.array(task['test'][0]['input'])

        # Monkey-patch activation iterations
        original_activate = cortex.activate
        def activate_wrapper(stimulus, context='both', iterations=iterations):
            return original_activate(stimulus, context, iterations=iterations)
        cortex.activate = activate_wrapper

        start = time.time()
        try:
            solution, confidence = solver.solve_task(train_pairs, test_input, time_limit=2.0)
            total_confidence += confidence
            total_time += time.time() - start
        except:
            continue

    avg_confidence = total_confidence / 5
    avg_time = total_time / 5

    results_iterations[iterations] = {
        'confidence': avg_confidence,
        'time': avg_time,
        'score': avg_confidence / avg_time if avg_time > 0 else 0
    }

    print(f"  Confidence: {avg_confidence:.3f}")
    print(f"  Time: {avg_time:.2f}s")
    print(f"  Efficiency: {results_iterations[iterations]['score']:.4f}")

print("\n📊 RESULTS - Propagation Iterations:")
best_iterations = max(results_iterations.items(), key=lambda x: x[1]['score'])
print(f"✅ BEST: iterations={best_iterations[0]} (score={best_iterations[1]['score']:.4f})")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "="*80)
print("🎯 OPTIMAL CONFIGURATION")
print("="*80)
print(f"Cortex Size: {best_size[0]}")
print(f"Connection Density: {best_density[0]}")
print(f"Propagation Iterations: {best_iterations[0]}")
print(f"\nExpected Performance:")
print(f"  Confidence: {best_iterations[1]['confidence']:.3f}")
print(f"  Time: {best_iterations[1]['time']:.2f}s")
print(f"  Efficiency Score: {best_iterations[1]['score']:.4f}")
print("="*80)

# Save results
results = {
    'optimal_config': {
        'size': int(best_size[0]),
        'density': float(best_density[0]),
        'iterations': int(best_iterations[0])
    },
    'performance': {
        'confidence': float(best_iterations[1]['confidence']),
        'time': float(best_iterations[1]['time']),
        'efficiency': float(best_iterations[1]['score'])
    },
    'all_results': {
        'size': {str(k): v for k, v in results_size.items()},
        'density': {str(k): v for k, v in results_density.items()},
        'iterations': {str(k): v for k, v in results_iterations.items()}
    }
}

with open('ablation_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n✅ Results saved to ablation_results.json")
print("Ready for Phase 2: 360° Vision System")
