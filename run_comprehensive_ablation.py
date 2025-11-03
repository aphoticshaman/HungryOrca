#!/usr/bin/env python3
"""
COMPREHENSIVE ABLATION TESTS - Full Parameter Sweeps
=====================================================

Tests EVERY parameter from MIN to MAX to find optimal settings.

Strategy: Test extremes first → narrow in on sweet spot
"""

import numpy as np
import json
import time
from unified_cortex import UnifiedCortex, CorticalSolver
from itertools import product

print("="*80)
print("COMPREHENSIVE ABLATION TEST MARATHON")
print("Testing all parameters from MIN → MAX")
print("="*80)

# Load ARC data
print("\nLoading ARC training data...")
with open('arc-agi_training_challenges.json') as f:
    training_tasks = json.load(f)

task_ids = list(training_tasks.keys())[:20]  # Use 20 tasks
test_tasks = [training_tasks[tid] for tid in task_ids]
print(f"Loaded {len(test_tasks)} tasks\n")

all_results = {}

# ============================================================================
# KNOB 1: CORTEX SIZE (MIN: 1k → MAX: 1M)
# ============================================================================

print("="*80)
print("🎛️  KNOB 1: CORTEX SIZE")
print("="*80)

# Test extremes first: MIN, LOW, MID, HIGH, MAX
cortex_sizes = [1_000, 5_000, 10_000, 50_000, 100_000, 500_000]

results_size = {}

for size in cortex_sizes:
    print(f"\n📊 Testing size={size:,} neurons...")

    scores = []
    times = []

    for i, task in enumerate(test_tasks[:10]):  # 10 tasks per config
        solver = CorticalSolver()
        solver.cortex = UnifiedCortex(size=size, connection_density=0.01)

        train_pairs = [(np.array(p['input']), np.array(p['output']))
                      for p in task['train']]
        test_input = np.array(task['test'][0]['input'])

        start = time.time()
        try:
            solution, confidence = solver.solve_task(train_pairs, test_input, time_limit=5.0)
            scores.append(confidence)
            times.append(time.time() - start)
        except Exception as e:
            print(f"  Task {i}: Error - {e}")
            scores.append(0.0)
            times.append(5.0)

    avg_score = np.mean(scores) if scores else 0.0
    avg_time = np.mean(times) if times else 0.0
    efficiency = avg_score / avg_time if avg_time > 0 else 0.0

    results_size[size] = {
        'score': float(avg_score),
        'time': float(avg_time),
        'efficiency': float(efficiency)
    }

    print(f"  ✓ Score: {avg_score:.4f}")
    print(f"  ✓ Time: {avg_time:.2f}s")
    print(f"  ✓ Efficiency: {efficiency:.4f}")

print("\n🎯 BEST CORTEX SIZE:")
best_size = max(results_size.items(), key=lambda x: x[1]['efficiency'])
print(f"  {best_size[0]:,} neurons (efficiency={best_size[1]['efficiency']:.4f})")

all_results['cortex_size'] = results_size

# ============================================================================
# KNOB 2: CONNECTION DENSITY (MIN: 0.001 → MAX: 0.1)
# ============================================================================

print("\n" + "="*80)
print("🎛️  KNOB 2: CONNECTION DENSITY")
print("="*80)

densities = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
results_density = {}
optimal_size = best_size[0]

for density in densities:
    print(f"\n📊 Testing density={density}...")

    scores = []
    times = []

    for i, task in enumerate(test_tasks[:10]):
        solver = CorticalSolver()
        solver.cortex = UnifiedCortex(size=optimal_size, connection_density=density)

        train_pairs = [(np.array(p['input']), np.array(p['output']))
                      for p in task['train']]
        test_input = np.array(task['test'][0]['input'])

        start = time.time()
        try:
            solution, confidence = solver.solve_task(train_pairs, test_input, time_limit=5.0)
            scores.append(confidence)
            times.append(time.time() - start)
        except Exception as e:
            print(f"  Task {i}: Error - {e}")
            scores.append(0.0)
            times.append(5.0)

    avg_score = np.mean(scores)
    avg_time = np.mean(times)
    efficiency = avg_score / avg_time if avg_time > 0 else 0.0

    results_density[density] = {
        'score': float(avg_score),
        'time': float(avg_time),
        'efficiency': float(efficiency)
    }

    print(f"  ✓ Score: {avg_score:.4f}")
    print(f"  ✓ Time: {avg_time:.2f}s")
    print(f"  ✓ Efficiency: {efficiency:.4f}")

print("\n🎯 BEST CONNECTION DENSITY:")
best_density = max(results_density.items(), key=lambda x: x[1]['efficiency'])
print(f"  {best_density[0]} (efficiency={best_density[1]['efficiency']:.4f})")

all_results['connection_density'] = results_density

# ============================================================================
# KNOB 3: PROPAGATION ITERATIONS (MIN: 1 → MAX: 100)
# ============================================================================

print("\n" + "="*80)
print("🎛️  KNOB 3: PROPAGATION ITERATIONS")
print("="*80)

iterations_list = [1, 5, 10, 20, 50, 100]
results_iterations = {}
optimal_density = best_density[0]

for iterations in iterations_list:
    print(f"\n📊 Testing iterations={iterations}...")

    scores = []
    times = []

    for i, task in enumerate(test_tasks[:10]):
        solver = CorticalSolver()
        cortex = UnifiedCortex(size=optimal_size, connection_density=optimal_density)
        solver.cortex = cortex

        train_pairs = [(np.array(p['input']), np.array(p['output']))
                      for p in task['train']]
        test_input = np.array(task['test'][0]['input'])

        # Override iterations
        original_activate = cortex.activate
        def activate_wrapper(stimulus, context='both', iters=iterations):
            return original_activate(stimulus, context, iterations=iters)
        cortex.activate = activate_wrapper

        start = time.time()
        try:
            solution, confidence = solver.solve_task(train_pairs, test_input, time_limit=5.0)
            scores.append(confidence)
            times.append(time.time() - start)
        except Exception as e:
            print(f"  Task {i}: Error - {e}")
            scores.append(0.0)
            times.append(5.0)

    avg_score = np.mean(scores)
    avg_time = np.mean(times)
    efficiency = avg_score / avg_time if avg_time > 0 else 0.0

    results_iterations[iterations] = {
        'score': float(avg_score),
        'time': float(avg_time),
        'efficiency': float(efficiency)
    }

    print(f"  ✓ Score: {avg_score:.4f}")
    print(f"  ✓ Time: {avg_time:.2f}s")
    print(f"  ✓ Efficiency: {efficiency:.4f}")

print("\n🎯 BEST ITERATIONS:")
best_iterations = max(results_iterations.items(), key=lambda x: x[1]['efficiency'])
print(f"  {best_iterations[0]} (efficiency={best_iterations[1]['efficiency']:.4f})")

all_results['propagation_iterations'] = results_iterations

# ============================================================================
# FINAL OPTIMAL CONFIGURATION
# ============================================================================

print("\n" + "="*80)
print("🏆 OPTIMAL CONFIGURATION FOUND")
print("="*80)
print(f"Cortex Size: {optimal_size:,} neurons")
print(f"Connection Density: {optimal_density}")
print(f"Propagation Iterations: {best_iterations[0]}")
print(f"\nExpected Performance:")
print(f"  Score: {best_iterations[1]['score']:.4f}")
print(f"  Time: {best_iterations[1]['time']:.2f}s/task")
print(f"  Efficiency: {best_iterations[1]['efficiency']:.4f}")
print("="*80)

# Save all results
final_results = {
    'optimal_config': {
        'cortex_size': int(optimal_size),
        'connection_density': float(optimal_density),
        'propagation_iterations': int(best_iterations[0])
    },
    'expected_performance': {
        'score': float(best_iterations[1]['score']),
        'time': float(best_iterations[1]['time']),
        'efficiency': float(best_iterations[1]['efficiency'])
    },
    'detailed_results': all_results,
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
}

with open('ablation_results_comprehensive.json', 'w') as f:
    json.dump(final_results, f, indent=2)

print("\n✅ Results saved to ablation_results_comprehensive.json")
print("="*80)
print("ABLATION MARATHON COMPLETE!")
print("Ready for Phase 2: 360° Vision System")
print("="*80)
