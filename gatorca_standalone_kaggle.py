#!/usr/bin/env python3
"""
PROJECT GATORCA - Standalone Kaggle Test
All code in one file - just run it!
"""

import json
import random
import time

# Paste the solver code directly here
SOLVER_CODE = """
# Copy the entire contents of gatorca_submission_compressed.py here
# Or we'll load it dynamically
"""

print("🐊 GatORCA - Quick Test")
print("="*60)

# Load solver
print("📦 Loading solver...")
with open('gatorca_submission_compressed.py', 'r') as f:
    exec(f.read())

# Load data
print("📁 Loading ARC data...")
with open('arc-agi_training_challenges.json', 'r') as f:
    data = json.load(f)

# Quick test on 5 tasks
print(f"🧪 Testing on 5 random tasks...\n")

test_ids = random.sample(list(data.keys()), 5)
results = []

for i, tid in enumerate(test_ids, 1):
    print(f"[{i}/5] {tid}...")
    task = data[tid]
    
    ops = get_all_operations()
    solver = OptimizedEvolutionarySolver(ops)
    result = solver.solve_task(task, max_generations=30, timeout_seconds=30)
    
    status = "✅" if result['solved'] else "❌"
    print(f"  {status} Fitness: {result['best_fitness']:.1%}\n")
    
    results.append({'id': tid, 'solved': result['solved'], 'fit': result['best_fitness']})

# Results
solved = sum(1 for r in results if r['solved'])
avg = sum(r['fit'] for r in results) / len(results)

print("="*60)
print(f"📊 Results: {solved}/5 solved ({solved/5:.0%})")
print(f"📊 Avg Fitness: {avg:.1%}")
print("="*60)
