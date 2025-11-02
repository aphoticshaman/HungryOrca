#!/usr/bin/env python3
"""
Validate v6-DataDriven on training data
Test the Big Three: Crop, ColorSwap, Pad
"""

import json
import numpy as np
from collections import defaultdict

# Load training data
print("Loading training data...")
with open('arc-agi_training_challenges.json', 'r') as f:
    challenges = json.load(f)
with open('arc-agi_training_solutions.json', 'r') as f:
    solutions = json.load(f)

print(f"Loaded {len(challenges)} training tasks\n")

# v6 Solver Functions (from v6-DataDriven.ipynb)
def dcr(td):
    """Detect crop"""
    if not td.get('train'):
        return None
    bg=None
    for p in td['train']:
        i,o=np.array(p['input']),np.array(p['output'])
        if o.shape[0]>i.shape[0] or o.shape[1]>i.shape[1]:
            return None
        b=np.argmax(np.bincount(i.flatten()))
        if bg is None:
            bg=b
        elif bg!=b:
            return None
        m=i!=bg
        if not m.any():
            return None
        r,c=np.where(m)
        cr=i[r.min():r.max()+1,c.min():c.max()+1]
        if not np.array_equal(o,cr):
            return None
    return {'bg':int(bg)}

def acr(ti,p):
    """Apply crop"""
    i=np.array(ti)
    m=i!=p['bg']
    if not m.any():
        return ti
    r,c=np.where(m)
    return i[r.min():r.max()+1,c.min():c.max()+1].tolist()

def dcs(td):
    """Detect color swap"""
    if not td.get('train'):
        return None
    gm=None
    for p in td['train']:
        i,o=np.array(p['input']),np.array(p['output'])
        if i.shape!=o.shape:
            return None
        m={}
        for iv,ov in zip(i.flatten(),o.flatten()):
            if iv in m:
                if m[iv]!=ov:
                    return None
            else:
                m[iv]=ov
        if all(k==v for k,v in m.items()):
            return None
        if gm is None:
            gm=m
        elif m!=gm:
            return None
    return {'m':{int(k):int(v) for k,v in gm.items()}} if gm else None

def acs(ti,p):
    """Apply color swap"""
    i=np.array(ti)
    o=np.copy(i)
    for k,v in p['m'].items():
        o[i==k]=v
    return o.tolist()

def dpd(td):
    """Detect pad"""
    if not td.get('train'):
        return None
    pp=[]
    for p in td['train']:
        i,o=np.array(p['input']),np.array(p['output'])
        if o.shape[0]<i.shape[0] or o.shape[1]<i.shape[1]:
            return None
        bg=np.argmax(np.bincount(o.flatten()))
        found=False
        for r in range(o.shape[0]-i.shape[0]+1):
            for c in range(o.shape[1]-i.shape[1]+1):
                if np.array_equal(o[r:r+i.shape[0],c:c+i.shape[1]],i):
                    t,l=r,c
                    b=o.shape[0]-i.shape[0]-t
                    rt=o.shape[1]-i.shape[1]-l
                    pp.append((int(bg),t,b,l,rt))
                    found=True
                    break
            if found:
                break
        if not found:
            return None
    if len(set(p[0] for p in pp))>1:
        return None
    if len(set(pp))>1:
        return None
    bg,t,b,l,r=pp[0]
    return {'bg':bg,'t':t,'b':b,'l':l,'r':r}

def apd(ti,p):
    """Apply pad"""
    i=np.array(ti)
    return np.pad(i,((p['t'],p['b']),(p['l'],p['r'])),constant_values=p['bg']).tolist()

# Validation
results = {
    'total_tasks': 0,
    'total_test_cases': 0,
    'solver_detected': defaultdict(int),
    'solver_triggered': defaultdict(int),
    'solver_correct': defaultdict(int),
    'solver_incorrect': defaultdict(int),
    'any_correct': 0,
    'all_incorrect': 0
}

print("Running validation...\n")

for task_id, task_data in challenges.items():
    results['total_tasks'] += 1

    # Detect patterns on training data
    crop_params = dcr(task_data)
    cswap_params = dcs(task_data)
    pad_params = dpd(task_data)

    if crop_params:
        results['solver_detected']['crop'] += 1
    if cswap_params:
        results['solver_detected']['cswap'] += 1
    if pad_params:
        results['solver_detected']['pad'] += 1

    # Test on test cases
    for test_idx, test_case in enumerate(task_data['test']):
        results['total_test_cases'] += 1
        test_input = test_case['input']
        correct_output = solutions[task_id][test_idx]

        any_solver_correct = False

        # Try crop
        if crop_params:
            results['solver_triggered']['crop'] += 1
            try:
                predicted = acr(test_input, crop_params)
                if np.array_equal(predicted, correct_output):
                    results['solver_correct']['crop'] += 1
                    any_solver_correct = True
                else:
                    results['solver_incorrect']['crop'] += 1
            except:
                results['solver_incorrect']['crop'] += 1

        # Try color swap
        if cswap_params:
            results['solver_triggered']['cswap'] += 1
            try:
                predicted = acs(test_input, cswap_params)
                if np.array_equal(predicted, correct_output):
                    results['solver_correct']['cswap'] += 1
                    any_solver_correct = True
                else:
                    results['solver_incorrect']['cswap'] += 1
            except:
                results['solver_incorrect']['cswap'] += 1

        # Try pad
        if pad_params:
            results['solver_triggered']['pad'] += 1
            try:
                predicted = apd(test_input, pad_params)
                if np.array_equal(predicted, correct_output):
                    results['solver_correct']['pad'] += 1
                    any_solver_correct = True
                else:
                    results['solver_incorrect']['pad'] += 1
            except:
                results['solver_incorrect']['pad'] += 1

        if any_solver_correct:
            results['any_correct'] += 1
        elif crop_params or cswap_params or pad_params:
            results['all_incorrect'] += 1

# Print results
print("="*70)
print("v6-DATADRIVEN VALIDATION RESULTS")
print("="*70)
print(f"\nTotal tasks: {results['total_tasks']}")
print(f"Total test cases: {results['total_test_cases']}")

print(f"\n{'Solver':<15} {'Detected':>10} {'Triggered':>10} {'Correct':>10} {'Incorrect':>10} {'Accuracy':>10}")
print("-"*70)

for solver in ['crop', 'cswap', 'pad']:
    detected = results['solver_detected'][solver]
    triggered = results['solver_triggered'][solver]
    correct = results['solver_correct'][solver]
    incorrect = results['solver_incorrect'][solver]
    accuracy = (correct / triggered * 100) if triggered > 0 else 0.0

    print(f"{solver:<15} {detected:>10} {triggered:>10} {correct:>10} {incorrect:>10} {accuracy:>9.1f}%")

# Overall stats
total_triggered = sum(results['solver_triggered'].values())
total_correct = results['any_correct']
coverage = (total_triggered / results['total_test_cases'] * 100) if results['total_test_cases'] > 0 else 0
contribution = (total_correct / results['total_test_cases'] * 100) if results['total_test_cases'] > 0 else 0

print("\n" + "="*70)
print("OVERALL PERFORMANCE")
print("="*70)
print(f"Coverage:     {total_triggered}/{results['total_test_cases']} ({coverage:.1f}%)")
print(f"Contribution: {total_correct}/{results['total_test_cases']} ({contribution:.1f}%)")
print(f"Accuracy:     {total_correct}/{total_triggered if total_triggered > 0 else 1} ({(total_correct/total_triggered*100) if total_triggered > 0 else 0:.1f}%)")

print("\n" + "="*70)
print("COMPARISON TO v5-Lite")
print("="*70)
print("v5-Lite:")
print("  Coverage: 40.4% (symmetry false positives)")
print("  Accuracy: 0.0%")
print("  Contribution: 0.0%")
print("\nv6-DataDriven:")
print(f"  Coverage: {coverage:.1f}%")
print(f"  Accuracy: {(total_correct/total_triggered*100) if total_triggered > 0 else 0:.1f}%")
print(f"  Contribution: {contribution:.1f}%")

improvement = contribution if contribution > 0 else 0
print(f"\nImprovement: {improvement:.1f} percentage points")
print("="*70)
