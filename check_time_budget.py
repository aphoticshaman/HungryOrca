#!/usr/bin/env python3
"""
Check if time budget actually fits within 7hr target / 8hr max
"""

import math

total_tasks = 1000

# Attempt 1: Inverted bell curve
max_time = 90.0
min_time = 2.0
decay_rate = 5.0 / total_tasks

total_time_attempt1 = 0
for i in range(total_tasks):
    timeout = max_time * math.exp(-decay_rate * i)
    timeout = max(timeout, min_time)
    total_time_attempt1 += timeout

print('ATTEMPT 1 TIME BUDGET CALCULATION:')
print(f'  Budget allocated: 5,400s (90 min)')
print(f'  Actual sum of timeouts: {total_time_attempt1:.0f}s ({total_time_attempt1/60:.1f} min)')
print(f'  Difference: {total_time_attempt1 - 5400:.0f}s')
fits1 = 'YES ✓' if total_time_attempt1 <= 5400 else 'NO - OVERBUDGET! ✗'
print(f'  Fits within budget: {fits1}')
print()

# Attempt 2: Linear ramp
total_time_attempt2 = 0
top_10_threshold = int(total_tasks * 0.9)

for i in range(total_tasks):
    if i < top_10_threshold:
        timeout = 1.0 + (39.0 * i / top_10_threshold)
    else:
        top_10_index = i - top_10_threshold
        timeout = 50.0 + (30.0 * top_10_index / (total_tasks - top_10_threshold))
    total_time_attempt2 += timeout

print('ATTEMPT 2 TIME BUDGET CALCULATION:')
print(f'  Budget allocated: 5,400s (90 min)')
print(f'  Actual sum of timeouts: {total_time_attempt2:.0f}s ({total_time_attempt2/60:.1f} min)')
print(f'  Difference: {total_time_attempt2 - 5400:.0f}s')
fits2 = 'YES ✓' if total_time_attempt2 <= 5400 else 'NO - OVERBUDGET! ✗'
print(f'  Fits within budget: {fits2}')
print()

total_combined = total_time_attempt1 + total_time_attempt2
print('='*70)
print('TOTAL TIME BUDGET CHECK:')
print('='*70)
print(f'  Combined timeout allocation: {total_combined:.0f}s ({total_combined/3600:.2f} hours)')
print(f'  Training budget: 3 hours = 10,800s')
print(f'  Target: 7 hours total = 25,200s')
print(f'  Max: 8 hours = 28,800s')
print()
fits_training = 'YES ✓' if total_combined <= 10800 else 'NO! ✗'
print(f'  Fits within training budget (3hr): {fits_training}')

if total_combined > 10800:
    overage_hours = (total_combined - 10800) / 3600
    print(f'  ⚠️  OVERBUDGET by: {total_combined - 10800:.0f}s ({overage_hours:.2f} hours)')
    print(f'\n  🚨 PROBLEM: Timeouts sum to MORE than available training time!')
    print(f'     Tasks will be cut off before completion.')

print('='*70)
