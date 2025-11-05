#!/usr/bin/env python3
"""
Calculate correctly-scaled timeout curves that fit within budget
"""

import math

total_tasks = 1000
attempt1_budget = 5400  # 90 minutes
attempt2_budget = 5400  # 90 minutes

print("="*70)
print("CALCULATING BUDGET-COMPLIANT TIMEOUT CURVES")
print("="*70)

# ==============================================================================
# ATTEMPT 1: Inverted Bell Curve (scaled to fit budget)
# ==============================================================================

print("\nATTEMPT 1: Inverted Bell Curve")
print("-"*70)

# Calculate weights for exponential decay
decay_rate = 5.0 / total_tasks
weights = []
for i in range(total_tasks):
    weight = math.exp(-decay_rate * i)
    weights.append(weight)

sum_weights = sum(weights)
print(f"Sum of exponential weights: {sum_weights:.2f}")

# Scale factor to make sum equal budget
scale_factor = attempt1_budget / sum_weights
print(f"Scale factor needed: {scale_factor:.2f}")

# Calculate actual timeouts
attempt1_timeouts = []
min_timeout = 0.5  # Minimum 0.5s per task
for i in range(total_tasks):
    timeout = scale_factor * weights[i]
    timeout = max(timeout, min_timeout)  # Floor at 0.5s
    attempt1_timeouts.append(timeout)

actual_sum_1 = sum(attempt1_timeouts)
print(f"Actual sum of timeouts: {actual_sum_1:.0f}s ({actual_sum_1/60:.1f} min)")
print(f"Budget: {attempt1_budget}s ({attempt1_budget/60:.1f} min)")
print(f"Difference: {actual_sum_1 - attempt1_budget:.0f}s")
print(f"Fits: {'YES ✓' if actual_sum_1 <= attempt1_budget else 'NO ✗'}")

print(f"\nSample timeouts:")
print(f"  Task 0 (easiest): {attempt1_timeouts[0]:.1f}s")
print(f"  Task 10: {attempt1_timeouts[10]:.1f}s")
print(f"  Task 50: {attempt1_timeouts[50]:.1f}s")
print(f"  Task 100: {attempt1_timeouts[100]:.1f}s")
print(f"  Task 500: {attempt1_timeouts[500]:.1f}s")
print(f"  Task 999 (hardest): {attempt1_timeouts[999]:.1f}s")

# ==============================================================================
# ATTEMPT 2: Linear Ramp (scaled to fit budget)
# ==============================================================================

print("\n" + "="*70)
print("ATTEMPT 2: Linear Ramp")
print("-"*70)

# For linear ramp: sum = n * (min + max) / 2
# We want: attempt2_budget = 1000 * (min_t + max_t) / 2
# Solve for max_t given min_t

min_timeout_2 = 0.5  # Minimum timeout
# budget = n * (min + max) / 2
# max = 2 * budget / n - min
max_timeout_regular = 2 * attempt2_budget * 0.9 / (total_tasks * 0.9) - min_timeout_2

print(f"Regular tasks (0-900):")
print(f"  Min timeout: {min_timeout_2:.1f}s")
print(f"  Max timeout: {max_timeout_regular:.1f}s")

# Top 10% gets extra budget
top_10_budget = attempt2_budget * 0.1
top_10_count = int(total_tasks * 0.1)
# If we want them to get 50-80s, check if budget allows
top_10_min = 50.0
top_10_max = 2 * top_10_budget / top_10_count - top_10_min

print(f"\nTop 10% hardest (901-1000):")
print(f"  Min timeout: {top_10_min:.1f}s")
print(f"  Max timeout (budget-limited): {top_10_max:.1f}s")

# Calculate actual timeouts
attempt2_timeouts = []
top_10_threshold = int(total_tasks * 0.9)

for i in range(total_tasks):
    if i < top_10_threshold:
        # Linear from min to max over first 90%
        progress = i / top_10_threshold
        timeout = min_timeout_2 + (max_timeout_regular - min_timeout_2) * progress
    else:
        # Linear from top_10_min to top_10_max over last 10%
        top_10_index = i - top_10_threshold
        progress = top_10_index / (total_tasks - top_10_threshold)
        timeout = top_10_min + (top_10_max - top_10_min) * progress

    attempt2_timeouts.append(timeout)

actual_sum_2 = sum(attempt2_timeouts)
print(f"\nActual sum of timeouts: {actual_sum_2:.0f}s ({actual_sum_2/60:.1f} min)")
print(f"Budget: {attempt2_budget}s ({attempt2_budget/60:.1f} min)")
print(f"Difference: {actual_sum_2 - attempt2_budget:.0f}s")
print(f"Fits: {'YES ✓' if actual_sum_2 <= attempt2_budget else 'NO ✗'}")

print(f"\nSample timeouts:")
print(f"  Task 0 (easiest): {attempt2_timeouts[0]:.1f}s")
print(f"  Task 100: {attempt2_timeouts[100]:.1f}s")
print(f"  Task 500: {attempt2_timeouts[500]:.1f}s")
print(f"  Task 900 (start of top 10%): {attempt2_timeouts[900]:.1f}s")
print(f"  Task 950: {attempt2_timeouts[950]:.1f}s")
print(f"  Task 999 (hardest): {attempt2_timeouts[999]:.1f}s")

# ==============================================================================
# TOTAL CHECK
# ==============================================================================

print("\n" + "="*70)
print("TOTAL BUDGET CHECK")
print("="*70)

total_time = actual_sum_1 + actual_sum_2
training_budget = 10800  # 3 hours

print(f"Total timeout allocation: {total_time:.0f}s ({total_time/3600:.2f} hours)")
print(f"Training budget: {training_budget}s ({training_budget/3600:.2f} hours)")
print(f"Difference: {total_time - training_budget:.0f}s")
print(f"Fits within training budget: {'YES ✓' if total_time <= training_budget else 'NO ✗'}")

print("\n" + "="*70)

# Save formulas for implementation
print("\nFORMULAS FOR IMPLEMENTATION:")
print(f"""
def calculate_attempt1_timeout_FIXED(task_index: int, total_tasks: int = 1000,
                                     budget: float = 5400.0) -> float:
    decay_rate = 5.0 / total_tasks
    weight = math.exp(-decay_rate * task_index)

    # Sum of all weights
    sum_weights = {sum_weights:.2f}  # Pre-calculated

    # Scale to fit budget
    timeout = (budget / sum_weights) * weight

    # Floor at 0.5s
    return max(timeout, 0.5)

def calculate_attempt2_timeout_FIXED(task_index: int, total_tasks: int = 1000,
                                     budget: float = 5400.0) -> float:
    top_10_threshold = int(total_tasks * 0.9)

    if task_index < top_10_threshold:
        # Regular tasks: linear 0.5s → {max_timeout_regular:.1f}s
        min_t = 0.5
        max_t = {max_timeout_regular:.1f}
        progress = task_index / top_10_threshold
        return min_t + (max_t - min_t) * progress
    else:
        # Top 10%: linear {top_10_min:.1f}s → {top_10_max:.1f}s
        top_10_index = task_index - top_10_threshold
        progress = top_10_index / (total_tasks - top_10_threshold)
        return {top_10_min:.1f} + ({top_10_max:.1f} - {top_10_min:.1f}) * progress
""")
