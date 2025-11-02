# ROUND 2 LESSONS: Ranker Gradient Descent (Failed Successfully)

**Date:** 2025-11-02
**Status:** Partial success - discovered deeper issue

---

## Novel Insight (Attempted)

**Gradient descent for ranker learning:**
Replace naive weight update with proper logistic regression gradient + momentum + L2 reg + LR decay.

**What we implemented:**
- Dynamic primitive discovery (20 primitives, not hardcoded 15)
- Proper gradient: error * sigmoid_deriv * features
- L2 regularization for weight decay
- Momentum (beta=0.9) for faster convergence
- Learning rate decay (0.995 per update)

---

## 3x Pros

1. **Dynamic discovery works perfectly**
   - Ranker now discovers all 20 primitives from executor
   - No hardcoded indices
   - Automatically handles new primitives from Round 1

2. **Gradient descent math is correct**
   - Proper logistic regression gradient
   - Momentum and regularization implemented
   - Learning rate decay working

3. **Code is more principled**
   - No more naive `weights += lr * reward * features`
   - Follows standard ML best practices
   - Easier to debug and extend

---

## 3x Cons

1. **FEATURES ARE INADEQUATE** (Major discovery!)
   - Current features: [size_ratio, color_change, shape_change, symmetry_change, connectivity]
   - For rotation task: **ALL ZEROS** [0, 0, 0, 0, 0]
   - Gradient = error * sigmoid_deriv * **[0,0,0,0,0]** = **[0,0,0,0,0]**
   - Can't learn what you can't measure!

2. **No learning happening**
   - Weights barely change (0.0002 over 10 updates)
   - Scores stay constant (0.5000)
   - Root cause: zero features → zero gradient → no update

3. **Wrong problem being solved**
   - Fixed gradient descent algorithm
   - But real issue is feature engineering
   - Classic ML mistake: optimize wrong thing

---

## Root Cause Analysis

**Debug output shows:**
```
Features: [0. 0. 0. 0. 0.]
Gradient (before reg): [0. 0. 0. 0. 0.]
Weight change: 0.000000
```

**Why features are zero for rotation:**
- size_ratio: log2(4/4) = 0 (size unchanged)
- color_change: {1,2,3,4} == {1,2,3,4}, diff=0 (colors unchanged)
- shape_change: |2/2 - 2/2| = 0 (square → square)
- symmetry_change: both non-symmetric = 0
- connectivity: 4-4 = 0 (same nonzero count)

**Rotations DON'T change these properties!**

---

## What Features Should Capture

**For spatial transforms (rotate, reflect, transpose):**
1. **Position correlation:** sum of (inp[i,j] - out[new_i, new_j])²
2. **Orientation change:** detect if rows become columns, etc.
3. **Corner movement:** track where corner pixels move
4. **Pairwise distance preservation:** do pixel distances stay same?
5. **Directional gradient:** horizontal vs vertical patterns

**Current features only work for:**
- Scale (changes size_ratio)
- Color ops (changes color_change)
- Crop (changes shape_change)

**They miss:**
- Rotations
- Reflections
- Transposes
- Most spatial transforms!

---

## Action Items

**Round 2.5 (NEW): Fix Feature Engineering**
- [ ] Add 5 new features for spatial transforms
- [ ] Implement position_correlation(inp, out)
- [ ] Implement orientation_change(inp, out)
- [ ] Test on rotation tasks - expect non-zero features
- [ ] Re-run Round 2 tests - expect learning to work

**Then continue:**
- [ ] Round 3: Beam search pruning
- [ ] Round 4: Program cache

---

## Meta-Lesson: Failed Successfully

**This is what iterative refinement looks like:**
1. Round 1: Added compositions → 100% coverage ✅
2. Round 2: Fixed gradient descent → Discovered feature inadequacy ✅
3. Round 2.5: Fix features → (upcoming)

**We didn't waste time:**
- Gradient descent code is good (reusable)
- Dynamic discovery works (needed for Round 1 compositions)
- Found REAL bottleneck (feature engineering)

**This is epistemic humility:**
- Hypothesis: Gradient descent will improve learning
- Test: Implement and measure
- Result: No improvement
- Analysis: Features inadequate
- Update: Fix features, re-test

**Not failure - discovery!**

---

## Metrics

**Before Round 2:**
- Primitives: 15 (hardcoded)
- Learning: Naive weight update
- Features: 5 (inadequate for spatial transforms)

**After Round 2:**
- Primitives: 20 (dynamic discovery) ✅
- Learning: Proper gradient descent ✅
- Features: 5 (still inadequate) ❌ ← Fix in Round 2.5

**Tests:**
- Dynamic discovery: ✅ PASS
- Weight updates: ❌ FAIL (features zero)
- Convergence: ❌ FAIL (no gradient)
- Compositions: ✅ PASS (by chance, not learning)

**Discovery:** Feature engineering is the bottleneck, not learning algorithm.

---

## Commitment for Round 2.5

**Will NOT move to Round 3 until:**
1. Features capture spatial transforms
2. Ranker learns on rotation tasks
3. Weights update measurably
4. Scores improve over time

**This is how you do science:**
Fix what's actually broken (features), not what you think is broken (learning algorithm).

---

## Cumulative Lessons

**Rounds completed:** 1 + 2 (partial) → 2.5 (needed)
**Total insights coded:** 2 (compositions + gradient descent)
**Total discoveries:** 1 (feature inadequacy)
**Trajectory:** Learning what doesn't work is progress
