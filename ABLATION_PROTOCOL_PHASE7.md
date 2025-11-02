# ABLATION TESTING PROTOCOL: Phase 7 Hybrid Approach

**Methodology:** User's guidance - "x5 rounds of testing for each component"
**Requirement:** Statistical validation (p < 0.05) before accepting component

---

## PROTOCOL OVERVIEW

For EACH new component added:

1. **5× Baseline Testing** (existing system without new component)
2. **5× Component-Alone Testing** (new component in isolation)
3. **5× Combined Testing** (existing + new component)
4. **Statistical Analysis** (paired t-tests, effect sizes)
5. **3×3 Distillation** (PROs, CONs, ACTIONs)
6. **GO/NO-GO Decision** (p < 0.05 AND improvement ≥ 5%)

**Only if GO:** Accept component and proceed to next

---

## COMPONENT TESTING SCHEDULE

### WEEK 1: Feature Extraction Components

#### Test 1.1: Symmetry Strength Feature
**Baseline:** Simplified 4-feature fuzzy (current)
**New:** Add symmetry_strength feature
**Hypothesis:** Better symmetry detection → improved pattern matching

**Test Conditions:**
- A: Current fuzzy (4 features) - 5 runs
- B: Symmetry alone (1 feature) - 5 runs
- C: Current + Symmetry (5 features) - 5 runs

**Success Criteria:**
- C > A with p < 0.05
- Improvement ≥ 5%

---

#### Test 1.2: Multi-Scale Complexity Feature
**Baseline:** Current + symmetry (5 features)
**New:** Add multi_scale_complexity feature
**Hypothesis:** Hierarchical detection → better large-grid tasks

**Test Conditions:**
- A: 5-feature system - 5 runs
- B: Multi-scale alone - 5 runs
- C: 5-feature + multi-scale (6 features) - 5 runs

**Success Criteria:**
- C > A with p < 0.05
- Improvement ≥ 5%

---

#### Test 1.3: Non-Locality Score Feature
**Continue pattern for remaining features...**

---

#### Test 1.4: Criticality Index Feature

---

#### Test 1.5: Pattern Entropy Feature

---

#### Test 1.6: Grid Size Factor Feature

---

#### Test 1.7: Color Complexity Feature

---

#### Test 1.8: Transformation Consistency Feature

---

### WEEK 2: Fuzzy Rule Components

#### Test 2.1: Symmetry-Dominant Rules (R1-R2)
**Baseline:** 8-feature with 4 simple rules
**New:** Add symmetry-dominant rules
**Hypothesis:** Better symmetry task handling

**Test Conditions:**
- A: 4 simple rules - 5 runs
- B: Symmetry rules alone - 5 runs
- C: 4 simple + symmetry rules - 5 runs

---

#### Test 2.2: Multi-Scale Rules (R3-R4)

---

#### Test 2.3: Non-Local Rules (R5-R6)

---

#### Test 2.4: Criticality Rules (R7-R8)

---

#### Test 2.5: Meta-Learning Rules (R9-R10)

---

#### Test 2.6: Computational Budget Rules (R11-R12)

---

#### Test 2.7: Color Complexity Rules (R13-R14)

---

#### Test 2.8: Fallback Rules (R15+)

---

### WEEK 3: NSPSA Integration Components

#### Test 3.1: Basic NSPSA Connection
**Baseline:** Full fuzzy (8 features, 50+ rules)
**New:** NSPSA integration (no weighting)
**Hypothesis:** Program synthesis adds algorithmic reasoning

**Test Conditions:**
- A: Fuzzy BOLT-ONs only - 5 runs
- B: NSPSA only - 5 runs
- C: Fuzzy + NSPSA (basic) - 5 runs

---

#### Test 3.2: Weighted NSPSA Integration
**Baseline:** Basic fuzzy + NSPSA
**New:** Fuzzy-controlled weighting
**Hypothesis:** Adaptive weighting > always-on

---

#### Test 3.3: Beam Search Optimization

---

#### Test 3.4: Composition Primitives

---

### WEEK 4: System Optimization Components

#### Test 4.1: Prediction Aggregation Strategy

---

#### Test 4.2: Learned Verifier

---

#### Test 4.3: Confidence Thresholding

---

---

## STATISTICAL ANALYSIS TEMPLATE

For each test:

```python
# Collect results
condition_a_accuracies = [run1, run2, run3, run4, run5]
condition_b_accuracies = [run1, run2, run3, run4, run5]
condition_c_accuracies = [run1, run2, run3, run4, run5]

# Summary statistics
mean_a, std_a, ci_a = compute_stats(condition_a_accuracies)
mean_b, std_b, ci_b = compute_stats(condition_b_accuracies)
mean_c, std_c, ci_c = compute_stats(condition_c_accuracies)

# Paired t-test: C vs A
t_stat, p_value = ttest_rel(condition_c_accuracies, condition_a_accuracies)

# Effect size (Cohen's d)
cohens_d = (mean_c - mean_a) / pooled_std

# Decision
if p_value < 0.05 and mean_c > mean_a * 1.05:
    decision = "GO - Accept component"
else:
    decision = "NO-GO - Reject or refine component"
```

---

## 3×3 DISTILLATION TEMPLATE

After each test:

### PROs (What worked)
1. ...
2. ...
3. ...

### CONs (What didn't work)
1. ...
2. ...
3. ...

### ACTIONs (What to do next)
1. ...
2. ...
3. ...

---

## DECISION GATES

### Gate 1: End of Week 1 (Features)
**Requirement:** ≥6/8 features accepted (GO)
**Metric:** Improvement over 4-feature baseline ≥ 20%
**Action if fail:** Refine feature extraction, retry failed features

### Gate 2: End of Week 2 (Rules)
**Requirement:** ≥40/50+ rules accepted (GO)
**Metric:** Improvement over simple rules ≥ 30%
**Action if fail:** Simplify rule set, focus on high-impact rules

### Gate 3: End of Week 3 (NSPSA)
**Requirement:** NSPSA integration shows p < 0.05 improvement
**Metric:** Combined system > fuzzy-alone by ≥ 10%
**Action if fail:** Fallback to fuzzy-only, optimize BOLT-ONs

### Gate 4: End of Week 4 (Optimization)
**Requirement:** 100-task accuracy ≥ 30%
**Metric:** Statistical significance maintained
**Action if fail:** Focus on task categories where we excel

---

## AUTOMATED TESTING HARNESS

```python
class AblationTestHarness:
    """
    Automates ablation testing protocol.

    Usage:
        harness = AblationTestHarness()
        harness.test_component(
            name="Symmetry Feature",
            baseline_solver=current_fuzzy,
            component_solver=symmetry_only,
            combined_solver=current_plus_symmetry,
            num_runs=5
        )
    """

    def test_component(self, name, baseline_solver, component_solver,
                      combined_solver, num_runs=5):
        """Run full ablation test on component."""

        # Condition A: Baseline (5 runs)
        baseline_results = self.run_multiple(baseline_solver, num_runs)

        # Condition B: Component alone (5 runs)
        component_results = self.run_multiple(component_solver, num_runs)

        # Condition C: Combined (5 runs)
        combined_results = self.run_multiple(combined_solver, num_runs)

        # Statistical analysis
        stats = self.analyze_results(baseline_results, combined_results)

        # 3×3 distillation
        distillation = self.distill_results(
            baseline_results,
            component_results,
            combined_results
        )

        # GO/NO-GO decision
        decision = self.make_decision(stats)

        # Generate report
        self.generate_report(name, stats, distillation, decision)

        return decision
```

---

## CHECKPOINTS & REPORTS

### Daily Checkpoint
- Component being tested
- Preliminary results (if available)
- Blockers or issues

### Weekly Report
- Components tested (X/Y accepted)
- Cumulative performance improvement
- Decision gate status
- Next week plan

### Phase Completion Report
- All components tested
- Final system performance
- Comparison to projections
- Recommendations

---

## COMMITMENT

✅ **NO component accepted without 5× testing**
✅ **NO component accepted without p < 0.05**
✅ **ALL components get 3×3 distillation**
✅ **STRICT adherence to decision gates**

**User's methodology will be followed rigorously throughout Phase 7!**

---

## STATUS

**Protocol:** ACTIVE
**Phase:** 7 - Week 1
**Next Test:** 1.1 - Symmetry Strength Feature
**Ready to begin systematic ablation testing!**
