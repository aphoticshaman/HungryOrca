# PHASE EXECUTION ROADMAP - 3 Phases to SOTA
## Bolt-On Feature Development with Iterative Ablation Testing

**Current Status:** 53.9% partial, 2 exact (20%)
**Target:** 80-90% partial, 60-70% exact (SOTA competitive)
**Strategy:** Bolt-on additions with x5 ablation per feature
**Architecture:** KEEP existing evolving specialist system ✅

---

## 🎯 REBUILD ASSESSMENT

### Do We Need Ground-Up Rebuild?

**Answer: NO! ❌**

**Why Architecture is Solid:**
```python
✅ Evolving specialist system works
✅ CollectiveKnowledge cross-learning proven
✅ Memory/adaptation functional
✅ Fixed flood-fill validated (+2.1%)
✅ Elite insight detection framework ready
✅ Proportional time management correct
✅ Structure preservation active
```

**What We Need:**
```
✅ BOLT-ON additions (not rebuild)
✅ x5 ablation testing per feature
✅ Incremental integration
✅ Keep what works, add what's missing
```

**Architecture Extensibility:**
```python
# Adding new specialist:
class NewSpecialist(EvolvingSpecialist):
    def __init__(self):
        super().__init__("NewSpecialist", "new_capability")

    def _try_strategy(self, strategy, train_pairs, test_input, time_budget):
        # New logic here
        return result, score, notes

# Add to solver:
solver.specialists.append(NewSpecialist())
# Done! No rebuild needed.
```

---

## 📋 THREE-PHASE EXECUTION PLAN

### PHASE 1: QUICK WINS (2-3 weeks)
**Target:** 60-70% partial, 5-8 exact (50-80%)
**Focus:** Trivial/low-complexity high-ROI bolt-ons

### PHASE 2: CORE ENHANCEMENTS (4-6 weeks cumulative)
**Target:** 70-80% partial, 12-18 exact (60-90%)
**Focus:** Medium-complexity fundamental improvements

### PHASE 3: ADVANCED METHODS (8-12 weeks cumulative)
**Target:** 80-90% partial, 25-35 exact (SOTA!)
**Focus:** Complex techniques + verification

---

## 🚀 PHASE 1: QUICK WINS (2-3 weeks)

### Bolt-On #1: Consistency Scoring
**Expected Gain:** +3-5%
**Complexity:** TRIVIAL (one line!)
**Priority:** IMMEDIATE

**Current Code:**
```python
# evolving_specialist_system.py, line ~XX
score = np.mean(scores)
```

**New Code:**
```python
score = np.mean(scores) * (1.0 - np.std(scores))
# Penalize inconsistency across examples
```

**Ablation Protocol:**
```
Condition A: Without consistency penalty (current)
Condition B: With consistency penalty (mean * (1 - std))
Condition C: With stronger penalty (mean * (1 - 2*std))

Test: 10 tasks × 5 runs × 3 conditions = 150 tests
Decision: Keep if B or C > A with p < 0.05
```

**Implementation Time:** 30 minutes
**Testing Time:** 2 hours

---

### Bolt-On #2: Multi-Step Chaining (Complete)
**Expected Gain:** +10-15%
**Complexity:** LOW (already 50% done)
**Priority:** HIGH

**Current Status:**
- multi_step_chaining_solver.py exists
- Tested HYBRID params (50.8%)
- Need: Integration into main solver

**Implementation:**
```python
# Add ChainingSpecialist to evolving_specialist_system.py
class ChainingSpecialist(EvolvingSpecialist):
    """Tries 2-3 step transformation sequences."""

    def __init__(self):
        super().__init__("ChainingSpecialist", "multi_step_composition")
        self.atomic_transforms = [
            'rotate_90', 'flip_h', 'flip_v',
            'color_map', 'fill', 'crop', 'scale'
        ]

    def _try_strategy(self, strategy, train_pairs, test_input, time_budget):
        # Try 2-step chains
        for t1 in self.atomic_transforms:
            for t2 in self.atomic_transforms:
                result = apply_transform(apply_transform(test_input, t1), t2)
                score = evaluate_on_training(result, train_pairs)
                if score > 0.8:
                    return result, score, f"Chain: {t1} → {t2}"
        return None, 0.0, "No chain found"
```

**Ablation Protocol:**
```
Condition A: Without ChainingSpecialist (current 5 specialists)
Condition B: With ChainingSpecialist (6 specialists)
Condition C: With ChainingSpecialist + longer chains (3-step)

Test: 10 tasks × 5 runs × 3 conditions = 150 tests
Decision: Keep if improvement > +5% with p < 0.05
```

**Implementation Time:** 4 hours
**Testing Time:** 3 hours

---

### Bolt-On #3: Object-Level Specialist
**Expected Gain:** +5-10%
**Complexity:** MEDIUM
**Priority:** HIGH

**New Specialist:**
```python
class ObjectSpecialist(EvolvingSpecialist):
    """Segments into objects, transforms per-object, recomposes."""

    def __init__(self):
        super().__init__("ObjectSpecialist", "object_level_transforms")

    def _try_strategy(self, strategy, train_pairs, test_input, time_budget):
        # Segment into connected components
        objects = self._segment_objects(test_input)

        # Try transforming each object independently
        for transform in ['rotate', 'flip', 'scale']:
            transformed_objects = [apply(obj, transform) for obj in objects]
            result = self._recompose(transformed_objects, test_input.shape)

            score = evaluate_on_training(result, train_pairs)
            if score > 0.7:
                return result, score, f"Object-level: {transform}"

        return None, 0.0, "No object transform found"

    def _segment_objects(self, grid):
        """Connected component analysis."""
        from collections import deque
        # Use existing component labeling
        labeled, num = label_components(grid != 0)

        objects = []
        for obj_id in range(1, num + 1):
            mask = (labeled == obj_id)
            objects.append((mask, grid[mask]))

        return objects

    def _recompose(self, transformed_objects, target_shape):
        """Place transformed objects back."""
        result = np.zeros(target_shape, dtype=int)
        for mask, values in transformed_objects:
            result[mask] = values
        return result
```

**Ablation Protocol:**
```
Condition A: Without ObjectSpecialist (current)
Condition B: With ObjectSpecialist (7 specialists)
Condition C: With ObjectSpecialist + enhanced segmentation

Test: 10 tasks × 5 runs × 3 conditions = 150 tests
Focus: Tasks with multiple disconnected objects
Decision: Keep if improvement on object-based tasks > +10%
```

**Implementation Time:** 6 hours
**Testing Time:** 4 hours

---

### Bolt-On #4: Adaptive Size Learning
**Expected Gain:** +5-8%
**Complexity:** LOW
**Priority:** MEDIUM

**New Specialist:**
```python
class SizeAdaptiveSpecialist(EvolvingSpecialist):
    """Learns input→output size relationships from training."""

    def __init__(self):
        super().__init__("SizeAdaptiveSpecialist", "size_relationships")

    def _observe_puzzle(self, train_pairs, test_input):
        """Learn size patterns."""
        chars = super()._observe_puzzle(train_pairs, test_input)

        # Learn size transformations
        size_rules = []
        for inp, out in train_pairs:
            rule = self._infer_size_rule(inp.shape, out.shape)
            size_rules.append(rule)

        # Find consistent rule
        if all(r == size_rules[0] for r in size_rules):
            chars['size_rule'] = size_rules[0]

        return chars

    def _infer_size_rule(self, in_shape, out_shape):
        """Infer size transformation rule."""
        h_in, w_in = in_shape
        h_out, w_out = out_shape

        # Test hypotheses
        if (h_out, w_out) == (h_in, w_in):
            return 'identity'
        elif (h_out, w_out) == (h_in * 2, w_in * 2):
            return 'double'
        elif (h_out, w_out) == (h_in * 3, w_in * 3):
            return 'triple'
        elif (h_out, w_out) == (h_in // 2, w_in // 2):
            return 'half'
        elif h_out < h_in and w_out < w_in:
            return 'crop_to_content'
        elif h_out > h_in or w_out > w_in:
            return f'tile_{h_out//h_in}x{w_out//w_in}'
        else:
            return 'unknown'

    def _try_strategy(self, strategy, train_pairs, test_input, time_budget):
        chars = self._observe_puzzle(train_pairs, test_input)

        if 'size_rule' not in chars:
            return None, 0.0, "No consistent size rule"

        rule = chars['size_rule']

        # Apply size transformation
        if rule == 'double':
            result = self._resize(test_input, scale=2)
        elif rule == 'triple':
            result = self._resize(test_input, scale=3)
        elif rule == 'crop_to_content':
            result = self._crop_to_content(test_input)
        # ... more rules

        return result, 0.85, f"Size rule: {rule}"
```

**Ablation Protocol:**
```
Condition A: Without SizeAdaptiveSpecialist
Condition B: With SizeAdaptiveSpecialist
Condition C: With SizeAdaptiveSpecialist + learned crop bounds

Test: 10 tasks × 5 runs × 3 conditions = 150 tests
Focus: Tasks where output size differs from input
Decision: Keep if improvement on size-changing tasks > +8%
```

**Implementation Time:** 5 hours
**Testing Time:** 3 hours

---

### PHASE 1 SUMMARY

| Bolt-On | Gain | Time (impl) | Time (test) | Complexity |
|---------|------|-------------|-------------|------------|
| #1: Consistency Scoring | +3-5% | 0.5h | 2h | TRIVIAL |
| #2: Multi-Step Chaining | +10-15% | 4h | 3h | LOW |
| #3: Object-Level | +5-10% | 6h | 4h | MEDIUM |
| #4: Adaptive Size | +5-8% | 5h | 3h | LOW |
| **TOTAL** | **+23-38%** | **15.5h** | **12h** | **~3 weeks** |

**Expected Performance After Phase 1:**
```
Current:  53.9% partial, 2 exact (20%)
Phase 1:  65-75% partial, 6-10 exact (60-100%)
```

---

## 🔬 PHASE 2: CORE ENHANCEMENTS (4-6 weeks cumulative)

### Bolt-On #5: CSP/SAT Solver Specialist
**Expected Gain:** +6-10%
**Complexity:** MEDIUM
**Priority:** HIGHEST (biggest single gain!)

**New Specialist:**
```python
class CSPSpecialist(EvolvingSpecialist):
    """Formulates puzzle as Constraint Satisfaction Problem."""

    def __init__(self):
        super().__init__("CSPSpecialist", "constraint_satisfaction")

    def _try_strategy(self, strategy, train_pairs, test_input, time_budget):
        # Extract constraints from training
        constraints = self._extract_constraints(train_pairs)

        if not constraints:
            return None, 0.0, "No constraints detected"

        # Formulate as CSP
        try:
            from z3 import *
            solver = Solver()

            # Create variables for each cell
            h, w = test_input.shape
            cells = [[Int(f'c_{i}_{j}') for j in range(w)] for i in range(h)]

            # Add constraints
            for constraint in constraints:
                solver.add(constraint.to_z3(cells))

            # Solve
            if solver.check() == sat:
                model = solver.model()
                result = self._extract_grid(model, cells)
                return result, 0.95, f"CSP solved: {len(constraints)} constraints"
            else:
                return None, 0.0, "CSP unsatisfiable"

        except ImportError:
            return None, 0.0, "Z3 not available"

    def _extract_constraints(self, train_pairs):
        """Extract constraints from training examples."""
        constraints = []

        for inp, out in train_pairs:
            # Check for Sudoku-like constraints
            if self._is_latin_square(out):
                constraints.append(LatinSquareConstraint())

            # Check for graph coloring
            if self._is_graph_coloring(out):
                constraints.append(NoAdjacentSameColorConstraint())

            # Check for sum constraints
            row_sums = [out[i, :].sum() for i in range(out.shape[0])]
            if len(set(row_sums)) == 1:
                constraints.append(ConstantRowSumConstraint(row_sums[0]))

        return constraints
```

**Ablation Protocol:**
```
Setup: pip install z3-solver (test on Kaggle notebook)

Condition A: Without CSPSpecialist
Condition B: With CSPSpecialist (basic constraints)
Condition C: With CSPSpecialist (advanced constraints)

Test: 20 tasks × 5 runs × 3 conditions = 300 tests
Focus: Sudoku-like, graph coloring, constraint-heavy puzzles
Decision: Keep if improvement on constraint tasks > +10%
```

**Implementation Time:** 8 hours
**Testing Time:** 5 hours
**External Dependency:** `z3-solver` (test Kaggle availability)

---

### Bolt-On #6: Spectral Analysis Specialist
**Expected Gain:** +4-6%
**Complexity:** MEDIUM-HIGH
**Priority:** MEDIUM

**New Specialist:**
```python
class SpectralSpecialist(EvolvingSpecialist):
    """Uses graph Laplacian for global structure analysis."""

    def __init__(self):
        super().__init__("SpectralSpecialist", "spectral_analysis")

    def _try_strategy(self, strategy, train_pairs, test_input, time_budget):
        # Build graph from grid
        A = self._build_adjacency(test_input)

        # Compute Laplacian
        D = np.diag(A.sum(axis=1))
        L = D - A

        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(L)

        # Use Fiedler vector (2nd eigenvector) for partitioning
        fiedler = eigenvectors[:, 1]

        # Partition based on sign
        h, w = test_input.shape
        partition = fiedler.reshape(h, w) > 0

        # Apply different colors to partitions
        result = test_input.copy()
        result[partition] = self._learn_partition_color(train_pairs, partition=True)
        result[~partition] = self._learn_partition_color(train_pairs, partition=False)

        score = self._score_on_training(result, train_pairs)
        return result, score, "Spectral partitioning"

    def _build_adjacency(self, grid):
        """Build adjacency matrix from grid (4-connectivity)."""
        h, w = grid.shape
        n = h * w
        A = np.zeros((n, n))

        for i in range(h):
            for j in range(w):
                idx = i * w + j

                # Connect to neighbors
                for di, dj in [(0,1), (1,0), (0,-1), (-1,0)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < h and 0 <= nj < w:
                        nidx = ni * w + nj
                        # Weight by color similarity
                        A[idx, nidx] = 1.0 if grid[i,j] == grid[ni,nj] else 0.5

        return A
```

**Ablation Protocol:**
```
Condition A: Without SpectralSpecialist
Condition B: With SpectralSpecialist (Fiedler partitioning)
Condition C: With SpectralSpecialist (multi-level clustering)

Test: 10 tasks × 5 runs × 3 conditions = 150 tests
Focus: Tasks with non-local spatial relationships
Decision: Keep if improvement on complex spatial tasks > +5%
```

**Implementation Time:** 10 hours
**Testing Time:** 5 hours

---

### Bolt-On #7: 3D Projection Specialist
**Expected Gain:** +3-5%
**Complexity:** MEDIUM
**Priority:** LOW-MEDIUM

**New Specialist:**
```python
class ProjectionSpecialist(EvolvingSpecialist):
    """Embeds 2D in 3D (color=depth), applies 3D transforms, projects back."""

    def __init__(self):
        super().__init__("ProjectionSpecialist", "3d_projection")

    def _try_strategy(self, strategy, train_pairs, test_input, time_budget):
        # Embed in 3D (color as z-coordinate)
        points_3d = self._embed_3d(test_input)

        # Try different 3D rotations
        for axis in ['x', 'y', 'z']:
            for angle in [90, 180, 270]:
                rotated = self._rotate_3d(points_3d, axis, angle)
                projected = self._project_2d(rotated)

                score = self._score_on_training(projected, train_pairs)
                if score > 0.7:
                    return projected, score, f"3D rotation: {axis} {angle}°"

        return None, 0.0, "No 3D transform found"

    def _embed_3d(self, grid):
        """Embed 2D grid in 3D (color = z)."""
        h, w = grid.shape
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        z = grid
        return np.stack([x, y, z], axis=-1)

    def _rotate_3d(self, points, axis, angle_deg):
        """Apply 3D rotation."""
        angle = np.radians(angle_deg)

        if axis == 'x':
            R = np.array([
                [1, 0, 0],
                [0, np.cos(angle), -np.sin(angle)],
                [0, np.sin(angle), np.cos(angle)]
            ])
        elif axis == 'y':
            R = np.array([
                [np.cos(angle), 0, np.sin(angle)],
                [0, 1, 0],
                [-np.sin(angle), 0, np.cos(angle)]
            ])
        else:  # z
            R = np.array([
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1]
            ])

        # Apply rotation
        rotated = np.einsum('ij,hwj->hwi', R, points)
        return rotated

    def _project_2d(self, points_3d):
        """Project 3D back to 2D (orthogonal projection)."""
        # Take z-coordinate as color
        return points_3d[:, :, 2].astype(int)
```

**Ablation Protocol:**
```
Condition A: Without ProjectionSpecialist
Condition B: With ProjectionSpecialist (basic rotations)
Condition C: With ProjectionSpecialist (perspective projection)

Test: 10 tasks × 5 runs × 3 conditions = 150 tests
Focus: Tasks with "impossible" 2D rotations
Decision: Keep if improvement on projection tasks > +5%
```

**Implementation Time:** 7 hours
**Testing Time:** 4 hours

---

### PHASE 2 SUMMARY

| Bolt-On | Gain | Time (impl) | Time (test) | Complexity |
|---------|------|-------------|-------------|------------|
| #5: CSP/SAT Solver | +6-10% | 8h | 5h | MEDIUM |
| #6: Spectral Analysis | +4-6% | 10h | 5h | MEDIUM-HIGH |
| #7: 3D Projection | +3-5% | 7h | 4h | MEDIUM |
| **TOTAL** | **+13-21%** | **25h** | **14h** | **~4 weeks** |

**Expected Performance After Phase 2:**
```
After Phase 1: 65-75% partial, 6-10 exact
After Phase 2: 75-85% partial, 15-22 exact (75-110%)
```

---

## 🚀 PHASE 3: ADVANCED METHODS (8-12 weeks cumulative)

### Bolt-On #8: Topological Invariant Specialist
**Expected Gain:** +3-5%
**Complexity:** HIGH
**Priority:** LOW

**Requires:** Full Betti number computation (homology library or manual implementation)

---

### Bolt-On #9: Nearest-Valid-Pattern Specialist
**Expected Gain:** +3-5%
**Complexity:** MEDIUM
**Priority:** MEDIUM

**Concept:** Learn manifold of valid patterns from training, project noisy input onto manifold

---

### Bolt-On #10: Iterative Refinement System
**Expected Gain:** +5-10%
**Complexity:** MEDIUM-HIGH
**Priority:** HIGH

**New System (not specialist, meta-layer):**
```python
class IterativeRefinementSystem:
    """Improves solution through multiple rounds."""

    def refine(self, initial_solution, train_pairs, test_input, max_iterations=5):
        current = initial_solution
        current_score = score_on_training(current, train_pairs)

        for iteration in range(max_iterations):
            # Identify wrong cells
            wrong_cells = self._find_wrong_cells(current, train_pairs)

            if not wrong_cells:
                break

            # Try fixing wrong cells
            improved = self._fix_cells(current, wrong_cells, train_pairs)
            new_score = score_on_training(improved, train_pairs)

            if new_score > current_score:
                current = improved
                current_score = new_score
            else:
                break  # No more improvement

        return current, current_score
```

---

### Bolt-On #11: Verification Integration
**Expected Gain:** +5-10% confidence boost
**Complexity:** HIGH
**Priority:** MEDIUM

**Integrate MASTER branch's verification framework:**
- SMT solver checks for high-confidence solutions
- Cell-by-cell verification with proofs
- Formal verification for 90%→100% boost

---

### PHASE 3 SUMMARY

| Bolt-On | Gain | Time (impl) | Time (test) | Complexity |
|---------|------|-------------|-------------|------------|
| #8: Topological | +3-5% | 12h | 6h | HIGH |
| #9: Nearest-Valid | +3-5% | 8h | 4h | MEDIUM |
| #10: Iterative Refinement | +5-10% | 10h | 5h | MEDIUM-HIGH |
| #11: Verification | +5-10% conf | 15h | 7h | HIGH |
| **TOTAL** | **+16-30%** | **45h** | **22h** | **~4 weeks** |

**Expected Performance After Phase 3:**
```
After Phase 2: 75-85% partial, 15-22 exact
After Phase 3: 85-95% partial, 30-45 exact (SOTA!)
```

---

## 📊 COMPLETE EXECUTION TIMELINE

### Cumulative Progress

| Phase | Weeks | Bolt-Ons | Impl Hours | Test Hours | Expected Partial | Expected Exact |
|-------|-------|----------|------------|------------|------------------|----------------|
| **Current** | 0 | 0 | - | - | 53.9% | 2 (20%) |
| **Phase 1** | 2-3 | 4 | 15.5 | 12 | 65-75% | 6-10 (60-100%) |
| **Phase 2** | 6-7 | 7 | 40.5 | 26 | 75-85% | 15-22 (75-110%) |
| **Phase 3** | 10-12 | 11 | 85.5 | 48 | **85-95%** | **30-45 (SOTA!)** |

### Time Breakdown
```
Implementation:  85.5 hours (~2 weeks full-time)
Testing:         48 hours (~1 week full-time)
Documentation:   20 hours (~2 days)
TOTAL:          153.5 hours (~4 weeks full-time, 12 weeks part-time)
```

---

## 🔬 ABLATION TESTING PROTOCOL

### Standard x5 Protocol for Each Bolt-On

**Setup:**
```
- Test set: 10-20 representative tasks
- Runs per condition: 5 (statistical validity)
- Conditions: Always 3 (A=baseline, B=new, C=variant)
- Total tests per bolt-on: 10 tasks × 5 runs × 3 conditions = 150 tests
```

**Execution:**
```python
def ablation_test_bolton(bolton_name, test_tasks=10, runs=5):
    """Standard ablation protocol."""

    results = {
        'A_baseline': [],
        'B_with_bolton': [],
        'C_variant': []
    }

    for task_id in test_tasks:
        for run in range(runs):
            # Condition A: Current system
            score_a = run_solver(task_id, include_bolton=False)
            results['A_baseline'].append(score_a)

            # Condition B: With new bolt-on
            score_b = run_solver(task_id, include_bolton=True, variant='standard')
            results['B_with_bolton'].append(score_b)

            # Condition C: Variant configuration
            score_c = run_solver(task_id, include_bolton=True, variant='enhanced')
            results['C_variant'].append(score_c)

    # Statistical analysis
    improvement_b = np.mean(results['B_with_bolton']) - np.mean(results['A_baseline'])
    improvement_c = np.mean(results['C_variant']) - np.mean(results['A_baseline'])

    # Paired t-test
    t_stat, p_value = paired_ttest(results['B_with_bolton'], results['A_baseline'])

    # Decision
    if p_value < 0.05 and improvement_b > 0.03:  # >3% gain, significant
        return 'ACCEPT', improvement_b
    else:
        return 'REJECT', improvement_b
```

**Decision Criteria:**
```
ACCEPT if:
  - p-value < 0.05 (statistically significant)
  - improvement > 3% (meaningful gain)
  - no regression on any task category

REJECT if:
  - p-value >= 0.05 (not significant)
  - improvement < 3% (not worth complexity)
  - causes regression on important tasks

ITERATE if:
  - marginal results (2-3% gain, p ~ 0.05-0.10)
  - try variant C or different parameters
```

---

## 🎯 RISK MITIGATION

### What If Bolt-On Fails Ablation?

**Backup Plans:**
1. **Try variant configuration** (Condition C)
2. **Reduce complexity** (simpler version)
3. **Different integration point** (different specialist)
4. **Shelve for later** (may need prerequisite)
5. **Document and move on** (not all insights work)

**Example:**
```
Bolt-On #6 (Spectral) fails ablation:
→ Try simpler Laplacian (Condition C)
→ Still fails? Try on specific task subset
→ Still fails? Document as "needs more research"
→ Move to Bolt-On #7, don't waste time
```

### Continuous Integration

**After Each Bolt-On:**
```python
# Run regression test suite
def regression_test():
    """Ensure no performance degradation."""

    # Test on known-good tasks
    baseline_tasks = ['009d5c81', '00d62c1b']  # Our 2 exact matches

    for task_id in baseline_tasks:
        score = run_solver(task_id)
        assert score >= previous_score[task_id], f"REGRESSION on {task_id}!"

    print("✅ No regression detected")
```

---

## 📋 IMPLEMENTATION CHECKLIST

### Phase 1 (Weeks 1-3)

- [ ] **Week 1: Bolt-On #1 (Consistency)**
  - [ ] Implement (30 min)
  - [ ] Run ablation x5 (2 hours)
  - [ ] Analyze results
  - [ ] Accept/reject decision
  - [ ] Commit if accepted

- [ ] **Week 1: Bolt-On #2 (Chaining)**
  - [ ] Implement ChainingSpecialist (4 hours)
  - [ ] Integrate into main solver
  - [ ] Run ablation x5 (3 hours)
  - [ ] Tune parameters if needed
  - [ ] Commit if accepted

- [ ] **Week 2: Bolt-On #3 (Object-Level)**
  - [ ] Implement ObjectSpecialist (6 hours)
  - [ ] Test segmentation quality
  - [ ] Run ablation x5 (4 hours)
  - [ ] Verify on object-heavy tasks
  - [ ] Commit if accepted

- [ ] **Week 2-3: Bolt-On #4 (Size Adaptive)**
  - [ ] Implement SizeAdaptiveSpecialist (5 hours)
  - [ ] Add all size rules
  - [ ] Run ablation x5 (3 hours)
  - [ ] Test on size-changing tasks
  - [ ] Commit if accepted

- [ ] **Week 3: Phase 1 Integration**
  - [ ] Regenerate submission.json
  - [ ] Full validation on training set
  - [ ] Document improvements
  - [ ] Push to branch

### Phase 2 (Weeks 4-7)

- [ ] **Week 4: Bolt-On #5 (CSP)**
  - [ ] Test z3-solver availability
  - [ ] Implement CSPSpecialist (8 hours)
  - [ ] Add constraint extractors
  - [ ] Run ablation x5 (5 hours)
  - [ ] Commit if accepted

- [ ] **Week 5-6: Bolt-On #6 (Spectral)**
  - [ ] Implement SpectralSpecialist (10 hours)
  - [ ] Test eigendecomposition
  - [ ] Run ablation x5 (5 hours)
  - [ ] Tune clustering parameters
  - [ ] Commit if accepted

- [ ] **Week 6-7: Bolt-On #7 (3D)**
  - [ ] Implement ProjectionSpecialist (7 hours)
  - [ ] Test 3D rotations
  - [ ] Run ablation x5 (4 hours)
  - [ ] Verify projection math
  - [ ] Commit if accepted

- [ ] **Week 7: Phase 2 Integration**
  - [ ] Regenerate submission.json
  - [ ] Compare to Phase 1 baseline
  - [ ] Document improvements
  - [ ] Push to branch

### Phase 3 (Weeks 8-12)

- [ ] **Week 8-9: Bolt-On #10 (Iterative Refinement)**
  - [ ] Implement refinement system (10 hours)
  - [ ] Test on near-miss tasks
  - [ ] Run ablation x5 (5 hours)
  - [ ] Tune iteration parameters
  - [ ] Commit if accepted

- [ ] **Week 9-10: Bolt-On #8, #9**
  - [ ] Implement remaining specialists
  - [ ] Run ablations
  - [ ] Accept/reject decisions

- [ ] **Week 11-12: Bolt-On #11 (Verification)**
  - [ ] Port verification from MASTER
  - [ ] Integrate SMT solver
  - [ ] Test on high-confidence cases
  - [ ] Document formal proofs
  - [ ] Final integration

- [ ] **Week 12: Final Validation**
  - [ ] Complete test suite
  - [ ] Performance report
  - [ ] Final submission.json
  - [ ] Documentation complete

---

## 🎯 FINAL TARGET

### Week 12 Expected Performance

```
Metric                    Current    Target     Improvement
────────────────────────────────────────────────────────────
Partial Match (avg)       53.9%      85-95%     +31-41%
Exact Match (count)       2/10       30-45/200  +1400-2150%
Exact Match (%)           20%        60-90%     +40-70%
High Similarity (>90%)    20%        70-85%     +50-65%

Status: SOTA COMPETITIVE! 🏆
```

### Competition Readiness

```
✅ Architecture: Solid, extensible, no rebuild needed
✅ Methodology: x5 ablation per feature, statistically valid
✅ Integration: Bolt-on approach, minimal risk
✅ Timeline: 12 weeks part-time to SOTA
✅ Validation: Continuous regression testing
✅ Documentation: Complete tracking of all decisions
```

---

## 🚀 CONCLUSION

**ARCHITECTURE DECISION: NO REBUILD!**

Your evolving specialist system is perfect for bolt-ons. Each new specialist:
- Inherits learning/memory/adaptation ✅
- Integrates via simple append ✅
- Tested independently via ablation ✅
- No disruption to working code ✅

**EXECUTION PLAN:**
- 11 bolt-ons over 3 phases
- Each tested with x5 ablation protocol
- Incremental gains: +3% to +15% per feature
- Total expected: +50-89% improvement
- **Target: 85-95% partial, 60-90% exact (SOTA!)**

**START IMMEDIATELY:** Bolt-On #1 (consistency scoring) - 30 minutes to implement! 🚀

---

*Phase Execution Roadmap v1.0*
*Created: 2025-11-02*
*Architecture: KEEP existing (extensible)*
*Strategy: Bolt-on with ablation testing*
*Timeline: 12 weeks to SOTA*
*Risk: LOW (incremental, tested)*
*Status: READY TO EXECUTE*
