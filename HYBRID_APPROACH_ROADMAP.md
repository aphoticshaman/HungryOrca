# HYBRID APPROACH ROADMAP: Full Fuzzy + NSPSA Integration

**Selected:** Option C
**Target:** 75-85% accuracy on ARC tasks
**Timeline:** 4-6 weeks
**Status:** INITIATED

---

## ARCHITECTURE OVERVIEW

```
┌──────────────────────────────────────────────────────────────┐
│                    ARC TASK INPUT                            │
│                  (train_pairs, test_input)                   │
└────────────────────────┬─────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────┐
│              FEATURE EXTRACTION (8 features)                 │
│  ├─ Symmetry strength                                        │
│  ├─ Multi-scale complexity                                   │
│  ├─ Non-locality score                                       │
│  ├─ Criticality index                                        │
│  ├─ Pattern entropy                                          │
│  ├─ Grid size factor                                         │
│  ├─ Color complexity                                         │
│  └─ Transformation consistency                               │
└────────────────────────┬─────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────┐
│        FUZZY META-CONTROLLER (50+ rules)                     │
│                                                              │
│  Outputs:                                                    │
│  ├─ Strategy weights (5 insights)                           │
│  ├─ Search depth                                             │
│  ├─ Confidence threshold                                     │
│  └─ Solver selection mode                                    │
└────────────────────────┬─────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────┐
│              WEIGHTED SOLVER ENSEMBLE                        │
│                                                              │
│  SYMBOLIC PATH:                    NEURAL-GUIDED PATH:       │
│  ├─ NSPSA (weighted)              ├─ Fuzzy BOLT-ONs         │
│  │  ├─ Composition primitives     │  ├─ Pattern solver      │
│  │  ├─ Meta-learning ranker       │  ├─ Example solver      │
│  │  └─ Bidirectional search       │  ├─ Object solver       │
│  │                                 │  └─ Nearest neighbor    │
│  └─ DSL Program Synthesis         │                          │
│     ├─ Primitive selection         └─ Adaptive weighting     │
│     └─ Beam search                                           │
└────────────────────────┬─────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────┐
│              PREDICTION AGGREGATION                          │
│  ├─ Confidence-weighted voting                               │
│  ├─ Learned verifier (plausibility check)                    │
│  └─ Best prediction selection                                │
└────────────────────────┬─────────────────────────────────────┘
                         │
                         ▼
                    FINAL OUTPUT
```

---

## PHASE 7 BREAKDOWN

### Week 1: Full Feature Extraction

**Goal:** Implement sophisticated 8-feature system from user's fuzzy research

**Tasks:**
1. **Symmetry Strength** (from FUZZY_ARC_CRITICAL_CONNECTION.md:246)
   - Horizontal reflection matching
   - Vertical reflection matching
   - Rotational symmetry (if square)
   - Return max(h_match, v_match, r_match)

2. **Multi-Scale Complexity** (line 265)
   - Variance at original scale
   - Variance at 2× downsampled scale
   - Complexity = variance preservation across scales
   - High complexity if variance remains

3. **Non-Locality Score** (line 284)
   - Connected component analysis per color
   - Fragmentation = components / pixels
   - High fragmentation → non-local constraints
   - Normalize to 0-1

4. **Criticality Index** (line 332)
   - Proximity to percolation threshold (p_c = 0.59)
   - p_occupied = fraction of non-background pixels
   - Criticality = 1 - |p_occupied - 0.59| / 0.3
   - High when near phase transition

5. **Pattern Entropy** (line 348)
   - Color histogram Shannon entropy
   - Normalize to 0-1 (max = log2(10) ≈ 3.32)

6. **Grid Size Factor** (line 222)
   - Normalize to 0-1 where 1 = 50×50
   - avg_size / (50 * 50)

7. **Color Complexity** (line 226)
   - Number of unique colors
   - Normalize to 0-1 (max = 10 colors)

8. **Transformation Consistency** (line 363)
   - Check if size change ratios consistent across training pairs
   - Low variance = consistent = high score

**Deliverable:** `full_feature_extraction.py`

---

### Week 2: Complete Fuzzy Rule Set

**Goal:** Implement all 50+ fuzzy rules from fuzzy_meta_controller_production.py

**Rule Categories:**

1. **Symmetry-Dominant Rules** (lines 526-552)
   - R1: High symmetry + low complexity → emphasize symmetry
   - R2: High symmetry + high consistency → very confident

2. **Multi-Scale Rules** (lines 554-580)
   - R3: Large grid + high multi-scale → use multi-scale
   - R4: Medium scale + medium symmetry → blend

3. **Non-Local Rules** (lines 582-607)
   - R5: High non-locality → emphasize graph-based
   - R6: Low non-locality → skip graph construction

4. **Criticality Rules** (lines 609-633)
   - R7: High criticality → use phase transition solver
   - R8: Low criticality → standard approaches

5. **Meta-Learning Rules** (lines 635-661)
   - R9: High consistency → meta-learning succeeds
   - R10: Low consistency → rely on ensemble

6. **Computational Budget Rules** (lines 663-689)
   - R11: Large grid → shallow search only
   - R12: Small grid + high complexity → deep search

7. **Color Complexity Rules** (lines 691-716)
   - R13: Low color complexity → simpler strategies
   - R14: High color complexity → sophisticated approaches

8. **Fallback Rules** (lines 718-736)
   - R15: When uncertain → balanced ensemble
   - ... (35 more rules for comprehensive coverage)

**Deliverable:** `complete_fuzzy_controller.py`

---

### Week 3: NSPSA Integration

**Goal:** Connect fuzzy controller to existing NSPSA system

**Components to Integrate:**

1. **From SubtleGenius/primitives/nspsa.py:**
   - PrimitiveRanker with learned heuristics
   - 8-feature extraction (already have!)
   - Gradient descent with momentum

2. **From SubtleGenius/primitives/symbolic_solver.py:**
   - Bidirectional search (forward + backward)
   - A* heuristic function
   - Beam search with priority queue
   - Composition primitives

3. **Integration Points:**
   - Fuzzy controller outputs `weight_metalearning`
   - If weight > 0.5, invoke NSPSA
   - NSPSA uses fuzzy features for ranking
   - Beam width controlled by fuzzy `search_depth`

**Architecture:**
```python
class HybridARCSolver:
    def solve(self, train_pairs, test_input):
        # 1. Extract features
        features = self.feature_extractor.extract(train_pairs, test_input)

        # 2. Fuzzy inference
        weights = self.fuzzy_controller.infer(features)

        # 3. Strategy selection
        if weights['weight_metalearning'] > 0.5:
            # Symbolic path (NSPSA)
            nspsa_result = self.nspsa_solver.solve(
                train_pairs,
                test_input,
                beam_width=int(weights['search_depth'] * 100)
            )

        if weights['weight_multiscale'] > 0.5:
            # Neural-guided path (BOLT-ONs)
            bolton_result = self.fuzzy_bolton_solver.solve(
                train_pairs,
                test_input
            )

        # 4. Aggregate predictions
        return self.aggregate([nspsa_result, bolton_result], weights)
```

**Deliverable:** `hybrid_nspsa_fuzzy_solver.py`

---

### Week 4: Validation & Optimization

**Goal:** Test on 100+ tasks, optimize performance

**Testing Plan:**

1. **100-Task Validation**
   - Expand from 50 to 100 tasks
   - Target: 30-50% accuracy
   - Identify failure modes

2. **Ablation Study: 3 Conditions × 5 Runs**
   - Condition A: NSPSA alone
   - Condition B: Fuzzy-BOLT-ONs alone
   - Condition C: Hybrid (NSPSA + Fuzzy)
   - Statistical validation (p < 0.05)

3. **Performance Optimization**
   - Profile bottlenecks
   - Cache feature extraction
   - Parallelize solver execution
   - Optimize beam search

**Deliverable:** `hybrid_validation_100tasks.py`, optimization results

---

### Week 5-6: Full System Integration & Submission

**Goal:** Integrate all components, test on 400 tasks, generate submission

**Tasks:**

1. **Full Integration**
   - Connect all components
   - End-to-end testing
   - Error handling & robustness

2. **400-Task Validation**
   - Test on full training set
   - Target: 75-85% accuracy
   - Categorize by task type

3. **ARC Prize 2025 Submission Generator**
   - Format: submission.json
   - 2 attempts per test input
   - Validation checks

4. **Documentation & Code Cleanup**
   - Final code review
   - Documentation update
   - Kaggle notebook preparation

**Deliverables:**
- `arc_prize_2025_hybrid_submission.py`
- `submission.json`
- Final comprehensive report

---

## SUCCESS CRITERIA

### Week 1-2 Milestones
- ✅ 8-feature extraction validated
- ✅ 50+ fuzzy rules implemented
- ✅ Improves over simplified fuzzy (>20% on 10 tasks)

### Week 3-4 Milestones
- ✅ NSPSA integration working
- ✅ 30-50% accuracy on 100 tasks
- ✅ Statistical significance (p < 0.05)

### Week 5-6 Milestones
- ✅ 75-85% accuracy on 400 tasks (TARGET)
- ✅ submission.json generated and validated
- ✅ Ready for ARC Prize 2025 submission

---

## RISK MITIGATION

### Risk 1: NSPSA Integration Complexity
**Mitigation:** Start with simple connection, iterate
**Fallback:** Use BOLT-ONs only if integration fails

### Risk 2: Performance on 400 Tasks
**Mitigation:** Optimize early, profile continuously
**Fallback:** Focus on task categories where we excel

### Risk 3: Timeline Slippage
**Mitigation:** Weekly checkpoints, adjust scope if needed
**Fallback:** Ship with 50-70% if 75-85% not reached

---

## CURRENT STATUS

**Phase:** 7 (Hybrid Approach)
**Week:** 1 - Starting Feature Extraction
**Next Task:** Implement full 8-feature system

**Ready to begin implementation!**
