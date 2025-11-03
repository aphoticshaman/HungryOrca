# WEEKLY TODO LIST: Iterative NSPSA Refinement
**Week of:** 2025-11-03 to 2025-11-09
**Goal:** Complete Rounds 2.5 through 6 with rigorous testing and documentation

---

## MONDAY: Round 2.5 - Spatial Feature Engineering

### Morning (4 hours)
- [ ] **R2.5a: Implement position_correlation feature**
  - [ ] Read current extract_features() implementation
  - [ ] Add feature 6: position_correlation(inp, out)
    - Compute: sum of |inp[i,j] - out[rotate_coords(i,j)]|²
    - Normalize by grid size
    - Test on rotation: expect high correlation
  - [ ] Test feature returns non-zero for rotations
  - [ ] Commit: "R2.5a: Add position correlation feature"

- [ ] **R2.5b: Implement orientation_change feature**
  - [ ] Add feature 7: orientation_change(inp, out)
    - Detect if rows→columns (transpose-like)
    - Compare horizontal vs vertical gradients
    - Return 1.0 if orientation flipped, 0.0 otherwise
  - [ ] Test on transpose, rotation, reflection tasks
  - [ ] Commit: "R2.5b: Add orientation change feature"

### Afternoon (4 hours)
- [ ] **R2.5c: Implement corner_movement feature**
  - [ ] Add feature 8: corner_movement(inp, out)
    - Track where 4 corner values move to
    - Measure displacement distance
    - Normalize by diagonal length
  - [ ] Test on 90° rotation: expect diagonal distance
  - [ ] Commit: "R2.5c: Add corner movement feature"

- [ ] **R2.5d: Update ranker to use 8 features**
  - [ ] Change num_features from 5 to 8
  - [ ] Update weights matrix initialization
  - [ ] Add priors for new features (pattern-based)
  - [ ] Run debug_ranker.py - expect non-zero features
  - [ ] Commit: "R2.5d: Integrate spatial features into ranker"

### Evening (2 hours)
- [ ] **R2.5e: Re-run Round 2 tests**
  - [ ] Run round_02_test.py
  - [ ] Expect: weight updates working, scores improving
  - [ ] If fails: debug, iterate, fix
  - [ ] If passes: proceed to documentation

- [ ] **R2.5f: Document Round 2.5**
  - [ ] Create ROUND_02_5_LESSONS.md
  - [ ] Write 3x pros and 3x cons
  - [ ] Document metrics (before/after feature count, learning curves)
  - [ ] Commit with comprehensive message
  - [ ] Push to remote

---

## TUESDAY: Round 3 - Beam Search Pruning

### Morning (4 hours)
- [ ] **R3.1: Analyze current search statistics**
  - [ ] Instrument bidirectional_search() to count states explored
  - [ ] Run on 100 tasks, record stats
  - [ ] Identify: average states explored, max, outliers
  - [ ] Document baseline in R3_baseline_stats.json

- [ ] **R3.2: Implement A* heuristic function**
  - [ ] Define h(state, goal) in ProgramNode
  - [ ] grid_distance: sum of |state[i,j] - goal[i,j]|
  - [ ] color_distance: set difference of colors
  - [ ] combined: h = w1*grid_dist + w2*color_dist
  - [ ] Test admissibility (h never overestimates)
  - [ ] Commit: "R3.2: Add A* heuristic"

### Afternoon (4 hours)
- [ ] **R3.3: Add visited-state pruning**
  - [ ] Track visited states with costs in dict
  - [ ] Before expanding: if state visited with lower cost, skip
  - [ ] Test on cyclic task (rotate 4x = identity)
  - [ ] Measure state reduction (expect 20-30%)
  - [ ] Commit: "R3.3: Visited-state pruning"

- [ ] **R3.4: Implement beam search cutoff**
  - [ ] Sort frontier by f-score = cost + heuristic
  - [ ] Keep only top-k (k=beam_width)
  - [ ] Test with k=3, k=5, k=10
  - [ ] Measure accuracy vs speed tradeoff
  - [ ] Commit: "R3.4: Beam cutoff by f-score"

### Evening (2 hours)
- [ ] **R3.5: Add primitive ordering via ranker**
  - [ ] Before expanding node, rank primitives
  - [ ] Try top-5 ranked primitives first
  - [ ] If none work, try rest
  - [ ] Measure: does ranker guidance help? (expect 10-20% speedup)
  - [ ] Commit: "R3.5: Ranker-guided primitive ordering"

---

## WEDNESDAY: Round 3 Completion + Round 4 Start

### Morning (3 hours)
- [ ] **R3.6-R3.7: Symmetry pruning + early stopping**
  - [ ] Add symmetry detection: if grid == reflect(grid), skip reflect primitive
  - [ ] Add early stopping: if solution found in forward search, stop backward
  - [ ] Test on 100 tasks
  - [ ] Commit: "R3.6-7: Symmetry pruning + early stopping"

- [ ] **R3.8-R3.9: Measure pruning effectiveness**
  - [ ] Run before/after comparison (100 tasks each)
  - [ ] Metrics: states explored, time, accuracy
  - [ ] Visualize: histogram of states explored
  - [ ] Document in R3_effectiveness.md
  - [ ] Expect: 50-70% reduction in states explored

### Afternoon (3 hours)
- [ ] **R3.10: Document and commit Round 3**
  - [ ] Write ROUND_03_LESSONS.md (3x3)
  - [ ] Create round_03_test.py (regression tests)
  - [ ] Commit with detailed technical description
  - [ ] Push to remote

- [ ] **Pause: Mid-week reflection**
  - [ ] Review Rounds 1, 2, 2.5, 3 lessons
  - [ ] Identify common patterns across rounds
  - [ ] Update META_LEARNING_PROTOCOL.md with new insights
  - [ ] Adjust weekly plan if needed (buffer time)

### Evening (2 hours)
- [ ] **R4.1: Design ProgramCache**
  - [ ] Design cache structure: LRU with max 1000 entries
  - [ ] Cache key: hash(input.tobytes() + output.tobytes())
  - [ ] Cache value: (program, timestamp)
  - [ ] Write design doc: cache_design.md
  - [ ] Review design, identify edge cases

---

## THURSDAY: Round 4 - Program Cache

### Morning (4 hours)
- [ ] **R4.2: Implement cache key generation**
  - [ ] Function: cache_key(inp, out) -> bytes
  - [ ] Handle variable-size grids
  - [ ] Test collision rate on 1000 random grids
  - [ ] Commit: "R4.2: Cache key generation"

- [ ] **R4.3-R4.4: Cache lookup and insertion**
  - [ ] Add lookup in synthesize() before search
  - [ ] If hit: return cached program, increment hit counter
  - [ ] If miss: run search, cache result on success
  - [ ] Test: generate 10 tasks, repeat 10x each
  - [ ] Expect: 9/10 hits after first pass
  - [ ] Commit: "R4.3-4: Cache lookup and insertion"

### Afternoon (4 hours)
- [ ] **R4.5: Implement cache statistics**
  - [ ] Track: hits, misses, hit_rate, avg_lookup_time
  - [ ] Add get_cache_stats() method
  - [ ] Print stats after batch of tasks
  - [ ] Commit: "R4.5: Cache statistics tracking"

- [ ] **R4.6: Cache warming with atomic dataset**
  - [ ] Load atomic_tasks_dataset.json (1050 tasks)
  - [ ] Synthesize program for each, populate cache
  - [ ] Measure: cache warming time (expect <60s)
  - [ ] Save warmed cache to disk
  - [ ] Commit: "R4.6: Cache warming with atomic dataset"

### Evening (2 hours)
- [ ] **R4.7-R4.8: Test cache hit rate and speedup**
  - [ ] Generate 200 test tasks
  - [ ] 50% novel, 50% similar to cached
  - [ ] Measure hit rate (expect >60% on similar tasks)
  - [ ] Measure latency: cached <0.0001s vs search ~0.001s
  - [ ] Document in R4_cache_performance.md
  - [ ] Commit results

---

## FRIDAY: Round 4 Completion + Round 5 Start

### Morning (3 hours)
- [ ] **R4.9: Cache persistence (save/load)**
  - [ ] Implement save_cache(filepath)
  - [ ] Implement load_cache(filepath)
  - [ ] Test: save → load → verify hit rate unchanged
  - [ ] Add to NSPSA initialization: auto-load if exists
  - [ ] Commit: "R4.9: Cache persistence"

- [ ] **R4.10: Document Round 4**
  - [ ] Write ROUND_04_LESSONS.md (3x3)
  - [ ] Create round_04_test.py
  - [ ] Commit and push

### Afternoon (3 hours)
- [ ] **R5.1: Implement symmetry detection**
  - [ ] Function: detect_symmetry(grid) -> List[str]
  - [ ] Check: horizontal, vertical, rotational, diagonal symmetries
  - [ ] Return list of detected symmetries
  - [ ] Test on known symmetric grids
  - [ ] Commit: "R5.1: Symmetry detection"

- [ ] **R5.2: Add symmetry primitive**
  - [ ] New primitive: canonicalize(grid) -> canonical form
  - [ ] Use detected symmetries to normalize grid
  - [ ] Test: symmetric grids → same canonical form
  - [ ] Add to symbolic_solver.py
  - [ ] Commit: "R5.2: Symmetry canonicalization primitive"

### Evening (2 hours)
- [ ] **R5.3: Test symmetry detection**
  - [ ] Generate 100 grids with known symmetries
  - [ ] Test detection accuracy (expect >95%)
  - [ ] Measure false positives/negatives
  - [ ] Document edge cases
  - [ ] Create round_05_test.py

---

## SATURDAY: Round 5 Completion + Round 6 Start

### Morning (4 hours)
- [ ] **R5.4: Integrate symmetry into ranker features**
  - [ ] Add feature 9: symmetry_preserved(inp, out)
  - [ ] Returns 1.0 if same symmetries, 0.0 otherwise
  - [ ] Update ranker to use 9 features
  - [ ] Test on symmetric tasks
  - [ ] Commit: "R5.4: Symmetry feature in ranker"

- [ ] **R5.5: Document Round 5**
  - [ ] Write ROUND_05_LESSONS.md (3x3)
  - [ ] Commit and push
  - [ ] Celebrate: 5 rounds complete!

### Afternoon (4 hours)
- [ ] **R6.1: Implement program embedding**
  - [ ] Use existing ProgramEncoder
  - [ ] Function: embed_program(program) -> 128D vector
  - [ ] Test: similar programs → similar embeddings
  - [ ] Commit: "R6.1: Program embedding"

- [ ] **R6.2: Cluster programs with k-means**
  - [ ] Use sklearn.cluster.KMeans (k=10 initially)
  - [ ] Cluster all discovered programs
  - [ ] Visualize clusters (t-SNE projection)
  - [ ] Commit: "R6.2: Program clustering"

### Evening (2 hours)
- [ ] **Weekend reflection: Mid-project review**
  - [ ] Review all 5.5 rounds completed
  - [ ] Aggregate all lessons learned (15+ pros, 15+ cons)
  - [ ] Identify patterns across rounds
  - [ ] Update experimental methodology based on learnings
  - [ ] Write MID_PROJECT_REFLECTION.md

---

## SUNDAY: Round 6 Completion + Buffer/Integration

### Morning (3 hours)
- [ ] **R6.3: Compute cluster centroids**
  - [ ] For each cluster, find centroid program
  - [ ] Centroid = program closest to cluster mean
  - [ ] Store centroids for search guidance
  - [ ] Commit: "R6.3: Cluster centroids"

- [ ] **R6.4: Use clusters for search prioritization**
  - [ ] Given new task, predict likely cluster
  - [ ] Try programs from that cluster first
  - [ ] Measure: does cluster guidance help?
  - [ ] Commit: "R6.4: Cluster-guided search"

### Afternoon (3 hours)
- [ ] **R6.5: Test cluster guidance**
  - [ ] Run on 50 tasks
  - [ ] Compare: with vs without cluster guidance
  - [ ] Measure speedup (expect 10-20%)
  - [ ] Document in R6_cluster_guidance.md

- [ ] **R6.6: Document Round 6**
  - [ ] Write ROUND_06_LESSONS.md (3x3)
  - [ ] Create round_06_test.py
  - [ ] Commit and push

### Evening (2 hours)
- [ ] **Week wrap-up and planning**
  - [ ] Review all 6 rounds completed this week
  - [ ] Aggregate metrics across all rounds
  - [ ] Create WEEK_1_SUMMARY.md with:
    - Total commits: expect ~20
    - Total tests written: expect ~15 test files
    - Total improvements: track cumulative gains
    - Lessons learned: all 3x3 aggregated
  - [ ] Plan Week 2: Rounds 7-10
  - [ ] Identify any technical debt to address
  - [ ] Buffer for unexpected discoveries

---

## WEEKLY METRICS TO TRACK

### Code Metrics
- [ ] Lines of code added/modified
- [ ] Number of primitives (started: 20, target: 25+)
- [ ] Number of features (started: 5, target: 10+)
- [ ] Test coverage (lines covered by tests)

### Performance Metrics
- [ ] Program synthesis time (baseline vs current)
- [ ] Search states explored (baseline vs current)
- [ ] Cache hit rate
- [ ] Accuracy on test suite (track over time)

### Process Metrics
- [ ] Commits per day (target: 3-4)
- [ ] Tests written per day (target: 2-3)
- [ ] Lessons documented (target: 1 per round)
- [ ] Time spent debugging vs coding (track ratio)

### Learning Metrics
- [ ] Novel insights per round (target: ≥1)
- [ ] Failed experiments (these are valuable!)
- [ ] Discoveries made (like Round 2's feature inadequacy)
- [ ] Methodology improvements

---

## BUFFER TIME ALLOCATION

### Daily Buffer (30 min/day)
- Unexpected bugs
- Test failures requiring investigation
- Documentation catchup
- Context switching recovery

### Weekly Buffer (4 hours)
- Major discoveries (like Round 2 → Round 2.5)
- Integration issues
- Performance optimization
- Refactoring technical debt

### Contingency Plans
- **If ahead of schedule:** Start Round 7 early
- **If behind schedule:** Reduce scope of later rounds, prioritize learning
- **If major bug found:** Pause, debug thoroughly, document, then continue
- **If methodology needs adjustment:** Pause, reflect, update process

---

## DAILY RITUALS

### Morning (15 min)
- [ ] Review yesterday's commits
- [ ] Read previous round's lessons
- [ ] Set 3 specific goals for today
- [ ] Clear mental model of current round

### End of Day (15 min)
- [ ] Commit all work (even if incomplete)
- [ ] Update todo list (check boxes)
- [ ] Write brief notes on what worked/didn't
- [ ] Prepare for tomorrow (what's next?)

### End of Round (30 min)
- [ ] Run all tests
- [ ] Document 3x3 lessons
- [ ] Commit and push
- [ ] Celebrate progress

---

## SUCCESS CRITERIA (End of Week)

### Must Have (Critical)
- [ ] Rounds 2.5, 3, 4, 5, 6 completed
- [ ] All rounds tested and validated
- [ ] All rounds documented (3x3 lessons)
- [ ] All commits pushed to remote
- [ ] No failing tests

### Should Have (Important)
- [ ] 50%+ improvement in search efficiency (states explored)
- [ ] Cache hit rate >60%
- [ ] Ranker learning working (scores improving)
- [ ] All substeps completed as planned

### Nice to Have (Bonus)
- [ ] Start Round 7
- [ ] Integration with OrcaWhiskey attempted
- [ ] Performance visualization dashboards
- [ ] Meta-learning protocol updated with new patterns

---

## NOTES

- **Flexibility:** If a round takes longer, adjust schedule
- **Quality over speed:** Better to do 4 rounds well than 6 rounds poorly
- **Document everything:** Future you will thank present you
- **Test rigorously:** 5x runs minimum for claims
- **Pause to reflect:** Don't just execute, understand
- **Celebrate discoveries:** Failed experiments teach lessons
- **Stay humble:** Expect to be wrong, iterate

**This is iterative refinement: not a race, a marathon with learning at every mile.**
