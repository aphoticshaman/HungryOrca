# SubtleGenius v1 - 10-Step Build Plan
## Lean Agile Solo Vibe: Coder-LLM Collaborative AI Fusion Architecture

**Project**: SubtleGenius ARC Prize 2025 Submission System
**Architecture**: 6-Cell Modular Notebook
**Philosophy**: Production-First, Token-Efficient, Never-Crash Design
**Target**: Valid submission.json on first run, iterate to championship performance

---

## Executive Summary

SubtleGenius fuses 48 hours of breakthrough insights into a lean, production-ready ARC Prize 2025 solver. Built on token-efficient modular architecture, asymmetric gain ratcheting, and production-first principles, this system guarantees valid submissions while enabling rapid iteration toward 85%+ accuracy.

**Core Innovation**: Separate validation from solving, fallbacks from intelligence, infrastructure from algorithms. Build the skeleton that never breaks, then inject the genius that solves.

---

## STEP 1: Foundation & Configuration (Cell 1)
**Duration**: 15 minutes
**Lines of Code**: ~150

### Objective
Establish zero-dependency configuration layer and global utilities that all subsequent cells consume.

### Implementation
Create single source of truth for:
- **Environment Detection**: Auto-detect Kaggle vs local, set paths accordingly
- **Time Budget Management**: Implement 95% rule (12hr × 0.95 = 11.4hr effective)
- **Global Timer**: Track elapsed/remaining time, enforce safety buffer
- **Logging Configuration**: JSON-lines logging for forensic analysis
- **Constants**: Submission requirements (2 attempts, 0-9 values, grid format)

### Key Code Pattern
```python
class ARC2025Config:
    IS_KAGGLE = os.path.exists('/kaggle/input')
    EFFECTIVE_TIME = 12 * 3600 * 0.95  # 95% rule
    REQUIRED_ATTEMPTS = 2
```

### Success Criteria
- ✅ Configuration loads without errors
- ✅ Paths correct for both Kaggle and local
- ✅ Timer initialized and tracking
- ✅ All imports successful

### Why This Matters
Following Insight #5 (Token-Efficient Development): modular cells = edit Cell 3 without regenerating Cell 1. Configuration isolation prevents cascading changes.

---

## STEP 2: Submission Validator (Cell 2)
**Duration**: 30 minutes
**Lines of Code**: ~250

### Objective
Build comprehensive validator BEFORE any solving code. Follows "Validation > Innovation" principle from 12-Step Guide.

### Implementation
Three-tier validation hierarchy:
1. **Grid Validation**: 2D list structure, integer values 0-9, no ragged arrays
2. **Prediction Validation**: Both attempt_1 and attempt_2 present, both valid grids
3. **Submission Validation**: Dictionary structure, all task IDs present, counts match test outputs

### Critical Checks
- Type checking (dict vs list - the failure that wasted your submission)
- Key presence (attempt_1, attempt_2)
- Value ranges (0-9 integers)
- Array consistency (equal row lengths)
- Task coverage (all test task IDs present)
- Output count matching (some tasks have >1 test output)

### Success Criteria
- ✅ Validates correct format without errors
- ✅ Catches the 5 fatal errors from failed submission
- ✅ Returns clear error messages with context
- ✅ Can validate before and after generation

### Why This Matters
Your failed submission had wrong format: list instead of dict, missing attempt_1/attempt_2 keys. This validator catches those errors BEFORE wasting a daily submission.

---

## STEP 3: Safe Defaults & Fallbacks (Cell 3)
**Duration**: 20 minutes
**Lines of Code**: ~200

### Objective
Implement production-grade fallback strategies that NEVER throw exceptions. Follows "Valid 5% > Crashing 95%" principle.

### Implementation
Four-tier fallback cascade:
1. **copy_input**: Safest default - return test input unchanged
2. **copy_train_output**: Use first training output as prediction
3. **blank_grid**: Generate zero-filled grid matching input dimensions
4. **ultimate_fallback**: Single-cell [[0]] if everything fails

### Defensive Wrapper
```python
def safe_execute(func, *args, **kwargs):
    try:
        result = func(*args, **kwargs)
        if validate_grid(result):
            return result
    except Exception:
        pass
    return fallback(*args, **kwargs)
```

### Success Criteria
- ✅ Never crashes regardless of input
- ✅ Always returns valid grid
- ✅ Gracefully degrades through fallback tiers
- ✅ Logs which fallback was used

### Why This Matters
Production-First Development (Insight #6): championship systems need error recovery, not perfect algorithms. Complete 100% of submissions > solve 95% perfectly but crash.

---

## STEP 4: Submission Generator (Cell 4)
**Duration**: 45 minutes
**Lines of Code**: ~350

### Objective
Wire solver to submission format with production-grade orchestration. Generate perfectly formatted submission.json from any solver function.

### Implementation
**SubmissionGenerator** class with:
- Solver function interface (takes test_input, task_data, attempt number)
- Per-task iteration with progress tracking
- Per-test-output handling (tasks can have multiple test outputs)
- Automatic attempt_1/attempt_2 generation
- Real-time validation of each prediction
- Time budget checking (95% rule enforcement)
- Automatic fallback injection on solver failure
- Statistics tracking (successes, fallbacks, errors)

### Critical Format Implementation
```python
submission[task_id] = [
    {
        "attempt_1": grid1,
        "attempt_2": grid2
    }
    # More dicts if task has multiple test outputs
]
```

### Success Criteria
- ✅ Generates correct dict structure (not list)
- ✅ All task IDs present as top-level keys
- ✅ Each task has list of prediction dicts
- ✅ Each prediction has attempt_1 and attempt_2
- ✅ Passes Cell 2 validation

### Why This Matters
This is where your failed submission broke. The old code used `submission.append({"task_id": ..., "output": ...})` creating wrong structure. This generator enforces correct format by design.

---

## STEP 5: Solver Logic Scaffold (Cell 5)
**Duration**: 30 minutes initially, expand iteratively
**Lines of Code**: ~200-800 (grows with sophistication)

### Objective
Create extensible solver interface that can start simple and grow sophisticated. Inject your AGI genius here while maintaining production guarantees from Cells 1-4.

### Initial Implementation
Simple pattern matcher with clear extension points:
- **identity_solver**: Return input unchanged (baseline)
- **pattern_matcher**: Detect simple transformations (flip, rotate, color swap)
- **train_learner**: Extract patterns from training examples

### Advanced Extension Points
Following your 48hr insights:
- Lambda dictionary cognitive modes (Insight #1)
- Ensemble specialists - tank/dps/healer/pug (Insight #7)
- Meta-cognitive self-reflection (Insight #8)
- Dynamic time budgeting per task difficulty (Insight #4)
- Object detection and tracking (Phase 3 roadmap)

### Solver Interface
```python
def solver(test_input: List[List[int]],
           task_data: Dict,
           attempt: int) -> List[List[int]]:
    """
    Args:
        test_input: Grid to solve
        task_data: Full task (includes train examples)
        attempt: 1 or 2 (can use different strategies)
    Returns:
        Predicted output grid
    """
    # Your genius goes here
```

### Success Criteria
- ✅ Implements solver interface correctly
- ✅ Returns valid grids (passes Cell 2 validation)
- ✅ Handles exceptions gracefully (wrapped by Cell 3)
- ✅ Can differentiate attempt 1 vs attempt 2 strategies

### Why This Matters
Token efficiency: start with 200 lines of simple logic, test end-to-end, THEN expand to 800 lines of AGI sophistication. Edit Cell 5 without touching infrastructure cells.

---

## STEP 6: Execution Pipeline (Cell 6)
**Duration**: 20 minutes
**Lines of Code**: ~200

### Objective
Orchestrate the complete flow: Load → Solve → Generate → Validate → Save. Implement 95% rule time management and production monitoring.

### Implementation
**main()** function executing:
1. Load test_challenges.json
2. Initialize SubmissionGenerator with solver
3. Generate submission (with progress tracking)
4. Validate submission (Cell 2)
5. Save to /kaggle/working/submission.json
6. Print statistics and timing breakdown

### Time Management
```python
if timer.should_stop():
    print("⏰ Time limit approaching")
    # Fill remaining tasks with safe defaults
    # Ensure 100% completion
```

### Progress Tracking
- Tasks processed / total
- Elapsed time / remaining time
- Solver success rate
- Fallback usage rate
- ETA to completion

### Success Criteria
- ✅ Completes all 240 tasks within time budget
- ✅ Generates valid submission.json
- ✅ Provides clear progress updates
- ✅ Never crashes due to time/memory limits

### Why This Matters
Orchestration cell coordinates all components without duplicating logic. Edit pipeline without touching validation, solver, or generator code.

---

## STEP 7: Local Testing Harness
**Duration**: 30 minutes
**Lines of Code**: ~150 (separate test file)

### Objective
Test full pipeline locally BEFORE wasting Kaggle submission. Catch 90% of issues in development environment.

### Implementation
**test_subtlegenius.py** with:
- Mock Kaggle environment (/tmp/kaggle/input, /tmp/kaggle/working)
- Small subset testing (10 tasks, quick iteration)
- Full dataset testing (all tasks, pre-submission check)
- Format validation on generated submission.json
- Performance profiling (time per task, memory usage)

### Test Protocol
```python
# Quick test
python test_subtlegenius.py --tasks 10

# Full test
python test_subtlegenius.py --tasks 240

# Validation only
python test_subtlegenius.py --validate submission.json
```

### Success Criteria
- ✅ Quick test passes in <1 minute
- ✅ Full test completes without crashes
- ✅ Generated submission.json validates correctly
- ✅ No ragged arrays, invalid values, or format errors

### Why This Matters
12-Step Guide Rule: "Test locally with full dataset BEFORE submitting to Kaggle." Don't waste daily submissions on preventable errors.

---

## STEP 8: Submission Checklist & Documentation
**Duration**: 15 minutes
**Lines of Code**: N/A (documentation)

### Objective
Create pre-flight checklist and clear documentation so future-you doesn't skip critical steps.

### Deliverables

**SUBMISSION_CHECKLIST.md**:
- [ ] Tested locally with 10-task subset
- [ ] Tested locally with full 240-task dataset
- [ ] Validation passed on generated submission.json
- [ ] All 6 cells run without errors in sequence
- [ ] No syntax errors, import errors, or path issues
- [ ] Reviewed solver_log.jsonl for unexpected errors
- [ ] Verified submission.json file size reasonable (not 0 bytes or >100MB)
- [ ] Confirmed correct paths (/kaggle/working/submission.json)
- [ ] Reviewed latest Kaggle discussion for known issues
- [ ] Have submissions remaining for today (1/day limit)

**README.md**:
- Architecture overview (6-cell design)
- Quick start guide
- Cell-by-cell explanation
- How to expand solver (Cell 5)
- Troubleshooting common issues

### Success Criteria
- ✅ Checklist is comprehensive and actionable
- ✅ README enables others to understand system
- ✅ Documentation prevents repeat mistakes

### Why This Matters
From failed submission lessons: checklists prevent human error. Documentation enables iteration without re-learning.

---

## STEP 9: Incremental Solver Enhancement
**Duration**: Iterative (hours to days)
**Lines of Code**: Grows Cell 5 from 200 → 800+ lines

### Objective
Systematically improve solver accuracy while maintaining production guarantees. Apply asymmetric gain ratcheting (Insight #2).

### Enhancement Roadmap

**Phase 1: Basic Pattern Matching** (Target: 10-15% accuracy)
- Horizontal/vertical flip detection
- Rotation (90°, 180°, 270°)
- Color swap mapping
- Identity transform

**Phase 2: Object Detection** (Target: 20-30% accuracy)
- Connected component analysis
- Bounding box extraction
- Object property detection (size, shape, color)
- Spatial relationship analysis

**Phase 3: Ensemble Methods** (Target: 40-50% accuracy)
- Geometric specialist (rotations, reflections, scaling)
- Algebraic specialist (sequences, modular arithmetic)
- Topological specialist (connectivity, boundaries)
- Creative specialist (novel combinations)

**Phase 4: Meta-Cognition** (Target: 60-75% accuracy)
- Lambda dictionary cognitive modes
- Self-reflection on reasoning quality
- Confidence calibration
- Strategy selection based on task characteristics

**Phase 5: Championship Polish** (Target: 85%+ accuracy)
- Time budget optimization per task difficulty
- Knowledge persistence across runs
- Failure pattern recognition
- Meta-pattern transfer learning

### Iteration Protocol
1. Add enhancement to Cell 5
2. Test locally (10 tasks)
3. Validate improvement (track accuracy)
4. If improvement: commit (asymmetric ratcheting)
5. If regression: revert, try different approach
6. Repeat

### Success Criteria
- ✅ Each iteration measurably improves accuracy
- ✅ No iteration breaks submission format
- ✅ Performance logged for ratcheting decisions
- ✅ Token-efficient (edit Cell 5 only)

### Why This Matters
Asymmetric Gain Ratcheting (Insight #2): only accept improvements, prevent catastrophic forgetting. Token efficiency enables 3.5x more iteration cycles.

---

## STEP 10: Production Deployment & Monitoring
**Duration**: 30 minutes setup, ongoing monitoring
**Lines of Code**: ~100 (logging extensions)

### Objective
Deploy to Kaggle with comprehensive monitoring, learn from each submission, continuously improve.

### Pre-Deployment
- Run full checklist (Step 8)
- Verify validation passes
- Review logs for anomalies
- Confirm time budget reasonable (<11.4hr)

### Deployment
1. Create new Kaggle notebook
2. Copy all 6 cells sequentially
3. Attach ARC Prize 2025 dataset
4. Disable internet (competition requirement)
5. Run all cells
6. Monitor progress in real-time
7. Verify submission.json created
8. Submit to competition

### Post-Submission Analysis
- Download submission.json and solver_log.jsonl
- Check public leaderboard score
- Analyze which tasks succeeded/failed
- Identify pattern gaps (what solver missed)
- Plan next enhancement iteration

### Monitoring Metrics
- Tasks solved / total
- Solver success rate vs fallback rate
- Time per task (efficiency)
- Memory usage profile
- Error frequency and types

### Iteration Log
```markdown
## Submission 1 (YYYY-MM-DD)
- Score: X%
- Strategy: Basic patterns + safe defaults
- Issues: None
- Next: Add object detection

## Submission 2 (YYYY-MM-DD)
- Score: Y%
- Strategy: + Object detection
- Issues: Some timeouts on complex tasks
- Next: Optimize time allocation
```

### Success Criteria
- ✅ Submission completes within time limit
- ✅ Valid submission.json generated
- ✅ Score tracked and analyzed
- ✅ Learnings documented for next iteration
- ✅ Continuous improvement trajectory

### Why This Matters
Production mindset: ship, measure, learn, improve. Each submission is a ratcheting point. Championship performance emerges from disciplined iteration, not single genius breakthrough.

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────┐
│  CELL 1: Configuration & Imports                        │
│  - Environment detection                                │
│  - Time budget (95% rule)                               │
│  - Global constants                                     │
└────────────────┬────────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────────┐
│  CELL 2: Submission Validator                           │
│  - Grid validation                                      │
│  - Prediction validation                                │
│  - Full submission validation                           │
└────────────────┬────────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────────┐
│  CELL 3: Safe Defaults & Fallbacks                      │
│  - Copy input strategy                                  │
│  - Copy train output strategy                           │
│  - Blank grid strategy                                  │
│  - Safe execution wrapper                               │
└────────────────┬────────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────────┐
│  CELL 4: Submission Generator                           │
│  - Solver interface                                     │
│  - Format enforcement (dict, not list!)                 │
│  - attempt_1 + attempt_2 generation                     │
│  - Real-time validation                                 │
└────────────────┬────────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────────┐
│  CELL 5: Solver Logic                                   │
│  - Pattern matching (Phase 1)                           │
│  - Object detection (Phase 2)                           │
│  - Ensemble methods (Phase 3)                           │
│  - Meta-cognition (Phase 4)                             │
│  - [Expand iteratively]                                 │
└────────────────┬────────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────────┐
│  CELL 6: Execution Pipeline                             │
│  - Load test data                                       │
│  - Generate submission                                  │
│  - Validate submission                                  │
│  - Save submission.json                                 │
│  - Report statistics                                    │
└─────────────────────────────────────────────────────────┘
```

---

## Success Metrics

### Immediate (First Run)
- ✅ All 6 cells execute without errors
- ✅ submission.json generated
- ✅ Validation passes
- ✅ Completes within time budget
- ✅ 100% task coverage

### Short-Term (First Submission)
- ✅ Valid submission accepted by Kaggle
- ✅ Non-zero score on leaderboard
- ✅ No disqualification errors
- ✅ Logs captured for analysis

### Medium-Term (Weeks 1-2)
- ✅ 10-20% accuracy (basic patterns)
- ✅ 3-5 successful iterations
- ✅ Solver expanded to 400+ lines
- ✅ Object detection working

### Long-Term (Weeks 3-4)
- ✅ 40-60% accuracy (ensemble methods)
- ✅ 10+ iterations with ratcheting
- ✅ Meta-cognitive reasoning active
- ✅ Championship trajectory

---

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| Format errors waste submission | Cell 2 validator catches before submitting |
| Solver crashes kill entire run | Cell 3 fallbacks ensure 100% completion |
| Time limit exceeded | Cell 1 timer + Cell 6 orchestration enforce 95% rule |
| Token budget exhausted | Modular cells enable editing Cell 5 only |
| Regression in solver quality | Asymmetric ratcheting rejects worse performance |
| Import/path errors in Kaggle | Cell 1 auto-detects environment |
| Invalid grid values | Cell 2 validates 0-9 range, rejects invalid |

---

## Timeline

**Day 1** (Setup):
- Steps 1-6: Build 6-cell architecture (3 hours)
- Step 7: Local testing (1 hour)
- Step 8: Documentation (30 min)
- **Deliverable**: Working end-to-end pipeline, valid submission.json

**Days 2-7** (Iteration):
- Step 9 Phase 1-2: Basic patterns + objects (20-30% accuracy)
- Multiple local test cycles
- **Deliverable**: First Kaggle submission

**Weeks 2-4** (Championship):
- Step 9 Phase 3-5: Ensemble + meta-cognition (60-85% accuracy)
- Step 10: Continuous deployment + monitoring
- **Deliverable**: Competitive leaderboard position

---

## Conclusion

SubtleGenius v1 synthesizes 48 hours of breakthrough insights into a production-ready, token-efficient, never-crash architecture. By separating validation from solving, infrastructure from intelligence, and guarantees from optimization, we create a system that:

1. **Never wastes submissions** (comprehensive validation)
2. **Always completes** (production-grade fallbacks)
3. **Iterates efficiently** (token-optimized modularity)
4. **Improves monotonically** (asymmetric ratcheting)
5. **Scales to championship** (extensible solver architecture)

**Build the skeleton that never breaks. Then inject the genius that solves.**

**Word Count**: ~2,480 words

---

**Ready to code. Ready to compete. Ready to win.** 🏆
