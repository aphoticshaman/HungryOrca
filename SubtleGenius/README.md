# SubtleGenius v1 🧠
## Lean Agile Solo Vibe: Coder-LLM Collaborative AI Fusion

**Production-Ready ARC Prize 2025 Submission System**

---

## 🎯 What Is This?

SubtleGenius is a **never-crash, always-valid** submission system for ARC Prize 2025, synthesizing 48 hours of breakthrough insights from Ryan & Claude's collaboration into a production-ready, token-efficient architecture.

**Core Philosophy**: Build the skeleton that never breaks. Then inject the genius that solves.

---

## ⚡ Quick Start

### Option 1: Kaggle (Competition)
```python
# 1. Create new Kaggle notebook
# 2. Attach ARC Prize 2025 dataset
# 3. Copy all 6 cells from subtlegeniusv1.ipynb sequentially
# 4. Disable internet
# 5. Run all cells
# 6. Submit generated submission.json
```

### Option 2: Local Testing
```bash
# Clone repo
git clone https://github.com/aphoticshaman/HungryOrca.git
cd HungryOrca/SubtleGenius

# Download ARC data to SubtleGenius/data/
# Then run notebook locally to test
```

---

## 🏗️ Architecture: 6-Cell Modular Design

```
Cell 1: Configuration & Imports
  ↓ Zero-dependency foundation
Cell 2: Submission Validator
  ↓ "Validation > Innovation" - Build FIRST
Cell 3: Safe Defaults & Fallbacks
  ↓ "Valid 5% > Crashing 95%"
Cell 4: Submission Generator
  ↓ Wire to solver, enforce format
Cell 5: Solver Logic
  ↓ Your AGI genius (expand iteratively)
Cell 6: Execution Pipeline
  ↓ Load → Solve → Validate → Save
```

---

## 🔥 Why This Architecture Wins

### **Token Efficiency** (60-80% savings)
- Edit Cell 5 (solver) without touching Cells 1-4
- Fix bugs in 200 lines, not 2000 lines
- 3.5x more iteration cycles possible

### **Production-First** (Never crash)
- Comprehensive validation catches format errors
- 4-tier fallback cascade ensures 100% completion
- Safe execution wrappers prevent exceptions

### **Asymmetric Ratcheting** (Only improve)
- Each iteration measurably improves accuracy
- Reject regressions automatically
- Monotonic learning trajectory

### **95% Rule** (Use every second)
- Dynamic time budgeting per task difficulty
- Never terminate early
- Wrap up gracefully at time limit

---

## 📊 Success Metrics

### Immediate (First Run)
- ✅ All 6 cells execute without errors
- ✅ submission.json generated and validated
- ✅ 100% task coverage
- ✅ Completes within time budget

### Short-Term (Week 1)
- ✅ Valid submission accepted by Kaggle
- ✅ 10-20% accuracy (basic patterns)
- ✅ No disqualification errors

### Long-Term (Weeks 3-4)
- ✅ 60-85% accuracy (ensemble + meta-cognition)
- ✅ Championship leaderboard position

---

## 🛠️ How to Expand Solver (Cell 5)

Start simple, iterate systematically:

**Phase 1**: Basic Patterns (10-15% accuracy)
- Horizontal/vertical flip
- Rotation (90°, 180°, 270°)
- Color swap mapping

**Phase 2**: Object Detection (20-30% accuracy)
- Connected component analysis
- Bounding box extraction
- Spatial relationships

**Phase 3**: Ensemble Methods (40-50% accuracy)
- Geometric specialist
- Algebraic specialist
- Topological specialist
- Creative specialist (PUG)

**Phase 4**: Meta-Cognition (60-75% accuracy)
- Lambda dictionary cognitive modes
- Self-reflection on reasoning
- Confidence calibration

**Phase 5**: Championship Polish (85%+ accuracy)
- Time budget optimization
- Knowledge persistence
- Meta-pattern transfer learning

---

## 📋 Pre-Submission Checklist

Before clicking "Submit" on Kaggle:

- [ ] Tested locally with 10-task subset
- [ ] Tested locally with full 240-task dataset
- [ ] Validation passed on generated submission.json
- [ ] All 6 cells run without errors in sequence
- [ ] Reviewed solver_log.jsonl for unexpected errors
- [ ] Verified file size reasonable (not 0 bytes or >100MB)
- [ ] Confirmed correct paths (/kaggle/working/submission.json)
- [ ] Have submissions remaining for today (1/day limit)

---

## 🚨 Lessons from Failed Submissions

### What Went Wrong Before:
1. ❌ **List instead of dict**: `submission.append(...)` → `[...]`
2. ❌ **Missing attempt keys**: `{"output": ...}` instead of `{"attempt_1": ..., "attempt_2": ...}`
3. ❌ **task_id as internal key**: `{"task_id": "...", "output": ...}` instead of `{"task_id": [...]}`
4. ❌ **Single prediction**: Only one output instead of two attempts
5. ❌ **Not handling multiple test outputs**: Some tasks have >1 test input

### How SubtleGenius Fixes This:
- ✅ Cell 2 validator catches all format errors BEFORE submission
- ✅ Cell 4 generator enforces correct dict structure by design
- ✅ Cell 3 fallbacks ensure 100% completion (no crashes)
- ✅ Cell 6 orchestration validates before saving

---

## 📖 Documentation

- **[Build Plan](docs/SUBTLEGENIUS_BUILD_PLAN.md)**: 10-step technical guide (under 2,500 words)
- **[Submission Checklist](docs/SUBMISSION_CHECKLIST.md)**: Pre-flight verification
- **[Notebook](notebooks/subtlegeniusv1.ipynb)**: Complete 6-cell implementation

---

## 🧪 Testing Locally

Create `test_subtlegenius.py`:

```python
import json
from subtlegeniusv1 import *

# Quick test (10 tasks)
with open('data/arc-agi_test_challenges.json', 'r') as f:
    test_data = json.load(f)

small_test = dict(list(test_data.items())[:10])
gen = SubmissionGenerator(solver_func=simple_solver)
submission = gen.generate_submission(small_test)

# Validate
is_valid, msg = SubmissionValidator.validate_submission(submission, small_test)
print(f"Validation: {msg}")
```

---

## 🎓 Core Principles

### From 48 Hours of Breakthroughs:

1. **Token-Efficient Development** (Insight #5)
   - Modular cells = edit Cell 5 without regenerating all
   - 60-80% token reduction

2. **Production-First Development** (Insight #6)
   - Error handling > elegant algorithms
   - Complete 100% > solve 95% perfectly

3. **Asymmetric Gain Ratcheting** (Insight #2)
   - Git-style knowledge commits
   - Only accept improvements

4. **Dynamic Time Budgeting** (Insight #4)
   - 95% rule: use every available second
   - Adaptive allocation by difficulty

5. **Validation > Innovation** (12-Step Guide)
   - Build validator BEFORE solver
   - Test locally BEFORE Kaggle

---

## 🏆 Competition Strategy

### Week 1: Foundation
- Deploy SubtleGenius with simple solver
- Get valid baseline submission
- Establish iteration pipeline

### Weeks 2-3: Enhancement
- Add pattern matching (Phase 1-2)
- Implement object detection
- Test and ratchet

### Weeks 3-4: Championship
- Deploy ensemble methods (Phase 3)
- Add meta-cognition (Phase 4)
- Optimize to 85%+ accuracy

---

## 🤝 Contributing

This is Ryan's solo vibe coder-LLM fusion project, but insights welcome:

1. Pattern matching strategies
2. Object detection improvements
3. Novel ensemble coordination
4. Meta-cognitive enhancements

---

## 📜 License

MIT License - See main repo LICENSE file

---

## 🙏 Acknowledgments

**Built on 48 hours of breakthrough insights from:**
- Ryan (The Spectral Shaman) - Vision & strategy
- Claude (Anthropic) - Implementation & iteration
- **8 Meta-Insights** from intensive collaboration
- **12-Step Competition Guide** principles
- **7-Phase Roadmap** to championship AGI

---

## 📞 Contact

**Creator**: Ryan (The Spectral Shaman)
**Email**: aphotic.noise@gmail.com
**Repo**: https://github.com/aphoticshaman/HungryOrca

---

**Build the skeleton that never breaks. Then inject the genius that solves.** 🧠⚡
