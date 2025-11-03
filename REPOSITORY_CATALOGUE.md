# HungryOrca Repository Catalogue
**Generated**: 2025-11-03
**Purpose**: Complete inventory of codebase structure for LLM+TTT integration

---

## 📁 Directory Structure

```
/home/user/HungryOrca/
├── .git/                              # Git repository
└── (flat structure - no subdirectories)
```

**Note**: All files are in root directory. Clean, simple structure.

---

## 🐍 Python Solvers (3 files, 2,995 total lines)

### 1. TurboOrcav9_iter2.py (987 lines, 40 KB) ✅ ACTIVE
**Purpose**: Current production solver - Geometric transforms + pattern learning
**Key Components**:
- `TurboOrcaV9` - Main solver class
- `NeuralSymbolicModel` (NSM) - Extract symbolic rules
- `SymbolicDifferentiableProgramModel` (SDPM) - Rule-to-program synthesis
- `CodeX3Strategies` - 3 parallel search strategies
- `PatternLearner` - Learn from training examples
- Time budget: Configurable (default 2 min, set via `TIME_BUDGET_MINUTES`)

**Architecture**: NSM → SDPM → CODE X3 pipeline
**Performance**: ~41% confidence on test tasks
**Status**: ✅ Running now (32.5 min elapsed, 117.5 min remaining)

### 2. advanced_toroid_physics_arc_insights.py (1,069 lines, 41 KB)
**Purpose**: Advanced physics-inspired solver (toroid magnetosphere model)
**Key Components**:
- `MagnetosphereConfig` - Earth magnetosphere modeling
- Physics-based reasoning from quantum/geology principles
- Target: ARC Prize 2026 (50×50 grids)

**Status**: ⚠️ Research/experimental code

### 3. fuzzy_meta_controller_production.py (939 lines, 34 KB)
**Purpose**: Fuzzy logic meta-controller for strategy blending
**Key Components**:
- `FuzzySet` - Fuzzy membership functions
- `FuzzyMetaController` - Adaptive strategy orchestration
- Designed to blend multiple solver strategies

**Status**: ⚠️ Available but not integrated with TurboOrca

---

## 📊 ARC Competition Data Files

### Training Data
- `arc-agi_training_challenges.json` (3.9 MB) - 1,000 training tasks
- `arc-agi_training_solutions.json` (644 KB) - Solutions for training

### Evaluation Data
- `arc-agi_evaluation_challenges.json` (962 KB) - 120 eval tasks
- `arc-agi_evaluation_solutions.json` (219 KB) - Eval solutions

### Test Data (Competition)
- `arc-agi_test_challenges.json` (992 KB) - 240 test tasks (no solutions)

### Submission Files
- `submission.json` (351 KB) - Current submission (being generated)
- `sample_submission.json` (20 KB) - Example format

---

## 📝 Documentation Files

### Core Documentation
- `README.md` (8.1 KB) - Project overview, consciousness-informed architecture
- `12_STEP_CLAUDE_CODE_GUIDE_FOR_RYAN.md` (31 KB) - Development guide
- `Claude's Lessons.txt` (71 KB) - Learning/insights log

### Research Documentation
- `FUZZY_ARC_CRITICAL_CONNECTION.md` (33 KB) - Fuzzy logic integration notes
- `MASTER_SUMMARY_PHYSICS_TO_AGI.md` (25 KB) - Physics → AGI insights

### Other Files
- `LICENSE` - Repository license
- `PivotOrcav2.ipynb` (176 KB) - Jupyter notebook (legacy?)
- `uberorcav2.1.ipynb` (20 KB) - Jupyter notebook (legacy?)
- `hungryorcav2_cell1_kaggle` (22 KB) - Kaggle cell export

---

## 🎯 Current State Analysis

### ✅ What's Working
1. **TurboOrcav9_iter2.py** is actively running (~41% performance)
2. Clean data pipeline with all ARC-AGI-2 datasets
3. Time budget management implemented
4. Submission generation in progress

### ⚠️ What's Not Integrated
1. **Fuzzy meta-controller** exists but not used by TurboOrca
2. **Physics-based solver** is standalone research code
3. **No LLM integration** - all current solvers are heuristic-based
4. **No test-time training (TTT)** - needed to reach 53%+ accuracy

### 🎯 Integration Opportunity
The fuzzy meta-controller (`fuzzy_meta_controller_production.py`) could be the **perfect bridge** between:
- Existing geometric solver (TurboOrca)
- New LLM+TTT solver (to be built)
- Physics-based insights (optional)

---

## 🚀 Recommended Architecture for Hybrid LLM+TTT

```
┌─────────────────────────────────────────────────────────┐
│           FUZZY META-CONTROLLER (EXISTS!)               │
│  Adaptive strategy selection & ensemble voting          │
└────────┬──────────────────┬─────────────────┬──────────┘
         │                  │                 │
    ┌────▼─────┐      ┌────▼─────┐     ┌────▼─────┐
    │ TurboOrca│      │ LLM+TTT  │     │ Physics  │
    │(Geometric)│      │ (NEW!)   │     │(Optional)│
    │ 41% conf │      │ ?% acc   │     │  N/A     │
    └──────────┘      └──────────┘     └──────────┘
         │                  │                 │
         └──────────┬───────┴─────────────────┘
                    │
            ┌───────▼────────┐
            │  Ensemble Vote  │
            │  Best Solution  │
            └────────────────┘
```

### File Organization Plan

**New files to create**:
```
/home/user/HungryOrca/
├── llm_solver.py              # LLM+TTT solver class
├── llm_utils.py               # Task formatting, augmentation
├── hybrid_solver.py           # Integrates all solvers
├── requirements_llm.txt       # New dependencies
└── HYBRID_ARCHITECTURE.md     # Design document
```

**Modified files**:
```
TurboOrcav9_iter2.py          # Add hybrid mode flag
fuzzy_meta_controller_production.py  # Connect to solvers
```

---

## 📦 Dependencies Analysis

### Current Dependencies (TurboOrca)
```python
numpy
json (stdlib)
time (stdlib)
os (stdlib)
sys (stdlib)
typing (stdlib)
datetime (stdlib)
collections (stdlib)
```

**Status**: ✅ Minimal, stdlib-based, fast

### New Dependencies Needed (LLM+TTT)
```python
transformers       # Hugging Face models
peft              # LoRA fine-tuning
bitsandbytes      # 4-bit quantization
accelerate        # GPU optimization
torch             # PyTorch backend
datasets          # Data handling
```

**Size Impact**: ~5-10 GB for model weights + libraries

---

## 🎯 Next Steps (10-15 min chunks)

### ✅ COMPLETED: Repository Catalogue (10 min)
- Mapped all files
- Identified integration points
- Planned architecture

### 🔄 NEXT: Create Hybrid Architecture Document (10 min)
- Detailed design doc
- Time budget allocation
- Component interfaces

### 🔜 AFTER THAT: Install Dependencies (10 min)
- Add requirements_llm.txt
- Test minimal transformers import
- Verify GPU availability

---

## 💡 Key Insights

1. **Clean slate**: No existing LLM code = no legacy baggage
2. **Meta-controller exists**: Don't reinvent the wheel, use fuzzy controller
3. **Good patterns**: TurboOrca's time budget management is solid
4. **Flat structure**: Easy to navigate, add new modules
5. **Production ready**: Already generating submissions, just needs boost

---

**END OF CATALOGUE**
