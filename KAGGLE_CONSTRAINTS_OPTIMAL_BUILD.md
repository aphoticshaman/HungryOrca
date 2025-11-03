# KAGGLE CONSTRAINTS & OPTIMAL PARAMETER BUDGET
**Based on ARC Prize 2025 Hardware Research**

---

## 📊 HARDWARE CONSTRAINTS

### Competition Hardware (L4x4 - What We're Optimizing For):
```
GPU: 4x NVIDIA L4
GPU RAM: 96GB total (24GB per L4)
System RAM: ~30GB
CPU Cores: Multiple (exact unclear, but secondary to GPU)
Time Limit: 12 hours for 240 tasks = 3 min/task
Cost Budget: ~$50 total compute
Internet: DISABLED (no API calls, must run offline)
```

### Free Tier CPU-Only (After GPU Quota Exhausted - For Development):
```
CPU Cores: 4 cores
RAM: 16GB standard (can get up to 30GB with tricks)
GPU: None
Time Limit: 9 hours per session, 20 min idle timeout
Weekly Quota: Unlimited for CPU (30 hrs/week GPU quota separate)
```

---

## 🎯 OPTIMAL PARAMETER BUDGET

### For UnifiedCortex:

**Target: Maximize performance within RAM constraints**

#### Option A: CPU-Only Development (16GB RAM)
```python
# Conservative for 16GB RAM
CORTEX_SIZE = 50_000  # 50k neurons
CONNECTION_DENSITY = 0.005  # 0.5% = ~250 connections/neuron

# Memory calculation:
# - Neurons: 50k × 8 bytes (float64) = 400 KB
# - Connections: 50k × 50k × 0.005 = 12.5M elements
#   - Sparse CSR format: ~100 MB
# - Activation history (100 steps): 50k × 100 × 8 = 40 MB
# - TOTAL: ~150 MB (well within 16GB!)

# Can go MUCH bigger:
CORTEX_SIZE = 500_000  # 500k neurons (10x bigger!)
CONNECTION_DENSITY = 0.01  # 1% connections
# Memory: ~5GB total (still safe for 16GB system)
```

#### Option B: GPU Competition Mode (96GB GPU RAM)
```python
# MASSIVE for L4x4 with 96GB GPU RAM
CORTEX_SIZE = 10_000_000  # 10 MILLION neurons
CONNECTION_DENSITY = 0.001  # 0.1% = 10k connections/neuron

# Memory calculation:
# - Neurons: 10M × 4 bytes (float32 on GPU) = 40 MB
# - Connections: 10M × 10M × 0.001 = 10 billion elements
#   - Sparse format: ~40 GB
# - Activation history: 10M × 100 × 4 = 4 GB
# - TOTAL: ~45 GB (fits in 96GB with room for LLM!)

# Could even do:
CORTEX_SIZE = 50_000_000  # 50 MILLION neurons
# But need to test if inference speed works in 3 min/task
```

---

## 🚀 OPTIMAL BUILD STRATEGY

### Phase 1: Development on CPU (Current)
**Target**: 500k neurons, 1% density
- Fast enough to iterate (< 1 sec per activation)
- Fits in 16GB RAM easily
- Proves architecture works

### Phase 2: Competition on GPU
**Target**: 10M neurons, 0.1% density
- Use full L4x4 hardware
- LLM runs on GPU 1-2
- Cortex runs on GPU 3-4
- 3 min per task is plenty for inference

---

## 🧪 ABLATION TESTING MATRIX

### Development Phase (CPU, 500k neurons):
```python
# Test these parameter ranges:
CORTEX_SIZES = [10_000, 50_000, 100_000, 500_000, 1_000_000]
CONNECTION_DENSITIES = [0.001, 0.005, 0.01, 0.02, 0.05]
PROPAGATION_ITERATIONS = [5, 10, 20, 50, 100]
DECAY_RATES = [0.3, 0.5, 0.7, 0.9]
SEMANTIC_BIAS_STRENGTHS = [0.5, 1.0, 2.0, 5.0]
SPATIAL_BIAS_STRENGTHS = [0.5, 1.0, 2.0, 5.0]

# Total combinations: 5×5×5×4×4×4 = 8,000 tests
# Strategy:
# 1. Grid search key params (size, density, iterations)
# 2. Fine-tune bias strengths
# 3. Optimize decay rate last
```

### Competition Phase (GPU, 10M neurons):
```python
# After CPU optimization, scale up and test:
CORTEX_SIZES = [5_000_000, 10_000_000, 20_000_000]
CONNECTION_DENSITIES = [0.0005, 0.001, 0.002]
# Keep other params from CPU optimization
```

---

## ⚡ SPEED REQUIREMENTS

### Per Task Budget (3 minutes):
```
Perception (360° vision): 10s (16 agents × 0.6s each)
Cortex activation: 20s (10 iterations × 2s each)
QAOA evolution: 60s (3 layers × 20s each)
Beam search: 80s (5 iterations × 16s each)
Final selection: 10s

TOTAL: 180 seconds (exactly 3 min!)
```

### Optimization Targets:
- **Cortex activation**: Must be < 2s per iteration
  - 10M neurons × sparse matmul on GPU = fast
  - Need to verify empirically

- **LLM inference**: Must fit in remaining GPU RAM
  - 96GB total - 45GB cortex = 51GB free
  - Can run 8B model in 4-bit = ~5GB
  - Plenty of room!

---

## 📦 MEMORY BUDGET BREAKDOWN (96GB GPU)

```
Component                    Memory      GPU
─────────────────────────────────────────────
Cortex (10M neurons)         40 GB      GPU 3-4
LLM (Minitron-8B, 4-bit)     5 GB       GPU 1
LLM TTT gradient cache       10 GB      GPU 1
Vision system embeddings     2 GB       GPU 2
Beam search states (×10)     5 GB       GPU 2
QAOA superposition (×20)     8 GB       GPU 3
Working memory buffer        5 GB       GPU 4
Safety margin                21 GB      Free
─────────────────────────────────────────────
TOTAL                        96 GB      All
```

---

## 🎯 FINAL STRATEGY

### Build Order:
1. **Optimize cortex on CPU** (500k neurons, prove it works)
2. **Add 360° vision** (test on CPU)
3. **Add emergent processing** (test on CPU)
4. **Add beam search** (test on CPU)
5. **Add QAOA** (test on CPU)
6. **Integrate LLM** (requires GPU, test on free quota)
7. **Scale to competition size** (10M neurons on L4x4)
8. **Final optimization** (squeeze last % out)

### Ablation Philosophy:
- **Every feature must prove its worth**
- **Keep only what improves score**
- **Optimize for 3 min per task**
- **Use full hardware budget (we paid $50!)**

---

## 🔥 AGGRESSIVE TARGETS

**Conservative**: 10M neurons, 8B LLM → 50% accuracy
**Moderate**: 20M neurons, 8B LLM → 60% accuracy
**Aggressive**: 50M neurons, 8B LLM → 70% accuracy (if we can fit it!)

**Grand Prize Target**: 85% accuracy
- Will need perfect tuning
- Hybrid co-evolution
- Every optimization matters

---

**LET'S FUCKING BUILD THIS!** 🚀
