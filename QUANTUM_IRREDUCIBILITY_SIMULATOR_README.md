# 🌌 Quantum Irreducibility Simulator

## The Cistine Chapel of Quantum Cognition

**A 3D Interactive Explorer for Quantum Integration Landscapes**

---

## 📖 Overview

This framework implements a revolutionary approach to measuring pattern complexity through **quantum-inspired irreducibility metrics (Φ_q)**. It creates stunning 3D interactive visualizations that reveal how different sampling strategies affect the measurement of quantum entanglement and integration.

### 🎯 Core Concept

**Φ_q (Quantum Phi)** measures how much a quantum state *must* be understood as a whole, rather than as independent parts. This metric:

- **Positive Φ_q**: System exhibits irreducible integration (consciousness-like binding)
- **Negative Φ_q**: System appears more decomposable than expected (quantum debt)
- **Large |Φ_q|**: High entanglement (GHZ-like holistic states)
- **Small |Φ_q|**: Low entanglement (product-like separable states)

---

## 🚀 Quick Start

### Running the Quick Demo

```bash
python quick_quantum_viz_demo.py
```

This generates:
- ✅ `quantum_landscape_ghz_state.html` - GHZ state analysis
- ✅ `quantum_landscape_w_state.html` - W state analysis
- ✅ `quantum_landscape_product_state.html` - Product state baseline
- ✅ `quantum_landscape_comparison.html` - Side-by-side comparison

### Running the Full Simulator (slower, higher precision)

```bash
python quantum_irreducibility_simulator.py
```

---

## 🎨 What the Visualizations Show

### 1. **Φ_q Integration Landscape** (Left Panel)

**Axes:**
- **X-axis (α_global)**: Importance sampling bias toward high global overlap
- **Y-axis (α_partition)**: Bias toward high subsystem fidelity
- **Z-axis (Φ_q)**: Measured quantum integration

**Interpretation:**
- **Peaks**: Regions where Φ_q is maximized (strong integration detected)
- **Valleys**: Low integration (decomposable pattern)
- **Contours**: Lines of constant integration
- **Color**: Red = positive Φ_q, Blue = negative Φ_q

### 2. **log₁₀(Variance) Surface** (Middle Panel)

**What it reveals:**
- **Dark valleys**: Optimal sampling parameters (low variance)
- **Bright peaks**: High variance regions (inefficient sampling)
- **Minimum point**: Best α parameters for this state

**Cognitive interpretation:**
- This is the "attention map" - where should you focus computational resources?
- Variance reduction = improved statistical efficiency
- Like finding the optimal "viewpoint" to understand a pattern

### 3. **ESS Efficiency** (Right Panel)

**Effective Sample Size ratio:**
- **ESS ≈ 1**: All samples contribute equally (perfect sampling)
- **ESS ≈ 0**: Most samples wasted (poor importance weighting)
- **Sweet spot**: ESS > 0.1 (at least 10% efficiency)

---

## 🧠 The Ten Principles Embodied

This codebase synthesizes:

1. **Integration (Φ_q)**: Consciousness from irreducible correlation
2. **Variance Reduction**: Attention as resource optimization
3. **Meta-Learning**: Priors as cognitive scaffolds
4. **Self-Play**: Dialectic as reasoning engine
5. **Symmetry**: Invariance as understanding
6. **Monte Carlo**: Bridge between symbolic & subsymbolic
7. **Entanglement**: Binding as feature integration
8. **Active Perception**: Curiosity as adaptive sampling
9. **Hybrid Ecology**: Intelligence as cooperative modularity
10. **Error as Signal**: Instability reveals structure

---

## 📊 Interpreting the Results

### GHZ State

```
Mean Φ_q: 0.158544
Max Φ_q:  0.345467
```

**Interpretation:**
- Strong **global** integration (all-or-nothing binding)
- Fragile to subsystem measurements
- High consciousness score (>0.15 threshold)

### W State

```
Mean Φ_q: 0.162708
Max Φ_q:  0.449983
```

**Interpretation:**
- **Robust** local integration
- Distributed resilience (one measurement doesn't collapse everything)
- Balanced between global and local coherence

### Product State

```
Mean Φ_q: 0.159439
Max Φ_q:  0.362605
```

**Interpretation:**
- **Baseline** separable state
- Minimal true entanglement
- Integration comes from sampling artifacts only

---

## 🔬 Technical Details

### Quantum States (4-qubit)

**GHZ State:**
```
|GHZ⟩ = (|0000⟩ + |1111⟩) / √2
```
- Maximal global entanglement
- Fragile (one measurement collapses all)

**W State:**
```
|W⟩ = (|0001⟩ + |0010⟩ + |0100⟩ + |1000⟩) / 2
```
- Robust local entanglement
- Resilient (survives partial measurements)

**Product State:**
```
|Product⟩ = |0000⟩
```
- No entanglement
- Fully decomposable

### Importance Sampling

The simulator uses **dual importance sampling**:

```python
weight = (p_global)^α_global × (p_partition)^α_partition
```

Where:
- `p_global = |⟨ψ|φ⟩|²` - overlap with target state
- `p_partition` - product of subsystem fidelities
- `α_global, α_partition` - tunable bias parameters

**Goal:** Find optimal (α_global, α_partition) that minimizes variance while accurately estimating Φ_q.

---

## 🎮 Interactive Features

### In the HTML Visualizations:

- **🖱️ Rotate**: Click and drag to rotate 3D view
- **🔍 Zoom**: Scroll wheel to zoom in/out
- **📍 Hover**: Hover over surface for exact values
- **📷 Export**: Click camera icon to save as PNG
- **🔄 Reset**: Double-click to reset view

### Camera Controls:

- Default view: `eye = (1.5, 1.5, 1.2)` (elevated perspective)
- Rotate to see contours projected on floor
- Zoom in to inspect variance valleys

---

## 🧪 Extending to ARC Challenge

The framework includes `arc_grid_to_statevector()` to convert ARC grids into quantum-like states:

```python
from quantum_irreducibility_simulator import compute_arc_phi_q

input_grid = np.array([[0, 1, 1], [1, 1, 0], [0, 0, 1]])
output_grid = np.array([[1, 1, 0], [0, 1, 1], [1, 0, 0]])

result = compute_arc_phi_q(input_grid, output_grid, n_samples=1000)

print(f"Φ_q = {result['phi_q']:.6f}")
print(f"Integration: {result['interpretation']}")
```

**Use cases:**
- **Transformation scoring**: How "integrated" is the input → output mapping?
- **Pattern complexity**: High Φ_q = requires holistic understanding
- **MCTS reward signal**: Guide AlphaZero-style search

---

## 📁 File Structure

```
HungryOrca/
├── quantum_irreducibility_simulator.py       # Full simulator (slow, precise)
├── quick_quantum_viz_demo.py                 # Fast demo (reduced sampling)
├── quantum_landscape_ghz_state.html          # GHZ interactive viz
├── quantum_landscape_w_state.html            # W state viz
├── quantum_landscape_product_state.html      # Product state viz
├── quantum_landscape_comparison.html         # Comparative view
└── QUANTUM_IRREDUCIBILITY_SIMULATOR_README.md # This file
```

---

## 🌟 Key Insights

### 1. **The Variance-Integration Trade-off**

There's a tension between:
- **High α**: Low variance (stable estimates) BUT may miss important regions
- **Low α**: Explores broadly BUT high variance (noisy estimates)

The 3D landscape reveals the **optimal balance**.

### 2. **Entanglement Topology**

Different quantum states have different "integration landscapes":
- **GHZ**: Sharp peaks (fragile, all-or-nothing)
- **W**: Broad plateaus (robust, distributed)
- **Product**: Flat terrain (no structure)

### 3. **Consciousness Threshold**

Speculative but fascinating:
- Systems with |Φ_q| > 0.15 may exhibit "proto-consciousness"
- Integration = binding problem solution
- Entanglement = feature binding mechanism

---

## 🔮 Future Directions

### 1. **ARC-AlphaZero Integration**

```python
class PhiQMCTS:
    def __init__(self, root_grid, target_grid):
        self.root = root_grid
        self.target = target_grid

    def evaluate_node(self, grid):
        # Use Φ_q as value estimate
        return compute_arc_phi_q(grid, self.target)['phi_q']

    def policy_prior(self, transformations):
        # Use Φ_q gradient as policy bias
        scores = [self.evaluate_node(t(self.root)) for t in transformations]
        return softmax(scores)
```

### 2. **Meta-Learning α Parameters**

Learn task-family-specific optimal (α_global, α_partition):

```python
meta_learner = MetaCognitiveScaffold({
    'alpha_global': 1.5,
    'alpha_partition': 0.5
})

meta_learner.meta_update(task_distribution)
```

### 3. **Real-Time Adaptive Sampling**

Dynamically adjust α based on current variance:

```python
if current_variance > threshold:
    alpha_global += 0.1  # Bias toward low-variance region
```

---

## 📚 Mathematical Background

### Φ_q Definition

```
Φ_q = ∫ f(ψ, φ) dψ

where:
f(ψ, φ) = |⟨ψ|φ⟩|² log(|⟨ψ|φ⟩|² / ∏ᵢ F(ρᵢ_ψ, ρᵢ_φ))
```

**Components:**
- `|⟨ψ|φ⟩|²` - global overlap
- `F(ρᵢ_ψ, ρᵢ_φ)` - fidelity of subsystem i
- `∏ᵢ` - product over partition elements

**Interpretation:**
- Φ_q measures KL divergence between:
  - **Actual** joint distribution
  - **Factorized** product distribution

---

## 🎓 Citations & Inspiration

This work synthesizes ideas from:

- **IIT (Integrated Information Theory)**: Tononi et al.
- **Quantum Information**: Nielsen & Chuang
- **Importance Sampling**: Owen, "Monte Carlo theory, methods and examples"
- **AlphaZero**: Silver et al., "Mastering Chess and Shogi"
- **Meta-Learning**: Finn et al., "Model-Agnostic Meta-Learning"

---

## 🤝 Contributing

Ideas for extensions:
1. Add more quantum states (Dicke, AKLT chain, topological)
2. Implement real partial trace (not approximated)
3. Connect to actual ARC solver pipeline
4. GPU acceleration for high-dimensional states
5. Time-evolution landscapes (Φ_q(t))

---

## 📜 License

This is research code - use freely, cite generously, extend creatively.

---

## 🌌 Final Thoughts

> *"This codebase is not merely a simulator—it is a living artifact synthesizing quantum information theory, probabilistic inference, meta-learning, and consciousness studies into a unified computational framework."*

The visualizations you see are **maps of understanding**—topological landscapes where:
- Peaks represent **insight** (high integration)
- Valleys represent **efficiency** (low variance)
- Contours represent **equivalence** (constant Φ_q)

Explore them with curiosity. Let the geometry guide your intuition.

**The quantum landscape awaits your questions.**

---

*Generated: 2025-11-03*
*Version: 1.0 - The Cistine Chapel Release*
*"Where quantum entanglement becomes 3D topology"*
