# Geometric Semantic Space Prototype

Experimental approach to tarot synthesis using 3D continuous semantic embeddings.

## Concept

Instead of discrete rule-based synthesis, model tarot cards as overlapping regions in 3D space:

- **X-axis**: Elemental polarity (Fire-Air [active] ↔ Water-Earth [receptive])
- **Y-axis**: Consciousness depth (Ego/Persona ↔ Shadow/Unconscious)
- **Z-axis**: Temporal focus (Past ↔ Present ↔ Future)

Cards exist as "fuzzy polygons" with influence radii. Overlaps create emergent meanings.

## Phase 1: Core Model ✅

- [x] 3D semantic space definition
- [x] Hand-coded embeddings for 10 sample cards
- [x] Euclidean distance & overlap calculations
- [x] User profile → query vector mapping
- [x] Theme extraction from geometric configuration

## Phase 2: Visualization ✅

- [x] Interactive 3D Plotly visualization
- [x] Card positions as sized spheres (size = influence radius)
- [x] Overlap connections (line thickness = strength)
- [x] Reading centroid display
- [x] User query vector overlay
- [x] Pairwise overlap heatmap

## Next Phases

- [ ] Phase 3: Integration with current synthesis engine
- [ ] Phase 4: Scale to all 78 cards

## Usage

**Run semantic space tests:**
```bash
pip install -r requirements.txt
python semantic_space.py
```

**Generate interactive visualizations:**
```bash
python visualize.py
# Opens semantic_space_3d.html and overlap_heatmap.html
```

## Sample Output

```
=== Sample 3-Card Reading ===
1. The Tower: [ 0.9 -0.6  0.1]
2. Death: [ 0.  -0.3  0. ]
3. The Star: [-0.4  0.6  0.8]

=== Pairwise Overlaps ===
The Tower ↔ Death: 0.831 (high - both transformational/shadow)
The Tower ↔ The Star: 0.465 (low - destruction vs hope)
Death ↔ The Star: 0.681 (moderate - transformation → renewal)

=== Dominant Themes ===
ego_shadow_balance: 0.900 (reading spans conscious/unconscious)
elemental_balance: 0.833 (spans fire to water energies)
present_focus: 0.700 (temporal center of gravity)
high_coherence: 0.659 (cards work together synergistically)
```

## Theory

This approach assumes tarot interpretation is fundamentally **geometric reasoning** over symbolic manifolds, not rule-based pattern matching.

Advantages:
- Continuous blending (no hard category boundaries)
- Emergent meanings from overlaps
- User profile as query vector (natural personalization)
- Scales to multi-card spreads (N-way overlaps)

Open questions:
- Can this outperform sophisticated heuristics?
- How to map geometric themes → natural language prose?
- Optimal embedding strategy (hand-coded vs learned)?
