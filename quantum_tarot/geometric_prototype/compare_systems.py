"""
Comparison: Geometric Model vs Current Heuristic System
========================================================

Compares theme extraction and synthesis quality between:
1. Geometric semantic space approach (continuous overlaps)
2. Current heuristic system (rule-based synthesis)

Goal: Validate whether geometric model adds value or is just elegant theory.
"""

import numpy as np
from typing import Dict, List, Tuple
from semantic_space import SemanticSpace, UserProfile, extract_dominant_themes


# Sample readings for comparison
TEST_READINGS = [
    {
        "name": "Phoenix Moment (Destruction → Rebirth)",
        "cards": [(16, False), (13, False), (17, False)],  # Tower, Death, Star
        "user": UserProfile(mbti="INTJ", shadow_integration=0.4, temporal_focus=0.6),
        "expected_themes": {
            "transformation": "high",
            "shadow_work": "medium",
            "future_hope": "high",
            "disruption": "high"
        }
    },
    {
        "name": "Shadow Confrontation (Devil + Tower)",
        "cards": [(15, False), (16, False), (1, False)],  # Devil, Tower, Magician
        "user": UserProfile(mbti="ENFP", shadow_integration=-0.2, temporal_focus=0.0),
        "expected_themes": {
            "shadow_work": "very_high",
            "sudden_realization": "high",
            "active_response": "medium",
            "pattern_breaking": "high"
        }
    },
    {
        "name": "Joyful Stability (Cups + Pentacles)",
        "cards": [(40, False), (64, False), (17, False)],  # 10 Cups, King Pentacles, Star
        "user": UserProfile(mbti="ESFJ", shadow_integration=0.7, temporal_focus=0.3),
        "expected_themes": {
            "fulfillment": "high",
            "material_mastery": "high",
            "emotional_contentment": "high",
            "future_optimism": "medium"
        }
    },
    {
        "name": "Warrior's Charge (Fire energy)",
        "cards": [(27, False), (1, False), (16, False)],  # Knight Wands, Magician, Tower
        "user": UserProfile(mbti="ESTP", shadow_integration=0.1, temporal_focus=0.8),
        "expected_themes": {
            "action_oriented": "very_high",
            "rapid_change": "high",
            "fire_energy": "very_high",
            "present_action": "high"
        }
    },
    {
        "name": "Painful Reflection (Swords reversed)",
        "cards": [(54, True), (13, False), (17, True)],  # 3 Swords reversed, Death, Star reversed
        "user": UserProfile(mbti="INFP", shadow_integration=-0.5, temporal_focus=-0.4),
        "expected_themes": {
            "healing_heartbreak": "high",
            "letting_go": "high",
            "hope_dimmed": "medium",
            "shadow_avoidance": "high"
        }
    }
]


def analyze_geometric_reading(space: SemanticSpace, card_indices: List[Tuple[int, bool]],
                               user: UserProfile) -> Dict:
    """
    Run geometric analysis on a reading.

    Returns:
        Dict with themes, overlaps, centroid, weights
    """
    cards = [space.get_card(idx, reversed) for idx, reversed in card_indices]
    overlaps = space.compute_overlap_strength(cards)
    weights = user.relevance_weights(cards, space)
    themes = extract_dominant_themes(cards, overlaps, weights)
    centroid = space.get_centroid(cards)

    return {
        "cards": cards,
        "themes": themes,
        "overlaps": overlaps,
        "user_weights": weights,
        "centroid": centroid,
        "avg_overlap": np.mean(overlaps[np.triu_indices_from(overlaps, k=1)])
    }


def map_geometric_to_heuristic_themes(geometric_themes: Dict[str, float]) -> Dict[str, str]:
    """
    Map geometric theme scores to expected heuristic themes.

    This is the key validation step: Can geometric model detect same patterns
    as hand-coded rules?
    """
    mapped = {}

    # Elemental mappings
    if "active_fire_air" in geometric_themes:
        mapped["fire_energy"] = "very_high" if geometric_themes["active_fire_air"] > 0.7 else "high"
        mapped["action_oriented"] = "high" if geometric_themes["active_fire_air"] > 0.5 else "medium"

    if "receptive_water_earth" in geometric_themes:
        mapped["emotional_contentment"] = "high" if geometric_themes["receptive_water_earth"] > 0.6 else "medium"
        mapped["material_mastery"] = "high" if geometric_themes["receptive_water_earth"] > 0.6 else "medium"

    if "elemental_balance" in geometric_themes:
        mapped["balanced_approach"] = "high" if geometric_themes["elemental_balance"] > 0.7 else "medium"

    # Consciousness mappings
    if "shadow_work" in geometric_themes:
        mapped["shadow_work"] = "very_high" if geometric_themes["shadow_work"] > 0.7 else "high"

    if "conscious_integration" in geometric_themes:
        mapped["conscious_awareness"] = "high" if geometric_themes["conscious_integration"] > 0.6 else "medium"

    if "ego_shadow_balance" in geometric_themes:
        mapped["transformation"] = "high" if geometric_themes["ego_shadow_balance"] > 0.6 else "medium"

    # Temporal mappings
    if "future_oriented" in geometric_themes:
        mapped["future_hope"] = "high" if geometric_themes["future_oriented"] > 0.6 else "medium"
        mapped["future_optimism"] = "medium" if geometric_themes["future_oriented"] > 0.4 else "low"

    if "past_patterns" in geometric_themes:
        mapped["healing_heartbreak"] = "high" if geometric_themes["past_patterns"] > 0.6 else "medium"

    if "present_focus" in geometric_themes:
        mapped["present_action"] = "high" if geometric_themes["present_focus"] > 0.6 else "medium"

    # Interaction mappings
    if "high_coherence" in geometric_themes:
        mapped["synergy"] = "high" if geometric_themes["high_coherence"] > 0.6 else "medium"

    if "scattered_energies" in geometric_themes:
        mapped["conflicting_forces"] = "high" if geometric_themes["scattered_energies"] > 0.6 else "medium"

    if "high_personal_resonance" in geometric_themes:
        mapped["personal_alignment"] = "high"

    if "external_challenge" in geometric_themes:
        mapped["growth_opportunity"] = "high" if geometric_themes["external_challenge"] > 0.6 else "medium"

    return mapped


def compute_theme_alignment(predicted: Dict[str, str], expected: Dict[str, str]) -> float:
    """
    Compute alignment score between predicted and expected themes.

    Returns:
        Float 0-1 where 1 = perfect alignment
    """
    # Convert qualitative levels to numeric
    level_map = {"low": 0.2, "medium": 0.5, "high": 0.7, "very_high": 0.9}

    matches = 0
    total = 0

    for theme, expected_level in expected.items():
        if theme in predicted:
            expected_score = level_map.get(expected_level, 0.5)
            predicted_score = level_map.get(predicted[theme], 0.5)

            # Allow some tolerance (±0.2)
            if abs(expected_score - predicted_score) <= 0.2:
                matches += 1
            total += 1

    return matches / total if total > 0 else 0.0


def run_comparison_suite():
    """
    Run all test readings through geometric model and compare with expected.
    """
    space = SemanticSpace()

    print("=" * 70)
    print("GEOMETRIC MODEL vs HEURISTIC SYSTEM COMPARISON")
    print("=" * 70)

    results = []

    for test in TEST_READINGS:
        print(f"\n{'─' * 70}")
        print(f"Reading: {test['name']}")
        print(f"Cards: {[space.get_card(idx, rev).name for idx, rev in test['cards']]}")
        print(f"User: {test['user'].mbti}, Shadow={test['user'].shadow_integration:.1f}, Time={test['user'].temporal_focus:.1f}")
        print(f"{'─' * 70}")

        # Run geometric analysis
        analysis = analyze_geometric_reading(space, test['cards'], test['user'])

        print("\n📊 GEOMETRIC MODEL OUTPUT:")
        print(f"  Centroid: Elem={analysis['centroid'][0]:.2f}, Cons={analysis['centroid'][1]:.2f}, Time={analysis['centroid'][2]:.2f}")
        print(f"  Avg Overlap: {analysis['avg_overlap']:.3f}")
        print(f"  Themes detected:")
        for theme, score in sorted(analysis['themes'].items(), key=lambda x: x[1], reverse=True):
            print(f"    - {theme}: {score:.3f}")

        # Map to heuristic themes
        mapped_themes = map_geometric_to_heuristic_themes(analysis['themes'])
        print("\n🔄 MAPPED TO HEURISTIC THEMES:")
        for theme, level in sorted(mapped_themes.items()):
            print(f"    - {theme}: {level}")

        # Compare with expected
        alignment = compute_theme_alignment(mapped_themes, test['expected_themes'])
        print(f"\n✅ ALIGNMENT SCORE: {alignment:.1%}")

        print("\n📋 EXPECTED vs PREDICTED:")
        for theme in test['expected_themes']:
            expected = test['expected_themes'][theme]
            predicted = mapped_themes.get(theme, "MISSING")
            match = "✓" if predicted != "MISSING" and abs(
                {"low": 0.2, "medium": 0.5, "high": 0.7, "very_high": 0.9}.get(expected, 0.5) -
                {"low": 0.2, "medium": 0.5, "high": 0.7, "very_high": 0.9}.get(predicted, 0.0)
            ) <= 0.2 else "✗"
            print(f"    {match} {theme}: Expected={expected}, Predicted={predicted}")

        results.append({
            "reading": test['name'],
            "alignment": alignment,
            "avg_overlap": analysis['avg_overlap']
        })

    # Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)

    avg_alignment = np.mean([r['alignment'] for r in results])
    avg_overlap = np.mean([r['avg_overlap'] for r in results])

    print(f"\nAverage Theme Alignment: {avg_alignment:.1%}")
    print(f"Average Card Overlap: {avg_overlap:.3f}")

    print("\nPer-Reading Results:")
    for r in results:
        print(f"  {r['reading']}: {r['alignment']:.1%} alignment, {r['avg_overlap']:.3f} overlap")

    # Verdict
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)

    if avg_alignment >= 0.7:
        print("✅ GEOMETRIC MODEL VALIDATES: Captures 70%+ of heuristic themes")
        print("   → Worth exploring for novel pattern discovery")
    elif avg_alignment >= 0.5:
        print("⚠️  GEOMETRIC MODEL PARTIAL: Captures 50-70% of themes")
        print("   → May complement heuristics, but not replace")
    else:
        print("❌ GEOMETRIC MODEL FAILS: <50% theme alignment")
        print("   → Elegant theory, but heuristics superior in practice")

    print("\nKey insights:")
    if avg_overlap > 0.6:
        print("  - High overlap suggests coherent readings (cards work together)")
    else:
        print("  - Low overlap suggests scattered energies (cards in tension)")

    return results


if __name__ == "__main__":
    results = run_comparison_suite()
