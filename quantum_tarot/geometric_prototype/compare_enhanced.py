"""
Re-run comparison with enhanced multi-scale theme extraction
"""

from compare_systems import TEST_READINGS, compute_theme_alignment
from semantic_space import SemanticSpace, UserProfile
from enhanced_themes import EnhancedThemeExtractor
import numpy as np


def map_enhanced_to_expected(enhanced_themes: dict) -> dict:
    """
    Map enhanced themes to expected heuristic theme format.

    This is more sophisticated than the naive centroid mapping.
    """
    mapped = {}

    themes = enhanced_themes['themes']

    # Direct mappings from card signatures
    if any(k in themes for k in ['transformation', 'endings', 'rebirth']):
        mapped['transformation'] = 'very_high' if themes.get('transformation', 0) > 0.7 else 'high'

    if any(k in themes for k in ['shadow_work_intense', 'shadow_confrontation', 'deep_shadow']):
        mapped['shadow_work'] = 'very_high'
    elif 'shadow_patterns' in themes or 'card_in_shadow_work' in themes:
        mapped['shadow_work'] = 'high'
    elif 'ego_shadow_integration' in themes:
        mapped['shadow_work'] = 'medium'

    if 'hope' in themes:
        mapped['future_hope'] = 'very_high' if themes['hope'] > 0.7 else 'high'
    elif 'hope_dimmed' in themes:
        mapped['hope_dimmed'] = 'high'
        mapped['future_hope'] = 'low'

    if any(k in themes for k in ['sudden_disruption', 'tower_moment', 'major_disruption']):
        mapped['disruption'] = 'very_high' if themes.get('sudden_disruption', 0) > 0.7 else 'high'

    if 'pattern_breaking' in themes or 'sudden_liberation' in themes:
        mapped['pattern_breaking'] = 'high'

    if any(k in themes for k in ['manifestation', 'action', 'willpower']):
        mapped['active_response'] = 'high'

    if 'revelation' in themes or 'sudden_realization' in themes:
        mapped['sudden_realization'] = 'high'

    # Elemental themes
    if 'fire_air_emphasis' in themes or themes.get('card_in_fire_zone', 0) > 0.5:
        mapped['fire_energy'] = 'very_high'
        mapped['action_oriented'] = 'very_high'

    if 'water_earth_emphasis' in themes or themes.get('card_in_water_zone', 0) > 0.5:
        mapped['emotional_contentment'] = 'high'
        mapped['material_mastery'] = 'high'

    # Temporal themes
    if 'future_focused' in themes or themes.get('card_in_future_oriented', 0) > 0.5:
        mapped['future_optimism'] = 'high'

    if 'past_processing' in themes or themes.get('card_in_past_bound', 0) > 0.5:
        mapped['healing_heartbreak'] = 'high'
        mapped['letting_go'] = 'high'

    if 'present_centered' in themes:
        mapped['present_action'] = 'high'

    # Specific card combinations
    if 'phoenix_moment' in themes or 'death_rebirth_cycle' in themes:
        mapped['transformation'] = 'very_high'
        mapped['future_hope'] = 'high'

    if 'emotional_fulfillment' in themes or 'family_joy' in themes:
        mapped['fulfillment'] = 'very_high'

    if 'heartbreak' in themes or 'sorrow' in themes:
        if 'healing_heartbreak' not in themes:
            mapped['heartbreak'] = 'high'

    if 'healing_heartbreak' in themes or 'recovery' in themes:
        mapped['healing_heartbreak'] = 'high'
        mapped['letting_go'] = 'high'

    if 'material_mastery' in themes or 'stability' in themes:
        mapped['material_mastery'] = 'very_high'

    # Coherence themes
    if 'high_synergy' in themes or 'reading_coherent' in themes:
        mapped['synergy'] = 'high'

    if 'conflicting_energies' in themes or 'reading_scattered' in themes:
        if themes.get('conflicting_energies', 0) > 0.5:
            mapped['rapid_change'] = 'high'

    return mapped


def run_enhanced_comparison():
    """Run comparison with enhanced multi-scale extraction."""

    space = SemanticSpace()
    extractor = EnhancedThemeExtractor()

    print("=" * 70)
    print("ENHANCED GEOMETRIC MODEL vs HEURISTIC SYSTEM")
    print("(Multi-Scale: Micro + Meso + Macro)")
    print("=" * 70)

    results = []

    for test in TEST_READINGS:
        print(f"\n{'─' * 70}")
        print(f"Reading: {test['name']}")
        print(f"Cards: {[space.get_card(idx, rev).name for idx, rev in test['cards']]}")
        print(f"{'─' * 70}")

        # Get cards and user
        cards = [space.get_card(idx, rev) for idx, rev in test['cards']]
        user = test['user']

        # Extract themes using enhanced multi-scale approach
        analysis = extractor.extract_all(space, cards, user)

        print("\n📊 ENHANCED EXTRACTION:")
        print(f"  Micro themes: {len(analysis['micro'])} detected")
        print(f"  Meso themes: {len(analysis['meso'])} detected")
        print(f"  Macro themes: {len(analysis['macro'])} detected")
        print(f"  Total weighted: {len(analysis['themes'])} themes")

        # Show top themes
        print("\n  Top 5 themes:")
        for theme, score in sorted(analysis['themes'].items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"    {score:.3f} - {theme}")

        # Map to expected format
        mapped = map_enhanced_to_expected(analysis)

        print("\n🔄 MAPPED THEMES:")
        for theme, level in sorted(mapped.items()):
            print(f"    - {theme}: {level}")

        # Compute alignment
        alignment = compute_theme_alignment(mapped, test['expected_themes'])

        print(f"\n✅ ALIGNMENT SCORE: {alignment:.1%}")

        # Detailed comparison
        print("\n📋 EXPECTED vs PREDICTED:")
        level_map = {"low": 0.2, "medium": 0.5, "high": 0.7, "very_high": 0.9}
        for theme in test['expected_themes']:
            expected = test['expected_themes'][theme]
            predicted = mapped.get(theme, "MISSING")

            if predicted != "MISSING":
                expected_score = level_map.get(expected, 0.5)
                predicted_score = level_map.get(predicted, 0.5)
                match = "✓" if abs(expected_score - predicted_score) <= 0.2 else "✗"
            else:
                match = "✗"

            print(f"    {match} {theme}: Expected={expected}, Predicted={predicted}")

        results.append({
            "reading": test['name'],
            "alignment": alignment,
            "avg_overlap": analysis['avg_overlap'],
            "theme_count": len(analysis['themes'])
        })

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)

    avg_alignment = np.mean([r['alignment'] for r in results])
    avg_overlap = np.mean([r['avg_overlap'] for r in results])

    print(f"\nAverage Theme Alignment: {avg_alignment:.1%}")
    print(f"Average Card Overlap: {avg_overlap:.3f}")

    print("\nPer-Reading Results:")
    for r in results:
        print(f"  {r['reading']}: {r['alignment']:.1%} alignment")

    # Verdict
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)

    if avg_alignment >= 0.7:
        print("✅ GEOMETRIC MODEL VALIDATES: Captures 70%+ of heuristic themes")
        print("   → Enhanced multi-scale extraction works!")
        print("   → Can complement or replace hand-coded rules")
    elif avg_alignment >= 0.5:
        print("⚠️  GEOMETRIC MODEL PARTIAL: Captures 50-70% of themes")
        print("   → Improvement over naive approach, but gaps remain")
    else:
        print("❌ GEOMETRIC MODEL INSUFFICIENT: <50% alignment")
        print("   → Heuristics remain superior")

    improvement = avg_alignment - 0.50  # Original naive alignment
    print(f"\nImprovement over naive centroid-only: +{improvement:.1%}")

    return results


if __name__ == "__main__":
    results = run_enhanced_comparison()
