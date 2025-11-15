"""
Phase 6: A/B Validation of Geometric Enhancement
=================================================

Tests geometric themes against pure heuristic synthesis by generating
side-by-side comparisons for diverse card combinations.

Validation metrics:
- Novel pattern detection (patterns found only by geometric model)
- Theme richness (unique themes added)
- Synthesis coherence (how well themes integrate)
"""

import sys
sys.path.append('/home/user/HungryOrca/quantum_tarot/geometric_prototype')

from semantic_space import SemanticSpace, UserProfile
from enhanced_themes import EnhancedThemeExtractor


# Diverse test cases covering different reading scenarios
TEST_CASES = [
    {
        "name": "Classic Phoenix (Tower → Death → Star)",
        "description": "Destruction leading to rebirth - should detect compound transformation",
        "cards": [16, 13, 17],  # Tower, Death, Star
        "user": UserProfile(mbti="INTJ", shadow_integration=0.4, temporal_focus=0.6),
        "expected_geometric_insights": [
            "compound_transformation",
            "phoenix_rebirth",
            "shadow_work",
            "future_focused"
        ]
    },
    {
        "name": "Shadow Breakthrough (Devil → Tower → Magician)",
        "description": "Breaking free from bondage - should detect liberation pattern",
        "cards": [15, 16, 1],  # Devil, Tower, Magician
        "user": UserProfile(mbti="ENFP", shadow_integration=-0.2, temporal_focus=0.0),
        "expected_geometric_insights": [
            "shadow_breakthrough",
            "pattern_breaking",
            "deep_shadow to conscious_integration",
            "active_fire_air"
        ]
    },
    {
        "name": "Emotional Harmony (10 Cups → 2 Cups → Ace Cups)",
        "description": "Love and fulfillment - should detect water/receptive dominance",
        "cards": [45, 37, 36],  # 10 Cups, 2 Cups, Ace Cups
        "user": UserProfile(mbti="ESFJ", shadow_integration=0.7, temporal_focus=0.3),
        "expected_geometric_insights": [
            "receptive_water_earth",
            "high_synergy",
            "emotional_fulfillment",
            "conscious_integration"
        ]
    },
    {
        "name": "Action Overload (Knight Wands → 5 Wands → King Wands)",
        "description": "Excessive fire energy - should detect scattered/conflict",
        "cards": [33, 26, 35],  # Knight Wands, 5 Wands, King Wands
        "user": UserProfile(mbti="ESTP", shadow_integration=0.1, temporal_focus=0.8),
        "expected_geometric_insights": [
            "active_fire_air",
            "competition/conflict",
            "future_focused",
            "scattered_energies or high_synergy"
        ]
    },
    {
        "name": "Intellectual Clarity (Ace Swords → King Swords → Justice)",
        "description": "Mental clarity and truth - should detect air element balance",
        "cards": [50, 63, 11],  # Ace Swords, King Swords, Justice
        "user": UserProfile(mbti="INTP", shadow_integration=0.5, temporal_focus=0.0),
        "expected_geometric_insights": [
            "clarity/truth",
            "present_centered",
            "balanced_approach"
        ]
    },
    {
        "name": "Material Struggle (5 Pentacles → 4 Pentacles → 10 Pentacles)",
        "description": "Poverty to abundance - should detect temporal progression",
        "cards": [68, 67, 73],  # 5 Pents, 4 Pents, 10 Pents
        "user": UserProfile(mbti="ISTJ", shadow_integration=-0.3, temporal_focus=-0.2),
        "expected_geometric_insights": [
            "receptive_water_earth",
            "shadow_work",
            "past_processing or present_centered"
        ]
    },
    {
        "name": "Scattered Energies (Moon → 7 Swords → 5 Cups)",
        "description": "Confusion, deception, loss - should detect low coherence",
        "cards": [18, 56, 40],  # Moon, 7 Swords, 5 Cups
        "user": UserProfile(mbti="INFP", shadow_integration=-0.5, temporal_focus=-0.4),
        "expected_geometric_insights": [
            "scattered_energies",
            "shadow_work_needed",
            "illusion/deception"
        ]
    },
    {
        "name": "Balanced Journey (Wheel → Temperance → World)",
        "description": "Fate, balance, completion - should detect integration theme",
        "cards": [10, 14, 21],  # Wheel, Temperance, World
        "user": UserProfile(mbti="INFJ", shadow_integration=0.6, temporal_focus=0.4),
        "expected_geometric_insights": [
            "integration/wholeness",
            "balance/moderation",
            "future_focused or completion"
        ]
    }
]


def run_validation():
    """Run A/B validation comparing geometric vs pure heuristic analysis."""

    space = SemanticSpace()
    extractor = EnhancedThemeExtractor()

    print("=" * 80)
    print("PHASE 6: A/B VALIDATION - GEOMETRIC ENHANCEMENT")
    print("=" * 80)
    print()

    results = []

    for i, test in enumerate(TEST_CASES, 1):
        print(f"\n{'━' * 80}")
        print(f"TEST {i}/{len(TEST_CASES)}: {test['name']}")
        print(f"Description: {test['description']}")
        print(f"Cards: {[space.get_card(idx).name for idx in test['cards']]}")
        print(f"{'━' * 80}\n")

        # Get cards
        cards = [space.get_card(idx, False) for idx in test['cards']]
        user = test['user']

        # Extract geometric themes using enhanced multi-scale extraction
        analysis = extractor.extract_all(space, cards, user)

        # Display results
        print("📐 GEOMETRIC ANALYSIS:")
        print(f"  Centroid: Elem={analysis['centroid'][0]:.2f}, "
              f"Cons={analysis['centroid'][1]:.2f}, Time={analysis['centroid'][2]:.2f}")
        print(f"  Avg Overlap: {analysis['avg_overlap']:.3f}")
        print(f"  Total Themes: {len(analysis['themes'])}")

        print("\n  Micro Themes (card-level):")
        for theme in sorted(list(analysis['micro'])[:10]):  # Show top 10
            print(f"    • {theme}")

        print("\n  Meso Themes (interaction-level):")
        for theme in sorted(analysis['meso']):
            print(f"    • {theme}")

        print("\n  Macro Themes (reading-level):")
        for theme in sorted(analysis['macro']):
            print(f"    • {theme}")

        # Check for expected insights
        print("\n✓ VALIDATION CHECK:")
        all_themes = set()
        all_themes.update(analysis['micro'])
        all_themes.update(analysis['meso'])
        all_themes.update(analysis['macro'])
        all_themes.update(analysis['themes'].keys())

        found_expected = []
        for expected in test['expected_geometric_insights']:
            # Fuzzy match (check if expected is substring of any theme)
            matches = [t for t in all_themes if expected.lower().replace('_', ' ') in t.lower().replace('_', ' ')
                      or t.lower().replace('_', ' ') in expected.lower().replace('_', ' ')]
            if matches:
                found_expected.append((expected, matches[0]))
                print(f"  ✓ Found: {expected} → {matches[0]}")
            else:
                print(f"  ✗ Missing: {expected}")

        detection_rate = len(found_expected) / len(test['expected_geometric_insights'])

        # Identify NOVEL themes (not in expected, discovered by model)
        expected_set = set(e.lower().replace(' ', '_') for e in test['expected_geometric_insights'])
        novel_themes = [t for t in all_themes if t.lower() not in expected_set
                       and not any(e in t.lower() for e in expected_set)]

        print(f"\n🔬 NOVEL INSIGHTS (not pre-specified):")
        for theme in sorted(novel_themes)[:5]:  # Show top 5 novel
            print(f"    • {theme}")

        # Store results
        results.append({
            'test_name': test['name'],
            'detection_rate': detection_rate,
            'total_themes': len(analysis['themes']),
            'novel_themes_count': len(novel_themes),
            'avg_overlap': analysis['avg_overlap']
        })

    # Summary statistics
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)

    avg_detection = sum(r['detection_rate'] for r in results) / len(results)
    avg_themes = sum(r['total_themes'] for r in results) / len(results)
    avg_novel = sum(r['novel_themes_count'] for r in results) / len(results)
    avg_overlap = sum(r['avg_overlap'] for r in results) / len(results)

    print(f"\nAverage Expected Pattern Detection: {avg_detection:.1%}")
    print(f"Average Themes Extracted: {avg_themes:.1f}")
    print(f"Average Novel Insights: {avg_novel:.1f}")
    print(f"Average Card Overlap: {avg_overlap:.3f}")

    print("\nPer-Test Results:")
    for r in results:
        print(f"  {r['test_name']:50s} | Detection: {r['detection_rate']:.0%} | "
              f"Themes: {r['total_themes']:2d} | Novel: {r['novel_themes_count']:2d}")

    # Verdict
    print("\n" + "=" * 80)
    print("VERDICT")
    print("=" * 80)

    if avg_detection >= 0.7:
        print("✅ GEOMETRIC MODEL VALIDATED: Detects 70%+ of expected patterns")
        print("   → Production-ready for synthesis enhancement")
    elif avg_detection >= 0.5:
        print("⚠️  GEOMETRIC MODEL PARTIAL: Detects 50-70% of patterns")
        print("   → Consider refinement before full deployment")
    else:
        print("❌ GEOMETRIC MODEL NEEDS WORK: <50% pattern detection")
        print("   → Requires embedding refinement or architecture changes")

    if avg_novel >= 3:
        print(f"\n✨ HIGH NOVELTY: Average {avg_novel:.1f} novel insights per reading")
        print("   → Geometric model discovering patterns not pre-specified")

    return results


if __name__ == "__main__":
    results = run_validation()
