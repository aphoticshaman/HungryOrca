"""
Enhanced Multi-Scale Theme Extraction
======================================

Addresses Phase 3 failures by implementing hierarchical pattern detection:
1. MICRO: Individual card semantic zones
2. MESO: Pairwise card interactions
3. MACRO: Reading-level centroid themes

Instead of just averaging to centroid (which hides details), we analyze
each scale separately and combine insights.
"""

import numpy as np
from typing import Dict, List, Set
from semantic_space import Card, SemanticSpace, UserProfile


class EnhancedThemeExtractor:
    """Multi-scale hierarchical theme extraction."""

    def __init__(self):
        # Define semantic zones in 3D space
        self.zones = {
            # Elemental zones (X-axis)
            "fire_zone": lambda pos: pos[0] > 0.6,  # Strong Fire-Air
            "water_zone": lambda pos: pos[0] < -0.6,  # Strong Water-Earth
            "air_zone": lambda pos: 0.3 < pos[0] <= 0.6 and pos[1] > 0,  # Air (active + conscious)
            "earth_zone": lambda pos: -0.6 <= pos[0] < -0.3 and pos[1] > 0,  # Earth (receptive + conscious)

            # Consciousness zones (Y-axis)
            "deep_shadow": lambda pos: pos[1] < -0.5,  # Deep unconscious
            "shadow_work": lambda pos: -0.5 <= pos[1] < -0.2,  # Active shadow integration
            "ego_conscious": lambda pos: 0.2 < pos[1] <= 0.6,  # Conscious awareness
            "superconscious": lambda pos: pos[1] > 0.6,  # Higher awareness/spiritual

            # Temporal zones (Z-axis)
            "past_bound": lambda pos: pos[2] < -0.3,  # Stuck in past
            "present_centered": lambda pos: -0.3 <= pos[2] <= 0.3,  # Present focus
            "future_oriented": lambda pos: pos[2] > 0.3,  # Future-looking
        }

        # Card-specific semantic signatures (all 78 cards)
        self.card_signatures = {
            # Major Arcana (0-21)
            0: {"curiosity", "new_beginnings", "leap_of_faith", "innocence"},
            1: {"manifestation", "willpower", "action", "mastery"},
            2: {"intuition", "mystery", "hidden_knowledge", "divine_feminine"},
            3: {"abundance", "nurturing", "fertility", "nature"},
            4: {"authority", "structure", "leadership", "stability"},
            5: {"tradition", "institutions", "conformity", "spiritual_wisdom"},
            6: {"choice", "love", "union", "alignment"},
            7: {"triumph", "determination", "willpower", "victory"},
            8: {"courage", "inner_strength", "compassion", "patience"},
            9: {"solitude", "introspection", "wisdom", "withdrawal"},
            10: {"fate", "cycles", "turning_point", "destiny"},
            11: {"fairness", "truth", "law", "balance"},
            12: {"surrender", "letting_go", "new_perspective", "sacrifice"},
            13: {"transformation", "endings", "rebirth", "transition"},
            14: {"balance", "moderation", "alchemy", "integration"},
            15: {"shadow_patterns", "addiction", "materialism", "bondage"},
            16: {"sudden_disruption", "tower_moment", "revelation", "chaos"},
            17: {"hope", "healing", "renewal", "inspiration"},
            18: {"illusion", "subconscious", "fear", "uncertainty"},
            19: {"joy", "success", "vitality", "clarity"},
            20: {"awakening", "reckoning", "absolution", "renewal"},
            21: {"completion", "integration", "wholeness", "achievement"},

            # Wands (Fire - Action, Passion, Creativity) 22-35
            22: {"inspiration", "potential", "creativity", "new_ventures"},
            23: {"planning", "future_vision", "progress", "discovery"},
            24: {"expansion", "foresight", "leadership", "collaboration"},
            25: {"celebration", "harmony", "homecoming", "community"},
            26: {"competition", "conflict", "tension", "struggle"},
            27: {"victory", "recognition", "progress", "success"},  # 6 Wands
            28: {"defensiveness", "perseverance", "challenge", "courage"},
            29: {"swift_action", "movement", "progress", "momentum"},
            30: {"burden", "responsibility", "perseverance", "overwhelm"},
            31: {"burnout", "burden", "responsibility", "completion_struggle"},
            32: {"enthusiasm", "adventure", "exploration", "curiosity"},
            33: {"passion", "adventure", "impulsiveness", "rush"},  # Knight Wands
            34: {"confidence", "independence", "determination", "charisma"},
            35: {"leadership", "vision", "entrepreneurship", "boldness"},

            # Cups (Water - Emotion, Relationships, Intuition) 36-49
            36: {"love", "new_relationship", "emotional_beginning", "compassion"},
            37: {"partnership", "unity", "connection", "mutual_respect"},
            38: {"celebration", "friendship", "community", "joy"},
            39: {"apathy", "contemplation", "reevaluation", "discontent"},
            40: {"grief", "loss", "disappointment", "regret"},  # 5 Cups
            41: {"nostalgia", "reunion", "innocence", "childhood"},
            42: {"illusion", "fantasy", "choices", "wishful_thinking"},
            43: {"moving_on", "transition", "leaving_behind", "journey"},
            44: {"contentment", "satisfaction", "wishes_fulfilled", "happiness"},
            45: {"emotional_fulfillment", "family_joy", "harmony", "love"},  # 10 Cups
            46: {"sensitivity", "intuition", "dreamer", "emotional_messages"},
            47: {"romance", "charm", "imagination", "idealism"},  # Knight Cups
            48: {"compassion", "empathy", "nurturing", "intuitive"},
            49: {"emotional_maturity", "balance", "diplomacy", "caring"},

            # Swords (Air - Intellect, Conflict, Truth) 50-63
            50: {"clarity", "breakthrough", "truth", "mental_clarity"},
            51: {"difficult_choice", "stalemate", "avoidance", "blocked"},
            52: {"heartbreak", "sorrow", "painful_truth", "grief"},  # 3 Swords
            53: {"rest", "recovery", "contemplation", "respite"},
            54: {"defeat", "loss", "conflict", "betrayal"},  # 5 Swords
            55: {"transition", "moving_forward", "recovery", "travel"},
            56: {"deception", "strategy", "sneakiness", "mental_game"},
            57: {"restriction", "isolation", "imprisonment", "helplessness"},
            58: {"anxiety", "worry", "nightmares", "fear"},
            59: {"rock_bottom", "endings", "painful_completion", "betrayal"},
            60: {"curiosity", "mental_energy", "vigilance", "ideas"},
            61: {"action", "impulsiveness", "haste", "determination"},  # Knight Swords
            62: {"independence", "clear_thinking", "directness", "perception"},
            63: {"authority", "intellectual_power", "truth", "clarity"},

            # Pentacles (Earth - Material, Work, Practicality) 64-77
            64: {"opportunity", "manifestation", "new_financial_beginning", "prosperity"},
            65: {"balance", "adaptability", "juggling", "time_management"},
            66: {"teamwork", "learning", "collaboration", "skill_building"},
            67: {"security", "control", "possessiveness", "saving"},
            68: {"hardship", "loss", "isolation", "poverty"},  # 5 Pentacles
            69: {"generosity", "charity", "sharing", "balance"},
            70: {"patience", "perseverance", "investment", "long_term_vision"},
            71: {"mastery", "skill", "dedication", "craftsmanship"},
            72: {"self_sufficiency", "luxury", "independence", "refinement"},
            73: {"legacy", "inheritance", "family", "wealth"},
            74: {"ambition", "diligence", "study", "manifestation"},
            75: {"reliability", "hard_work", "routine", "patience"},  # Knight Pentacles
            76: {"nurturing", "practical", "abundance", "security"},
            77: {"material_mastery", "wealth", "stability", "success"},  # King Pentacles
        }

    def extract_micro_themes(self, cards: List[Card]) -> Set[str]:
        """
        MICRO: Analyze individual card positions in semantic space.

        Returns themes based on which zones each card occupies.
        """
        themes = set()

        for card in cards:
            pos = card.embedding

            # Check which zones this card occupies
            for zone_name, zone_func in self.zones.items():
                if zone_func(pos):
                    themes.add(f"card_in_{zone_name}")

            # Add card-specific signatures
            if card.index in self.card_signatures:
                themes.update(self.card_signatures[card.index])

            # Reversal handling
            if card.reversed:
                # Reversed cards often mean blocked/inverted energy
                if card.index == 17:  # Star reversed
                    themes.discard("hope")
                    themes.add("hope_dimmed")
                    themes.add("despair")
                if card.index == 54:  # 3 of Swords reversed
                    themes.discard("heartbreak")
                    themes.add("healing_heartbreak")
                    themes.add("recovery")

        return themes

    def extract_meso_themes(self, cards: List[Card], overlaps: np.ndarray) -> Set[str]:
        """
        MESO: Analyze pairwise card interactions based on overlap strength.

        High overlap = cards reinforce each other
        Low overlap = cards in tension
        """
        themes = set()
        n = len(cards)

        for i in range(n):
            for j in range(i + 1, n):
                overlap = overlaps[i, j]
                card_i, card_j = cards[i], cards[j]

                # Check for specific powerful combinations
                if {card_i.index, card_j.index} == {16, 13}:  # Tower + Death
                    if overlap > 0.7:
                        themes.add("compound_transformation")
                        themes.add("double_ending")
                    themes.add("major_disruption")

                if {card_i.index, card_j.index} == {15, 16}:  # Devil + Tower
                    themes.add("shadow_confrontation")
                    themes.add("pattern_breaking")
                    if overlap > 0.6:
                        themes.add("sudden_liberation")

                if {card_i.index, card_j.index} == {13, 17}:  # Death + Star
                    themes.add("death_rebirth_cycle")
                    themes.add("phoenix_moment")
                    if overlap > 0.6:
                        themes.add("transformative_hope")

                # General interaction patterns
                if overlap > 0.7:
                    # Cards in same semantic region (reinforcing)
                    themes.add("high_synergy")
                elif overlap < 0.4:
                    # Cards in different regions (tension)
                    themes.add("conflicting_energies")

                # Shadow card pairs
                shadow_cards = {15, 16, 13, 54}  # Devil, Tower, Death, 3 Swords
                if card_i.index in shadow_cards and card_j.index in shadow_cards:
                    themes.add("shadow_work_intense")

        return themes

    def extract_macro_themes(self, centroid: np.ndarray, avg_overlap: float) -> Set[str]:
        """
        MACRO: Analyze reading-level patterns from centroid and overall coherence.

        This is the original approach - still valid for broad categorization.
        """
        themes = set()

        x, y, z = centroid

        # Elemental themes
        if x > 0.5:
            themes.add("active_energy_dominant")
            themes.add("fire_air_emphasis")
        elif x < -0.5:
            themes.add("receptive_energy_dominant")
            themes.add("water_earth_emphasis")
        else:
            themes.add("elemental_balance")

        # Consciousness themes
        if y > 0.5:
            themes.add("conscious_awareness_high")
        elif y < -0.5:
            themes.add("shadow_work_needed")
        else:
            themes.add("ego_shadow_integration")

        # Temporal themes
        if z > 0.4:
            themes.add("future_focused")
        elif z < -0.4:
            themes.add("past_processing")
        else:
            themes.add("present_centered")

        # Coherence themes
        if avg_overlap > 0.6:
            themes.add("reading_coherent")
            themes.add("clear_message")
        elif avg_overlap < 0.45:
            themes.add("reading_scattered")
            themes.add("multiple_directions")

        return themes

    def synthesize_themes(self, micro: Set[str], meso: Set[str], macro: Set[str],
                         user: UserProfile) -> Dict[str, float]:
        """
        Combine micro, meso, macro themes into weighted final theme set.

        Weighting:
        - Micro: 0.5 (card-specific meanings are most important)
        - Meso: 0.3 (interactions are important)
        - Macro: 0.2 (overall tone is background)
        """
        final_themes = {}

        # Micro themes (highest weight)
        for theme in micro:
            final_themes[theme] = final_themes.get(theme, 0) + 0.5

        # Meso themes (medium weight)
        for theme in meso:
            final_themes[theme] = final_themes.get(theme, 0) + 0.3

        # Macro themes (lower weight - provides context)
        for theme in macro:
            final_themes[theme] = final_themes.get(theme, 0) + 0.2

        # User alignment boost
        # If user is shadow-averse (Y < 0) but reading has shadow themes, boost those
        if user.shadow_integration < -0.2:
            for theme in list(final_themes.keys()):
                if "shadow" in theme:
                    final_themes[theme] *= 1.3  # Boost shadow themes for resistant users

        # Normalize to 0-1 range
        if final_themes:
            max_score = max(final_themes.values())
            final_themes = {k: v / max_score for k, v in final_themes.items()}

        return final_themes

    def extract_all(self, space: SemanticSpace, cards: List[Card],
                   user: UserProfile) -> Dict[str, float]:
        """
        Full multi-scale theme extraction pipeline.

        Returns:
            Dict mapping theme names to confidence scores (0-1)
        """
        # Compute geometric features
        overlaps = space.compute_overlap_strength(cards)
        centroid = space.get_centroid(cards)
        avg_overlap = np.mean(overlaps[np.triu_indices_from(overlaps, k=1)])

        # Extract themes at each scale
        micro = self.extract_micro_themes(cards)
        meso = self.extract_meso_themes(cards, overlaps)
        macro = self.extract_macro_themes(centroid, avg_overlap)

        # Synthesize into final weighted themes
        final = self.synthesize_themes(micro, meso, macro, user)

        return {
            "themes": final,
            "micro": micro,
            "meso": meso,
            "macro": macro,
            "centroid": centroid,
            "avg_overlap": avg_overlap
        }


if __name__ == "__main__":
    # Test enhanced extraction
    from semantic_space import CARD_EMBEDDINGS

    space = SemanticSpace()
    extractor = EnhancedThemeExtractor()

    # Test case: Shadow Confrontation (previously failed with 0% alignment)
    print("=" * 70)
    print("TEST: Shadow Confrontation (Devil + Tower + Magician)")
    print("Expected: shadow_work (very_high), pattern_breaking (high)")
    print("=" * 70)

    cards = [
        space.get_card(15, False),  # The Devil
        space.get_card(16, False),  # The Tower
        space.get_card(1, False),   # The Magician
    ]
    user = UserProfile(mbti="ENFP", shadow_integration=-0.2, temporal_focus=0.0)

    result = extractor.extract_all(space, cards, user)

    print("\nMICRO themes (card-level):")
    for theme in sorted(result['micro']):
        print(f"  - {theme}")

    print("\nMESO themes (interaction-level):")
    for theme in sorted(result['meso']):
        print(f"  - {theme}")

    print("\nMACRO themes (reading-level):")
    for theme in sorted(result['macro']):
        print(f"  - {theme}")

    print("\nFINAL WEIGHTED themes:")
    for theme, score in sorted(result['themes'].items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {score:.3f} - {theme}")

    print(f"\nCentroid: {result['centroid']}")
    print(f"Avg Overlap: {result['avg_overlap']:.3f}")

    # Check if we now detect expected themes
    final_themes = set(result['themes'].keys())
    if "shadow_work_intense" in final_themes or "shadow_confrontation" in final_themes:
        print("\n✅ SUCCESS: Shadow work themes detected!")
    else:
        print("\n❌ FAIL: Shadow work themes still missing")

    if "pattern_breaking" in final_themes or "sudden_liberation" in final_themes:
        print("✅ SUCCESS: Pattern breaking detected!")
    else:
        print("❌ FAIL: Pattern breaking missing")
