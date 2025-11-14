"""
Geometric Semantic Space for Tarot Synthesis
=============================================

3D continuous space where cards exist as overlapping polygons/regions:
- X-axis: Elemental polarity (Fire-Air [active] ← → Water-Earth [receptive])
- Y-axis: Consciousness depth (Ego/Persona [+1] ← → Shadow/Unconscious [-1])
- Z-axis: Temporal focus (Past [-1] ← → Present [0] ← → Future [+1])

Each card occupies a region in this space. Overlaps create emergent meanings.
User profile acts as a query vector that weights card relevance.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class Card:
    """Tarot card with semantic embedding."""
    index: int
    name: str
    embedding: np.ndarray  # 3D vector [x, y, z]
    radius: float = 0.2    # Influence radius (for overlap calculations)
    reversed: bool = False


# All 78 Tarot Card Embeddings
# Auto-generated with hand-tuned Major Arcana
# Format: [elemental, consciousness, temporal]
# Range: [-1, 1] for each axis
CARD_EMBEDDINGS: Dict[int, Tuple[str, np.ndarray, float]] = {
    # Major Arcana (0-21) - Hand-tuned archetypal positions
    0: ("The Fool", np.array([0.100, 0.700, 0.900]), 0.30),
    1: ("The Magician", np.array([0.500, 0.800, 0.500]), 0.25),
    2: ("The High Priestess", np.array([-0.300, 0.600, -0.200]), 0.35),
    3: ("The Empress", np.array([-0.700, 0.700, 0.400]), 0.40),
    4: ("The Emperor", np.array([0.600, 0.500, 0.300]), 0.35),
    5: ("The Hierophant", np.array([0.200, 0.400, -0.100]), 0.30),
    6: ("The Lovers", np.array([0.400, 0.500, 0.200]), 0.35),
    7: ("The Chariot", np.array([0.700, 0.600, 0.500]), 0.30),
    8: ("Strength", np.array([0.500, 0.700, 0.300]), 0.35),
    9: ("The Hermit", np.array([0.100, 0.300, -0.300]), 0.30),
    10: ("Wheel of Fortune", np.array([0.000, 0.000, 0.000]), 0.40),
    11: ("Justice", np.array([0.300, 0.500, 0.000]), 0.30),
    12: ("The Hanged Man", np.array([-0.200, -0.200, -0.100]), 0.35),
    13: ("Death", np.array([0.000, -0.300, 0.000]), 0.40),
    14: ("Temperance", np.array([-0.100, 0.600, 0.400]), 0.35),
    15: ("The Devil", np.array([0.700, -0.800, -0.200]), 0.30),
    16: ("The Tower", np.array([0.900, -0.600, 0.100]), 0.35),
    17: ("The Star", np.array([-0.400, 0.600, 0.800]), 0.40),
    18: ("The Moon", np.array([-0.500, -0.400, 0.200]), 0.35),
    19: ("The Sun", np.array([0.600, 0.800, 0.700]), 0.40),
    20: ("Judgement", np.array([0.300, 0.500, 0.600]), 0.35),
    21: ("The World", np.array([0.000, 0.700, 0.500]), 0.40),
    # Wands (Fire) - Suit 22-35
    22: ("Ace of Wands", np.array([0.800, 0.600, -0.400]), 0.20),
    23: ("Two of Wands", np.array([0.800, 0.400, -0.300]), 0.20),
    24: ("Three of Wands", np.array([0.800, 0.500, -0.200]), 0.20),
    25: ("Four of Wands", np.array([0.800, 0.300, -0.100]), 0.20),
    26: ("Five of Wands", np.array([0.800, -0.200, 0.000]), 0.20),
    27: ("Six of Wands", np.array([0.800, 0.400, 0.100]), 0.20),
    28: ("Seven of Wands", np.array([0.800, -0.100, 0.000]), 0.20),
    29: ("Eight of Wands", np.array([0.800, 0.200, 0.200]), 0.20),
    30: ("Nine of Wands", np.array([0.800, 0.100, 0.300]), 0.20),
    31: ("Ten of Wands", np.array([0.800, 0.000, 0.400]), 0.25),
    32: ("Page of Wands", np.array([0.800, 0.500, 0.100]), 0.25),
    33: ("Knight of Wands", np.array([0.900, 0.400, 0.500]), 0.30),
    34: ("Queen of Wands", np.array([0.700, 0.600, 0.200]), 0.25),
    35: ("King of Wands", np.array([0.900, 0.500, 0.300]), 0.25),
    # Cups (Water) - Suit 36-49
    36: ("Ace of Cups", np.array([-0.800, 0.700, -0.400]), 0.20),
    37: ("Two of Cups", np.array([-0.800, 0.500, -0.300]), 0.20),
    38: ("Three of Cups", np.array([-0.800, 0.600, -0.200]), 0.20),
    39: ("Four of Cups", np.array([-0.800, 0.400, -0.100]), 0.20),
    40: ("Five of Cups", np.array([-0.800, -0.100, 0.000]), 0.20),
    41: ("Six of Cups", np.array([-0.800, 0.500, 0.100]), 0.20),
    42: ("Seven of Cups", np.array([-0.800, 0.000, 0.000]), 0.20),
    43: ("Eight of Cups", np.array([-0.800, 0.300, 0.200]), 0.20),
    44: ("Nine of Cups", np.array([-0.800, 0.200, 0.300]), 0.20),
    45: ("Ten of Cups", np.array([-0.800, 0.100, 0.400]), 0.25),
    46: ("Page of Cups", np.array([-0.800, 0.600, 0.100]), 0.25),
    47: ("Knight of Cups", np.array([-0.900, 0.500, 0.500]), 0.30),
    48: ("Queen of Cups", np.array([-0.700, 0.700, 0.200]), 0.25),
    49: ("King of Cups", np.array([-0.700, 0.600, 0.300]), 0.25),
    # Swords (Air) - Suit 50-63
    50: ("Ace of Swords", np.array([0.300, 0.600, -0.400]), 0.20),
    51: ("Two of Swords", np.array([0.300, 0.400, -0.300]), 0.20),
    52: ("Three of Swords", np.array([0.300, 0.500, -0.200]), 0.20),
    53: ("Four of Swords", np.array([0.300, 0.300, -0.100]), 0.20),
    54: ("Five of Swords", np.array([0.300, -0.200, 0.000]), 0.20),
    55: ("Six of Swords", np.array([0.300, 0.400, 0.100]), 0.20),
    56: ("Seven of Swords", np.array([0.300, -0.100, 0.000]), 0.20),
    57: ("Eight of Swords", np.array([0.300, 0.200, 0.200]), 0.20),
    58: ("Nine of Swords", np.array([0.300, 0.100, 0.300]), 0.20),
    59: ("Ten of Swords", np.array([0.300, 0.000, 0.400]), 0.25),
    60: ("Page of Swords", np.array([0.300, 0.500, 0.100]), 0.25),
    61: ("Knight of Swords", np.array([0.400, 0.400, 0.500]), 0.30),
    62: ("Queen of Swords", np.array([0.200, 0.600, 0.200]), 0.25),
    63: ("King of Swords", np.array([0.400, 0.500, 0.300]), 0.25),
    # Pentacles (Earth) - Suit 64-77
    64: ("Ace of Pentacles", np.array([-0.600, 0.500, -0.400]), 0.20),
    65: ("Two of Pentacles", np.array([-0.600, 0.300, -0.300]), 0.20),
    66: ("Three of Pentacles", np.array([-0.600, 0.400, -0.200]), 0.20),
    67: ("Four of Pentacles", np.array([-0.600, 0.200, -0.100]), 0.20),
    68: ("Five of Pentacles", np.array([-0.600, -0.300, 0.000]), 0.20),
    69: ("Six of Pentacles", np.array([-0.600, 0.300, 0.100]), 0.20),
    70: ("Seven of Pentacles", np.array([-0.600, -0.200, 0.000]), 0.20),
    71: ("Eight of Pentacles", np.array([-0.600, 0.100, 0.200]), 0.20),
    72: ("Nine of Pentacles", np.array([-0.600, 0.000, 0.300]), 0.20),
    73: ("Ten of Pentacles", np.array([-0.600, -0.100, 0.400]), 0.25),
    74: ("Page of Pentacles", np.array([-0.600, 0.400, 0.100]), 0.25),
    75: ("Knight of Pentacles", np.array([-0.700, 0.300, 0.500]), 0.30),
    76: ("Queen of Pentacles", np.array([-0.500, 0.500, 0.200]), 0.25),
    77: ("King of Pentacles", np.array([-0.500, 0.400, 0.300]), 0.25),
}


class SemanticSpace:
    """3D continuous semantic space for tarot interpretation."""

    def __init__(self):
        self.cards: Dict[int, Card] = {}
        self._initialize_cards()

    def _initialize_cards(self):
        """Load card embeddings into Card objects."""
        for idx, (name, embedding, radius) in CARD_EMBEDDINGS.items():
            self.cards[idx] = Card(
                index=idx,
                name=name,
                embedding=embedding,
                radius=radius
            )

    def get_card(self, index: int, reversed: bool = False) -> Optional[Card]:
        """Get card by index. Reversal inverts Y-axis (consciousness)."""
        if index not in self.cards:
            return None

        card = self.cards[index]
        if reversed:
            # Reversed cards flip consciousness axis (ego ↔ shadow)
            embedding = card.embedding.copy()
            embedding[1] = -embedding[1]
            return Card(
                index=card.index,
                name=f"{card.name} (Reversed)",
                embedding=embedding,
                radius=card.radius,
                reversed=True
            )
        return card

    def euclidean_distance(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute Euclidean distance between two vectors."""
        return np.linalg.norm(vec1 - vec2)

    def semantic_similarity(self, card1: Card, card2: Card) -> float:
        """
        Compute semantic similarity (0-1 scale).
        Uses inverse distance with radius-adjusted overlap.
        """
        distance = self.euclidean_distance(card1.embedding, card2.embedding)
        combined_radius = card1.radius + card2.radius

        # If within combined radii, they overlap
        if distance < combined_radius:
            # Similarity decreases with distance, max at 1.0 when overlapping
            overlap = 1.0 - (distance / combined_radius)
            return overlap
        else:
            # Beyond overlap zone, diminishing similarity
            return 1.0 / (1.0 + (distance - combined_radius))

    def compute_overlap_strength(self, cards: List[Card]) -> np.ndarray:
        """
        Compute pairwise overlap matrix for a set of cards.
        Returns NxN matrix where entry [i,j] = overlap strength.
        """
        n = len(cards)
        overlaps = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                if i == j:
                    overlaps[i, j] = 1.0  # Card fully overlaps with itself
                else:
                    overlaps[i, j] = self.semantic_similarity(cards[i], cards[j])

        return overlaps

    def get_centroid(self, cards: List[Card]) -> np.ndarray:
        """
        Compute geometric centroid of multiple cards.
        Represents the "center of gravity" of the reading.
        """
        embeddings = np.array([c.embedding for c in cards])
        return np.mean(embeddings, axis=0)

    def weighted_centroid(self, cards: List[Card], weights: np.ndarray) -> np.ndarray:
        """
        Compute weighted centroid (e.g., weighted by MCQ resonance).
        """
        embeddings = np.array([c.embedding for c in cards])
        return np.average(embeddings, axis=0, weights=weights)


class UserProfile:
    """User profile mapped to 3D semantic query vector."""

    def __init__(self, mbti: str = "XXXX",
                 shadow_integration: float = 0.0,
                 temporal_focus: float = 0.0):
        """
        Args:
            mbti: 4-letter MBTI type (e.g., "INTJ")
            shadow_integration: -1 (avoiding) to +1 (integrating)
            temporal_focus: -1 (past) to +1 (future)
        """
        self.mbti = mbti
        self.shadow_integration = shadow_integration
        self.temporal_focus = temporal_focus
        self.vector = self._compute_vector()

    def _compute_vector(self) -> np.ndarray:
        """Convert user profile to 3D query vector."""
        # X-axis (elemental): Thinking-Feeling + Sensing-Intuition
        # T/F: Thinking = Air/Fire (+), Feeling = Water/Earth (-)
        # S/N: Sensing = Earth (-), Intuition = Air/Fire (+)
        x_tf = 0.5 if 'T' in self.mbti else -0.5
        x_sn = 0.5 if 'N' in self.mbti else -0.5
        x = (x_tf + x_sn) / 2

        # Y-axis (consciousness): Shadow integration level
        y = self.shadow_integration

        # Z-axis (temporal): Temporal focus
        z = self.temporal_focus

        return np.array([x, y, z])

    def distance_to_card(self, card: Card, space: SemanticSpace) -> float:
        """Compute how close user is to a card's semantic region."""
        return space.euclidean_distance(self.vector, card.embedding)

    def relevance_weights(self, cards: List[Card], space: SemanticSpace) -> np.ndarray:
        """
        Compute relevance weight for each card based on user profile.
        Closer cards = higher weight.
        """
        distances = np.array([self.distance_to_card(c, space) for c in cards])

        # Inverse distance weighting (add small constant to avoid division by zero)
        weights = 1.0 / (distances + 0.1)

        # Normalize to sum to 1
        return weights / weights.sum()


def extract_dominant_themes(cards: List[Card], overlaps: np.ndarray,
                            user_weights: np.ndarray) -> Dict[str, float]:
    """
    Extract dominant themes from geometric configuration.

    Returns:
        Dict mapping theme names to strength scores (0-1).
    """
    themes = {}

    # Compute reading centroid
    centroid = np.mean([c.embedding for c in cards], axis=0)

    # Elemental theme (X-axis)
    if centroid[0] > 0.3:
        themes['active_fire_air'] = min(1.0, centroid[0])
    elif centroid[0] < -0.3:
        themes['receptive_water_earth'] = min(1.0, abs(centroid[0]))
    else:
        themes['elemental_balance'] = 1.0 - abs(centroid[0])

    # Consciousness theme (Y-axis)
    if centroid[1] > 0.3:
        themes['conscious_integration'] = min(1.0, centroid[1])
    elif centroid[1] < -0.3:
        themes['shadow_work'] = min(1.0, abs(centroid[1]))
    else:
        themes['ego_shadow_balance'] = 1.0 - abs(centroid[1])

    # Temporal theme (Z-axis)
    if centroid[2] > 0.3:
        themes['future_oriented'] = min(1.0, centroid[2])
    elif centroid[2] < -0.3:
        themes['past_patterns'] = min(1.0, abs(centroid[2]))
    else:
        themes['present_focus'] = 1.0 - abs(centroid[2])

    # Interaction theme (based on overlap strength)
    avg_overlap = np.mean(overlaps[np.triu_indices_from(overlaps, k=1)])
    if avg_overlap > 0.5:
        themes['high_coherence'] = avg_overlap
    else:
        themes['scattered_energies'] = 1.0 - avg_overlap

    # User alignment theme
    max_weight = np.max(user_weights)
    if max_weight > 0.4:
        themes['high_personal_resonance'] = max_weight
    else:
        themes['external_challenge'] = 1.0 - max_weight

    return themes


if __name__ == "__main__":
    # Test basic functionality
    space = SemanticSpace()

    # Sample reading: The Tower + Death + The Star
    cards = [
        space.get_card(16, reversed=False),  # The Tower
        space.get_card(13, reversed=False),  # Death
        space.get_card(17, reversed=False),  # The Star
    ]

    print("=== Sample 3-Card Reading ===")
    for i, card in enumerate(cards):
        print(f"{i+1}. {card.name}: {card.embedding}")

    # Compute overlaps
    overlaps = space.compute_overlap_strength(cards)
    print("\n=== Pairwise Overlaps ===")
    for i in range(len(cards)):
        for j in range(i+1, len(cards)):
            print(f"{cards[i].name} ↔ {cards[j].name}: {overlaps[i,j]:.3f}")

    # User profile: INTJ, moderate shadow work, future-focused
    user = UserProfile(mbti="INTJ", shadow_integration=0.4, temporal_focus=0.6)
    print(f"\n=== User Profile ===")
    print(f"MBTI: {user.mbti}")
    print(f"Vector: {user.vector}")

    # Compute user relevance
    weights = user.relevance_weights(cards, space)
    print("\n=== User Relevance Weights ===")
    for i, (card, weight) in enumerate(zip(cards, weights)):
        print(f"{card.name}: {weight:.3f}")

    # Extract themes
    themes = extract_dominant_themes(cards, overlaps, weights)
    print("\n=== Dominant Themes ===")
    for theme, strength in sorted(themes.items(), key=lambda x: x[1], reverse=True):
        print(f"{theme}: {strength:.3f}")

    # Centroid
    centroid = space.get_centroid(cards)
    print(f"\n=== Reading Centroid ===")
    print(f"Position: {centroid}")
    print(f"Interpretation: Elemental={centroid[0]:.2f}, Consciousness={centroid[1]:.2f}, Temporal={centroid[2]:.2f}")
