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


# Hand-coded embeddings for 10 representative cards
# Format: [elemental, consciousness, temporal]
# Range: [-1, 1] for each axis
CARD_EMBEDDINGS: Dict[int, Tuple[str, np.ndarray, float]] = {
    # Major Arcana
    0: ("The Fool", np.array([0.1, 0.7, 0.9]), 0.3),     # Air, conscious, future-oriented
    1: ("The Magician", np.array([0.5, 0.8, 0.5]), 0.25), # Fire-Air, conscious, present
    13: ("Death", np.array([0.0, -0.3, 0.0]), 0.4),      # Neutral, shadow, transformative present
    15: ("The Devil", np.array([0.7, -0.8, -0.2]), 0.3), # Fire, deep shadow, past-patterns
    16: ("The Tower", np.array([0.9, -0.6, 0.1]), 0.35), # Fire, shadow, sudden present
    17: ("The Star", np.array([-0.4, 0.6, 0.8]), 0.4),   # Water, conscious-hope, future

    # Minor Arcana samples
    40: ("Ten of Cups", np.array([-0.8, 0.7, 0.3]), 0.25),    # Water, conscious-joy, present-future
    54: ("Three of Swords", np.array([0.2, -0.5, -0.4]), 0.3), # Air-pain, shadow, past-hurt
    64: ("King of Pentacles", np.array([-0.6, 0.5, 0.2]), 0.2), # Earth, conscious-mastery, stable
    27: ("Knight of Wands", np.array([0.9, 0.4, 0.7]), 0.3),   # Fire, action-oriented, future
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
