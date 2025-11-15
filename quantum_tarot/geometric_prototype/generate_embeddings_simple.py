"""
Simple 78-Card Embedding Generator
===================================

Uses tarot structure knowledge instead of parsing cardDatabase.js
"""

import numpy as np
from typing import Dict, Tuple


# Major Arcana (0-21) - Hand-tuned archetypal positions
MAJOR_ARCANA_EMBEDDINGS = {
    0: ("The Fool", (0.1, 0.7, 0.9), 0.30),           # Air, conscious, future
    1: ("The Magician", (0.5, 0.8, 0.5), 0.25),       # Air, mastery, present
    2: ("The High Priestess", (-0.3, 0.6, -0.2), 0.35), # Water, intuition, past-mystery
    3: ("The Empress", (-0.7, 0.7, 0.4), 0.40),       # Earth, nurturing, present-future
    4: ("The Emperor", (0.6, 0.5, 0.3), 0.35),        # Fire, structure, present
    5: ("The Hierophant", (0.2, 0.4, -0.1), 0.30),    # Earth/Spirit, tradition, past
    6: ("The Lovers", (0.4, 0.5, 0.2), 0.35),         # Air, choice, present
    7: ("The Chariot", (0.7, 0.6, 0.5), 0.30),        # Water/Will, triumph, future
    8: ("Strength", (0.5, 0.7, 0.3), 0.35),           # Fire, integration, present
    9: ("The Hermit", (0.1, 0.3, -0.3), 0.30),        # Earth, introspection, past
    10: ("Wheel of Fortune", (0.0, 0.0, 0.0), 0.40),  # All elements, fate, eternal-present
    11: ("Justice", (0.3, 0.5, 0.0), 0.30),           # Air, balance, present
    12: ("The Hanged Man", (-0.2, -0.2, -0.1), 0.35), # Water, surrender, suspended
    13: ("Death", (0.0, -0.3, 0.0), 0.40),            # Water/Spirit, transformation, threshold
    14: ("Temperance", (-0.1, 0.6, 0.4), 0.35),       # Water/Fire, alchemy, integration
    15: ("The Devil", (0.7, -0.8, -0.2), 0.30),       # Earth/Fire, shadow, past-patterns
    16: ("The Tower", (0.9, -0.6, 0.1), 0.35),        # Fire, disruption, sudden-present
    17: ("The Star", (-0.4, 0.6, 0.8), 0.40),         # Water, hope, future
    18: ("The Moon", (-0.5, -0.4, 0.2), 0.35),        # Water, illusion, night-journey
    19: ("The Sun", (0.6, 0.8, 0.7), 0.40),           # Fire, clarity, bright-future
    20: ("Judgement", (0.3, 0.5, 0.6), 0.35),         # Fire/Spirit, awakening, rebirth
    21: ("The World", (0.0, 0.7, 0.5), 0.40),         # All elements, completion, integration
}


def generate_minor_arcana() -> Dict[int, Tuple[str, Tuple[float, float, float], float]]:
    """
    Generate all 56 Minor Arcana cards (22-77) algorithmically.

    Structure:
    - Wands (Fire): 22-35
    - Cups (Water): 36-49
    - Swords (Air): 50-63
    - Pentacles (Earth): 64-77

    Each suit: Ace, 2-10, Page, Knight, Queen, King
    """
    suits = [
        ('Wands', 'fire', 0.8),      # Active, yang
        ('Cups', 'water', -0.8),     # Receptive, emotional
        ('Swords', 'air', 0.3),      # Active, mental
        ('Pentacles', 'earth', -0.6) # Receptive, material
    ]

    ranks = [
        ('Ace', 0.6, -0.4, 0.20),
        ('Two', 0.4, -0.3, 0.20),
        ('Three', 0.5, -0.2, 0.20),
        ('Four', 0.3, -0.1, 0.20),
        ('Five', -0.2, 0.0, 0.20),
        ('Six', 0.4, 0.1, 0.20),
        ('Seven', -0.1, 0.0, 0.20),
        ('Eight', 0.2, 0.2, 0.20),
        ('Nine', 0.1, 0.3, 0.20),
        ('Ten', 0.0, 0.4, 0.25),
        ('Page', 0.5, 0.1, 0.25),
        ('Knight', 0.4, 0.5, 0.30),
        ('Queen', 0.6, 0.2, 0.25),
        ('King', 0.5, 0.3, 0.25),
    ]

    embeddings = {}
    card_id = 22  # Start after Major Arcana

    for suit_name, suit_element, x_base in suits:
        for rank_name, y_val, z_val, radius in ranks:
            # X-axis: Base elemental + slight rank variation
            if rank_name in ['Knight', 'King']:  # More active
                x = x_base + 0.1
            elif rank_name in ['Queen']:  # More receptive
                x = x_base - 0.1
            else:
                x = x_base

            # Y-axis: Rank consciousness
            y = y_val

            # Suit adjustments to Y
            if suit_name == 'Cups':
                y += 0.1  # Emotional awareness boost
            elif suit_name == 'Pentacles':
                y -= 0.1  # Grounded, less transcendent

            # Z-axis: Rank temporal
            z = z_val

            # Clamp
            x = max(-1.0, min(1.0, x))
            y = max(-1.0, min(1.0, y))
            z = max(-1.0, min(1.0, z))

            name = f"{rank_name} of {suit_name}"
            embeddings[card_id] = (name, (x, y, z), radius)
            card_id += 1

    return embeddings


def generate_all_78_cards() -> Dict[int, Tuple[str, Tuple[float, float, float], float]]:
    """Generate complete 78-card deck embeddings."""
    all_cards = {}

    # Add Major Arcana
    all_cards.update(MAJOR_ARCANA_EMBEDDINGS)

    # Add Minor Arcana
    all_cards.update(generate_minor_arcana())

    return all_cards


def format_as_python_dict(embeddings: Dict) -> str:
    """Format embeddings as Python dict for semantic_space.py"""
    lines = ["CARD_EMBEDDINGS: Dict[int, Tuple[str, np.ndarray, float]] = {"]

    for card_id in sorted(embeddings.keys()):
        name, (x, y, z), radius = embeddings[card_id]
        # Escape quotes in name
        name_safe = name.replace('"', '\\"')
        lines.append(f'    {card_id}: ("{name_safe}", np.array([{x:.3f}, {y:.3f}, {z:.3f}]), {radius:.2f}),')

    lines.append("}")

    return "\n".join(lines)


if __name__ == "__main__":
    print("Generating all 78 card embeddings...")
    embeddings = generate_all_78_cards()

    print(f"\n✓ Generated {len(embeddings)} cards")

    # Show samples from each category
    print("\nSamples:")
    samples = [0, 1, 15, 16, 22, 36, 50, 64, 35, 49, 63, 77]
    for card_id in samples:
        if card_id in embeddings:
            name, (x, y, z), r = embeddings[card_id]
            print(f"  {card_id:2d}. {name:25s} → ({x:+.2f}, {y:+.2f}, {z:+.2f}) r={r:.2f}")

    # Write to file
    output_path = "embeddings_78_cards.py"
    with open(output_path, 'w') as f:
        f.write("import numpy as np\n")
        f.write("from typing import Dict, Tuple\n\n")
        f.write("# All 78 Tarot Card Embeddings\n")
        f.write("# X: Elemental (Fire/Air + , Water/Earth -)\n")
        f.write("# Y: Consciousness (Shadow -, Light +)\n")
        f.write("# Z: Temporal (Past -, Future +)\n\n")
        f.write(format_as_python_dict(embeddings))

    print(f"\n✓ Saved to {output_path}")

    # Validation stats
    print("\nDistribution stats:")
    xs = [emb[1][0] for emb in embeddings.values()]
    ys = [emb[1][1] for emb in embeddings.values()]
    zs = [emb[1][2] for emb in embeddings.values()]

    print(f"  X (Elemental):    mean={np.mean(xs):+.2f}, std={np.std(xs):.2f}, range=[{min(xs):+.2f}, {max(xs):+.2f}]")
    print(f"  Y (Consciousness): mean={np.mean(ys):+.2f}, std={np.std(ys):.2f}, range=[{min(ys):+.2f}, {max(ys):+.2f}]")
    print(f"  Z (Temporal):     mean={np.mean(zs):+.2f}, std={np.std(zs):.2f}, range=[{min(zs):+.2f}, {max(zs):+.2f}]")
