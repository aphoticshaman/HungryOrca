"""
Algorithmic 78-Card Embedding Generator
========================================

Derives 3D semantic embeddings from cardDatabase.js metadata.

Embedding Logic:
- X-axis (Elemental): Fire/Air = +, Water/Earth = -, Spirit = 0
- Y-axis (Consciousness): Shadow vs Light, Jungian archetype depth
- Z-axis (Temporal): Numerology progression, rank evolution

Derivation rules:
1. Major Arcana: Hand-tuned archetypal positions (0-21)
2. Minor Arcana: Algorithmic from suit + rank + numerology
"""

import json
import re
from typing import Dict, Tuple
import numpy as np


# Elemental mappings (X-axis: -1 to +1)
ELEMENT_MAP = {
    'fire': 0.8,      # Active, masculine, yang
    'air': 0.5,       # Active, mental
    'water': -0.7,    # Receptive, emotional
    'earth': -0.6,    # Receptive, material
    'spirit': 0.0,    # Transcendent, balanced
}

# Suit elemental associations
SUIT_ELEMENTS = {
    'wands': 'fire',
    'cups': 'water',
    'swords': 'air',
    'pentacles': 'earth',
}

# Consciousness depth mappings (Y-axis: -1 to +1)
# Shadow = negative, Light = positive
ARCHETYPE_CONSCIOUSNESS = {
    # Major Arcana archetypes (from jungian field)
    'puer_aeternus': 0.7,        # The Fool - innocent consciousness
    'magician_archetype': 0.8,   # The Magician - mastery
    'high_priestess': 0.6,       # Intuitive consciousness
    'empress': 0.7,              # Nurturing consciousness
    'emperor': 0.5,              # Structured ego
    'hierophant': 0.4,           # Traditional consciousness
    'lovers': 0.5,               # Choice/duality
    'chariot': 0.6,              # Willpower
    'strength': 0.7,             # Integrated power
    'hermit': 0.3,               # Introspective (shadow edge)
    'wheel_of_fortune': 0.0,     # Neutral/fate
    'justice': 0.5,              # Balanced judgment
    'hanged_man': -0.2,          # Surrender (shadow work)
    'death': -0.3,               # Transformation (shadow)
    'temperance': 0.6,           # Integration
    'devil': -0.8,               # Deep shadow
    'tower': -0.6,               # Destructive shadow
    'star': 0.6,                 # Hope/healing
    'moon': -0.4,                # Illusion/subconscious
    'sun': 0.8,                  # Pure light
    'judgement': 0.5,            # Awakening
    'world': 0.7,                # Completion/integration
}

# Rank consciousness progression (for Minor Arcana)
RANK_CONSCIOUSNESS = {
    'ace': 0.6,      # Pure potential (conscious gift)
    '2': 0.4,        # Balance/duality
    '3': 0.5,        # Creation/growth
    '4': 0.3,        # Stability (grounded, less transcendent)
    '5': -0.2,       # Conflict/challenge (shadow edge)
    '6': 0.4,        # Harmony restored
    '7': -0.1,       # Testing/introspection (shadow work)
    '8': 0.2,        # Mastery through effort
    '9': 0.1,        # Refinement/near completion
    '10': 0.0,       # Completion/burden (neutral - can be overwhelming)
    'page': 0.5,     # Youthful consciousness
    'knight': 0.4,   # Active pursuit
    'queen': 0.6,    # Mature receptive
    'king': 0.5,     # Mature active
}

# Temporal progression (Z-axis: -1 = past, 0 = present, +1 = future)
def calculate_temporal(numerology: int, rank: str, arcana: str) -> float:
    """
    Calculate temporal position based on numerology and rank.

    Logic:
    - Low numbers (Ace-3) = foundations/past (-0.4 to -0.2)
    - Mid numbers (4-7) = present challenges (-0.1 to 0.1)
    - High numbers (8-10) = future/completion (0.2 to 0.4)
    - Court cards = active present to future (0.1 to 0.5)
    - Major Arcana = archetypal time (varies by card)
    """
    if arcana == 'major':
        # Major Arcana: Map 0-21 to temporal arc
        # 0-7: Past/foundation (-0.3 to 0)
        # 8-14: Present transformation (0 to 0.2)
        # 15-21: Future/completion (0.2 to 0.8)
        if numerology <= 7:
            return -0.3 + (numerology / 7) * 0.3  # -0.3 to 0
        elif numerology <= 14:
            return 0.0 + ((numerology - 8) / 6) * 0.2  # 0 to 0.2
        else:
            return 0.2 + ((numerology - 15) / 6) * 0.6  # 0.2 to 0.8

    # Minor Arcana temporal mapping
    RANK_TEMPORAL = {
        'ace': -0.4,    # Seeds/beginnings (past potential)
        '2': -0.3,
        '3': -0.2,
        '4': -0.1,      # Stability (present-leaning)
        '5': 0.0,       # Conflict (present)
        '6': 0.1,
        '7': 0.0,       # Testing (present)
        '8': 0.2,
        '9': 0.3,
        '10': 0.4,      # Completion (future arrived)
        'page': 0.1,    # Student energy (near-present)
        'knight': 0.5,  # Active pursuit (future-oriented)
        'queen': 0.2,   # Mastery (present-stable)
        'king': 0.3,    # Authority (present-established)
    }

    return RANK_TEMPORAL.get(rank, 0.0)


def generate_embedding(card: Dict) -> Tuple[float, float, float]:
    """
    Generate 3D embedding from card metadata.

    Returns:
        (x, y, z) where:
        - x: Elemental polarity (-1 to +1)
        - y: Consciousness depth (-1 to +1)
        - z: Temporal focus (-1 to +1)
    """
    # X-axis: Elemental
    if card.get('element'):
        x = ELEMENT_MAP.get(card['element'], 0.0)
    elif card.get('suit'):
        element = SUIT_ELEMENTS.get(card['suit'], 'air')
        x = ELEMENT_MAP[element]
    else:
        x = 0.0  # Default for cards without element

    # Y-axis: Consciousness
    if card['arcana'] == 'major':
        # Try jungian archetype first
        jungian = card.get('jungian', '').lower().replace(' ', '_')
        y = ARCHETYPE_CONSCIOUSNESS.get(jungian, 0.0)
    else:
        # Minor Arcana: Use rank consciousness
        rank = card.get('rank', '').lower()
        y = RANK_CONSCIOUSNESS.get(rank, 0.0)

        # Adjust based on suit (Cups/Pentacles more grounded)
        if card.get('suit') == 'cups':
            y += 0.1  # Emotional awareness boost
        elif card.get('suit') == 'pentacles':
            y -= 0.1  # Material grounding (less transcendent)

    # Z-axis: Temporal
    z = calculate_temporal(
        card.get('numerology', 0),
        card.get('rank', '').lower(),
        card['arcana']
    )

    # Clamp to [-1, 1] range
    x = max(-1.0, min(1.0, x))
    y = max(-1.0, min(1.0, y))
    z = max(-1.0, min(1.0, z))

    return (x, y, z)


def generate_all_embeddings_from_json(json_path: str) -> Dict[int, Tuple[str, Tuple[float, float, float], float]]:
    """
    Generate embeddings for all 78 cards from cardDatabase JSON.

    Returns:
        Dict mapping card_id to (name, embedding, radius)
    """
    # Read cardDatabase
    with open(json_path, 'r') as f:
        content = f.read()

    # Extract CARD_DATABASE array (it's exported JS, not pure JSON)
    # Find the array between [ and ]
    match = re.search(r'export const CARD_DATABASE = (\[[\s\S]*?\]);', content)
    if not match:
        raise ValueError("Could not find CARD_DATABASE in file")

    # Parse as JSON (after removing trailing semicolon)
    cards_json = match.group(1)
    cards = json.loads(cards_json)

    embeddings = {}

    for card in cards:
        card_id = card['id']
        name = card['name']
        embedding = generate_embedding(card)

        # Radius based on card significance
        if card['arcana'] == 'major':
            radius = 0.35  # Major Arcana have larger influence
        else:
            # Court cards larger than pips
            rank = card.get('rank', '').lower()
            if rank in ['page', 'knight', 'queen', 'king']:
                radius = 0.25
            else:
                radius = 0.20

        embeddings[card_id] = (name, embedding, radius)

    return embeddings


def format_as_python_dict(embeddings: Dict) -> str:
    """Format embeddings as Python dict for semantic_space.py"""
    lines = ["CARD_EMBEDDINGS: Dict[int, Tuple[str, np.ndarray, float]] = {"]

    for card_id in sorted(embeddings.keys()):
        name, (x, y, z), radius = embeddings[card_id]
        lines.append(f'    {card_id}: ("{name}", np.array([{x:.3f}, {y:.3f}, {z:.3f}]), {radius:.2f}),')

    lines.append("}")

    return "\n".join(lines)


if __name__ == "__main__":
    import sys

    # Path to cardDatabase.js
    db_path = "/home/user/HungryOrca/quantum_tarot/mobile/quantum-tarot-mvp/src/data/cardDatabase.js"

    print("Generating embeddings from cardDatabase.js...")
    embeddings = generate_all_embeddings_from_json(db_path)

    print(f"\n✓ Generated {len(embeddings)} card embeddings")

    # Show sample
    print("\nSample embeddings:")
    for card_id in [0, 15, 16, 40, 54]:
        if card_id in embeddings:
            name, (x, y, z), r = embeddings[card_id]
            print(f"  {card_id:2d}. {name:25s} → ({x:+.2f}, {y:+.2f}, {z:+.2f}) r={r:.2f}")

    # Write to file
    output_path = "embeddings_78_cards.py"
    with open(output_path, 'w') as f:
        f.write("import numpy as np\n")
        f.write("from typing import Dict, Tuple\n\n")
        f.write("# Auto-generated from cardDatabase.js\n")
        f.write("# X: Elemental (Fire/Air + , Water/Earth -)\n")
        f.write("# Y: Consciousness (Shadow -, Light +)\n")
        f.write("# Z: Temporal (Past -, Future +)\n\n")
        f.write(format_as_python_dict(embeddings))

    print(f"\n✓ Saved to {output_path}")

    # Validation: Compare with hand-coded
    print("\nValidation against hand-coded embeddings:")
    hand_coded = {
        0: (0.1, 0.7, 0.9),   # The Fool
        1: (0.5, 0.8, 0.5),   # The Magician
        15: (0.7, -0.8, -0.2), # The Devil
        16: (0.9, -0.6, 0.1),  # The Tower
    }

    for card_id, (hx, hy, hz) in hand_coded.items():
        name, (gx, gy, gz), _ = embeddings[card_id]
        diff = np.sqrt((hx-gx)**2 + (hy-gy)**2 + (hz-hz)**2)
        match = "✓" if diff < 0.3 else "⚠"
        print(f"  {match} {name:20s} hand=({hx:+.1f},{hy:+.1f},{hz:+.1f}) gen=({gx:+.1f},{gy:+.1f},{gz:+.1f}) Δ={diff:.2f}")
