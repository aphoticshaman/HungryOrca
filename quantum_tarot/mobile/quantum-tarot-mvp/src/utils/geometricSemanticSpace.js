/**
 * Geometric Semantic Space for Tarot
 * ====================================
 *
 * 3D continuous semantic embeddings for enhanced theme extraction.
 * Ported from Python prototype (validated at 75% alignment).
 *
 * X-axis: Elemental polarity (Fire/Air + , Water/Earth -)
 * Y-axis: Consciousness depth (Shadow -, Light +)
 * Z-axis: Temporal focus (Past -, Future +)
 */

// Full 78-card embeddings (ported from Python prototype)
// Format: { cardIndex: { name, position: [x, y, z], radius } }
export const CARD_EMBEDDINGS = {
  // Major Arcana (0-21) - Hand-tuned archetypal positions
  0: { name: "The Fool", position: [0.100, 0.700, 0.900], radius: 0.30 },
  1: { name: "The Magician", position: [0.500, 0.800, 0.500], radius: 0.25 },
  2: { name: "The High Priestess", position: [-0.300, 0.600, -0.200], radius: 0.35 },
  3: { name: "The Empress", position: [-0.700, 0.700, 0.400], radius: 0.40 },
  4: { name: "The Emperor", position: [0.600, 0.500, 0.300], radius: 0.35 },
  5: { name: "The Hierophant", position: [0.200, 0.400, -0.100], radius: 0.30 },
  6: { name: "The Lovers", position: [0.400, 0.500, 0.200], radius: 0.35 },
  7: { name: "The Chariot", position: [0.700, 0.600, 0.500], radius: 0.30 },
  8: { name: "Strength", position: [0.500, 0.700, 0.300], radius: 0.35 },
  9: { name: "The Hermit", position: [0.100, 0.300, -0.300], radius: 0.30 },
  10: { name: "Wheel of Fortune", position: [0.000, 0.000, 0.000], radius: 0.40 },
  11: { name: "Justice", position: [0.300, 0.500, 0.000], radius: 0.30 },
  12: { name: "The Hanged Man", position: [-0.200, -0.200, -0.100], radius: 0.35 },
  13: { name: "Death", position: [0.000, -0.300, 0.000], radius: 0.40 },
  14: { name: "Temperance", position: [-0.100, 0.600, 0.400], radius: 0.35 },
  15: { name: "The Devil", position: [0.700, -0.800, -0.200], radius: 0.30 },
  16: { name: "The Tower", position: [0.900, -0.600, 0.100], radius: 0.35 },
  17: { name: "The Star", position: [-0.400, 0.600, 0.800], radius: 0.40 },
  18: { name: "The Moon", position: [-0.500, -0.400, 0.200], radius: 0.35 },
  19: { name: "The Sun", position: [0.600, 0.800, 0.700], radius: 0.40 },
  20: { name: "Judgement", position: [0.300, 0.500, 0.600], radius: 0.35 },
  21: { name: "The World", position: [0.000, 0.700, 0.500], radius: 0.40 },
  // Wands (22-35) - Fire/Active
  22: { name: "Ace of Wands", position: [0.800, 0.600, -0.400], radius: 0.20 },
  23: { name: "Two of Wands", position: [0.800, 0.400, -0.300], radius: 0.20 },
  24: { name: "Three of Wands", position: [0.800, 0.500, -0.200], radius: 0.20 },
  25: { name: "Four of Wands", position: [0.800, 0.300, -0.100], radius: 0.20 },
  26: { name: "Five of Wands", position: [0.800, -0.200, 0.000], radius: 0.20 },
  27: { name: "Six of Wands", position: [0.800, 0.400, 0.100], radius: 0.20 },
  28: { name: "Seven of Wands", position: [0.800, -0.100, 0.000], radius: 0.20 },
  29: { name: "Eight of Wands", position: [0.800, 0.200, 0.200], radius: 0.20 },
  30: { name: "Nine of Wands", position: [0.800, 0.100, 0.300], radius: 0.20 },
  31: { name: "Ten of Wands", position: [0.800, 0.000, 0.400], radius: 0.25 },
  32: { name: "Page of Wands", position: [0.800, 0.500, 0.100], radius: 0.25 },
  33: { name: "Knight of Wands", position: [0.900, 0.400, 0.500], radius: 0.30 },
  34: { name: "Queen of Wands", position: [0.700, 0.600, 0.200], radius: 0.25 },
  35: { name: "King of Wands", position: [0.900, 0.500, 0.300], radius: 0.25 },
  // Cups (36-49) - Water/Receptive
  36: { name: "Ace of Cups", position: [-0.800, 0.700, -0.400], radius: 0.20 },
  37: { name: "Two of Cups", position: [-0.800, 0.500, -0.300], radius: 0.20 },
  38: { name: "Three of Cups", position: [-0.800, 0.600, -0.200], radius: 0.20 },
  39: { name: "Four of Cups", position: [-0.800, 0.400, -0.100], radius: 0.20 },
  40: { name: "Five of Cups", position: [-0.800, -0.100, 0.000], radius: 0.20 },
  41: { name: "Six of Cups", position: [-0.800, 0.500, 0.100], radius: 0.20 },
  42: { name: "Seven of Cups", position: [-0.800, 0.000, 0.000], radius: 0.20 },
  43: { name: "Eight of Cups", position: [-0.800, 0.300, 0.200], radius: 0.20 },
  44: { name: "Nine of Cups", position: [-0.800, 0.200, 0.300], radius: 0.20 },
  45: { name: "Ten of Cups", position: [-0.800, 0.100, 0.400], radius: 0.25 },
  46: { name: "Page of Cups", position: [-0.800, 0.600, 0.100], radius: 0.25 },
  47: { name: "Knight of Cups", position: [-0.900, 0.500, 0.500], radius: 0.30 },
  48: { name: "Queen of Cups", position: [-0.700, 0.700, 0.200], radius: 0.25 },
  49: { name: "King of Cups", position: [-0.700, 0.600, 0.300], radius: 0.25 },
  // Swords (50-63) - Air/Mental
  50: { name: "Ace of Swords", position: [0.300, 0.600, -0.400], radius: 0.20 },
  51: { name: "Two of Swords", position: [0.300, 0.400, -0.300], radius: 0.20 },
  52: { name: "Three of Swords", position: [0.300, 0.500, -0.200], radius: 0.20 },
  53: { name: "Four of Swords", position: [0.300, 0.300, -0.100], radius: 0.20 },
  54: { name: "Five of Swords", position: [0.300, -0.200, 0.000], radius: 0.20 },
  55: { name: "Six of Swords", position: [0.300, 0.400, 0.100], radius: 0.20 },
  56: { name: "Seven of Swords", position: [0.300, -0.100, 0.000], radius: 0.20 },
  57: { name: "Eight of Swords", position: [0.300, 0.200, 0.200], radius: 0.20 },
  58: { name: "Nine of Swords", position: [0.300, 0.100, 0.300], radius: 0.20 },
  59: { name: "Ten of Swords", position: [0.300, 0.000, 0.400], radius: 0.25 },
  60: { name: "Page of Swords", position: [0.300, 0.500, 0.100], radius: 0.25 },
  61: { name: "Knight of Swords", position: [0.400, 0.400, 0.500], radius: 0.30 },
  62: { name: "Queen of Swords", position: [0.200, 0.600, 0.200], radius: 0.25 },
  63: { name: "King of Swords", position: [0.400, 0.500, 0.300], radius: 0.25 },
  // Pentacles (64-77) - Earth/Material
  64: { name: "Ace of Pentacles", position: [-0.600, 0.500, -0.400], radius: 0.20 },
  65: { name: "Two of Pentacles", position: [-0.600, 0.300, -0.300], radius: 0.20 },
  66: { name: "Three of Pentacles", position: [-0.600, 0.400, -0.200], radius: 0.20 },
  67: { name: "Four of Pentacles", position: [-0.600, 0.200, -0.100], radius: 0.20 },
  68: { name: "Five of Pentacles", position: [-0.600, -0.300, 0.000], radius: 0.20 },
  69: { name: "Six of Pentacles", position: [-0.600, 0.300, 0.100], radius: 0.20 },
  70: { name: "Seven of Pentacles", position: [-0.600, -0.200, 0.000], radius: 0.20 },
  71: { name: "Eight of Pentacles", position: [-0.600, 0.100, 0.200], radius: 0.20 },
  72: { name: "Nine of Pentacles", position: [-0.600, 0.000, 0.300], radius: 0.20 },
  73: { name: "Ten of Pentacles", position: [-0.600, -0.100, 0.400], radius: 0.25 },
  74: { name: "Page of Pentacles", position: [-0.600, 0.400, 0.100], radius: 0.25 },
  75: { name: "Knight of Pentacles", position: [-0.700, 0.300, 0.500], radius: 0.30 },
  76: { name: "Queen of Pentacles", position: [-0.500, 0.500, 0.200], radius: 0.25 },
  77: { name: "King of Pentacles", position: [-0.500, 0.400, 0.300], radius: 0.25 },
};

/**
 * Calculate Euclidean distance between two 3D positions
 */
function euclideanDistance(pos1, pos2) {
  const dx = pos1[0] - pos2[0];
  const dy = pos1[1] - pos2[1];
  const dz = pos1[2] - pos2[2];
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

/**
 * Calculate semantic overlap between two cards
 * Returns 0-1 score where 1 = perfect overlap, 0 = no overlap
 */
export function calculateOverlap(card1Idx, card2Idx) {
  const emb1 = CARD_EMBEDDINGS[card1Idx];
  const emb2 = CARD_EMBEDDINGS[card2Idx];

  if (!emb1 || !emb2) return 0;

  const distance = euclideanDistance(emb1.position, emb2.position);
  const combinedRadius = emb1.radius + emb2.radius;

  // If within combined radii, they overlap
  if (distance < combinedRadius) {
    return 1.0 - (distance / combinedRadius);
  }

  // Beyond overlap zone, diminishing similarity
  return 1.0 / (1.0 + (distance - combinedRadius));
}

/**
 * Calculate geometric centroid of cards in reading
 * Returns [x, y, z] position representing reading's center of gravity
 */
export function calculateCentroid(cardIndices) {
  if (!cardIndices || cardIndices.length === 0) return [0, 0, 0];

  let sumX = 0, sumY = 0, sumZ = 0;
  let count = 0;

  cardIndices.forEach(idx => {
    const emb = CARD_EMBEDDINGS[idx];
    if (emb) {
      sumX += emb.position[0];
      sumY += emb.position[1];
      sumZ += emb.position[2];
      count++;
    }
  });

  if (count === 0) return [0, 0, 0];

  return [sumX / count, sumY / count, sumZ / count];
}

/**
 * Extract geometric themes from card positions
 * Multi-scale analysis: Micro (individual) + Meso (pairs) + Macro (overall)
 */
export function extractGeometricThemes(cards) {
  if (!cards || cards.length === 0) {
    return { themes: [], centroid: [0, 0, 0], avgOverlap: 0 };
  }

  const cardIndices = cards.map(c => c.cardIndex);
  const themes = [];

  // MACRO: Centroid analysis (overall reading energy)
  const centroid = calculateCentroid(cardIndices);
  const [x, y, z] = centroid;

  // Elemental themes (X-axis)
  if (x > 0.5) {
    themes.push({ type: 'elemental', theme: 'active_fire_air', strength: Math.min(1.0, x) });
  } else if (x < -0.5) {
    themes.push({ type: 'elemental', theme: 'receptive_water_earth', strength: Math.min(1.0, Math.abs(x)) });
  } else {
    themes.push({ type: 'elemental', theme: 'elemental_balance', strength: 1.0 - Math.abs(x) });
  }

  // Consciousness themes (Y-axis)
  if (y > 0.5) {
    themes.push({ type: 'consciousness', theme: 'conscious_integration', strength: Math.min(1.0, y) });
  } else if (y < -0.5) {
    themes.push({ type: 'consciousness', theme: 'shadow_work_needed', strength: Math.min(1.0, Math.abs(y)) });
  } else {
    themes.push({ type: 'consciousness', theme: 'ego_shadow_balance', strength: 1.0 - Math.abs(y) });
  }

  // Temporal themes (Z-axis)
  if (z > 0.4) {
    themes.push({ type: 'temporal', theme: 'future_focused', strength: Math.min(1.0, z) });
  } else if (z < -0.4) {
    themes.push({ type: 'temporal', theme: 'past_processing', strength: Math.min(1.0, Math.abs(z)) });
  } else {
    themes.push({ type: 'temporal', theme: 'present_centered', strength: 1.0 - Math.abs(z) });
  }

  // MESO: Pairwise overlaps (card interaction patterns)
  let totalOverlap = 0;
  let pairCount = 0;

  for (let i = 0; i < cardIndices.length; i++) {
    for (let j = i + 1; j < cardIndices.length; j++) {
      const overlap = calculateOverlap(cardIndices[i], cardIndices[j]);
      totalOverlap += overlap;
      pairCount++;

      // Detect specific powerful combinations
      const pair = new Set([cardIndices[i], cardIndices[j]]);

      if (pair.has(16) && pair.has(13)) { // Tower + Death
        themes.push({ type: 'interaction', theme: 'compound_transformation', strength: overlap });
      }
      if (pair.has(15) && pair.has(16)) { // Devil + Tower
        themes.push({ type: 'interaction', theme: 'shadow_breakthrough', strength: overlap });
      }
      if (pair.has(13) && pair.has(17)) { // Death + Star
        themes.push({ type: 'interaction', theme: 'phoenix_rebirth', strength: overlap });
      }
    }
  }

  const avgOverlap = pairCount > 0 ? totalOverlap / pairCount : 0;

  // Coherence theme based on average overlap
  if (avgOverlap > 0.6) {
    themes.push({ type: 'coherence', theme: 'high_synergy', strength: avgOverlap });
  } else if (avgOverlap < 0.45) {
    themes.push({ type: 'coherence', theme: 'scattered_energies', strength: 1.0 - avgOverlap });
  }

  return {
    themes,
    centroid,
    avgOverlap
  };
}

/**
 * Generate natural language description of geometric themes
 * Integrates with existing synthesis narratives
 */
export function describeGeometricThemes(geometricAnalysis) {
  if (!geometricAnalysis || !geometricAnalysis.themes || geometricAnalysis.themes.length === 0) {
    return null;
  }

  const { themes, centroid, avgOverlap } = geometricAnalysis;
  const descriptions = [];

  // Find strongest theme in each category
  const elementalTheme = themes.find(t => t.type === 'elemental');
  const consciousnessTheme = themes.find(t => t.type === 'consciousness');
  const temporalTheme = themes.find(t => t.type === 'temporal');
  const interactionThemes = themes.filter(t => t.type === 'interaction');
  const coherenceTheme = themes.find(t => t.type === 'coherence');

  // Elemental description
  if (elementalTheme) {
    if (elementalTheme.theme === 'active_fire_air') {
      descriptions.push('These cards pulse with active, yang energy—fire and air, action and intellect.');
    } else if (elementalTheme.theme === 'receptive_water_earth') {
      descriptions.push('The energy here is receptive, grounded—water and earth, emotion and matter.');
    } else {
      descriptions.push('Your cards show elemental balance, integrating all four forces.');
    }
  }

  // Consciousness description
  if (consciousnessTheme) {
    if (consciousnessTheme.theme === 'shadow_work_needed') {
      descriptions.push('There\'s shadow work here—unconscious patterns demanding integration.');
    } else if (consciousnessTheme.theme === 'conscious_integration') {
      descriptions.push('You\'re operating from conscious awareness, bringing light to what was hidden.');
    }
  }

  // Interaction descriptions (most powerful)
  if (interactionThemes.length > 0) {
    interactionThemes.forEach(theme => {
      if (theme.theme === 'compound_transformation') {
        descriptions.push('The Tower and Death together signal compound disruption—one ending triggering another. This isn\'t subtle.');
      } else if (theme.theme === 'shadow_breakthrough') {
        descriptions.push('Devil and Tower: shadow patterns suddenly collapsing. Liberation through crisis.');
      } else if (theme.theme === 'phoenix_rebirth') {
        descriptions.push('Death transforming into the Star—classic phoenix energy. What burns away makes room for hope.');
      }
    });
  }

  // Coherence description
  if (coherenceTheme) {
    if (coherenceTheme.theme === 'high_synergy') {
      descriptions.push('These cards work together seamlessly—a clear, coherent message.');
    } else if (coherenceTheme.theme === 'scattered_energies') {
      descriptions.push('The cards are pulling in different directions. You\'re navigating complexity, not a single path.');
    }
  }

  // Combine into single paragraph
  if (descriptions.length > 0) {
    return descriptions.join(' ');
  }

  return null;
}
