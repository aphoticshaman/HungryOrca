/**
 * CARD SYNERGY MATRIX
 *
 * Professional tarot readers don't just interpret cards individually—they read the
 * INTERACTIONS and EMERGENT MEANINGS when cards appear together.
 *
 * "The Tower next to The Lovers means sudden upheaval in relationship"
 * "Three Swords cards = communication crisis or mental breakdown"
 * "Death followed by Ace = ending births powerful new beginning"
 *
 * This creates SPECIFICITY that makes readings feel more authentic and insightful.
 */

import { CARD_DATABASE } from '../data/cardDatabase';

/**
 * Analyze synergies between cards in a spread
 * @param {Array} cards - Array of card objects
 * @returns {Array} - Array of synergy insights
 */
export function analyzeCardSynergies(cards) {
  const insights = [];

  // Pair interactions (adjacent cards)
  for (let i = 0; i < cards.length - 1; i++) {
    const pair = analyzePair(cards[i], cards[i + 1], i);
    if (pair) insights.push(pair);
  }

  // Triad patterns (all three for 3-card spreads)
  if (cards.length === 3) {
    const triad = analyzeTriad(cards[0], cards[1], cards[2]);
    if (triad) insights.push(triad);
  }

  // Suit clustering
  const suitCluster = analyzeSuitClusters(cards);
  if (suitCluster) insights.push(suitCluster);

  // Elemental dominance or conflict
  const elementalPattern = analyzeElementalFlow(cards);
  if (elementalPattern) insights.push(elementalPattern);

  // Archetype echoes (Major Arcana patterns)
  const archetypePattern = analyzeArchetypePatterns(cards);
  if (archetypePattern) insights.push(archetypePattern);

  // Number patterns
  const numberPattern = analyzeNumerology(cards);
  if (numberPattern) insights.push(numberPattern);

  return insights;
}

/**
 * Analyze pair of adjacent cards
 */
function analyzePair(card1, card2, position) {
  const data1 = CARD_DATABASE[card1.cardIndex];
  const data2 = CARD_DATABASE[card2.cardIndex];

  const name1 = data1.name;
  const name2 = data2.name;

  // HIGH-IMPACT MAJOR ARCANA PAIRS
  const majorPairs = {
    // Death + transformation cards
    'Death_The Tower': 'DOUBLE ENDING: Complete ego death and structural collapse happening simultaneously. This isn\'t renovation—this is demolition down to the foundation. Terrifying. Necessary. Liberating.',
    'Death_The Fool': 'The phoenix moment. Death of old self, immediate rebirth into beginner\'s mind. You\'re about to start completely fresh in a way you can\'t yet imagine.',
    'Death_Judgement': 'Resurrection energy. What you thought was dead is coming back transformed. Past life karma resolving. Ancestral patterns breaking.',

    // Love cards together
    'The Lovers_Two of Cups': 'SOULMATE INDICATOR: This is beyond normal attraction. Twin flame, karmic partner, or deeply fated connection manifesting NOW.',
    'The Lovers_The Devil': 'Toxic attachment disguised as love. The chemistry is real, but so is the codependency. Passion vs. prison—know the difference.',
    'The Lovers_The Empress': 'Fertile love. This relationship creates something—a child, a business, art, a home. Union bears fruit.',

    // Crisis combinations
    'The Tower_The Devil': 'The addiction/toxic pattern is being FORCIBLY removed. You don\'t get to hold onto this anymore. The universe is intervening.',
    'The Tower_Five of Pentacles': 'Material/financial crisis. Sudden loss. But look—the church window is lit. Help exists if you ask.',
    'The Tower_Ten of Swords': 'Rock bottom has a basement. This is the full collapse before the rebuild. It gets worse before it gets better.',

    // Success combinations
    'The Sun_The World': 'MASSIVE SUCCESS. Everything coming together. Joy + completion. This is your victory lap.',
    'The Sun_Ace of Pentacles': 'Golden opportunity manifesting. Money, job, physical abundance arriving with joy.',
    'The Star_The Sun': 'From hope to manifestation. What you wished for under the stars is now blazing in daylight.',

    // Spiritual awakening
    'The High Priestess_The Hermit': 'Deep mystical initiation. Powerful downloads coming through solitude and inner work. Trust your inner knowing.',
    'The High Priestess_The Moon': 'Psychic activation. Dreams become prophetic. Veil is THIN. You\'re accessing other realms.',

    // Justice + consequences
    'Justice_Eight of Swords': 'Legal troubles or feeling trapped by "fairness." The rules that bind you may not be as fixed as you think.',
    'Justice_Two of Swords': 'Decision time. The scales must tip. Avoidance is no longer possible. Choose.',

    // Power dynamics
    'The Emperor_The Empress': 'Divine masculine + feminine in balance. Building together. Structure meets nurture. Power couple energy.',
    'The Emperor_Five of Swords': 'Tyranny. Authoritarian abuse. Control freak behavior. Winning at all costs creates hollow victory.',

    // Fortune + fate
    'Wheel of Fortune_The Hanged Man': 'Forced surrender to fate. Karmic timeout. You\'re suspended while the wheel turns.',
    'Wheel of Fortune_Ace of Pentacles': 'Luck is turning in your favor materially. Money/opportunity arrives through "random" chance.'
  };

  // Check both orderings
  const key1 = `${name1}_${name2}`;
  const key2 = `${name2}_${name1}`;

  if (majorPairs[key1]) {
    return {
      type: 'major_pair',
      cards: [name1, name2],
      position: position,
      insight: `${name1} → ${name2}: ${majorPairs[key1]}`
    };
  }
  if (majorPairs[key2]) {
    return {
      type: 'major_pair',
      cards: [name2, name1],
      position: position,
      insight: `${name2} → ${name1}: ${majorPairs[key2]}`
    };
  }

  // Suit-based pairs (elemental flow)
  const element1 = data1.element;
  const element2 = data2.element;

  if (element1 && element2 && element1 !== element2) {
    const elementalPairs = {
      'Fire_Water': 'Fire and water: passion meets emotion. Either steam (transformation) or the fire gets put out. Integration is key.',
      'Fire_Air': 'Fire and air: ideas fuel action. Intellectual energy becomes tangible movement. Perfect for manifestation.',
      'Fire_Earth': 'Fire and earth: willpower meets practicality. Dreams become plans become reality. Grounded action.',
      'Water_Air': 'Water and air: emotion meets thought. Feelings need articulation. Intuition needs logic. Balance head and heart.',
      'Water_Earth': 'Water and earth: emotion becomes tangible. Feelings create form. Nurturing produces results.',
      'Air_Earth': 'Air and earth: theory meets practice. Thoughts become things. Strategy executed.'
    };

    const elemKey1 = `${element1}_${element2}`;
    const elemKey2 = `${element2}_${element1}`;

    if (elementalPairs[elemKey1]) {
      return {
        type: 'elemental_pair',
        cards: [name1, name2],
        insight: elementalPairs[elemKey1]
      };
    }
    if (elementalPairs[elemKey2]) {
      return {
        type: 'elemental_pair',
        cards: [name1, name2],
        insight: elementalPairs[elemKey2]
      };
    }
  }

  // Number echoes (same number, different suits)
  const num1 = getCardNumber(card1.cardIndex);
  const num2 = getCardNumber(card2.cardIndex);

  if (num1 === num2 && num1 <= 10) {
    return {
      type: 'number_echo',
      number: num1,
      cards: [name1, name2],
      insight: `The number ${num1} echoing: ${getNumerologyMeaning(num1)} This theme is AMPLIFIED across multiple life areas.`
    };
  }

  return null;
}

/**
 * Analyze three-card narrative arc
 */
function analyzeTriad(card1, card2, card3) {
  const data1 = CARD_DATABASE[card1.cardIndex];
  const data2 = CARD_DATABASE[card2.cardIndex];
  const data3 = CARD_DATABASE[card3.cardIndex];

  // Story arcs: Past → Present → Future tells a STORY
  const archetypes = [data1, data2, data3].map(d => d.archetype || d.name);

  // Check for "journey" patterns
  if (data1.name.includes('Fool') && data3.name.includes('World')) {
    return {
      type: 'journey_complete',
      insight: 'The Fool\'s Journey from start to finish: You\'re completing a major life cycle. Beginner to Master.'
    };
  }

  // Problem → Obstacle → Resolution patterns
  const hasCrisisCard = [data1, data2, data3].some(d =>
    ['The Tower', 'Ten of Swords', 'Five of Pentacles', 'Three of Swords'].includes(d.name)
  );

  const hasHopeCard = [data1, data2, data3].some(d =>
    ['The Star', 'The Sun', 'Ace of Pentacles', 'Ace of Wands', 'Six of Wands'].includes(d.name)
  );

  if (hasCrisisCard && hasHopeCard) {
    return {
      type: 'crisis_to_hope',
      insight: 'Classic transformation arc: Crisis → Processing → Hope. You\'re moving from breakdown to breakthrough.'
    };
  }

  // All court cards = people-focused reading
  const allCourt = [card1, card2, card3].every(c => isCourtCard(c.cardIndex));
  if (allCourt) {
    return {
      type: 'people_drama',
      insight: 'Three court cards: This is about PEOPLE and relationships. Who\'s who? What roles are they playing in your story?'
    };
  }

  // All major arcana = FATED events
  const allMajor = [card1, card2, card3].every(c => c.cardIndex <= 21);
  if (allMajor) {
    return {
      type: 'fated_path',
      insight: 'Three Major Arcana: This is KARMIC, FATED, DESTINED. Free will exists within constraints. Big soul lessons unfolding.'
    };
  }

  return null;
}

/**
 * Analyze suit clustering
 */
function analyzeSuitClusters(cards) {
  const suits = { Wands: 0, Cups: 0, Swords: 0, Pentacles: 0 };

  cards.forEach(card => {
    const data = CARD_DATABASE[card.cardIndex];
    if (data.suit) {
      suits[data.suit]++;
    }
  });

  for (const [suit, count] of Object.entries(suits)) {
    if (count >= 2) {
      const meanings = {
        Wands: `${count} Wands: FIRE OVERLOAD. Passion, action, willpower dominating. You're in execution mode. But are you burning out?`,
        Cups: `${count} Cups: EMOTIONAL FLOOD. Feelings, relationships, intuition taking center stage. Logic takes backseat. How's that serving you?`,
        Swords: `${count} Swords: MENTAL WARFARE. Overthinking, anxiety, communication issues, or mental clarity cutting through fog. Double-edged.`,
        Pentacles: `${count} Pentacles: MATERIAL FOCUS. Money, body, resources, practical reality demands attention. Get grounded.`
      };

      return {
        type: 'suit_cluster',
        suit: suit,
        count: count,
        insight: meanings[suit]
      };
    }
  }

  return null;
}

/**
 * Analyze elemental flow (balance or conflict)
 */
function analyzeElementalFlow(cards) {
  const elements = cards.map(c => CARD_DATABASE[c.cardIndex].element).filter(Boolean);

  const counts = { Fire: 0, Water: 0, Air: 0, Earth: 0 };
  elements.forEach(e => counts[e]++);

  // Elemental imbalance
  const maxElement = Object.entries(counts).sort((a, b) => b[1] - a[1])[0];

  if (maxElement[1] >= 2) {
    const meanings = {
      Fire: 'Fire dominates: Yang, active, passionate, potentially aggressive. Where\'s the water (emotion) or earth (grounding)?',
      Water: 'Water dominates: Yin, passive, emotional, intuitive. Are you drowning? Where\'s the air (clarity) or fire (action)?',
      Air: 'Air dominates: Mental, communicative, detached. Are you in your head? Where\'s the earth (body) or water (feeling)?',
      Earth: 'Earth dominates: Practical, material, slow. Are you stuck? Where\'s the fire (passion) or air (new ideas)?'
    };

    return {
      type: 'elemental_imbalance',
      element: maxElement[0],
      insight: meanings[maxElement[0]]
    };
  }

  return null;
}

/**
 * Analyze Major Arcana patterns
 */
function analyzeArchetypePatterns(cards) {
  const majors = cards.filter(c => c.cardIndex <= 21);

  if (majors.length === 0) {
    return {
      type: 'no_majors',
      insight: 'No Major Arcana: This is mundane, everyday energy. Personal will has more influence than fate here.'
    };
  }

  // Map majors to Hero's Journey stages
  const stages = majors.map(c => {
    const idx = c.cardIndex;
    if (idx <= 7) return 'initiation';
    if (idx <= 14) return 'trials';
    return 'return';
  });

  const initiation = stages.filter(s => s === 'initiation').length;
  const trials = stages.filter(s => s === 'trials').length;
  const returnPhase = stages.filter(s => s === 'return').length;

  if (trials >= 2) {
    return {
      type: 'hero_trials',
      insight: 'You\'re in the TRIALS phase of the Hero\'s Journey. Tests, challenges, crossing thresholds. This is the hard middle.'
    };
  }

  return null;
}

/**
 * Analyze numerological patterns
 */
function analyzeNumerology(cards) {
  const numbers = cards.map(c => getCardNumber(c.cardIndex)).filter(n => n <= 10);

  // Check for sequences (1,2,3 or 7,8,9)
  const sorted = [...numbers].sort((a, b) => a - b);
  let isSequence = true;
  for (let i = 0; i < sorted.length - 1; i++) {
    if (sorted[i + 1] - sorted[i] !== 1) {
      isSequence = false;
      break;
    }
  }

  if (isSequence && numbers.length >= 2) {
    return {
      type: 'number_sequence',
      sequence: sorted,
      insight: `Number sequence ${sorted.join('→')}: Progressive development. Step-by-step evolution. You're moving through stages in order.`
    };
  }

  return null;
}

/**
 * Helper: Get card number
 */
function getCardNumber(cardIndex) {
  if (cardIndex <= 21) return cardIndex;
  const minorIndex = cardIndex - 22;
  return (minorIndex % 14);
}

/**
 * Helper: Check if court card
 */
function isCourtCard(cardIndex) {
  if (cardIndex <= 21) return false;
  const minorIndex = cardIndex - 22;
  const cardValue = minorIndex % 14;
  return cardValue >= 0 && cardValue <= 3; // Page, Knight, Queen, King
}

/**
 * Helper: Get numerology meaning
 */
function getNumerologyMeaning(num) {
  const meanings = {
    1: 'New beginnings, initiation, unity',
    2: 'Duality, choice, partnership',
    3: 'Creativity, growth, expansion',
    4: 'Stability, structure, foundation',
    5: 'Conflict, change, instability',
    6: 'Harmony, balance, healing',
    7: 'Spirituality, introspection, mystery',
    8: 'Power, manifestation, infinity',
    9: 'Completion, wisdom, endings',
    10: 'Cycles complete, new level unlocked'
  };
  return meanings[num] || '';
}

/**
 * Generate synergy summary for synthesis
 */
export function generateSynergySummary(synergies, quantumSeed) {
  if (synergies.length === 0) return '';

  const intros = [
    'Here\'s what\'s crucial about how these cards interact:',
    'The cards don\'t exist in isolation—watch how they speak to each other:',
    'The real story emerges in the spaces BETWEEN cards:',
    'Card combinations reveal what individual interpretations miss:'
  ];

  const intro = intros[Math.floor(quantumSeed * intros.length)];

  const insights = synergies.map(s => s.insight).join(' ');

  return `\n\n${intro} ${insights}\n\n`;
}
