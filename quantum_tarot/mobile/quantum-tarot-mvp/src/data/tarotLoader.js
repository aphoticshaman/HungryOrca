/**
 * Tarot Card Loader
 * Helper functions to access card data
 */

import tarotData from './tarotCards.json' with { type: 'json' };

/**
 * Complete deck array (all 78 cards)
 * NOTE: This sample has just the aces + major arcana examples
 * In production, tarotCards.json will have all 78 cards
 */
export const COMPLETE_DECK = [
  ...tarotData.majorArcana,
  ...tarotData.wands,
  ...tarotData.cups,
  ...tarotData.swords,
  ...tarotData.pentacles
];

/**
 * Get card by index (0-77)
 */
export function getCardByIndex(index) {
  try {
    // Validate index
    if (typeof index !== 'number' || isNaN(index)) {
      throw new Error(`Invalid card index type: ${index}`);
    }

    // Bounds check
    if (index < 0 || index >= COMPLETE_DECK.length) {
      console.warn(`Card index ${index} out of bounds (0-${COMPLETE_DECK.length - 1}), using fallback`);
      // Fallback to a valid index (wrap around)
      index = Math.abs(index) % COMPLETE_DECK.length;
    }

    const card = COMPLETE_DECK[index];
    if (!card) {
      throw new Error(`Card at index ${index} is null/undefined`);
    }

    return card;
  } catch (error) {
    console.error('Failed to get card by index:', error);
    // Ultimate fallback: return The Fool (index 0)
    return COMPLETE_DECK[0] || {
      name: 'The Fool',
      number: 0,
      suit: 'major_arcana',
      uprightMeaning: 'New beginnings, innocence, spontaneity',
      reversedMeaning: 'Recklessness, fearlessness, risk',
      uprightKeywords: ['beginnings', 'innocence', 'spontaneity', 'free spirit'],
      reversedKeywords: ['recklessness', 'risk-taking', 'inconsideration']
    };
  }
}

/**
 * Get card by name
 */
export function getCardByName(name) {
  try {
    if (!name || typeof name !== 'string') {
      throw new Error('Invalid card name');
    }

    const card = COMPLETE_DECK.find(card =>
      card && card.name && card.name.toLowerCase() === name.toLowerCase()
    );

    return card || null;
  } catch (error) {
    console.error('Failed to get card by name:', error);
    return null;
  }
}

/**
 * Get all cards of a specific suit
 */
export function getCardsBySuit(suit) {
  return COMPLETE_DECK.filter(card => card.suit === suit);
}

/**
 * Get interpretation for card based on reading type
 */
export function getInterpretation(card, readingType, isReversed = false) {
  try {
    // Validate inputs
    if (!card || typeof card !== 'object') {
      throw new Error('Invalid card object');
    }
    if (!readingType || typeof readingType !== 'string') {
      throw new Error('Invalid reading type');
    }

    // If reversed and not a specific reading type, use reversed meaning
    if (isReversed && readingType === 'general') {
      return card.reversedMeaning || card.uprightMeaning || 'No interpretation available';
    }

    // Get specific interpretation
    let interpretation;
    switch (readingType) {
      case 'career':
        interpretation = card.careerInterpretation;
        break;
      case 'romance':
        interpretation = card.romanceInterpretation;
        break;
      case 'wellness':
        interpretation = card.wellnessInterpretation;
        break;
      case 'family':
        interpretation = card.familyInterpretation;
        break;
      case 'self_growth':
        interpretation = card.selfGrowthInterpretation;
        break;
      case 'school':
        interpretation = card.schoolInterpretation;
        break;
      default:
        interpretation = isReversed ? card.reversedMeaning : card.uprightMeaning;
    }

    // Fallback if specific interpretation missing
    return interpretation || card.uprightMeaning || 'No interpretation available';
  } catch (error) {
    console.error('Failed to get interpretation:', error);
    return 'No interpretation available';
  }
}

/**
 * Get keywords for card
 */
export function getKeywords(card, isReversed = false) {
  try {
    if (!card || typeof card !== 'object') {
      throw new Error('Invalid card object');
    }

    const keywords = isReversed ? card.reversedKeywords : card.uprightKeywords;

    // Validate keywords is an array
    if (!keywords || !Array.isArray(keywords)) {
      console.warn('Missing or invalid keywords, using fallback');
      return ['mystery', 'potential', 'unknown'];
    }

    return keywords;
  } catch (error) {
    console.error('Failed to get keywords:', error);
    return ['mystery', 'potential', 'unknown'];
  }
}

/**
 * Deck statistics
 */
export const DECK_INFO = {
  totalCards: 78, // Will be actual count when full deck loaded
  majorArcana: 22,
  minorArcana: 56,
  suits: ['wands', 'cups', 'swords', 'pentacles'],
  elements: {
    wands: 'fire',
    cups: 'water',
    swords: 'air',
    pentacles: 'earth',
    major_arcana: 'spirit'
  }
};

/**
 * Get card image path
 * Cards should be stored in assets/cards/{suit}/{number}_{name}.jpg
 */
export function getCardImagePath(card, aesthetic = 'soft_mystical') {
  const fileName = `${String(card.number).padStart(2, '0')}_${card.name.toLowerCase().replace(/\s+/g, '_')}`;
  return `assets/cards/${aesthetic}/${card.suit}/${fileName}.jpg`;
}

// Export everything
export default {
  COMPLETE_DECK,
  DECK_INFO,
  getCardByIndex,
  getCardByName,
  getCardsBySuit,
  getInterpretation,
  getKeywords,
  getCardImagePath
};
