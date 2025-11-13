/**
 * Tarot Card Loader
 * Helper functions to access card data
 */

import tarotData from './tarotCards.json';

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
  if (index < 0 || index >= COMPLETE_DECK.length) {
    throw new Error(`Invalid card index: ${index}`);
  }
  return COMPLETE_DECK[index];
}

/**
 * Get card by name
 */
export function getCardByName(name) {
  return COMPLETE_DECK.find(card =>
    card.name.toLowerCase() === name.toLowerCase()
  );
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
  // If reversed and not a specific reading type, use reversed meaning
  if (isReversed && readingType === 'general') {
    return card.reversedMeaning;
  }

  // Get specific interpretation
  switch (readingType) {
    case 'career':
      return card.careerInterpretation;
    case 'romance':
      return card.romanceInterpretation;
    case 'wellness':
      return card.wellnessInterpretation;
    case 'family':
      return card.familyInterpretation;
    case 'self_growth':
      return card.selfGrowthInterpretation;
    case 'school':
      return card.schoolInterpretation;
    default:
      return isReversed ? card.reversedMeaning : card.uprightMeaning;
  }
}

/**
 * Get keywords for card
 */
export function getKeywords(card, isReversed = false) {
  return isReversed ? card.reversedKeywords : card.uprightKeywords;
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
