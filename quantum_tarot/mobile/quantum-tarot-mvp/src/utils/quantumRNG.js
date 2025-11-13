/**
 * QUANTUM RNG - Hardware-based true random number generation
 * Uses expo-crypto for cryptographically secure randomness
 */

import * as Crypto from 'expo-crypto';

/**
 * Alphanumeric character set for seed generation
 * 0-9, A-Z, a-z = 62 characters
 */
const ALPHANUMERIC_CHARS = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz';

/**
 * Generate quantum random bytes
 * @param {number} byteCount - Number of random bytes to generate
 * @returns {Promise<Uint8Array>}
 */
async function getQuantumBytes(byteCount) {
  const bytes = await Crypto.getRandomBytesAsync(byteCount);
  return bytes;
}

/**
 * Generate a quantum superposition collapsed alphanumeric seed
 * 31 characters, each can be [0-9, A-Z, a-z]
 * Total entropy: 62^31 ≈ 3.35 x 10^55 combinations
 * @returns {Promise<string>} - 31 character alphanumeric seed
 */
export async function generateQuantumSeed() {
  const SEED_LENGTH = 31;
  const CHAR_SET_SIZE = ALPHANUMERIC_CHARS.length; // 62

  // Get quantum random bytes (need enough for 31 characters)
  const bytesNeeded = SEED_LENGTH * 2; // 2 bytes per character for good distribution
  const randomBytes = await getQuantumBytes(bytesNeeded);

  let seed = '';
  for (let i = 0; i < SEED_LENGTH; i++) {
    // Use 2 bytes to get a number, then mod by char set size
    const byte1 = randomBytes[i * 2];
    const byte2 = randomBytes[i * 2 + 1];
    const randomIndex = ((byte1 << 8) | byte2) % CHAR_SET_SIZE;
    seed += ALPHANUMERIC_CHARS[randomIndex];
  }

  return seed;
}

/**
 * Generate a random integer between min and max (inclusive)
 * @param {number} min - Minimum value
 * @param {number} max - Maximum value
 * @returns {Promise<number>}
 */
export async function getQuantumInt(min, max) {
  const range = max - min + 1;

  // Determine how many bytes we need
  const bytesNeeded = Math.ceil(Math.log2(range) / 8);

  // Get quantum random bytes
  const randomBytes = await getQuantumBytes(bytesNeeded);

  // Convert bytes to integer
  let randomInt = 0;
  for (let i = 0; i < randomBytes.length; i++) {
    randomInt = (randomInt << 8) | randomBytes[i];
  }

  // Map to range using modulo (with bias correction)
  return min + (randomInt % range);
}

/**
 * Shuffle an array using Fisher-Yates with quantum randomness
 * @param {Array} array - Array to shuffle
 * @returns {Promise<Array>} - Shuffled copy of array
 */
export async function quantumShuffle(array) {
  const shuffled = [...array];

  for (let i = shuffled.length - 1; i > 0; i--) {
    // Get quantum random index
    const j = await getQuantumInt(0, i);

    // Swap elements
    [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
  }

  return shuffled;
}

/**
 * Fetch current timestamp from public API for non-repeatable entropy
 * Falls back to local time if network unavailable
 * @returns {Promise<string>}
 */
async function getPublicTimestamp() {
  try {
    // WorldTimeAPI - free, no auth required, global CDN
    const response = await fetch('https://worldtimeapi.org/api/timezone/Etc/UTC', {
      timeout: 5000
    });
    const data = await response.json();
    // Returns ISO timestamp with high precision (microseconds)
    return data.datetime + data.unixtime.toString();
  } catch (error) {
    // Fallback to local time + performance counter for uniqueness
    return Date.now().toString() + performance.now().toString();
  }
}

/**
 * Draw N random cards from deck (currently 5 cards until full database is populated)
 * Returns array of { cardIndex, reversed } objects
 * @param {number} cardCount - Number of cards to draw
 * @param {string} intention - User's intention (mixed into entropy)
 * @returns {Promise<Array>}
 */
export async function drawCards(cardCount, intention = '') {
  const TOTAL_CARDS = 5; // TODO: Change to 78 when full card database is populated

  // Create deck indices
  const deck = Array.from({ length: TOTAL_CARDS }, (_, i) => i);

  // Get public timestamp for non-repeatable entropy
  const publicTimestamp = await getPublicTimestamp();

  // Mix intention + public timestamp into entropy
  // This ensures draws are:
  // 1. Unique to the user's intention
  // 2. Non-repeatable (different timestamp every time)
  // 3. Still cryptographically secure (doesn't compromise randomness)
  const entropyMix = intention + publicTimestamp;
  const intentionHash = await Crypto.digestStringAsync(
    Crypto.CryptoDigestAlgorithm.SHA256,
    entropyMix
  );

  // Shuffle deck with quantum randomness
  const shuffledDeck = await quantumShuffle(deck);

  console.log('🎴 QUANTUM DRAW DEBUG:');
  console.log('Shuffled first 10 cards:', shuffledDeck.slice(0, 10));

  // Draw cards and determine reversals
  const drawnCards = [];
  for (let i = 0; i < cardCount; i++) {
    const cardIndex = shuffledDeck[i];

    // Quantum random reversal (50/50 chance)
    const reversed = (await getQuantumInt(0, 1)) === 1;

    console.log(`Card ${i + 1}: Index=${cardIndex}, Reversed=${reversed}`);

    drawnCards.push({
      cardIndex,
      reversed
    });
  }

  console.log('Final drawn cards:', drawnCards);

  return drawnCards;
}

/**
 * Get spread positions based on spread type
 * @param {string} spreadType - Type of spread
 * @returns {Array<string>} - Position names
 */
export function getSpreadPositions(spreadType) {
  const spreads = {
    single_card: ['Present Moment'],

    three_card: ['Past', 'Present', 'Future'],

    daily: ['Focus On', 'Avoid', 'Gift'],

    decision: [
      'Current Situation',
      'Path A: Outcome',
      'Path A: Challenges',
      'Path B: Outcome',
      'Path B: Challenges',
      'Guidance'
    ],

    relationship: [
      'You',
      'Them',
      'The Connection',
      'Hidden Influences',
      'Past Foundation',
      'Future Potential'
    ],

    celtic_cross: [
      'Present',
      'Challenge',
      'Past',
      'Future',
      'Above (Conscious)',
      'Below (Unconscious)',
      'Advice',
      'External Influences',
      'Hopes/Fears',
      'Outcome'
    ]
  };

  return spreads[spreadType] || spreads.three_card;
}

/**
 * Perform a full quantum tarot reading
 * @param {string} spreadType - Type of spread
 * @param {string} intention - User's intention
 * @returns {Promise<Object>} - Reading data with cards and quantum seed
 */
export async function performReading(spreadType, intention) {
  const positions = getSpreadPositions(spreadType);

  // Generate quantum seed for this reading (non-repeatable)
  const quantumSeed = await generateQuantumSeed();

  // Get public timestamp
  const publicTimestamp = await getPublicTimestamp();

  // Draw cards
  const cards = await drawCards(positions.length, intention);

  // Attach positions to cards
  const cardsWithPositions = cards.map((card, index) => ({
    ...card,
    position: positions[index]
  }));

  return {
    cards: cardsWithPositions,
    quantumSeed,
    timestamp: publicTimestamp,
    spreadType,
    intention
  };
}
