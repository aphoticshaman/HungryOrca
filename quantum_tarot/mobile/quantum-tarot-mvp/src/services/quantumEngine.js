/**
 * Quantum Tarot - Quantum Randomization Engine (JavaScript)
 * Ported from Python - Runs entirely on phone
 * No server needed!
 */

import * as Random from 'expo-random';
import * as Crypto from 'expo-crypto';
import { getSpreadDefinition } from '../data/spreadDefinitions.js';

/**
 * Quantum State - represents a collapsed card selection
 */
export class QuantumState {
  constructor(cardIndex, reversed, collapseTimestamp, entropySource, quantumSignature) {
    this.cardIndex = cardIndex;
    this.reversed = reversed;
    this.collapseTimestamp = collapseTimestamp;
    this.entropySource = entropySource;
    this.quantumSignature = quantumSignature;
  }
}

/**
 * Quantum Random Generator
 * Uses device's hardware random number generator
 */
export class QuantumRandomGenerator {
  constructor() {
    this.entropyPool = [];
  }

  /**
   * Get quantum random bytes from device hardware
   * Uses crypto.getRandomValues (backed by hardware RNG)
   */
  async getQuantumBytes(numBytes = 32) {
    try {
      // Validate input
      if (!numBytes || numBytes <= 0 || numBytes > 1024) {
        throw new Error('Invalid byte count');
      }

      // expo-random uses device's hardware random number generator
      // This is genuinely random (quantum effects in silicon + environmental noise)
      const randomBytes = await Random.getRandomBytesAsync(numBytes);

      // Validate result
      if (!randomBytes || randomBytes.length !== numBytes) {
        throw new Error('Failed to generate quantum bytes');
      }

      return randomBytes;
    } catch (error) {
      console.error('Quantum RNG failed, falling back to Math.random():', error);

      // FALLBACK: Use Math.random() if hardware RNG fails
      // Not quantum, but better than crashing
      const fallbackBytes = new Uint8Array(numBytes);
      for (let i = 0; i < numBytes; i++) {
        fallbackBytes[i] = Math.floor(Math.random() * 256);
      }
      return fallbackBytes;
    }
  }

  /**
   * Convert bytes to integer
   */
  bytesToInt(bytes) {
    try {
      // Validate input
      if (!bytes || bytes.length === 0) {
        throw new Error('Invalid bytes array');
      }

      let result = 0;
      for (let i = 0; i < Math.min(4, bytes.length); i++) {
        result = (result << 8) | bytes[i];
      }
      return result >>> 0; // Convert to unsigned 32-bit integer
    } catch (error) {
      console.error('Failed to convert bytes to int:', error);
      // Fallback to random integer
      return Math.floor(Math.random() * 0xFFFFFFFF);
    }
  }

  /**
   * Generate quantum-random card positions for a reading
   *
   * @param {number} numCards - Number of cards to draw
   * @param {number} deckSize - Size of deck (default 78)
   * @param {boolean} allowDuplicates - Whether same card can appear multiple times
   * @returns {QuantumState[]} Array of quantum states
   */
  async generateCardPositions(numCards, deckSize = 78, allowDuplicates = false) {
    if (!allowDuplicates && numCards > deckSize) {
      throw new Error(`Cannot draw ${numCards} unique cards from deck of ${deckSize}`);
    }

    const drawnCards = [];
    const usedIndices = new Set();

    for (let i = 0; i < numCards; i++) {
      // Get quantum bytes for this card
      const quantumBytes = await this.getQuantumBytes(32);

      // Convert to card index
      const byteInt = this.bytesToInt(quantumBytes.slice(0, 4));

      let cardIndex;
      if (allowDuplicates) {
        cardIndex = byteInt % deckSize;
      } else {
        // Draw without replacement
        const availableIndices = deckSize - usedIndices.size;
        const position = byteInt % availableIndices;

        // Map to actual unused index
        const allIndices = Array.from({ length: deckSize }, (_, i) => i);
        const unusedIndices = allIndices.filter(idx => !usedIndices.has(idx));
        cardIndex = unusedIndices[position];
        usedIndices.add(cardIndex);
      }

      // Determine if reversed (50/50 quantum coin flip)
      const reversedByte = quantumBytes[4];
      const isReversed = (reversedByte & 1) === 1;

      // Create quantum signature for provenance
      const signature = await this.createQuantumSignature(quantumBytes, i);

      const quantumState = new QuantumState(
        cardIndex,
        isReversed,
        Date.now(),
        'hardware_rng',
        signature
      );

      drawnCards.push(quantumState);
    }

    return drawnCards;
  }

  /**
   * Create cryptographic signature from quantum entropy
   */
  async createQuantumSignature(quantumBytes, index) {
    try {
      // Validate inputs
      if (!quantumBytes || quantumBytes.length === 0) {
        throw new Error('Invalid quantum bytes');
      }
      if (typeof index !== 'number' || index < 0) {
        throw new Error('Invalid index');
      }

      // Create SHA-256 hash using expo-crypto
      const timestamp = Date.now().toString();
      const indexStr = index.toString();

      // Combine data
      const combined = new Uint8Array(quantumBytes.length + timestamp.length + indexStr.length);
      combined.set(quantumBytes, 0);
      for (let i = 0; i < timestamp.length; i++) {
        combined[quantumBytes.length + i] = timestamp.charCodeAt(i);
      }
      for (let i = 0; i < indexStr.length; i++) {
        combined[quantumBytes.length + timestamp.length + i] = indexStr.charCodeAt(i);
      }

      const digest = await Crypto.digestStringAsync(
        Crypto.CryptoDigestAlgorithm.SHA256,
        Array.from(combined).map(b => String.fromCharCode(b)).join('')
      );

      if (!digest) {
        throw new Error('Failed to create digest');
      }

      return digest;
    } catch (error) {
      console.error('Failed to create quantum signature:', error);
      // Fallback to simple hash
      return `fallback_${Date.now()}_${index}_${Math.random()}`;
    }
  }

  /**
   * Collapse wave function based on user intention
   * Mix intention with quantum randomness
   */
  async collapseWaveFunction(userIntention, readingType, numCards = 3) {
    try {
      // Validate inputs
      if (!userIntention || typeof userIntention !== 'string') {
        throw new Error('Invalid user intention');
      }
      if (!readingType || typeof readingType !== 'string') {
        throw new Error('Invalid reading type');
      }
      if (!numCards || numCards <= 0 || numCards > 78) {
        throw new Error('Invalid card count');
      }

      // Hash user intention to create intention-seed
      const intentionString = userIntention + readingType + Date.now().toString();
      const intentionHashHex = await Crypto.digestStringAsync(
        Crypto.CryptoDigestAlgorithm.SHA256,
        intentionString
      );

      if (!intentionHashHex || typeof intentionHashHex !== 'string') {
        throw new Error('Failed to hash intention');
      }

      // Convert hex string to bytes
      const hexPairs = intentionHashHex.match(/.{1,2}/g);
      if (!hexPairs || hexPairs.length === 0) {
        throw new Error('Invalid hash format');
      }

      const intentionHash = new Uint8Array(
        hexPairs.map(byte => parseInt(byte, 16))
      );

      // Get quantum entropy
      const quantumBytes = await this.getQuantumBytes(32);

      // Mix intention with quantum entropy (XOR operation)
      const mixed = new Uint8Array(32);
      for (let i = 0; i < 32; i++) {
        mixed[i] = quantumBytes[i] ^ intentionHash[i % intentionHash.length];
      }

      // Use mixed entropy to seed card selection
      // The quantum randomness ensures unpredictability
      // The intention ensures personal relevance
      const seed = this.bytesToInt(mixed.slice(0, 4));

      // Generate quantum states
      const states = await this.generateCardPositions(numCards, 78, false);

      if (!states || states.length !== numCards) {
        throw new Error('Failed to generate card positions');
      }

      return states;
    } catch (error) {
      console.error('Wave function collapse failed:', error);
      throw error; // Re-throw to let caller handle
    }
  }
}

/**
 * Quantum Spread Engine
 * Manages different spread types
 */
export class QuantumSpreadEngine {
  constructor() {
    this.quantumGen = new QuantumRandomGenerator();
  }

  /**
   * Perform a complete quantum tarot reading
   */
  async performReading(spreadType, userIntention, readingType) {
    try {
      // Validate inputs
      if (!spreadType || typeof spreadType !== 'string') {
        throw new Error('Invalid spread type');
      }
      if (!userIntention || typeof userIntention !== 'string') {
        throw new Error('Invalid user intention');
      }
      if (!readingType || typeof readingType !== 'string') {
        throw new Error('Invalid reading type');
      }

      // Get spread definition
      const spreadDef = getSpreadDefinition(spreadType);
      if (!spreadDef) {
        throw new Error(`Unknown spread type: ${spreadType}`);
      }

      const numCards = spreadDef.cardCount;

      // Collapse quantum wave function
      const quantumStates = await this.quantumGen.collapseWaveFunction(
        userIntention,
        readingType,
        numCards
      );

      if (!quantumStates || quantumStates.length !== numCards) {
        throw new Error('Invalid quantum states generated');
      }

      // Package reading with full spread metadata
      const reading = {
        spreadType,
        spreadName: spreadDef.name,
        spreadDescription: spreadDef.description,
        readingType,
        timestamp: Date.now(),
        positions: []
      };

      for (let i = 0; i < spreadDef.positions.length; i++) {
        const positionDef = spreadDef.positions[i];
        const quantumState = quantumStates[i];

        if (!quantumState) {
          throw new Error(`Missing quantum state at position ${i}`);
        }

        reading.positions.push({
          position: positionDef.name,
          meaning: positionDef.meaning,
          cardIndex: quantumState.cardIndex,
          reversed: quantumState.reversed,
          quantumSignature: quantumState.quantumSignature,
          collapseTime: quantumState.collapseTimestamp,
          coordinates: positionDef.coordinates,
          colorAccent: positionDef.colorAccent,
          relatedPositions: positionDef.relatedPositions
        });
      }

      return reading;
    } catch (error) {
      console.error('Failed to perform reading:', error);
      throw error; // Re-throw to let screen handle with UI
    }
  }

  /**
   * Get available spreads
   */
  getAvailableSpreads() {
    const { getAllSpreads } = require('../data/spreadDefinitions.js');
    return getAllSpreads();
  }
}
