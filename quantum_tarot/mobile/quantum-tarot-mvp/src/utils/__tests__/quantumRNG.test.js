/**
 * Unit tests for Quantum RNG system
 */

import {
  generateQuantumSeed,
  getQuantumInt,
  quantumShuffle,
  drawCards,
  performReading
} from '../quantumRNG';

describe('Quantum RNG', () => {
  describe('generateQuantumSeed', () => {
    it('should generate a 31-character alphanumeric seed', async () => {
      const seed = await generateQuantumSeed();
      expect(seed).toHaveLength(31);
      expect(seed).toMatch(/^[0-9A-Za-z]+$/);
    });

    it('should generate unique seeds on multiple calls', async () => {
      const seed1 = await generateQuantumSeed();
      const seed2 = await generateQuantumSeed();
      expect(seed1).not.toBe(seed2);
    });

    it('should use full alphanumeric character set', async () => {
      const seeds = await Promise.all(
        Array(100).fill(0).map(() => generateQuantumSeed())
      );
      const allChars = seeds.join('');

      // Should contain numbers
      expect(allChars).toMatch(/[0-9]/);
      // Should contain uppercase
      expect(allChars).toMatch(/[A-Z]/);
      // Should contain lowercase
      expect(allChars).toMatch(/[a-z]/);
    });
  });

  describe('getQuantumInt', () => {
    it('should return integer within range', async () => {
      for (let i = 0; i < 100; i++) {
        const num = await getQuantumInt(0, 10);
        expect(num).toBeGreaterThanOrEqual(0);
        expect(num).toBeLessThanOrEqual(10);
        expect(Number.isInteger(num)).toBe(true);
      }
    });

    it('should handle single value range', async () => {
      const num = await getQuantumInt(5, 5);
      expect(num).toBe(5);
    });

    it('should distribute across range', async () => {
      const results = {};
      for (let i = 0; i < 1000; i++) {
        const num = await getQuantumInt(0, 9);
        results[num] = (results[num] || 0) + 1;
      }

      // Should use all values in range
      expect(Object.keys(results).length).toBe(10);

      // Rough distribution check (each should appear ~100 times ± 50)
      Object.values(results).forEach(count => {
        expect(count).toBeGreaterThan(50);
        expect(count).toBeLessThan(150);
      });
    });
  });

  describe('quantumShuffle', () => {
    it('should shuffle array without losing elements', async () => {
      const original = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
      const shuffled = await quantumShuffle(original);

      expect(shuffled).toHaveLength(original.length);
      expect(shuffled.sort()).toEqual(original);
    });

    it('should actually shuffle (not return same order)', async () => {
      const original = Array.from({ length: 78 }, (_, i) => i);
      const shuffled = await quantumShuffle(original);

      // Extremely unlikely to get same order
      expect(shuffled).not.toEqual(original);
    });

    it('should preserve original array', async () => {
      const original = [1, 2, 3, 4, 5];
      const originalCopy = [...original];
      await quantumShuffle(original);

      expect(original).toEqual(originalCopy);
    });
  });

  describe('drawCards', () => {
    it('should draw requested number of cards', async () => {
      const cards = await drawCards(3, 'test intention');
      expect(cards).toHaveLength(3);
    });

    it('should return unique cards (no duplicates)', async () => {
      const cards = await drawCards(10, 'test intention');
      const indices = cards.map(c => c.cardIndex);
      const uniqueIndices = [...new Set(indices)];

      expect(uniqueIndices).toHaveLength(10);
    });

    it('should include reversal information', async () => {
      const cards = await drawCards(5, 'test intention');

      cards.forEach(card => {
        expect(card).toHaveProperty('cardIndex');
        expect(card).toHaveProperty('reversed');
        expect(typeof card.reversed).toBe('boolean');
      });
    });

    it('should work with different intentions', async () => {
      const cards1 = await drawCards(3, 'love question');
      const cards2 = await drawCards(3, 'career question');

      expect(cards1).toHaveLength(3);
      expect(cards2).toHaveLength(3);
      // Different intentions should give different results
      expect(cards1).not.toEqual(cards2);
    });

    it('should have roughly 50/50 reversal distribution', async () => {
      const cards = await drawCards(100, 'distribution test');
      const reversedCount = cards.filter(c => c.reversed).length;

      // Should be roughly 50 ± 20
      expect(reversedCount).toBeGreaterThan(30);
      expect(reversedCount).toBeLessThan(70);
    });
  });

  describe('performReading', () => {
    it('should return complete reading object', async () => {
      const reading = await performReading('three_card', 'test intention');

      expect(reading).toHaveProperty('cards');
      expect(reading).toHaveProperty('quantumSeed');
      expect(reading).toHaveProperty('timestamp');
      expect(reading).toHaveProperty('spreadType');
      expect(reading).toHaveProperty('intention');
    });

    it('should draw correct number of cards for spread', async () => {
      const singleReading = await performReading('single_card', 'test');
      expect(singleReading.cards).toHaveLength(1);

      const threeCardReading = await performReading('three_card', 'test');
      expect(threeCardReading.cards).toHaveLength(3);
    });

    it('should attach position information to cards', async () => {
      const reading = await performReading('three_card', 'test');

      reading.cards.forEach(card => {
        expect(card).toHaveProperty('position');
        expect(card).toHaveProperty('cardIndex');
        expect(card).toHaveProperty('reversed');
      });
    });

    it('should generate unique quantum seed per reading', async () => {
      const reading1 = await performReading('single_card', 'test');
      const reading2 = await performReading('single_card', 'test');

      expect(reading1.quantumSeed).not.toBe(reading2.quantumSeed);
    });
  });
});
