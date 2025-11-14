/**
 * Unit tests for Card Database
 */

import { CARD_DATABASE, getCardData } from '../cardDatabase';

describe('Card Database', () => {
  describe('Database Structure', () => {
    it('should contain all 78 tarot cards', () => {
      expect(CARD_DATABASE).toHaveLength(78);
    });

    it('should have 22 major arcana cards', () => {
      const majorArcana = CARD_DATABASE.filter(card => card.arcana === 'major');
      expect(majorArcana).toHaveLength(22);
    });

    it('should have 56 minor arcana cards', () => {
      const minorArcana = CARD_DATABASE.filter(card => card.arcana === 'minor');
      expect(minorArcana).toHaveLength(56);
    });

    it('should have 14 cards per suit', () => {
      const suits = ['wands', 'cups', 'swords', 'pentacles'];

      suits.forEach(suit => {
        const suitCards = CARD_DATABASE.filter(card => card.suit === suit);
        expect(suitCards).toHaveLength(14);
      });
    });

    it('should have unique IDs from 0-77', () => {
      const ids = CARD_DATABASE.map(card => card.id);
      const uniqueIds = [...new Set(ids)];

      expect(uniqueIds).toHaveLength(78);
      expect(Math.min(...ids)).toBe(0);
      expect(Math.max(...ids)).toBe(77);
    });
  });

  describe('Card Data Completeness', () => {
    it('should have all required fields for each card', () => {
      const requiredFields = [
        'id', 'name', 'arcana', 'element', 'numerology',
        'symbols', 'archetypes', 'themes', 'keywords',
        'jungian', 'chakra', 'advice', 'light', 'questions', 'description'
      ];

      CARD_DATABASE.forEach(card => {
        requiredFields.forEach(field => {
          expect(card).toHaveProperty(field);
          expect(card[field]).toBeDefined();
          expect(card[field]).not.toBeNull();
        });
      });
    });

    it('should have upright and reversed keywords', () => {
      CARD_DATABASE.forEach(card => {
        expect(card.keywords).toHaveProperty('upright');
        expect(card.keywords).toHaveProperty('reversed');
        expect(Array.isArray(card.keywords.upright)).toBe(true);
        expect(Array.isArray(card.keywords.reversed)).toBe(true);
        expect(card.keywords.upright.length).toBeGreaterThan(0);
        expect(card.keywords.reversed.length).toBeGreaterThan(0);
      });
    });

    it('should have valid elements', () => {
      const validElements = ['fire', 'water', 'air', 'earth', 'spirit'];

      CARD_DATABASE.forEach(card => {
        expect(validElements).toContain(card.element);
      });
    });

    it('should have valid chakras', () => {
      const validChakras = [
        'root', 'sacral', 'solar_plexus', 'heart',
        'throat', 'third_eye', 'crown'
      ];

      CARD_DATABASE.forEach(card => {
        expect(validChakras).toContain(card.chakra);
      });
    });

    it('should have symbols array', () => {
      CARD_DATABASE.forEach(card => {
        expect(Array.isArray(card.symbols)).toBe(true);
        expect(card.symbols.length).toBeGreaterThan(0);
      });
    });

    it('should have archetypes array', () => {
      CARD_DATABASE.forEach(card => {
        expect(Array.isArray(card.archetypes)).toBe(true);
        expect(card.archetypes.length).toBeGreaterThan(0);
      });
    });

    it('should have themes array', () => {
      CARD_DATABASE.forEach(card => {
        expect(Array.isArray(card.themes)).toBe(true);
        expect(card.themes.length).toBeGreaterThan(0);
      });
    });

    it('should have reflection questions array', () => {
      CARD_DATABASE.forEach(card => {
        expect(Array.isArray(card.questions)).toBe(true);
        expect(card.questions.length).toBeGreaterThan(0);
      });
    });
  });

  describe('Major Arcana', () => {
    it('should start with The Fool (0)', () => {
      const fool = CARD_DATABASE.find(card => card.id === 0);
      expect(fool.name).toBe('The Fool');
      expect(fool.arcana).toBe('major');
      expect(fool.number).toBe(0);
    });

    it('should end with The World (21)', () => {
      const world = CARD_DATABASE.find(card => card.id === 21);
      expect(world.name).toBe('The World');
      expect(world.arcana).toBe('major');
      expect(world.number).toBe(21);
    });

    it('should have null suit and rank for major arcana', () => {
      const majorArcana = CARD_DATABASE.filter(card => card.arcana === 'major');

      majorArcana.forEach(card => {
        expect(card.suit).toBeNull();
        expect(card.rank).toBeNull();
      });
    });
  });

  describe('Minor Arcana', () => {
    it('should have correct suit assignments', () => {
      const minorArcana = CARD_DATABASE.filter(card => card.arcana === 'minor');

      minorArcana.forEach(card => {
        expect(['wands', 'cups', 'swords', 'pentacles']).toContain(card.suit);
      });
    });

    it('should have ace through king in each suit', () => {
      const ranks = ['ace', '2', '3', '4', '5', '6', '7', '8', '9', '10',
                     'page', 'knight', 'queen', 'king'];
      const suits = ['wands', 'cups', 'swords', 'pentacles'];

      suits.forEach(suit => {
        const suitCards = CARD_DATABASE.filter(card => card.suit === suit);
        const suitRanks = suitCards.map(card => card.rank);

        expect(suitRanks.sort()).toEqual(ranks.sort());
      });
    });
  });

  describe('getCardData', () => {
    it('should retrieve card by index', () => {
      const card = getCardData(0);
      expect(card.name).toBe('The Fool');
    });

    it('should return null for invalid index', () => {
      expect(getCardData(-1)).toBeNull();
      expect(getCardData(78)).toBeNull();
      expect(getCardData(999)).toBeNull();
    });

    it('should return complete card object', () => {
      const card = getCardData(10);
      expect(card).toHaveProperty('id');
      expect(card).toHaveProperty('name');
      expect(card).toHaveProperty('keywords');
      expect(card).toHaveProperty('description');
    });
  });

  describe('Data Quality', () => {
    it('should have non-empty advice for all cards', () => {
      CARD_DATABASE.forEach(card => {
        expect(card.advice.length).toBeGreaterThan(10);
      });
    });

    it('should have non-empty description for all cards', () => {
      CARD_DATABASE.forEach(card => {
        expect(card.description.length).toBeGreaterThan(20);
      });
    });

    it('should have at least 3 keywords upright and reversed', () => {
      CARD_DATABASE.forEach(card => {
        expect(card.keywords.upright.length).toBeGreaterThanOrEqual(3);
        expect(card.keywords.reversed.length).toBeGreaterThanOrEqual(3);
      });
    });

    it('should have at least 2 reflection questions', () => {
      CARD_DATABASE.forEach(card => {
        expect(card.questions.length).toBeGreaterThanOrEqual(2);
      });
    });
  });
});
