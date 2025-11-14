/**
 * Card Query Engine
 * Enables AGI to query card database and detect patterns
 * Powers meta-analysis and sophisticated interpretation
 */

import { CARD_DATABASE } from '../data/cardDatabase';

/**
 * Card Query Engine
 * Provides structured queries for the LunatIQ Tarot AGI
 */
export class CardQueryEngine {
  constructor(spread) {
    this.spread = spread; // Array of {cardIndex, reversed, position}
    this.cards = spread.map(pos => ({
      data: CARD_DATABASE[pos.cardIndex],
      reversed: pos.reversed,
      position: pos.position
    }));
  }

  // ═══════════════════════════════════════════════════════════
  // ELEMENTAL ANALYSIS
  // ═══════════════════════════════════════════════════════════

  getElementalBreakdown() {
    const elements = { fire: 0, water: 0, air: 0, earth: 0, spirit: 0 };
    this.cards.forEach(c => {
      elements[c.data.element]++;
    });
    return elements;
  }

  getDominantElement() {
    const breakdown = this.getElementalBreakdown();
    const dominant = Object.entries(breakdown)
      .sort((a, b) => b[1] - a[1])[0];
    return dominant[1] > 0 ? dominant[0] : null;
  }

  getElementalImbalance() {
    const breakdown = this.getElementalBreakdown();
    const total = this.cards.length;
    const avg = total / 5;

    const imbalances = {};
    Object.entries(breakdown).forEach(([elem, count]) => {
      const diff = count - avg;
      if (Math.abs(diff) > 1) {
        imbalances[elem] = diff > 0 ? 'excess' : 'deficient';
      }
    });
    return imbalances;
  }

  // ═══════════════════════════════════════════════════════════
  // ARCANA ANALYSIS
  // ═══════════════════════════════════════════════════════════

  getMajorArcanaCount() {
    return this.cards.filter(c => c.data.arcana === 'major').length;
  }

  getMinorArcanaCount() {
    return this.cards.filter(c => c.data.arcana === 'minor').length;
  }

  getMajorArcanaRatio() {
    return this.getMajorArcanaCount() / this.cards.length;
  }

  // High major arcana = soul-level, fated events
  // High minor arcana = everyday choices, personal agency
  getArcanaSignificance() {
    const ratio = this.getMajorArcanaRatio();
    if (ratio >= 0.6) return 'fated_spiritual';
    if (ratio <= 0.2) return 'mundane_choice';
    return 'balanced';
  }

  // ═══════════════════════════════════════════════════════════
  // SUIT ANALYSIS
  // ═══════════════════════════════════════════════════════════

  getSuitBreakdown() {
    const suits = { wands: 0, cups: 0, swords: 0, pentacles: 0 };
    this.cards.forEach(c => {
      if (c.data.suit) suits[c.data.suit]++;
    });
    return suits;
  }

  getDominantSuit() {
    const breakdown = this.getSuitBreakdown();
    const dominant = Object.entries(breakdown)
      .sort((a, b) => b[1] - a[1])[0];
    return dominant[1] > 0 ? dominant[0] : null;
  }

  // ═══════════════════════════════════════════════════════════
  // REVERSAL ANALYSIS
  // ═══════════════════════════════════════════════════════════

  getReversalCount() {
    return this.cards.filter(c => c.reversed).length;
  }

  getReversalRatio() {
    return this.getReversalCount() / this.cards.length;
  }

  // High reversals = internal blocks, resistance, shadow work needed
  getReversalSignificance() {
    const ratio = this.getReversalRatio();
    if (ratio >= 0.7) return 'heavy_blockage';
    if (ratio >= 0.5) return 'moderate_resistance';
    if (ratio >= 0.3) return 'some_shadow';
    return 'flow_state';
  }

  // ═══════════════════════════════════════════════════════════
  // NUMEROLOGY ANALYSIS
  // ═══════════════════════════════════════════════════════════

  getNumerologyPattern() {
    const numbers = this.cards.map(c => c.data.numerology);
    const counts = {};
    numbers.forEach(n => {
      counts[n] = (counts[n] || 0) + 1;
    });
    return counts;
  }

  // Repeating numbers (e.g., three 3s = creativity cycle)
  getRepeatingNumbers() {
    const pattern = this.getNumerologyPattern();
    return Object.entries(pattern)
      .filter(([num, count]) => count >= 2)
      .map(([num, count]) => ({ number: parseInt(num), count }));
  }

  // ═══════════════════════════════════════════════════════════
  // COURT CARD ANALYSIS
  // ═══════════════════════════════════════════════════════════

  getCourtCards() {
    return this.cards.filter(c =>
      c.data.rank && ['page', 'knight', 'queen', 'king'].includes(c.data.rank)
    );
  }

  getCourtCardRatio() {
    return this.getCourtCards().length / this.cards.length;
  }

  // High court cards = people/relationships are significant
  getCourtCardSignificance() {
    const ratio = this.getCourtCardRatio();
    if (ratio >= 0.5) return 'people_dominant';
    if (ratio >= 0.3) return 'relational_focus';
    return 'individual_journey';
  }

  // ═══════════════════════════════════════════════════════════
  // THEMATIC ANALYSIS
  // ═══════════════════════════════════════════════════════════

  getCommonThemes() {
    const themeCount = {};
    this.cards.forEach(c => {
      c.data.themes.forEach(theme => {
        themeCount[theme] = (themeCount[theme] || 0) + 1;
      });
    });

    return Object.entries(themeCount)
      .filter(([theme, count]) => count >= 2)
      .sort((a, b) => b[1] - a[1])
      .map(([theme, count]) => ({ theme, count }));
  }

  getCommonArchetypes() {
    const archetypeCount = {};
    this.cards.forEach(c => {
      c.data.archetypes.forEach(arch => {
        archetypeCount[arch] = (archetypeCount[arch] || 0) + 1;
      });
    });

    return Object.entries(archetypeCount)
      .filter(([arch, count]) => count >= 2)
      .sort((a, b) => b[1] - a[1])
      .map(([arch, count]) => ({ archetype: arch, count }));
  }

  // ═══════════════════════════════════════════════════════════
  // CHAKRA ANALYSIS
  // ═══════════════════════════════════════════════════════════

  getChakraBreakdown() {
    const chakras = {};
    this.cards.forEach(c => {
      const chakra = c.data.chakra;
      if (chakra) chakras[chakra] = (chakras[chakra] || 0) + 1;
    });
    return chakras;
  }

  getDominantChakra() {
    const breakdown = this.getChakraBreakdown();
    const dominant = Object.entries(breakdown)
      .sort((a, b) => b[1] - a[1])[0];
    return dominant ? dominant[0] : null;
  }

  // ═══════════════════════════════════════════════════════════
  // ASTROLOGICAL ANALYSIS
  // ═══════════════════════════════════════════════════════════

  getAstrologicalInfluences() {
    const astro = {};
    this.cards.forEach(c => {
      const influence = c.data.astrology;
      if (influence) astro[influence] = (astro[influence] || 0) + 1;
    });
    return astro;
  }

  // ═══════════════════════════════════════════════════════════
  // COMPOSITE QUERIES
  // ═══════════════════════════════════════════════════════════

  /**
   * Get full spread meta-analysis
   * Returns comprehensive pattern detection
   */
  getMetaAnalysis() {
    return {
      // Elemental
      elements: this.getElementalBreakdown(),
      dominantElement: this.getDominantElement(),
      elementalImbalance: this.getElementalImbalance(),

      // Arcana
      majorCount: this.getMajorArcanaCount(),
      minorCount: this.getMinorArcanaCount(),
      arcanaSignificance: this.getArcanaSignificance(),

      // Suits
      suits: this.getSuitBreakdown(),
      dominantSuit: this.getDominantSuit(),

      // Reversals
      reversalCount: this.getReversalCount(),
      reversalRatio: this.getReversalRatio(),
      reversalSignificance: this.getReversalSignificance(),

      // Numerology
      numerologyPattern: this.getNumerologyPattern(),
      repeatingNumbers: this.getRepeatingNumbers(),

      // Court cards
      courtCardCount: this.getCourtCards().length,
      courtCardSignificance: this.getCourtCardSignificance(),

      // Themes
      commonThemes: this.getCommonThemes(),
      commonArchetypes: this.getCommonArchetypes(),

      // Chakras
      chakras: this.getChakraBreakdown(),
      dominantChakra: this.getDominantChakra(),

      // Astrology
      astrologicalInfluences: this.getAstrologicalInfluences()
    };
  }

  /**
   * Get human-readable summary of patterns
   */
  getSummary() {
    const meta = this.getMetaAnalysis();
    const insights = [];

    // Elemental insights
    if (meta.dominantElement) {
      const elementMeanings = {
        fire: "action, passion, and willpower",
        water: "emotions, intuition, and relationships",
        air: "thoughts, communication, and mental activity",
        earth: "material matters, body, and practical concerns",
        spirit: "transcendence and divine connection"
      };
      insights.push(`Strong ${meta.dominantElement} energy suggests focus on ${elementMeanings[meta.dominantElement]}.`);
    }

    // Arcana insights
    if (meta.arcanaSignificance === 'fated_spiritual') {
      insights.push("High Major Arcana presence indicates soul-level lessons and fated events.");
    } else if (meta.arcanaSignificance === 'mundane_choice') {
      insights.push("Mostly Minor Arcana suggests everyday choices and personal agency are key.");
    }

    // Reversal insights
    if (meta.reversalSignificance === 'heavy_blockage') {
      insights.push("Many reversed cards point to internal blocks or resistance. Shadow work may be needed.");
    } else if (meta.reversalSignificance === 'flow_state') {
      insights.push("Mostly upright cards suggest you're in alignment and flow.");
    }

    // Court card insights
    if (meta.courtCardSignificance === 'people_dominant') {
      insights.push("Multiple court cards indicate relationships and other people are central to this situation.");
    }

    // Numerology insights
    if (meta.repeatingNumbers.length > 0) {
      meta.repeatingNumbers.forEach(({ number, count }) => {
        const numMeanings = {
          1: "new beginnings and individuality",
          2: "duality and partnership",
          3: "creativity and expression",
          4: "stability and foundation",
          5: "change and freedom",
          6: "harmony and responsibility",
          7: "spirituality and introspection",
          8: "power and manifestation",
          9: "completion and wisdom",
          10: "endings and new cycles"
        };
        insights.push(`Repeating ${number}s (${count} cards) emphasize ${numMeanings[number]}.`);
      });
    }

    // Theme insights
    if (meta.commonThemes.length > 0) {
      const topTheme = meta.commonThemes[0];
      insights.push(`Recurring theme: ${topTheme.theme.replace(/_/g, ' ')} appears ${topTheme.count} times.`);
    }

    return insights;
  }
}

/**
 * Example usage:
 */
export function testQueryEngine() {
  const mockSpread = [
    { cardIndex: 0, reversed: false, position: 'Past' },
    { cardIndex: 1, reversed: false, position: 'Present' },
    { cardIndex: 2, reversed: true, position: 'Future' }
  ];

  const engine = new CardQueryEngine(mockSpread);
  const meta = engine.getMetaAnalysis();
  const summary = engine.getSummary();

  console.log('Meta-analysis:', meta);
  console.log('Summary:', summary);
}
