/**
 * LUNATIQ FUZZY ORCHESTRATOR
 * ===========================
 *
 * Port of Ryan's fuzzy meta-controller from ARC AGI work
 * Applied to symbolic tarot interpretation
 *
 * Architecture:
 * - 5 reasoning modalities (from 5x insights)
 * - Fuzzy logic for adaptive strategy selection
 * - Offline, no LLM dependencies
 * - Based on METAMORPHOSIS + XYZA frameworks
 */

// ═══════════════════════════════════════════════════════════════════════════════
// FUZZY LOGIC PRIMITIVES
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Fuzzy membership function (triangular)
 */
class FuzzySet {
  constructor(name, points) {
    this.name = name;
    this.points = points.sort((a, b) => a[0] - b[0]);
  }

  membership(x) {
    if (x <= this.points[0][0]) return this.points[0][1];
    if (x >= this.points[this.points.length - 1][0]) {
      return this.points[this.points.length - 1][1];
    }

    // Linear interpolation
    for (let i = 0; i < this.points.length - 1; i++) {
      const [x1, y1] = this.points[i];
      const [x2, y2] = this.points[i + 1];

      if (x1 <= x && x <= x2) {
        if (x2 === x1) return y1;
        return y1 + (y2 - y1) * (x - x1) / (x2 - x1);
      }
    }

    return 0;
  }
}

/**
 * Linguistic variable with multiple fuzzy sets
 */
class FuzzyVariable {
  constructor(name, rangeMin, rangeMax) {
    this.name = name;
    this.range = [rangeMin, rangeMax];
    this.sets = {};
  }

  addSet(name, points) {
    this.sets[name] = new FuzzySet(name, points);
  }

  fuzzify(value) {
    const memberships = {};
    for (const [setName, fuzzySet] of Object.entries(this.sets)) {
      memberships[setName] = fuzzySet.membership(value);
    }
    return memberships;
  }
}

/**
 * Fuzzy rule: IF antecedents THEN consequents
 */
class FuzzyRule {
  constructor(antecedents, consequents, weight = 1.0) {
    this.antecedents = antecedents; // {varName: setName}
    this.consequents = consequents; // {varName: setName}
    this.weight = weight;
  }

  evaluate(fuzzyInputs) {
    // Min operator for AND
    let activation = 1.0;

    for (const [varName, setName] of Object.entries(this.antecedents)) {
      const membership = fuzzyInputs[varName]?.[setName] || 0;
      activation = Math.min(activation, membership);
    }

    return activation * this.weight;
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TAROT FEATURE EXTRACTION (Multi-Modal)
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Extract symbolic features from cards
 * Modality 1: Symbolic-Archetypal
 */
class SymbolicAnalyzer {
  analyze(cards) {
    // Archetypal intensity based on card numbers
    const majorArcanaCount = cards.filter(c => c.number <= 21).length;
    const archetypeIntensity = majorArcanaCount / cards.length;

    // Suit diversity (minor arcana)
    const suits = new Set(cards.map(c => c.suit));
    const suitDiversity = (suits.size - 1) / 3; // 0-1 scale

    // Narrative flow (sequential numbers)
    let sequentialPairs = 0;
    for (let i = 0; i < cards.length - 1; i++) {
      if (Math.abs(cards[i].number - cards[i + 1].number) === 1) {
        sequentialPairs++;
      }
    }
    const narrativeFlow = sequentialPairs / Math.max(cards.length - 1, 1);

    return {
      archetypeIntensity,
      suitDiversity,
      narrativeFlow
    };
  }
}

/**
 * Extract relational features between cards
 * Modality 2: Graph-Relational
 */
class RelationalAnalyzer {
  analyze(cards, positions) {
    // Card contrast (opposing meanings)
    const hasReversals = positions.some(p => p.reversed);
    const reversalTension = hasReversals ? 0.7 : 0.3;

    // Position relationships (past-present-future)
    const positionCoherence = this.calculatePositionCoherence(cards, positions);

    // Elemental balance (suits as elements)
    const elementalBalance = this.calculateElementalBalance(cards);

    return {
      reversalTension,
      positionCoherence,
      elementalBalance
    };
  }

  calculatePositionCoherence(cards, positions) {
    // Simple heuristic: cards in sequence are more coherent
    let coherence = 0.5; // baseline
    if (cards.length >= 3) {
      // Check if past → present → future forms logical progression
      const avgNumberChange = Math.abs(
        (cards[1].number - cards[0].number) +
        (cards[2].number - cards[1].number)
      ) / 2;
      coherence = Math.max(0, 1 - avgNumberChange / 20);
    }
    return coherence;
  }

  calculateElementalBalance(cards) {
    const elements = {
      wands: 0,    // Fire
      cups: 0,     // Water
      swords: 0,   // Air
      pentacles: 0 // Earth
    };

    cards.forEach(card => {
      if (elements.hasOwnProperty(card.suit)) {
        elements[card.suit]++;
      }
    });

    // Calculate entropy (more balanced = higher)
    const total = cards.length;
    let entropy = 0;
    for (const count of Object.values(elements)) {
      if (count > 0) {
        const p = count / total;
        entropy -= p * Math.log2(p);
      }
    }

    return entropy / 2; // Normalize to ~0-1
  }
}

/**
 * Extract psychological/energetic features
 * Modality 3: Energetic-Psychological
 */
class EnergeticAnalyzer {
  analyze(cards, userProfile, intention) {
    // Emotional intensity from keywords
    const emotionalIntensity = this.calculateEmotionalIntensity(cards);

    // User resonance (profile alignment)
    const userResonance = this.calculateUserResonance(cards, userProfile);

    // Intention alignment
    const intentionAlignment = this.calculateIntentionAlignment(cards, intention);

    return {
      emotionalIntensity,
      userResonance,
      intentionAlignment
    };
  }

  calculateEmotionalIntensity(cards) {
    // High intensity keywords
    const intensityKeywords = [
      'death', 'tower', 'devil', 'transformation', 'crisis',
      'passion', 'love', 'fear', 'power', 'change'
    ];

    let intensity = 0;
    cards.forEach(card => {
      const keywords = card.uprightKeywords.concat(card.reversedKeywords || []);
      keywords.forEach(keyword => {
        if (intensityKeywords.some(k => keyword.toLowerCase().includes(k))) {
          intensity += 0.2;
        }
      });
    });

    return Math.min(intensity / cards.length, 1);
  }

  calculateUserResonance(cards, profile) {
    // Match card energy to personality profile
    let resonance = 0.5; // baseline

    // Analytical users resonate with structured cards
    if (profile.analytical > 0.7) {
      const structuredCards = cards.filter(c =>
        c.suit === 'swords' || c.suit === 'pentacles'
      ).length;
      resonance += 0.2 * (structuredCards / cards.length);
    }

    // Mystical users resonate with cups/major arcana
    if (profile.mystical > 0.7) {
      const mysticalCards = cards.filter(c =>
        c.suit === 'cups' || c.number <= 21
      ).length;
      resonance += 0.2 * (mysticalCards / cards.length);
    }

    return Math.min(resonance, 1);
  }

  calculateIntentionAlignment(cards, intention) {
    // Simple keyword matching for now
    const intentionLower = intention.toLowerCase();
    let alignment = 0.5;

    // Career intentions → pentacles
    if (intentionLower.includes('career') || intentionLower.includes('work')) {
      const pentacles = cards.filter(c => c.suit === 'pentacles').length;
      alignment = 0.3 + 0.4 * (pentacles / cards.length);
    }

    // Love intentions → cups
    if (intentionLower.includes('love') || intentionLower.includes('relationship')) {
      const cups = cards.filter(c => c.suit === 'cups').length;
      alignment = 0.3 + 0.4 * (cups / cards.length);
    }

    return alignment;
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// FUZZY ORCHESTRATOR (THE CORE)
// ═══════════════════════════════════════════════════════════════════════════════

export class LunatiQOrchestrator {
  constructor() {
    this.symbolicAnalyzer = new SymbolicAnalyzer();
    this.relationalAnalyzer = new RelationalAnalyzer();
    this.energeticAnalyzer = new EnergeticAnalyzer();

    this.setupFuzzyVariables();
    this.setupFuzzyRules();
  }

  setupFuzzyVariables() {
    // Input variables
    this.archetypeIntensity = new FuzzyVariable('archetypeIntensity', 0, 1);
    this.archetypeIntensity.addSet('low', [[0, 1], [0.3, 1], [0.5, 0]]);
    this.archetypeIntensity.addSet('medium', [[0.3, 0], [0.5, 1], [0.7, 0]]);
    this.archetypeIntensity.addSet('high', [[0.5, 0], [0.7, 1], [1, 1]]);

    this.emotionalIntensity = new FuzzyVariable('emotionalIntensity', 0, 1);
    this.emotionalIntensity.addSet('calm', [[0, 1], [0.3, 1], [0.5, 0]]);
    this.emotionalIntensity.addSet('moderate', [[0.3, 0], [0.5, 1], [0.7, 0]]);
    this.emotionalIntensity.addSet('intense', [[0.5, 0], [0.7, 1], [1, 1]]);

    this.userResonance = new FuzzyVariable('userResonance', 0, 1);
    this.userResonance.addSet('weak', [[0, 1], [0.4, 1], [0.6, 0]]);
    this.userResonance.addSet('strong', [[0.4, 0], [0.6, 1], [1, 1]]);

    // Output variables (agent activations)
    this.archetypeDepth = new FuzzyVariable('archetypeDepth', 0, 1);
    this.archetypeDepth.addSet('surface', [[0, 1], [0.3, 1], [0.5, 0]]);
    this.archetypeDepth.addSet('deep', [[0.3, 0], [0.7, 1], [1, 1]]);

    this.practicalGuidance = new FuzzyVariable('practicalGuidance', 0, 1);
    this.practicalGuidance.addSet('minimal', [[0, 1], [0.3, 1], [0.5, 0]]);
    this.practicalGuidance.addSet('extensive', [[0.3, 0], [0.7, 1], [1, 1]]);

    this.psychologicalInsight = new FuzzyVariable('psychologicalInsight', 0, 1);
    this.psychologicalInsight.addSet('light', [[0, 1], [0.3, 1], [0.5, 0]]);
    this.psychologicalInsight.addSet('profound', [[0.3, 0], [0.7, 1], [1, 1]]);
  }

  setupFuzzyRules() {
    this.rules = [];

    // Rule 1: High archetype intensity → Deep archetypal analysis
    this.rules.push(new FuzzyRule(
      { archetypeIntensity: 'high' },
      { archetypeDepth: 'deep' },
      0.9
    ));

    // Rule 2: High emotional intensity → Profound psychological insight
    this.rules.push(new FuzzyRule(
      { emotionalIntensity: 'intense' },
      { psychologicalInsight: 'profound' },
      0.8
    ));

    // Rule 3: Strong user resonance → Extensive practical guidance
    this.rules.push(new FuzzyRule(
      { userResonance: 'strong' },
      { practicalGuidance: 'extensive' },
      0.85
    ));

    // Rule 4: Low archetype + calm → Surface + extensive practical
    this.rules.push(new FuzzyRule(
      { archetypeIntensity: 'low', emotionalIntensity: 'calm' },
      { archetypeDepth: 'surface', practicalGuidance: 'extensive' },
      0.7
    ));

    // Rule 5: High archetype + intense → Deep + profound
    this.rules.push(new FuzzyRule(
      { archetypeIntensity: 'high', emotionalIntensity: 'intense' },
      { archetypeDepth: 'deep', psychologicalInsight: 'profound' },
      1.0
    ));

    // Additional rules for coverage...
    this.rules.push(new FuzzyRule(
      { archetypeIntensity: 'medium', userResonance: 'strong' },
      { archetypeDepth: 'deep', practicalGuidance: 'extensive' },
      0.75
    ));

    this.rules.push(new FuzzyRule(
      { emotionalIntensity: 'moderate', userResonance: 'weak' },
      { psychologicalInsight: 'light', practicalGuidance: 'minimal' },
      0.6
    ));
  }

  /**
   * Main orchestration method
   * Returns activation levels for each interpretation mode
   */
  computeActivations(cards, positions, userProfile, intention) {
    // Extract features from multiple modalities
    const symbolic = this.symbolicAnalyzer.analyze(cards);
    const relational = this.relationalAnalyzer.analyze(cards, positions);
    const energetic = this.energeticAnalyzer.analyze(cards, userProfile, intention);

    // Fuzzify inputs
    const fuzzyInputs = {
      archetypeIntensity: this.archetypeIntensity.fuzzify(symbolic.archetypeIntensity),
      emotionalIntensity: this.emotionalIntensity.fuzzify(energetic.emotionalIntensity),
      userResonance: this.userResonance.fuzzify(energetic.userResonance)
    };

    // Evaluate all rules
    const ruleActivations = {
      archetypeDepth: {},
      practicalGuidance: {},
      psychologicalInsight: {}
    };

    this.rules.forEach(rule => {
      const activation = rule.evaluate(fuzzyInputs);

      for (const [outputVar, outputSet] of Object.entries(rule.consequents)) {
        if (!ruleActivations[outputVar][outputSet]) {
          ruleActivations[outputVar][outputSet] = [];
        }
        ruleActivations[outputVar][outputSet].push(activation);
      }
    });

    // Defuzzify (max aggregation + centroid)
    const activations = {
      archetypal: this.defuzzify(ruleActivations.archetypeDepth),
      practical: this.defuzzify(ruleActivations.practicalGuidance),
      psychological: this.defuzzify(ruleActivations.psychologicalInsight),
      // Additional modes
      relational: relational.positionCoherence,
      mystical: symbolic.archetypeIntensity * energetic.userResonance
    };

    return {
      activations,
      features: {
        symbolic,
        relational,
        energetic
      }
    };
  }

  defuzzify(outputMemberships) {
    // Max aggregation
    let maxActivation = 0;
    for (const activations of Object.values(outputMemberships)) {
      const max = Math.max(...activations);
      maxActivation = Math.max(maxActivation, max);
    }

    // Simple defuzzification: return max activation
    return Math.min(Math.max(maxActivation, 0.3), 1.0); // Clamp to [0.3, 1.0]
  }
}
