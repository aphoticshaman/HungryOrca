/**
 * ENHANCED LUNATIQ ENGINE
 * Integration of Card Database + Query Engine with LunatiQ AGI
 *
 * Adds meta-analysis capabilities to interpretation:
 * - Spread-level pattern detection
 * - Elemental/suit/reversal analysis
 * - Cross-card theme recognition
 * - Database-driven symbolism
 */

import { CardQueryEngine } from './cardQueryEngine';
import { getCardData } from '../data/cardDatabase';
import { LunatiQOrchestrator } from './fuzzyOrchestrator';
import {
  ArchetypalAgent,
  PracticalAgent,
  PsychologicalAgent,
  RelationalAgent,
  MysticalAgent
} from './interpretationAgents';

/**
 * Enhanced Interpretation Generator
 * Uses card database + query engine for richer interpretations
 */
export class EnhancedLunatiQEngine {
  constructor() {
    this.orchestrator = new LunatiQOrchestrator();
    this.agents = {
      archetypal: new ArchetypalAgent(),
      practical: new PracticalAgent(),
      psychological: new PsychologicalAgent(),
      relational: new RelationalAgent(),
      mystical: new MysticalAgent()
    };
  }

  /**
   * Generate interpretation with database enhancement
   */
  generateCardInterpretation(card, position, reversed, commProfile, readingType, spreadContext) {
    // Get rich card data from database
    const cardData = getCardData(card.index);
    if (!cardData) {
      console.warn(`Card ${card.index} not found in database, using basic data`);
      return this.fallbackInterpretation(card, position, reversed);
    }

    // Build enhanced context
    const enhancedContext = {
      card: cardData,
      position,
      reversed,
      readingType,
      spreadContext, // Other cards in spread
      keywords: reversed ? cardData.keywords.reversed : cardData.keywords.upright,
      symbols: cardData.symbols,
      archetypes: cardData.archetypes,
      themes: cardData.themes,
      element: cardData.element,
      astrology: cardData.astrology,
      numerology: cardData.numerology,
      chakra: cardData.chakra
    };

    // Generate interpretation using agents (existing logic)
    const interpretation = this.generateWithAgents(enhancedContext, commProfile);

    // Add database-driven insights
    return this.enrichWithDatabase(interpretation, cardData, reversed);
  }

  /**
   * Generate spread-level meta-analysis
   */
  generateSpreadMetaAnalysis(spread, commProfile) {
    // Create query engine
    const queryEngine = new CardQueryEngine(spread);
    const meta = queryEngine.getMetaAnalysis();
    const summary = queryEngine.getSummary();

    // Build synthesis paragraph
    const synthesis = this.synthesizeMetaAnalysis(meta, summary, commProfile);

    return {
      meta,
      summary,
      synthesis,
      patterns: this.detectAdvancedPatterns(queryEngine, spread)
    };
  }

  /**
   * Detect advanced patterns across spread
   */
  detectAdvancedPatterns(queryEngine, spread) {
    const patterns = [];

    // Elemental flow (e.g., fire -> water -> earth = passion cooled into action)
    const elementSequence = spread.map(pos => {
      const cardData = getCardData(pos.cardIndex);
      return cardData ? cardData.element : null;
    }).filter(e => e !== null);

    if (elementSequence.length >= 3) {
      patterns.push({
        type: 'elemental_flow',
        sequence: elementSequence,
        meaning: this.interpretElementalFlow(elementSequence)
      });
    }

    // Numerological progression (e.g., 1-2-3 = building energy)
    const numSequence = spread.map(pos => {
      const cardData = getCardData(pos.cardIndex);
      return cardData ? cardData.numerology : null;
    }).filter(n => n !== null);

    if (this.isProgression(numSequence)) {
      patterns.push({
        type: 'numerological_progression',
        sequence: numSequence,
        meaning: "Sequential numbers suggest natural progression and building energy"
      });
    }

    // Archetypal repetition
    const commonArchetypes = queryEngine.getCommonArchetypes();
    if (commonArchetypes.length > 0) {
      patterns.push({
        type: 'archetypal_theme',
        archetypes: commonArchetypes,
        meaning: this.interpretArchetypalPattern(commonArchetypes)
      });
    }

    // Chakra alignment
    const chakras = queryEngine.getChakraBreakdown();
    const chakraPattern = this.analyzeChakraPattern(chakras);
    if (chakraPattern) {
      patterns.push({
        type: 'chakra_focus',
        ...chakraPattern
      });
    }

    // Reversal pattern (all reversed vs mixed)
    const reversalSig = queryEngine.getReversalSignificance();
    if (reversalSig !== 'flow_state') {
      patterns.push({
        type: 'reversal_pattern',
        significance: reversalSig,
        meaning: this.interpretReversalPattern(reversalSig)
      });
    }

    return patterns;
  }

  /**
   * Synthesize meta-analysis into human-readable narrative
   */
  synthesizeMetaAnalysis(meta, summary, commProfile) {
    const voice = commProfile.primaryVoice;
    const insights = [];

    // Opening based on voice
    const openings = {
      analytical_guide: "Analysis of the spread reveals:",
      intuitive_mystic: "The cards weave together a tapestry showing:",
      supportive_friend: "Looking at your spread as a whole, here's what I see:",
      direct_coach: "Big picture:",
      gentle_nurturer: "Sweetheart, when we step back and look at everything together:",
      wise_mentor: "Consider the spread as a teacher offering these lessons:",
      playful_explorer: "Ooh, check out what's happening across all the cards!",
      balanced_sage: "The spread as a whole speaks to:"
    };

    insights.push(openings[voice] || openings.balanced_sage);

    // Add summary insights
    summary.forEach(insight => insights.push(insight));

    // Add advanced pattern insights
    const dominant = meta.dominantElement;
    const arcana = meta.arcanaSignificance;

    if (arcana === 'fated_spiritual') {
      insights.push("The universe has something important to show you here.");
    }

    // Combine into narrative
    return insights.join(' ');
  }

  /**
   * Helper: Interpret elemental flow
   */
  interpretElementalFlow(sequence) {
    // Map sequences to meanings
    const flows = {
      'fire_water': "Passion cooling into emotion",
      'water_earth': "Feelings manifesting into reality",
      'earth_air': "Practical foundation leading to new ideas",
      'air_fire': "Thoughts igniting into action",
      'fire_earth': "Passionate action becoming tangible",
      'water_air': "Emotions transforming through understanding"
    };

    // Check first two elements
    const key = `${sequence[0]}_${sequence[1]}`;
    return flows[key] || "Elemental transformation is occurring";
  }

  /**
   * Helper: Check if sequence is progression
   */
  isProgression(nums) {
    for (let i = 1; i < nums.length; i++) {
      if (nums[i] !== nums[i-1] + 1) return false;
    }
    return true;
  }

  /**
   * Helper: Interpret archetypal pattern
   */
  interpretArchetypalPattern(archetypes) {
    const top = archetypes[0];
    const archetypeMeanings = {
      innocent: "innocence and new beginnings",
      wanderer: "seeking and exploration",
      creator: "manifestation and building",
      mother: "nurturing and abundance",
      father: "structure and authority",
      trickster: "change and adaptation"
    };

    return `The ${top.archetype} archetype repeats, emphasizing ${archetypeMeanings[top.archetype] || 'this energy'}.`;
  }

  /**
   * Helper: Analyze chakra pattern
   */
  analyzeChakraPattern(chakras) {
    const entries = Object.entries(chakras);
    if (entries.length === 0) return null;

    const dominant = entries.sort((a, b) => b[1] - a[1])[0];
    const chakraMeanings = {
      root: "survival, security, groundedness",
      sacral: "creativity, sexuality, pleasure",
      solar_plexus: "power, will, identity",
      heart: "love, compassion, connection",
      throat: "communication, expression, truth",
      third_eye: "intuition, vision, insight",
      crown: "spirituality, transcendence, unity"
    };

    return {
      dominant: dominant[0],
      count: dominant[1],
      meaning: `Energy concentrated in ${dominant[0]} chakra - focus on ${chakraMeanings[dominant[0]]}`
    };
  }

  /**
   * Helper: Interpret reversal pattern
   */
  interpretReversalPattern(significance) {
    const meanings = {
      heavy_blockage: "Many blocks suggest deep shadow work or resistance to change",
      moderate_resistance: "Some internal obstacles to work through",
      some_shadow: "A few shadow elements to integrate"
    };
    return meanings[significance] || "";
  }

  /**
   * Enrich interpretation with database insights
   */
  enrichWithDatabase(interpretation, cardData, reversed) {
    // Add symbols if highly symbolic card
    if (cardData.symbols.length > 3) {
      const symbolStr = cardData.symbols.slice(0, 3).join(', ');
      interpretation += ` Key symbols: ${symbolStr}.`;
    }

    // Add shadow/light if reversed
    if (reversed && cardData.shadow) {
      interpretation += ` In reversal, watch for: ${cardData.shadow}.`;
    }

    return interpretation;
  }

  /**
   * Fallback if card not in database
   */
  fallbackInterpretation(card, position, reversed) {
    return `${card.name}${reversed ? ' (reversed)' : ''} appears in the ${position} position, offering its unique guidance.`;
  }

  /**
   * Generate with agents (delegates to existing agent system)
   */
  generateWithAgents(context, commProfile) {
    // This would integrate with existing agent code
    // Placeholder for now
    const { card, position, keywords } = context;
    const keywordStr = keywords.slice(0, 3).join(', ');
    return `${card.name} in ${position} suggests ${keywordStr}`;
  }
}

/**
 * Example usage
 */
export function testEnhancedLunatiQ() {
  const engine = new EnhancedLunatiQEngine();

  const mockSpread = [
    { cardIndex: 0, reversed: false, position: 'Past' },
    { cardIndex: 1, reversed: false, position: 'Present' },
    { cardIndex: 2, reversed: true, position: 'Future' }
  ];

  const mockCommProfile = { primaryVoice: 'balanced_sage' };

  const meta = engine.generateSpreadMetaAnalysis(mockSpread, mockCommProfile);
  console.log('Meta-analysis:', meta);
}
