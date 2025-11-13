/**
 * LUNATIQ AGI ENGINE - Multi-layer tarot interpretation
 * Offline AGI that adapts to personality profile + astrological context
 */

import { CARD_DATABASE } from '../data/cardDatabase';
import { getAstrologicalContext } from './astrology';

/**
 * Generate interpretation for a single card
 * @param {Object} card - Card data { cardIndex, reversed, position }
 * @param {string} intention - User's intention
 * @param {string} readingType - Type of reading
 * @param {Object} context - Additional context (zodiacSign, birthdate, etc.)
 * @returns {Object} - Interpretation layers
 */
export function interpretCard(card, intention, readingType, context = {}) {
  const cardData = CARD_DATABASE[card.cardIndex] || CARD_DATABASE[0];
  const { reversed, position } = card;

  // LAYER 1: ARCHETYPAL - Universal symbolic meaning
  const archetypal = generateArchetypalLayer(cardData, reversed, position);

  // LAYER 2: CONTEXTUAL - Adapted to reading type and intention
  const contextual = generateContextualLayer(cardData, reversed, position, readingType, intention);

  // LAYER 3: PSYCHOLOGICAL - Shadow work and deeper insights
  const psychological = generatePsychologicalLayer(cardData, reversed, position, context);

  // LAYER 4: PRACTICAL - Actionable guidance
  const practical = generatePracticalLayer(cardData, reversed, position, readingType, intention);

  // LAYER 5: SYNTHESIS - Integrated interpretation
  const synthesis = generateSynthesis(archetypal, contextual, psychological, practical);

  return {
    cardData,
    position,
    reversed,
    layers: {
      archetypal,
      contextual,
      psychological,
      practical,
      synthesis
    }
  };
}

/**
 * LAYER 1: ARCHETYPAL - Universal patterns and symbols
 */
function generateArchetypalLayer(cardData, reversed, position) {
  const orientation = reversed ? 'reversed' : 'upright';
  const keywords = cardData.keywords?.[orientation] || [];

  return {
    name: cardData.name,
    arcana: cardData.arcana,
    element: cardData.element,
    numerology: cardData.numerology,
    symbols: cardData.symbols || [],
    keywords,
    core_meaning: cardData.description || 'No description available',
    shadow_aspect: reversed ? cardData.shadow_work : null
  };
}

/**
 * LAYER 2: CONTEXTUAL - Adapted to reading type and intention
 */
function generateContextualLayer(cardData, reversed, position, readingType, intention) {
  // Map reading type to interpretation focus
  const focusMap = {
    career: 'professional growth, ambition, work-life balance',
    romance: 'emotional connection, intimacy, relationship dynamics',
    wellness: 'physical health, mental wellbeing, energy management',
    finance: 'resources, abundance mindset, material security',
    personal_growth: 'self-awareness, transformation, inner wisdom',
    decision: 'choices, pathways, consequences',
    general: 'overall life guidance, universal wisdom',
    shadow_work: 'unconscious patterns, hidden aspects, healing'
  };

  const focus = focusMap[readingType] || focusMap.general;

  return {
    position_significance: `In the ${position} position, this card speaks to ${focus.toLowerCase()}.`,
    intention_alignment: analyzeIntentionAlignment(cardData, intention, reversed),
    reading_type_focus: focus,
    temporal_aspect: analyzeTemporalAspect(position),
    energy_quality: reversed ? 'blocked, internal, or inverted' : 'flowing, external, or direct'
  };
}

/**
 * LAYER 3: PSYCHOLOGICAL - Deep patterns and shadow work
 */
function generatePsychologicalLayer(cardData, reversed, position, context) {
  return {
    shadow_work: cardData.shadow_work || 'Explore what this card reveals about hidden aspects of yourself.',
    integration_path: cardData.integration || 'Acknowledge and integrate this energy consciously.',
    emotional_resonance: analyzeEmotionalResonance(cardData, reversed),
    zodiac_connection: analyzeZodiacConnection(cardData, context.zodiacSign),
    growth_opportunity: `This card invites you to ${reversed ? 'address resistance or blocks' : 'embrace and embody'} the energy it represents.`
  };
}

/**
 * LAYER 4: PRACTICAL - Concrete actions and guidance
 */
function generatePracticalLayer(cardData, reversed, position, readingType, intention) {
  const advice = cardData.advice || 'Trust your intuition and move forward mindfully.';

  return {
    action_steps: generateActionSteps(cardData, reversed, readingType),
    what_to_focus_on: cardData.keywords?.upright?.[0] || 'Present moment awareness',
    what_to_avoid: reversed ? 'Getting stuck in this pattern' : 'Overextending this energy',
    timing_guidance: generateTimingGuidance(position),
    practical_advice: advice
  };
}

/**
 * LAYER 5: SYNTHESIS - Integrated multi-layer interpretation
 */
function generateSynthesis(archetypal, contextual, psychological, practical) {
  return {
    core_message: `${archetypal.name} speaks to the archetypal pattern of ${archetypal.keywords.slice(0, 3).join(', ')}. ${contextual.position_significance}`,
    integration: `${psychological.integration_path} ${practical.practical_advice}`,
    deeper_insight: psychological.shadow_work,
    next_steps: practical.action_steps.join(' ')
  };
}

// Helper functions

function analyzeIntentionAlignment(cardData, intention, reversed) {
  // Stub: In full implementation, would use NLP to analyze intention-card alignment
  return `This card ${reversed ? 'challenges or complicates' : 'supports and clarifies'} your question about "${intention.substring(0, 50)}..."`;
}

function analyzeTemporalAspect(position) {
  if (position.toLowerCase().includes('past')) return 'This represents influences from your history';
  if (position.toLowerCase().includes('present')) return 'This is active in your current moment';
  if (position.toLowerCase().includes('future')) return 'This energy is emerging or approaching';
  return 'This aspect influences your journey';
}

function analyzeEmotionalResonance(cardData, reversed) {
  const element = cardData.element;
  const resonanceMap = {
    fire: reversed ? 'anger, frustration, burnout' : 'passion, motivation, courage',
    water: reversed ? 'emotional overwhelm, confusion' : 'intuition, empathy, flow',
    air: reversed ? 'mental fog, overthinking' : 'clarity, communication, insight',
    earth: reversed ? 'stagnation, rigidity' : 'stability, growth, abundance'
  };
  return resonanceMap[element] || 'complex emotional landscape';
}

function analyzeZodiacConnection(cardData, zodiacSign) {
  if (!zodiacSign) return null;
  // Stub: Full implementation would have detailed zodiac-tarot correspondences
  return `As a ${zodiacSign}, this card resonates with your natural tendencies.`;
}

function generateActionSteps(cardData, reversed, readingType) {
  // Stub: Full implementation would have detailed action frameworks
  if (reversed) {
    return [
      'Identify where this energy feels blocked.',
      'Journal on what internal resistance is present.',
      'Take one small step to release or transform this pattern.'
    ];
  }
  return [
    `Embrace the energy of ${cardData.name}.`,
    'Take aligned action in your ' + readingType + ' area.',
    'Trust the process and observe what unfolds.'
  ];
}

function generateTimingGuidance(position) {
  if (position.toLowerCase().includes('past')) return 'Reflect on this before moving forward';
  if (position.toLowerCase().includes('present')) return 'Act on this now';
  if (position.toLowerCase().includes('future')) return 'Prepare for this emerging energy';
  return 'Consider this timing in your own rhythm';
}

/**
 * Generate full reading interpretation with astrological context
 * @param {Array} cards - Array of drawn cards
 * @param {string} spreadType - Type of spread
 * @param {string} intention - User's intention
 * @param {Object} context - Reading context (zodiacSign, birthdate, readingType, etc.)
 * @returns {Object} - Full interpretation with astrological data
 */
export function interpretReading(cards, spreadType, intention, context = {}) {
  // Get comprehensive astrological context
  const astroContext = getAstrologicalContext({
    birthdate: context.birthdate,
    zodiacSign: context.zodiacSign
  });

  // Interpret each card with full context
  const interpretations = cards.map(card =>
    interpretCard(card, intention, context.readingType || 'general', {
      ...context,
      astrology: astroContext
    })
  );

  return {
    interpretations,
    astrologicalContext: astroContext,
    spreadType,
    intention,
    summary: generateReadingSummary(interpretations, astroContext)
  };
}

/**
 * Generate overall reading summary
 */
function generateReadingSummary(interpretations, astroContext) {
  const cardNames = interpretations.map(i => i.cardData.name).slice(0, 3).join(', ');

  return {
    cards_drawn: cardNames + (interpretations.length > 3 ? '...' : ''),
    astrological_influence: astroContext.summary,
    moon_phase: astroContext.moonPhase.name,
    mercury_status: astroContext.mercuryRetrograde.isRetrograde ? 'Retrograde' : 'Direct',
    overall_energy: `${astroContext.planetaryInfluences.dominantPlanet} energy dominant`
  };
}
