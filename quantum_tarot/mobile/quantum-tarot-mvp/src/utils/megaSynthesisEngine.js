/**
 * MEGA SYNTHESIS ENGINE
 * Orchestrates ALL context to generate 600-1500 word UNIQUE syntheses
 *
 * Integrates:
 * - Card meanings (archetypal, positional, elemental)
 * - User profile (MBTI, sun sign, birthday)
 * - MCQ answers (cognitive dissonance detection, pattern recognition)
 * - Advanced astrology (Lilith, Chiron, Nodes, moon phase, transits, time of day)
 * - Reading type and intention
 * - Quantum narrative variation (NO repetition)
 *
 * Based on Tina Gong's philosophy:
 * - Narrative storytelling over keywords
 * - Action-oriented, present-focused
 * - Shadow integration (discomfort = growth)
 * - Context-woven, psychologically sophisticated
 */

import { CARD_DATABASE } from '../data/cardDatabase';
import { getCardQuote } from '../data/cardQuotes';
import { analyzeMCQAnswers, getSynthesisGuidance } from './postCardQuestions';
import { getFullAstrologicalContext, getTimeOfDayEnergy } from './advancedAstrology';
import { getMBTIInterpretationGuidelines } from './mbtiTest';
import { generateQuantumNarrative } from './quantumNarrativeEngine';
import { generateQuantumSeed } from './quantumRNG';
import { BalancedWisdomIntegration, getModerationWisdom } from './balancedWisdom';

/**
 * Generate comprehensive synthesis
 * @param {Object} readingData - Complete reading context
 * @returns {String} - 600-1500 word synthesis
 */
export async function generateMegaSynthesis(readingData) {
  try {
    // Validate input
    if (!readingData) {
      console.error('generateMegaSynthesis: readingData is undefined');
      return 'Error: No reading data provided.';
    }

    const {
      cards = [], // Array of { cardIndex, reversed, position, positionMeaning }
      mcqAnswers = [], // All MCQ answers from post-card questions
      userProfile = {}, // { name, birthday, zodiacSign, mbtiType, pronouns }
      intention = 'Personal growth', // User's stated intention
      readingType = 'general', // 'career', 'romance', 'wellness', etc.
      spreadType = 'three_card' // 'celtic_cross', 'three_card', etc.
    } = readingData;

    console.log('📊 Synthesis input validation:', {
      hasCards: cards?.length > 0,
      cardCount: cards?.length,
      hasMCQAnswers: mcqAnswers?.length > 0,
      mcqCount: mcqAnswers?.length,
      hasUserProfile: !!userProfile,
      mbtiType: userProfile?.mbtiType
    });

    // 1. ANALYZE MCQ ANSWERS (with safety checks)
    console.log('🔍 Step 1: Analyzing MCQ answers...');
    const mcqAnalysis = analyzeMCQAnswers(mcqAnswers || []);
    const synthesisGuidance = getSynthesisGuidance(mcqAnalysis, userProfile?.mbtiType || 'INFP');

    // 2. GET ASTROLOGICAL CONTEXT (with safety checks)
    console.log('🔍 Step 2: Getting astrological context...');
    const astroContext = getFullAstrologicalContext(
      userProfile?.birthday || '2000-01-01',
      userProfile?.zodiacSign || 'Aries'
    );
    const timeEnergy = getTimeOfDayEnergy();

    // 3. GET MBTI INTERPRETATION GUIDELINES (with safety checks)
    console.log('🔍 Step 3: Getting MBTI guidelines...');
    const mbtiGuidelines = getMBTIInterpretationGuidelines(userProfile?.mbtiType || 'INFP');

    // 4. GENERATE QUANTUM NARRATIVE FRAMEWORK
    console.log('🔍 Step 4: Generating quantum narrative...');
    const quantumSeed = generateQuantumSeed();
    const narrative = generateQuantumNarrative(cards, {
      userProfile,
      astroContext,
      mcqAnalysis,
      readingType
    }, quantumSeed);

    // 5. BUILD SYNTHESIS
    console.log('🔍 Step 5: Building synthesis...');
    const synthesis = buildSynthesis({
      cards,
      mcqAnswers,
      mcqAnalysis,
      astroContext,
      timeEnergy,
      mbtiGuidelines,
      synthesisGuidance,
      narrative,
      userProfile,
      intention,
      readingType,
      spreadType,
      quantumSeed
    });

    console.log('✅ Synthesis generated successfully, length:', synthesis?.length);
    return synthesis;
  } catch (error) {
    console.error('❌ generateMegaSynthesis ERROR:', error);
    console.error('Error stack:', error.stack);
    console.error('Error name:', error.name);
    console.error('Error message:', error.message);
    throw error; // Re-throw so we can see the actual error
  }
}

/**
 * Build the actual synthesis text
 */
function buildSynthesis(context) {
  try {
    const {
      cards,
      mcqAnswers = [],
      mcqAnalysis,
      astroContext,
      timeEnergy,
      mbtiGuidelines,
      synthesisGuidance,
      narrative,
      userProfile,
      intention,
      readingType,
      spreadType,
      quantumSeed
    } = context;

    console.log('🔨 buildSynthesis starting with:', {
      cardsCount: cards?.length,
      mcqAnswersCount: mcqAnswers?.length,
      hasNarrative: !!narrative,
      hasUserProfile: !!userProfile,
      userName: userProfile?.name
    });

    let synthesis = '';

  // ═══════════════════════════════════════════════════════════
  // OPENING (150-250 words)
  // ═══════════════════════════════════════════════════════════
  const opening = narrative.getOpening(readingType, userProfile?.name || 'Seeker');
  if (opening) {
    synthesis += `${opening}\n\n`;
  }

  // Weave in intention
  synthesis += `You came to this reading seeking clarity on ${intention || 'your path forward'}. `;

  // Add astrological/temporal context
  const astroRef = narrative.getAstroRef({
    sunSign: astroContext?.sunSign,
    mbtiType: userProfile?.mbtiType,
    lilith: astroContext?.lilith,
    chiron: astroContext?.chiron,
    moonPhase: astroContext?.moonPhase
  });
  if (astroRef) {
    synthesis += `${astroRef} `;
  }

  if (timeEnergy?.period && timeEnergy?.energy) {
    synthesis += `It's ${timeEnergy.period.toLowerCase()}, when ${timeEnergy.energy.toLowerCase()}. `;
  }
  if (timeEnergy?.advice) {
    synthesis += `${timeEnergy.advice}\n\n`;
  } else {
    synthesis += `\n\n`;
  }

  // ═══════════════════════════════════════════════════════════
  // CARD-BY-CARD INTERPRETATION (300-600 words)
  // ═══════════════════════════════════════════════════════════

  // Analyze overall reading patterns first
  const patterns = analyzeReadingPatterns(cards);
  if (patterns.length > 0) {
    synthesis += `Before we dive into individual cards, notice this: ${patterns.join(' ')} This sets the stage for everything that follows.\n\n`;
  }

  // Interpret each card with quantum variation
  cards.forEach((card, index) => {
    const cardData = CARD_DATABASE[card.cardIndex];
    const keywords = card.reversed ? cardData.keywords?.reversed : cardData.keywords?.upright;
    const primaryKeyword = keywords?.[0] || 'transformation';

    // Add transition (except for first card)
    if (index > 0) {
      const transition = narrative.getTransition();
      if (transition) {
        synthesis += `${transition} `;
      }
    }

    // Main card interpretation
    const cardName = `${cardData?.name || 'Unknown Card'}${card.reversed ? ' Reversed' : ''}`;
    const position = card.position || `position ${index + 1}`;
    const positionMeaning = card.positionMeaning || '';

    // Add pop culture quote hook (quantum-seeded for variety)
    const sentenceSeed = narrative?.sentenceSeeds?.[index] || Math.random();
    const quote = getCardQuote(card.cardIndex, sentenceSeed);
    if (quote?.text && quote?.source) {
      synthesis += `\n\n**"${quote.text}"**  \n—${quote.source}\n\n`;

      // Reference the quote in the interpretation
      const quoteIntegrations = [
        `This wisdom speaks directly to ${cardName}'s message. `,
        `Let that sink in as we explore ${cardName}. `,
        `Keep those words close as ${cardName} unfolds its meaning. `,
        `That quote? That's ${cardName} speaking through culture. `,
        `${cardName} echoes this sentiment: `,
        `These words capture the essence of ${cardName} perfectly. `,
        `As ${cardName} reveals itself, remember: `,
      ];
      const quoteIntegrationIndex = Math.floor(
        (sentenceSeed * quoteIntegrations.length) % quoteIntegrations.length
      );
      synthesis += quoteIntegrations[quoteIntegrationIndex];
    }

    // Generate quantum-varied sentence
    const sentence = narrative.getSentence(
      cardName,
      `${primaryKeyword} in the realm of ${positionMeaning || position}`,
      position
    );
    if (sentence) {
      synthesis += `${sentence} `;
    }

    // Add depth based on card-specific MCQ answers
    const cardMCQ = mcqAnswers.find(a => a.cardIndex === index);
    if (cardMCQ) {
      synthesis += weaveMCQInsights(cardMCQ, cardData, narrative);
    }

    // Add elemental/archetypal layer
    synthesis += weaveCardLayers(cardData, card.reversed, narrative);

    // Add balanced wisdom pillar (every 2-3 cards to avoid repetition)
    if (index % 3 === 0 || cards.length <= 3) {
      const pillarGuidance = BalancedWisdomIntegration.getCardPillar(
        card.cardIndex,
        primaryKeyword,
        context.narrative.sentenceSeeds[index * 2] || generateQuantumSeed()
      );
      if (pillarGuidance) {
        synthesis += `${pillarGuidance.wisdom} `;
      }
    }

    synthesis += `\n\n`;
  });

  // ═══════════════════════════════════════════════════════════
  // PATTERN SYNTHESIS (200-400 words)
  // ═══════════════════════════════════════════════════════════

  synthesis += `Now let's ${narrative.getWord('examine')} how these cards speak to each other.\n\n`;

  // Cognitive dissonance detection
  if (mcqAnalysis.overallResonance < 2.5) {
    synthesis += detectCognitiveDissonance(mcqAnalysis, cards, narrative, astroContext);
  }

  // Shadow work opportunities (Lilith integration)
  if (astroContext.lilith) {
    synthesis += integrateLilithShadow(astroContext.lilith, cards, mcqAnalysis, narrative);
  }

  // Chiron wound activation
  if (astroContext.chiron) {
    synthesis += integrateChironHealing(astroContext.chiron, cards, readingType, narrative);
  }

  // North Node evolutionary pull
  if (astroContext.northNode) {
    synthesis += integrateNodalAxis(astroContext.northNode, astroContext.southNode, cards, narrative);
  }

  // Moon phase timing
  if (astroContext.moonPhase) {
    synthesis += integrateMoonPhase(astroContext.moonPhase, mcqAnalysis, narrative);
  }

  // Active transits
  if (astroContext.currentTransits) {
    synthesis += integrateTransits(astroContext.currentTransits, cards, readingType, narrative);
  }

  // Moderation wisdom (Middle Way principles)
  const moderationSeed = (quantumSeed * 0.9876) % 1;
  const moderationWisdom = getModerationWisdom(moderationSeed);
  if (moderationWisdom) {
    synthesis += `\nA word on balance: ${moderationWisdom} `;
    synthesis += `The cards aren't asking for perfection or extremism—they're inviting you into the middle way.\n\n`;
  }

  // ═══════════════════════════════════════════════════════════
  // MBTI-SPECIFIC GUIDANCE (100-200 words)
  // ═══════════════════════════════════════════════════════════

  const examineWord = narrative.getWord('examine') || 'consider';
  synthesis += `\n\nGiven your ${userProfile?.mbtiType || 'personality'}, here's what to ${examineWord}:\n\n`;
  const mbtiGuidance = generateMBTIGuidance(mbtiGuidelines, cards, mcqAnalysis, narrative);
  if (mbtiGuidance) {
    synthesis += mbtiGuidance;
  }

  // ═══════════════════════════════════════════════════════════
  // ACTION STEPS (100-150 words)
  // ═══════════════════════════════════════════════════════════

  synthesis += `\n\n## What To Do Now\n\n`;
  const actionSteps = generateActionSteps(cards, mcqAnalysis, synthesisGuidance, readingType, narrative);
  if (actionSteps) {
    synthesis += actionSteps;
  }

  // ═══════════════════════════════════════════════════════════
  // CLOSING (50-100 words)
  // ═══════════════════════════════════════════════════════════

  // Add balanced wisdom closing
  const dominantElement = getDominantElement(cards);
  const actionReadiness = mcqAnalysis.actionReadiness || 'medium';
  const balancedClosing = BalancedWisdomIntegration.getClosing(
    dominantElement,
    actionReadiness,
    (quantumSeed * 0.111) % 1,
    (quantumSeed * 0.222) % 1
  );

  console.log('🔍 balancedClosing:', {
    hasModeration: !!balancedClosing?.moderation,
    hasPillar: !!balancedClosing?.pillar,
    pillarWisdom: balancedClosing?.pillar?.wisdom
  });

  if (balancedClosing?.moderation) {
    synthesis += `\n\n${balancedClosing.moderation}\n\n`;
  }

  if (balancedClosing?.pillar?.wisdom) {
    synthesis += `${balancedClosing.pillar.wisdom}\n\n`;
  }

  const closing = narrative.getClosing();
  if (closing) {
    synthesis += `${closing}\n`;
  }

  console.log('✅ buildSynthesis completed successfully, final length:', synthesis?.length);
  return synthesis;
  } catch (error) {
    console.error('❌ buildSynthesis ERROR:', error);
    console.error('Error at:', error.stack);
    throw error;
  }
}

/**
 * Get dominant element from reading
 */
function getDominantElement(cards) {
  const elements = { Fire: 0, Water: 0, Air: 0, Earth: 0 };

  cards.forEach(card => {
    const cardData = CARD_DATABASE[card.cardIndex];
    if (cardData.element) {
      elements[cardData.element]++;
    }
  });

  const dominant = Object.entries(elements).reduce((a, b) =>
    elements[a[0]] > elements[b[0]] ? a : b
  );

  return dominant[0];
}

/**
 * Analyze overall patterns in the reading
 */
function analyzeReadingPatterns(cards) {
  const patterns = [];

  // Suit dominance
  const suits = { wands: 0, cups: 0, swords: 0, pentacles: 0, major: 0 };
  cards.forEach(card => {
    const cardData = CARD_DATABASE[card.cardIndex];
    if (cardData.suit) suits[cardData.suit.toLowerCase()]++;
    else suits.major++;
  });

  const dominantSuit = Object.entries(suits).reduce((a, b) => suits[a[0]] > suits[b[0]] ? a : b);
  if (dominantSuit[1] >= cards.length * 0.4) {
    const suitMeanings = {
      wands: 'Fire energy dominates - this is about passion, willpower, creative action',
      cups: 'Water flows through this reading - emotion, intuition, relationship are central',
      swords: 'Air cuts through - your mind, communication, and thought patterns are key',
      pentacles: 'Earth grounds this - material reality, body, resources demand attention',
      major: 'Major Arcana clusters signal MAJOR life lessons unfolding'
    };
    patterns.push(suitMeanings[dominantSuit[0]]);
  }

  // Reversal ratio
  const reversals = cards.filter(c => c.reversed).length;
  if (reversals >= cards.length * 0.6) {
    patterns.push('Heavy reversals suggest blocked energy or extreme thinking - the universe is asking you to find balance');
  }

  return patterns;
}

/**
 * Weave MCQ insights into interpretation
 */
function weaveMCQInsights(cardMCQ, cardData, narrative) {
  let text = '';

  if (cardMCQ.resonance && cardMCQ.resonance < 2) {
    text += `You rated this card's resonance as low. That disconnection? That's data. `;
    const avoidWord = narrative?.getWord?.('avoid') || 'avoiding';
    text += `What are you ${avoidWord}ing by not seeing yourself here? `;
  } else if (cardMCQ.resonance && cardMCQ.resonance >= 4) {
    const soulWord = narrative?.getWord?.('soul') || 'soul';
    text += `This card hit hard for you. That visceral reaction is your ${soulWord} recognizing itself. `;
  }

  if (cardMCQ.emotion === 'resistance') {
    text += `Your resistance to this card is the doorway. What you resist persists. `;
  } else if (cardMCQ.emotion === 'validation') {
    text += `This card validates what you already knew. Trust that inner knowing. `;
  }

  return text;
}

/**
 * Weave card's elemental/archetypal layers
 */
function weaveCardLayers(cardData, reversed, narrative) {
  let text = '';

  // Archetypal layer
  if (cardData.archetypes && cardData.archetypes.length > 0) {
    const archetype = cardData.archetypes[0];
    text += `The ${archetype} archetype lives in you whether you acknowledge it or not. `;
  }

  // Elemental wisdom
  if (cardData.element) {
    const elementWisdom = {
      Fire: 'This fire asks: where do you need to ${narrative.getWord("act")} with courage?',
      Water: 'These waters ask: what emotions are you ${narrative.getWord("avoid")}ing?',
      Air: 'This air asks: what ${narrative.getWord("truth")} needs to be spoken?',
      Earth: 'This earth asks: what tangible reality needs your attention?'
    };
    text += elementWisdom[cardData.element] || '';
  }

  return text;
}

/**
 * Detect and address cognitive dissonance
 */
function detectCognitiveDissonance(mcqAnalysis, cards, narrative, astroContext) {
  return `Here's the uncomfortable ${narrative.getWord('truth')}: your stated priorities don't match your emotional reactions to these cards. ` +
    `That's cognitive dissonance - the gap between what you tell yourself you want and what you actually ${narrative.getWord('feel')}. ` +
    `Your ${astroContext.lilith.sign} Lilith knows this well: ${astroContext.lilith.meaning.split('.')[0]}. ` +
    `The cards are calling bullshit. Listen to your body's wisdom over your mind's stories.\n\n`;
}

/**
 * Integrate Lilith shadow work
 */
function integrateLilithShadow(lilith, cards, mcqAnalysis, narrative) {
  return `Your Black Moon Lilith in ${lilith.sign} is screaming through this reading. ` +
    `${lilith.meaning} ` +
    `Look at where you dimmed your power, played small, or people-pleased. ` +
    `These cards are permission to ${narrative.getWord('reclaim')} what was taken.\n\n`;
}

/**
 * Integrate Chiron healing
 */
function integrateChironHealing(chiron, cards, readingType, narrative) {
  return `Your Chiron in ${chiron.sign} is activated here. ` +
    `${chiron.meaning} ` +
    `This ${readingType} situation is touching that tender wound. But here's the gift: ` +
    `where you're wounded is where you become the healer. Your pain has purpose.\n\n`;
}

/**
 * Integrate nodal axis (destiny vs comfort zone)
 */
function integrateNodalAxis(northNode, southNode, cards, narrative) {
  return `Your North Node in ${northNode.sign} is calling: ${northNode.meaning} ` +
    `But your South Node in ${southNode.sign} wants to pull you back to old patterns. ` +
    `These cards are showing you where you're defaulting to the South Node. ` +
    `Growth happens when you lean toward the North Node edge, even when it's terrifying.\n\n`;
}

/**
 * Integrate moon phase timing
 */
function integrateMoonPhase(moonPhase, mcqAnalysis, narrative) {
  return `The ${moonPhase.phaseName} isn't coincidental. ${moonPhase.phaseEnergy}. ` +
    `${moonPhase.phaseAdvice} ` +
    `The cosmic timing supports what these cards are asking of you.\n\n`;
}

/**
 * Integrate active transits
 */
function integrateTransits(transits, cards, readingType, narrative) {
  let text = '';

  Object.entries(transits).forEach(([transitName, transit]) => {
    if (transit.active && transit.meaning) {
      text += `${transit.meaning} `;
      text += `This transit is WHY your ${readingType} situation feels so intense right now. `;
    }
  });

  if (text) text += '\n\n';
  return text;
}

/**
 * Generate MBTI-specific guidance
 */
function generateMBTIGuidance(mbtiGuidelines, cards, mcqAnalysis, narrative) {
  let text = '';

  // Emphasize areas
  if (mbtiGuidelines.emphasize) {
    text += `Your strengths lie in ${mbtiGuidelines.emphasize.slice(0, 2).join(' and ')}. `;
    text += `${narrative.getWord('Use')} those. `;
  }

  // Avoid areas (blind spots)
  if (mbtiGuidelines.avoid) {
    text += `Watch out for ${mbtiGuidelines.avoid[0]}—that's often a ${narrative.getWord('shadow')} blind spot for your type. `;
  }

  // Tone guidance
  text += `The ${mbtiGuidelines.tone} approach will serve you best here.`;

  return text;
}

/**
 * Generate concrete action steps (with balanced wisdom integration)
 */
function generateActionSteps(cards, mcqAnalysis, synthesisGuidance, readingType, narrative) {
  let text = '';

  const actionLevel = synthesisGuidance.actionLevel;
  const actionGuidance = BalancedWisdomIntegration.getActionGuidance(
    actionLevel,
    generateQuantumSeed()
  );

  if (actionLevel === 'high') {
    text += `You're ready to ${narrative.getWord('act')}. Here's how:\n\n`;
    text += `**Guiding principle**: ${actionGuidance.wisdom}\n\n`;
    text += `1. **TODAY**: ${getImmediateAction(cards[0], readingType, narrative)}\n`;
    text += `2. **THIS WEEK**: ${getWeekAction(cards, readingType, narrative)}\n`;
    text += `3. **THIS MONTH**: ${getMonthAction(cards, readingType, narrative)}\n\n`;
    text += `Remember: sustainable effort beats burnout. Marathon pace, not sprint pace.\n`;
  } else if (actionLevel === 'low') {
    text += `You're not ready to ${narrative.getWord('act')} yet. That's okay. Process first:\n\n`;
    text += `**Guiding principle**: ${actionGuidance.wisdom}\n\n`;
    text += `1. **Journal**: ${getJournalingPrompt(cards, narrative)}\n`;
    text += `2. **Reflect**: ${getReflectionPrompt(cards, narrative)}\n`;
    text += `3. **When ready**: ${getEventualAction(cards, readingType, narrative)}\n\n`;
    text += `Clarity comes before action. Honor where you are.\n`;
  } else {
    text += `Balance reflection with action:\n\n`;
    text += `**Guiding principle**: ${actionGuidance.wisdom}\n\n`;
    text += `1. **Reflect**: ${getReflectionPrompt(cards, narrative)}\n`;
    text += `2. **Act**: ${getImmediateAction(cards[0], readingType, narrative)}\n`;
    text += `3. **Integrate**: ${getIntegrationAction(cards, narrative)}\n\n`;
    text += `The middle way: neither rushing ahead nor hiding in analysis paralysis.\n`;
  }

  return text;
}

// Helper functions for action steps
function getImmediateAction(card, readingType, narrative) {
  const cardData = CARD_DATABASE[card.cardIndex];
  return `${narrative.getWord('Act')} on ${cardData.keywords?.upright?.[0] || 'this energy'} in your ${readingType} life. Small step. Today.`;
}

function getWeekAction(cards, readingType, narrative) {
  return `Have the conversation, make the decision, or take the risk these cards are pointing toward.`;
}

function getMonthAction(cards, readingType, narrative) {
  return `Build on the foundation. Make this ${narrative.getWord('change')} a pattern, not a one-off.`;
}

function getJournalingPrompt(cards, narrative) {
  return `Write about what ${narrative.getWord('trigger')}ed you in this reading. The charge is the clue.`;
}

function getReflectionPrompt(cards, narrative) {
  return `Sit with the discomfort. Don't fix, solve, or spiritually bypass. Just ${narrative.getWord('feel')} it.`;
}

function getEventualAction(cards, readingType, narrative) {
  const cardData = CARD_DATABASE[cards[cards.length - 1].cardIndex];
  return `When you're ready, ${narrative.getWord('act')} on ${cardData.keywords?.upright?.[0] || 'the final card'}.`;
}

function getIntegrationAction(cards, narrative) {
  return `Check back in a week. Notice what shifted. Adjust course as needed.`;
}
