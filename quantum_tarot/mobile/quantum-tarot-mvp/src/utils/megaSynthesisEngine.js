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
import { getCardQuote, CARD_QUOTES_EXPANDED } from '../data/cardQuotes_EXPANDED';
import { analyzeMCQAnswers, getSynthesisGuidance } from './postCardQuestions';
import { getFullAstrologicalContext, getTimeOfDayEnergy } from './advancedAstrology';
import { getMBTIInterpretationGuidelines } from './mbtiTest';
import { generateQuantumNarrative } from './quantumNarrativeEngine';
import { generateQuantumSeed } from './quantumRNG';
import { BalancedWisdomIntegration, getModerationWisdom } from './balancedWisdom';
import { getChineseZodiac, getSpiritualSynthesisMessage, getSpiritualGrowthInsight } from './chineseZodiac';
import { generateTimingPrediction, generateReadingTimeframe, generatePredictiveMarker } from './temporalPredictions';
import { analyzeCardSynergies, generateSynergySummary } from './cardSynergyMatrix';
import { weaveColdReadingElements } from './coldReadingEnhancer';
import { composeNarrativeArc, generateStoryOpener } from './narrativeArcComposer';

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

    // Calculate Chinese zodiac from birth year
    const birthYear = userProfile?.birthday
      ? new Date(userProfile.birthday).getFullYear()
      : new Date().getFullYear() - 25;
    const chineseZodiac = getChineseZodiac(birthYear);
    console.log('🐉 Chinese Zodiac:', chineseZodiac.fullSign);

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

    // 4B. ANALYZE CARD SYNERGIES (card interaction patterns)
    console.log('🔍 Step 4B: Analyzing card synergies...');
    const cardSynergies = analyzeCardSynergies(cards);

    // 4C. COMPOSE NARRATIVE ARC (story structure)
    console.log('🔍 Step 4C: Composing narrative arc...');
    const narrativeArc = composeNarrativeArc(cards, mcqAnalysis, readingType, userProfile, quantumSeed);

    // 4D. GENERATE COLD READING ELEMENTS (personalization)
    console.log('🔍 Step 4D: Generating cold reading elements...');
    const coldReading = weaveColdReadingElements(userProfile, cards, readingType, quantumSeed);

    // 5. BUILD SYNTHESIS
    console.log('🔍 Step 5: Building synthesis...');
    const synthesis = buildSynthesis({
      cards,
      mcqAnswers,
      mcqAnalysis,
      astroContext,
      timeEnergy,
      chineseZodiac,
      mbtiGuidelines,
      synthesisGuidance,
      narrative,
      userProfile,
      intention,
      readingType,
      spreadType,
      quantumSeed,
      cardSynergies,
      narrativeArc,
      coldReading
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
      chineseZodiac,
      mbtiGuidelines,
      synthesisGuidance,
      narrative,
      userProfile,
      intention,
      readingType,
      spreadType,
      quantumSeed,
      cardSynergies = [],
      narrativeArc = {},
      coldReading = {}
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
  // OPENING (150-250 words) - NARRATIVE ARC APPROACH
  // ═══════════════════════════════════════════════════════════

  // Use story-based opener if we have narrative arc
  const opening = narrativeArc?.structure
    ? generateStoryOpener(userProfile?.name || 'Seeker', readingType, quantumSeed)
    : narrative.getOpening(readingType, userProfile?.name || 'Seeker');

  if (opening) {
    synthesis += `${opening}\n\n`;
  }

  // Add BARNUM STATEMENT early for personalization hook
  if (coldReading?.barnum && coldReading.barnum.length > 0) {
    synthesis += `${coldReading.barnum[0]}\n\n`;
  }

  // Weave in intention with timeframe prediction (QUOTED)
  const quotedIntention = intention ? `"${intention}"` : 'your path forward';
  synthesis += `You came to this reading seeking clarity on ${quotedIntention}. `;

  // Add reading timeframe (TEMPORAL PREDICTION)
  const timeframe = generateReadingTimeframe(cards, astroContext, readingType, quantumSeed);
  if (timeframe) {
    synthesis += `${timeframe} `;
  }

  // Add astrological/temporal context with flow
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

  // Temporal context
  if (timeEnergy?.period && timeEnergy?.energy) {
    synthesis += `It's ${timeEnergy.period.toLowerCase()}, when ${timeEnergy.energy.toLowerCase()}. `;
  }
  if (timeEnergy?.advice) {
    synthesis += `${timeEnergy.advice}`;
  }

  // Paragraph break before zodiac
  synthesis += `\n\n`;

  // Chinese zodiac integration with smoother flow
  if (chineseZodiac) {
    synthesis += `Born in the year of the ${chineseZodiac.fullSign}, you carry the archetype of ${chineseZodiac.traits.archetype}. `;
    if (chineseZodiac.traits.wisdom) {
      synthesis += `${chineseZodiac.traits.wisdom} `;
    }
    if (chineseZodiac.elementInfluence?.lesson) {
      synthesis += `Your ${chineseZodiac.element} element adds another dimension: ${chineseZodiac.elementInfluence.lesson}`;
    }
    synthesis += `\n\n`;
  }

  // Add INTUITIVE HOOK for psychic vibe (with transition)
  if (coldReading?.intuitiveHook) {
    synthesis += `${coldReading.intuitiveHook}\n\n`;
  }

  // HARDENING #9: Meta-Skepticism Integration
  // Acknowledge skepticism and make it part of the reading
  const skepticismStatements = [
    'If part of you thinks this is bullshit, notice that reaction. What are you defending against? Sometimes skepticism is wisdom. Sometimes it\'s fear of being seen.',
    'You might be tempted to dismiss this as confirmation bias. Fair enough. But ask yourself: why does that bother you? What would it mean if this WAS meaningful?',
    'Tarot works whether you believe in it or not—because it\'s a mirror, not a magic 8-ball. What you see in these cards says more about you than them.',
    'If you\'re skeptical, good. Blind faith is dangerous. But so is cynicism that protects you from uncomfortable truths.'
  ];
  const skepticismIdx = Math.floor((quantumSeed * 0.999) * skepticismStatements.length);
  if (Math.random() > 0.5) { // 50% of readings get meta-skepticism acknowledgment
    synthesis += `${skepticismStatements[skepticismIdx]}\n\n`;
  }

  // HARDENING #10: Decision Empowerment Language
  // Combat decision avoidance - emphasize agency over passive waiting
  const empowermentStatements = [
    'These cards don\'t tell you what WILL happen—they show you what\'s possible if you choose it. The universe doesn\'t decide for you. You decide, and the universe responds.',
    'Reading tarot isn\'t about predicting a fixed future. It\'s about seeing your options clearly so you can make better choices NOW. Your agency is the point, not a variable.',
    'Don\'t use this reading as permission to wait. Use it as intel to act. The cards show terrain, but you choose the path. Sitting still IS a choice—just usually the wrong one.',
    'If you came here hoping the cards would make the decision for you, they won\'t. They\'ll show you what you already know but haven\'t admitted. Then YOU choose.',
    'The future isn\'t written. These cards show momentum and probability based on your current trajectory. Don\'t like where it\'s headed? Change direction. That power is yours.'
  ];
  const empowermentIdx = Math.floor((quantumSeed * 0.888) * empowermentStatements.length);
  if (Math.random() > 0.4) { // 60% of readings get decision empowerment
    synthesis += `${empowermentStatements[empowermentIdx]}\n\n`;
  }

  // HARDENING #8: Crisis Detection & Ethical Redirection
  // If crisis signals detected, provide supportive resources BEFORE the reading
  if (mcqAnalysis?.crisisSignals?.detected) {
    const severity = mcqAnalysis.crisisSignals.severity;

    if (severity === 'severe' || severity === 'moderate') {
      synthesis += `**A moment of care**: I'm sensing you're carrying a heavy weight right now. `;

      if (severity === 'severe') {
        synthesis += `Before we continue, I want to acknowledge that tarot can offer perspective, but it's not a substitute for human support when you're in crisis.\n\n`;
        synthesis += `**If you're in immediate danger or experiencing thoughts of self-harm**:\n`;
        synthesis += `- **Crisis Text Line**: Text HOME to 741741 (US)\n`;
        synthesis += `- **National Suicide Prevention Lifeline**: 988 (US)\n`;
        synthesis += `- **International Association for Suicide Prevention**: https://www.iasp.info/resources/Crisis_Centres/\n\n`;
        synthesis += `**For non-emergency support**:\n`;
        synthesis += `- **SAMHSA National Helpline**: 1-800-662-4357 (mental health/substance abuse, 24/7, free, confidential)\n`;
        synthesis += `- **Therapy directories**: Psychology Today, BetterHelp, OpenPath Collective\n\n`;
        synthesis += `The cards below may still offer insight, but please—reach out to someone trained to help. You deserve real, human support.\n\n`;
      } else if (severity === 'moderate') {
        synthesis += `Tarot can illuminate, but it can't replace therapy, medical care, or trusted relationships. If you're struggling more than usual, consider reaching out:\n\n`;
        synthesis += `- **SAMHSA National Helpline**: 1-800-662-4357 (free, confidential, 24/7)\n`;
        synthesis += `- **Crisis Text Line**: Text HOME to 741741\n`;
        synthesis += `- Find a therapist: Psychology Today directory, OpenPath Collective (affordable options)\n\n`;
        synthesis += `The reading continues below, but please honor if you need more than cards right now.\n\n`;
      }
    } else if (severity === 'mild') {
      synthesis += `I'm noticing some tension or heaviness in your responses. That's okay—hard seasons happen. If this reading brings up more than you're ready for, it's okay to step away and return when you're in a better place. Self-care isn't selfish.\n\n`;
    }
  }

  // ═══════════════════════════════════════════════════════════
  // CARD-BY-CARD INTERPRETATION (300-600 words) - WITH NARRATIVE ARC
  // ═══════════════════════════════════════════════════════════

  // Transition to cards section
  const cardTransitions = [
    'Now, let\'s look at what the cards themselves are saying.',
    'The cards laid out before you tell a specific story.',
    'Here\'s where we get into the heart of your reading.',
    'Let me walk you through what each card reveals.'
  ];
  const transitionIdx = Math.floor((quantumSeed * 0.999) * cardTransitions.length);
  synthesis += `${cardTransitions[transitionIdx]}\n\n`;

  // Add CARD SYNERGY ANALYSIS first (shows we read interactions, not just individual cards)
  const synergySummary = generateSynergySummary(cardSynergies, quantumSeed * 0.111);
  if (synergySummary) {
    synthesis += synergySummary;
  }

  // Analyze overall reading patterns
  const patterns = analyzeReadingPatterns(cards);
  if (patterns.length > 0) {
    synthesis += `Before we dive into individual cards, notice this: ${patterns.join(' ')} This sets the stage for everything that follows.\n\n`;
  }

  // Add narrative arc climax if we're in a 3+ card reading
  if (narrativeArc?.climax && cards.length >= 3) {
    const climaxIndex = Math.floor(cards.length / 2);
    // We'll add this before the climax card
  }

  // Interpret each card with quantum variation - FOCUS ON MCQ INSIGHTS
  cards.forEach((card, index) => {
    const cardData = CARD_DATABASE[card.cardIndex];
    const keywords = card.reversed ? cardData.keywords?.reversed : cardData.keywords?.upright;
    const primaryKeyword = keywords?.[0] || 'transformation';

    // Add CHAPTER HEADING for narrative arc
    if (narrativeArc?.chapterHeadings && narrativeArc.chapterHeadings[index]) {
      synthesis += `\n## ${narrativeArc.chapterHeadings[index]}\n\n`;
    }

    // Add CLIMAX marker before middle card
    const climaxIndex = Math.floor(cards.length / 2);
    if (index === climaxIndex && narrativeArc?.climax) {
      synthesis += narrativeArc.climax;
    }

    // Add transition (except for first card)
    if (index > 0) {
      const transition = narrative.getTransition();
      if (transition) {
        synthesis += `${transition} `;
      }
    }

    // Main card interpretation WITH STRUCTURAL VARIATION (Hardening #1)
    const cardName = `${cardData?.name || 'Unknown Card'}${card.reversed ? ' Reversed' : ''}`;
    const position = card.position || `position ${index + 1}`;
    const positionMeaning = card.positionMeaning || '';

    // Generate context-aware sentence
    const sentence = narrative.getSentence(
      cardName,
      `${primaryKeyword} in the realm of ${positionMeaning || position}`,
      position
    );

    // STRUCTURAL VARIATION: 10+ different ways to introduce card
    const structureVariations = [
      // Standard format (20% of time)
      `**${cardName}** in ${positionMeaning || position}: ${sentence} `,

      // Position-first formats
      `In ${positionMeaning || position}, **${cardName}** emerges. ${sentence} `,
      `The ${positionMeaning || position} position holds **${cardName}**. ${sentence} `,
      `${positionMeaning || position}: Here we find **${cardName}**. ${sentence} `,

      // Card-first, no position header
      `**${cardName}** speaks to ${positionMeaning || position}. ${sentence} `,
      `**${cardName}**. ${sentence} This is the energy of ${positionMeaning || position}. `,

      // Direct dive (no header at all)
      `${sentence} This is **${cardName}** in ${positionMeaning || position}. `,
      sentence + ` **${cardName}** has appeared in ${positionMeaning || position}. `,

      // Conversational
      `Let's talk about **${cardName}** in ${positionMeaning || position}. ${sentence} `,
      `**${cardName}** showed up in ${positionMeaning || position}—and that's significant. ${sentence} `,

      // Mystical/dramatic
      `The cards reveal **${cardName}** in ${positionMeaning || position}. ${sentence} `,
      `**${cardName}**. ${positionMeaning || position}. ${sentence} `
    ];

    const structureIdx = Math.floor((quantumSeed * (index + 1) * 0.789) * structureVariations.length) % structureVariations.length;
    if (sentence) {
      synthesis += structureVariations[structureIdx];
    }

    // Add TIMING PREDICTION for this card
    const timingPrediction = generateTimingPrediction(card, astroContext, readingType, quantumSeed * (index + 1));
    if (timingPrediction && index < cards.length - 1) {
      synthesis += `${timingPrediction} `;
    }

    // PRIORITIZE MCQ INSIGHTS - This is where the real personalization happens
    const cardMCQAnswers = mcqAnswers.filter(a => a.cardIndex === index);
    if (cardMCQAnswers.length > 0) {
      const mcqInsight = weaveMCQInsights(cardMCQAnswers, cardData, narrative, mcqAnalysis, readingType);
      if (mcqInsight) {
        synthesis += mcqInsight;
      }
    } else {
      // If no MCQ, add brief elemental context
      synthesis += weaveCardLayers(cardData, card.reversed, narrative);
    }

    // Only add pop culture quote for first card or when MCQ shows high resonance
    const hasHighResonance = cardMCQAnswers.some(a =>
      a.questionType === 'resonance' && a.selectedOptionIndex >= 3
    );
    const shouldAddQuote = index === 0 || hasHighResonance;
    if (shouldAddQuote) {
      const sentenceSeed = narrative?.sentenceSeeds?.[index] || Math.random();
      const quote = getCardQuote(card.cardIndex, sentenceSeed);
      if (quote?.text && quote?.source) {
        synthesis += `\n\n*"${quote.text}"* — ${quote.source}\n\n`;
      }
    }

    synthesis += `\n\n`;
  });

  // ═══════════════════════════════════════════════════════════
  // PATTERN SYNTHESIS (200-400 words)
  // ═══════════════════════════════════════════════════════════

  // Transition to deeper analysis
  const patternTransitions = [
    `Now let's ${narrative.getWord('examine')} how these cards speak to each other and what patterns emerge.`,
    'Individual cards tell part of the story. The spaces between them tell the rest.',
    'The cards you drew aren\'t random. They form a constellation of meaning.',
    'Beyond the surface reading, there are deeper currents at work here.'
  ];
  const patternTransIdx = Math.floor((quantumSeed * 0.888) * patternTransitions.length);
  synthesis += `\n\n${patternTransitions[patternTransIdx]}\n\n`;

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

  // Transition to spiritual synthesis
  const spiritualTransitions = [
    '\n\nThis brings me to something bigger—the spiritual landscape you\'re navigating.',
    '\n\nLet\'s zoom out to the spiritual framework underpinning all of this.',
    '\n\nNow, about the spiritual territory you\'re crossing.',
    '\n\nThere\'s a meta-level worth addressing here.'
  ];
  const spiritTransIdx = Math.floor((quantumSeed * 0.777) * spiritualTransitions.length);

  // Spiritual synthesis - emphasizing that all traditions contain partial truths
  if (chineseZodiac && astroContext?.sunSign) {
    synthesis += spiritualTransitions[spiritTransIdx];

    const spiritualMessage = getSpiritualSynthesisMessage(chineseZodiac, astroContext.sunSign);
    if (spiritualMessage) {
      synthesis += `\n\n${spiritualMessage}\n\n`;
    }

    // Add spiritual growth insight
    const growthInsight = getSpiritualGrowthInsight(chineseZodiac);
    if (growthInsight) {
      synthesis += `${growthInsight}\n\n`;
    }
  }

  // Add SENSORY DETAILS for emotional hooks and specificity
  if (coldReading?.sensory && coldReading.sensory.length > 0) {
    const sensoryDetail = coldReading.sensory[0]; // Color detail
    if (sensoryDetail?.text) {
      synthesis += `${sensoryDetail.text} `;
    }
  }

  // Add PREDICTIVE MARKER (what to watch for)
  const predictiveMarker = generatePredictiveMarker(cards[0], readingType, quantumSeed * 0.444);
  if (predictiveMarker) {
    synthesis += `${predictiveMarker}\n\n`;
  }

  // Moderation wisdom (Middle Way principles)
  const moderationSeed = (quantumSeed * 0.9876) % 1;
  const moderationWisdom = getModerationWisdom(moderationSeed);
  if (moderationWisdom) {
    synthesis += `\nA word on balance: ${moderationWisdom} `;
    synthesis += `The cards aren't asking for perfection or extremism—they're inviting you into the middle way.\n\n`;
  }

  // Add CONFIRMATION PROMPT (yes/no question for engagement)
  if (coldReading?.confirmationPrompt) {
    synthesis += `\nQuick check: ${coldReading.confirmationPrompt}\n\n`;
  }

  // ═══════════════════════════════════════════════════════════
  // MBTI-SPECIFIC GUIDANCE (100-200 words)
  // ═══════════════════════════════════════════════════════════

  // Transition to personalized guidance
  const mbtiTransitions = [
    `\n\nNow, let's talk about how your ${userProfile?.mbtiType || 'personality'} type processes all of this.`,
    `\n\nGiven your ${userProfile?.mbtiType || 'personality type'}, there are specific things you need to know.`,
    `\n\nYour ${userProfile?.mbtiType || 'cognitive wiring'} matters here. Here's why.`,
    `\n\nAs a${userProfile?.mbtiType ? 'n' : ''} ${userProfile?.mbtiType || 'person'}, you'll want to pay attention to this.`
  ];
  const mbtiTransIdx = Math.floor((quantumSeed * 0.666) * mbtiTransitions.length);
  synthesis += mbtiTransitions[mbtiTransIdx];

  const mbtiGuidance = generateMBTIGuidance(mbtiGuidelines, cards, mcqAnalysis, narrative);
  if (mbtiGuidance) {
    synthesis += `\n\n${mbtiGuidance}`;
  }

  // ═══════════════════════════════════════════════════════════
  // ACTION STEPS (100-150 words)
  // ═══════════════════════════════════════════════════════════

  // Transition to actionable steps
  const actionTransitions = [
    '\n\n## What To Do Now',
    '\n\n## Your Next Steps',
    '\n\n## From Insight to Action',
    '\n\n## Making This Real'
  ];
  const actionTransIdx = Math.floor((quantumSeed * 0.555) * actionTransitions.length);
  synthesis += `${actionTransitions[actionTransIdx]}\n\n`;

  const actionSteps = generateActionSteps(cards, mcqAnalysis, synthesisGuidance, readingType, narrative);
  if (actionSteps) {
    synthesis += actionSteps;
  }

  // ═══════════════════════════════════════════════════════════
  // CLOSING (50-100 words) - WITH NARRATIVE DENOUEMENT
  // ═══════════════════════════════════════════════════════════

  // Transition to closing
  const closingTransitions = [
    '\n\n* * *\n\n',
    '\n\n---\n\n',
    '\n\n•••\n\n'
  ];
  const closingTransIdx = Math.floor((quantumSeed * 0.444) * closingTransitions.length);
  synthesis += closingTransitions[closingTransIdx];

  // Add DENOUEMENT (story wisdom)
  if (narrativeArc?.denouement) {
    synthesis += narrativeArc.denouement;
  }

  // Add more SENSORY DETAILS for memorable specificity
  if (coldReading?.sensory && coldReading.sensory.length > 1) {
    const numberDetail = coldReading.sensory.find(d => d.type === 'number');
    const animalDetail = coldReading.sensory.find(d => d.type === 'animal');

    if (numberDetail?.text) {
      synthesis += `${numberDetail.text} `;
    }
    if (animalDetail?.text) {
      synthesis += `${animalDetail.text} `;
    }
    synthesis += '\n\n';
  }

  // Add FLATTERY WITH EDGE (makes user feel seen and called out)
  if (coldReading?.flattery) {
    synthesis += `${coldReading.flattery}\n\n`;
  }

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
    synthesis += `${balancedClosing.moderation}\n\n`;
  }

  if (balancedClosing?.pillar?.wisdom) {
    synthesis += `${balancedClosing.pillar.wisdom}\n\n`;
  }

  // Add second BARNUM statement for closing hook
  if (coldReading?.barnum && coldReading.barnum.length > 1) {
    synthesis += `${coldReading.barnum[1]}\n\n`;
  }

  // Final closing statement
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
 * Weave MCQ insights into interpretation - DEEP ANALYSIS
 * @param {Array} cardMCQAnswers - Array of MCQ answers for this specific card
 */
function weaveMCQInsights(cardMCQAnswers, cardData, narrative, mcqAnalysis, readingType) {
  let text = '';

  if (!cardMCQAnswers || cardMCQAnswers.length === 0) {
    return text;
  }

  // Look for cognitive dissonance patterns in user's ACTUAL responses
  cardMCQAnswers.forEach(answer => {
    const questionType = answer.questionType;
    const selected = answer.selectedOption;

    if (questionType === 'resonance') {
      const resonanceLevel = answer.selectedOptionIndex + 1; // 1-5 scale
      if (resonanceLevel <= 2) {
        text += `You felt disconnect from this card. In ${readingType || 'this area'}, that gap between card and reaction reveals where you might be avoiding something. `;
      } else if (resonanceLevel >= 4) {
        text += `This card struck a chord. That visceral response is your psyche signaling: "Pay attention here." `;
      }
    }

    if (questionType === 'emotion') {
      const emotion = typeof selected === 'string' ? selected : selected?.text;
      if (emotion?.toLowerCase().includes('resist')) {
        text += `Your resistance to this card's message is the portal. Shadow work begins exactly where discomfort lives. `;
      } else if (emotion?.toLowerCase().includes('excit') || emotion?.toLowerCase().includes('valid')) {
        text += `You felt validated by this card. That alignment between inner knowing and external reflection? That's confirmation you're on track. `;
      } else if (emotion?.toLowerCase().includes('confus')) {
        text += `The confusion you felt suggests cognitive dissonance between what you think you want and what your deeper self needs. `;
      }
    }

    if (questionType === 'action') {
      const actionText = typeof selected === 'string' ? selected : selected?.text;
      if (actionText?.toLowerCase().includes('immediate') || actionText?.toLowerCase().includes('ready')) {
        text += `You're ready to move. That readiness is crucial—don't let overthinking steal your momentum. `;
      } else if (actionText?.toLowerCase().includes('not ready') || actionText?.toLowerCase().includes('overwhelm')) {
        text += `You signaled you're not ready to act yet. Honoring that hesitation is wise. Integration takes time. `;
      }
    }

    if (questionType === 'situation') {
      const situationText = typeof selected === 'string' ? selected : selected?.text;
      // Connect the card to their specific life situation
      text += `You connected this to ${situationText?.toLowerCase() || 'a specific situation'}. `;
      text += `The card isn't speaking abstractly—it's addressing this exact dynamic in your life right now. `;
    }
  });

  // Add MBTI-specific insight based on mcqAnalysis patterns
  if (mcqAnalysis?.dominantEmotions?.length > 0) {
    const blockedEmotions = mcqAnalysis.dominantEmotions.filter(e =>
      e.energy === 'blocked' || e.energy === 'contracted'
    );
    if (blockedEmotions.length > 0) {
      text += `Your pattern across cards shows emotional contraction. This card is asking you to soften, to let feeling move through you. `;
    }
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
    text += `You're ready to ${narrative.getWord('act')}. Don't wait for permission—here's how:\n\n`;
    text += `**Guiding principle**: ${actionGuidance.wisdom}\n\n`;
    text += `1. **TODAY**: ${getImmediateAction(cards[0], readingType, narrative)}\n`;
    text += `2. **THIS WEEK**: ${getWeekAction(cards, readingType, narrative)}\n`;
    text += `3. **THIS MONTH**: ${getMonthAction(cards, readingType, narrative)}\n\n`;
    text += `Remember: sustainable effort beats burnout. Marathon pace, not sprint pace. But START.\n`;
  } else if (actionLevel === 'low') {
    text += `You're not ready to ${narrative.getWord('act')} yet. That's okay—but don't confuse processing with procrastination:\n\n`;
    text += `**Guiding principle**: ${actionGuidance.wisdom}\n\n`;
    text += `1. **Journal**: ${getJournalingPrompt(cards, narrative)}\n`;
    text += `2. **Reflect**: ${getReflectionPrompt(cards, narrative)}\n`;
    text += `3. **Then decide**: ${getEventualAction(cards, readingType, narrative)}\n\n`;
    text += `Clarity comes before action. But set a deadline for your clarity—don't hide there forever.\n`;
  } else {
    text += `Balance reflection with action—but don't let balance become an excuse for neither:\n\n`;
    text += `**Guiding principle**: ${actionGuidance.wisdom}\n\n`;
    text += `1. **Reflect**: ${getReflectionPrompt(cards, narrative)}\n`;
    text += `2. **Act**: ${getImmediateAction(cards[0], readingType, narrative)}\n`;
    text += `3. **Integrate**: ${getIntegrationAction(cards, narrative)}\n\n`;
    text += `The middle way isn't paralysis. It's thoughtful movement. Choose a direction and GO.\n`;
  }

  return text;
}

// Helper functions for action steps
function getImmediateAction(card, readingType, narrative) {
  const cardData = CARD_DATABASE[card.cardIndex];
  return `${narrative.getWord('Act')} on ${cardData.keywords?.upright?.[0] || 'this energy'} in your ${readingType} life. Small step. Today.`;
}

function getWeekAction(cards, readingType, narrative) {
  return `Have the conversation, make the decision, or take the risk these cards are pointing toward. Stop rehearsing—execute.`;
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
  return `Decide when you'll ${narrative.getWord('act')} on ${cardData.keywords?.upright?.[0] || 'the final card'}. Put it on the calendar. Commit.`;
}

function getIntegrationAction(cards, narrative) {
  return `Check back in a week. Notice what shifted. Don't just observe—course-correct. You're the captain, not a passenger.`;
}
