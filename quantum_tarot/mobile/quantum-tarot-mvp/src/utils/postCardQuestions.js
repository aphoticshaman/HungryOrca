/**
 * POST-CARD MCQ GENERATOR
 * Generates 1-3 dynamic multiple-choice questions after each card draw
 * Uses card context, user intention, and card position to create hyper-relevant questions
 *
 * Flow:
 * Card 1 → MCQs → Card 2 → MCQs → ... → Last Card → MCQs → FINAL SYNTHESIS
 */

import { CARD_DATABASE } from '../data/cardDatabase';

/**
 * Generate 1-3 MCQs for a card that was just drawn
 * @param {Object} card - { cardIndex, reversed, position }
 * @param {string} intention - User's intention
 * @param {string} readingType - Type of reading
 * @param {number} cardNumber - Which card in the spread (1st, 2nd, 3rd, etc.)
 * @param {number} totalCards - Total cards in spread
 * @param {Array} previousAnswers - Answers to previous card MCQs
 * @returns {Array} - Array of 1-3 MCQ objects
 */
export function generatePostCardQuestions(card, intention, readingType, cardNumber, totalCards, previousAnswers = []) {
  const cardData = CARD_DATABASE[card.cardIndex] || CARD_DATABASE[0];
  const { reversed, position } = card;

  const questions = [];
  const numQuestions = cardNumber === 1 ? 3 : (cardNumber === totalCards ? 2 : randomInt(1, 3));

  // CARD 1: Context-setting questions
  if (cardNumber === 1) {
    questions.push(generateResonanceQuestion(cardData, reversed));
    questions.push(generateAspectQuestion(cardData, reversed, position, readingType));
    questions.push(generateEmotionalQuestion(cardData, reversed));
  }
  // MIDDLE CARDS: Refinement questions
  else if (cardNumber < totalCards) {
    questions.push(generateConfirmationQuestion(cardData, reversed, previousAnswers));
    if (numQuestions >= 2) {
      questions.push(generateSituationQuestion(cardData, reversed, intention, readingType));
    }
    if (numQuestions === 3) {
      questions.push(generateActionQuestion(cardData, reversed, readingType));
    }
  }
  // LAST CARD: Pre-synthesis questions
  else {
    questions.push(generateTakeawayQuestion(totalCards));
    questions.push(generateReadinessQuestion());
  }

  return questions;
}

/**
 * QUESTION TYPE 1: Resonance (strength of connection)
 * Used for first card to gauge emotional/intuitive connection
 */
function generateResonanceQuestion(cardData, reversed) {
  const orientation = reversed ? 'reversed' : 'upright';
  const cardName = cardData.name;

  return {
    id: `resonance_${cardData.id}`,
    type: 'resonance',
    question: `${cardName}${reversed ? ' (reversed)' : ''} and your current situation—what's the connection?`,
    options: [
      { text: 'Direct hit - this is exactly what I needed to see', value: 5, emotional: 'validated' },
      { text: 'Strong connection - it relates to my situation', value: 4, emotional: 'connected' },
      { text: 'Some connection - I see partial relevance', value: 3, emotional: 'curious' },
      { text: 'Weak or unclear connection', value: 2, emotional: 'confused' },
      { text: 'No connection - this doesn\'t fit', value: 1, emotional: 'disconnected' }
    ]
  };
}

/**
 * QUESTION TYPE 2: Aspect relevance (what part of card matters most)
 * Identifies which dimension user connects to
 */
function generateAspectQuestion(cardData, reversed, position, readingType) {
  const keywords = reversed ? (cardData.keywords?.reversed || []) : (cardData.keywords?.upright || []);
  const element = cardData.element || 'spirit';
  const symbols = cardData.symbols || [];

  const aspects = [
    { text: `The ${element} element energy (${getElementMeaning(element)})`, value: 'element', data: element },
    { text: `The core keywords: ${keywords.slice(0, 3).join(', ')}`, value: 'keywords', data: keywords[0] },
    { text: `The ${position} position timing/significance`, value: 'position', data: position }
  ];

  if (symbols.length > 0) {
    aspects.push({
      text: `The symbolism: ${symbols.slice(0, 2).join(', ')}`,
      value: 'symbols',
      data: symbols[0]
    });
  }

  if (cardData.archetypes && cardData.archetypes.length > 0) {
    aspects.push({
      text: `The archetype: ${cardData.archetypes[0]}`,
      value: 'archetype',
      data: cardData.archetypes[0]
    });
  }

  return {
    id: `aspect_${cardData.id}`,
    type: 'aspect',
    question: `If any part of this card connects—what is it?`,
    options: aspects.slice(0, 4) // Max 4 options
  };
}

/**
 * QUESTION TYPE 3: Emotional response
 * Captures visceral feeling about the card
 */
function generateEmotionalQuestion(cardData, reversed) {
  const emotions = reversed ? [
    { text: 'Resistance or discomfort - I don\'t want to face this', value: 'resistance', energy: 'blocked' },
    { text: 'Relief - finally naming what I\'ve been avoiding', value: 'relief', energy: 'releasing' },
    { text: 'Fear or anxiety - this worries me', value: 'fear', energy: 'contracted' },
    { text: 'Curiosity - I want to understand this better', value: 'curiosity', energy: 'open' },
    { text: 'Numbness - I\'m not feeling much about it', value: 'numb', energy: 'disconnected' }
  ] : [
    { text: 'Excitement or hope - this energizes me', value: 'excitement', energy: 'expanded' },
    { text: 'Calm or peace - this feels right', value: 'peace', energy: 'centered' },
    { text: 'Validation - confirms what I already knew', value: 'validation', energy: 'aligned' },
    { text: 'Confusion - I don\'t understand yet', value: 'confusion', energy: 'seeking' },
    { text: 'Skepticism - not sure I believe this applies', value: 'skepticism', energy: 'guarded' }
  ];

  return {
    id: `emotion_${cardData.id}`,
    type: 'emotion',
    question: `What emotion comes up when you see this card?`,
    options: emotions
  };
}

/**
 * QUESTION TYPE 4: Confirmation (relates to previous cards)
 * Middle cards - shows pattern recognition
 */
function generateConfirmationQuestion(cardData, reversed, previousAnswers) {
  return {
    id: `confirmation_${cardData.id}`,
    type: 'confirmation',
    question: `This card and the previous card(s)—what's the relationship, if any?`,
    options: [
      { text: 'Confirms and amplifies the same message', value: 'amplify', pattern: 'reinforcement' },
      { text: 'Adds a new dimension or perspective', value: 'expand', pattern: 'expansion' },
      { text: 'Contradicts or challenges the previous message', value: 'contradict', pattern: 'tension' },
      { text: 'Shows the next step or progression', value: 'progress', pattern: 'sequence' },
      { text: 'No clear connection - they feel separate', value: 'unclear', pattern: 'fragmented' }
    ]
  };
}

/**
 * QUESTION TYPE 5: Situation specificity
 * Links card to concrete life situations
 */
function generateSituationQuestion(cardData, reversed, intention, readingType) {
  const cardName = cardData.name;

  // Generate contextual situation prompts based on reading type
  const situationTemplates = {
    career: [
      { text: 'Current job or work project', value: 'current_work', specificity: 'high' },
      { text: 'Career transition or job search', value: 'transition', specificity: 'high' },
      { text: 'Relationship with boss or colleagues', value: 'work_relationships', specificity: 'medium' },
      { text: 'Long-term career path or calling', value: 'career_path', specificity: 'low' },
      { text: 'Work-life balance or boundaries', value: 'balance', specificity: 'medium' }
    ],
    romance: [
      { text: 'Current romantic relationship', value: 'current_relationship', specificity: 'high' },
      { text: 'Dating or meeting new people', value: 'dating', specificity: 'high' },
      { text: 'Healing from past relationship', value: 'healing', specificity: 'medium' },
      { text: 'Self-love and personal readiness', value: 'self_love', specificity: 'low' },
      { text: 'Communication or intimacy patterns', value: 'patterns', specificity: 'medium' }
    ],
    wellness: [
      { text: 'Physical health or body', value: 'physical', specificity: 'high' },
      { text: 'Mental health or emotional state', value: 'mental', specificity: 'high' },
      { text: 'Energy levels or vitality', value: 'energy', specificity: 'medium' },
      { text: 'Self-care practices or routines', value: 'self_care', specificity: 'medium' },
      { text: 'Spiritual or existential wellbeing', value: 'spiritual', specificity: 'low' }
    ],
    personal_growth: [
      { text: 'Current growth edge or challenge', value: 'growth_edge', specificity: 'high' },
      { text: 'Shadow work or healing old wounds', value: 'shadow', specificity: 'high' },
      { text: 'Developing new skills or capacities', value: 'skills', specificity: 'medium' },
      { text: 'Life purpose or meaning', value: 'purpose', specificity: 'low' },
      { text: 'Relationship with self', value: 'self_relationship', specificity: 'medium' }
    ],
    decision: [
      { text: 'The decision itself and options', value: 'decision_options', specificity: 'high' },
      { text: 'Fear or resistance around deciding', value: 'resistance', specificity: 'medium' },
      { text: 'Consequences of each path', value: 'consequences', specificity: 'high' },
      { text: 'What I\'m avoiding or denying', value: 'avoidance', specificity: 'medium' },
      { text: 'Deeper values or priorities', value: 'values', specificity: 'low' }
    ],
    general: [
      { text: 'Specific situation or event happening now', value: 'specific_event', specificity: 'high' },
      { text: 'Relationship or person in my life', value: 'relationship', specificity: 'high' },
      { text: 'Internal pattern or belief', value: 'internal', specificity: 'medium' },
      { text: 'Overall life phase or season', value: 'life_phase', specificity: 'low' },
      { text: 'Multiple areas simultaneously', value: 'multiple', specificity: 'medium' }
    ]
  };

  const situations = situationTemplates[readingType] || situationTemplates.general;

  return {
    id: `situation_${cardData.id}`,
    type: 'situation',
    question: `Does this card connect to a specific area of your life? If so, which?`,
    options: situations
  };
}

/**
 * QUESTION TYPE 6: Action readiness
 * Middle cards - gauges willingness to act
 */
function generateActionQuestion(cardData, reversed, readingType) {
  const keywords = reversed ? (cardData.keywords?.reversed || []) : (cardData.keywords?.upright || []);
  const primaryKeyword = keywords[0] || 'this energy';

  return {
    id: `action_${cardData.id}`,
    type: 'action',
    question: `If you had to embody "${primaryKeyword}" TODAY, what would you do?`,
    options: [
      { text: 'Take immediate bold action', value: 'immediate', readiness: 'high' },
      { text: 'Make a plan to act within this week', value: 'planned', readiness: 'medium-high' },
      { text: 'Reflect and journal about it first', value: 'reflect', readiness: 'medium' },
      { text: 'Talk to someone about it', value: 'discuss', readiness: 'medium-low' },
      { text: 'Not ready to act on this yet', value: 'not_ready', readiness: 'low' }
    ]
  };
}

/**
 * QUESTION TYPE 7: Overall takeaway (after last card)
 * Synthesis preparation - what stands out
 */
function generateTakeawayQuestion(totalCards) {
  return {
    id: 'takeaway_overall',
    type: 'takeaway',
    question: `Looking at all ${totalCards} cards together—what are you noticing?`,
    options: [
      { text: 'I see a clear pattern or message emerging', value: 'pattern', clarity: 'high' },
      { text: 'One specific card really stands out', value: 'single_card', clarity: 'medium-high' },
      { text: 'The cards confirm what I already knew deep down', value: 'confirmation', clarity: 'high' },
      { text: 'I\'m seeing new perspectives I hadn\'t considered', value: 'new_perspective', clarity: 'medium' },
      { text: 'Still processing - nothing clear yet', value: 'processing', clarity: 'low' }
    ]
  };
}

/**
 * QUESTION TYPE 8: Action readiness (final)
 * Determines synthesis tone - push vs support
 */
function generateReadinessQuestion() {
  return {
    id: 'readiness_final',
    type: 'readiness',
    question: `How do you feel about taking action based on this reading?`,
    options: [
      { text: 'Ready to act NOW - give me concrete next steps', value: 'ready', tone: 'directive' },
      { text: 'Need to process and integrate first', value: 'process', tone: 'reflective' },
      { text: 'Want to explore deeper before deciding', value: 'explore', tone: 'exploratory' },
      { text: 'Feeling overwhelmed - need support', value: 'overwhelmed', tone: 'supportive' },
      { text: 'Skeptical but curious', value: 'skeptical', tone: 'evidence-based' }
    ]
  };
}

// Helper functions

function getElementMeaning(element) {
  const meanings = {
    fire: 'passion, action, willpower',
    water: 'emotion, intuition, flow',
    air: 'thought, communication, clarity',
    earth: 'practicality, manifestation, grounding',
    spirit: 'divine connection, transcendence'
  };
  return meanings[element.toLowerCase()] || 'transformative energy';
}

function randomInt(min, max) {
  return Math.floor(Math.random() * (max - min + 1)) + min;
}

/**
 * Analyze all MCQ answers to extract patterns
 * Used by synthesis engine to personalize narrative
 * HARDENING #8: Includes crisis detection
 */
export function analyzeMCQAnswers(allAnswers) {
  const analysis = {
    overallResonance: 0,
    dominantEmotions: [],
    dominantAspects: [],
    situationSpecificity: 'medium',
    actionReadiness: 'medium',
    patternRecognition: 'unclear',
    synthesisClarity: 'medium',
    synthesesTone: 'balanced',
    crisisSignals: {
      detected: false,
      severity: 'none', // none, mild, moderate, severe
      indicators: []
    }
  };

  let resonanceCount = 0;
  let resonanceSum = 0;
  const emotions = [];
  const aspects = [];
  let readiness = null;

  allAnswers.forEach(answer => {
    const { question, selectedOption } = answer;

    // Analyze by question type
    if (question.type === 'resonance') {
      resonanceSum += selectedOption.value;
      resonanceCount++;
    }
    else if (question.type === 'emotion') {
      emotions.push({
        emotion: selectedOption.value,
        energy: selectedOption.energy
      });
    }
    else if (question.type === 'aspect') {
      aspects.push({
        aspect: selectedOption.value,
        data: selectedOption.data
      });
    }
    else if (question.type === 'situation') {
      // Track highest specificity
      if (selectedOption.specificity === 'high') {
        analysis.situationSpecificity = 'high';
      }
    }
    else if (question.type === 'action') {
      if (selectedOption.readiness === 'high') {
        analysis.actionReadiness = 'high';
      } else if (selectedOption.readiness === 'low') {
        analysis.actionReadiness = 'low';
      }
    }
    else if (question.type === 'confirmation') {
      analysis.patternRecognition = selectedOption.pattern;
    }
    else if (question.type === 'takeaway') {
      analysis.synthesisClarity = selectedOption.clarity;
    }
    else if (question.type === 'readiness') {
      analysis.synthesesTone = selectedOption.tone;
      readiness = selectedOption.value;
    }
  });

  // Calculate average resonance (1-5 scale)
  if (resonanceCount > 0) {
    analysis.overallResonance = resonanceSum / resonanceCount;
  }

  analysis.dominantEmotions = emotions;
  analysis.dominantAspects = aspects;
  analysis.finalReadiness = readiness;

  // HARDENING #8: Crisis Detection
  // Detect patterns that indicate user may be in crisis or vulnerable state
  const crisisIndicators = [];

  // Check for consistent fear/anxiety/overwhelm emotions
  const fearEmotions = emotions.filter(e =>
    e.emotion === 'fear' || e.emotion === 'resistance' || e.energy === 'contracted'
  );
  if (fearEmotions.length >= 2) {
    crisisIndicators.push('repeated_fear');
  }

  // Check for "overwhelmed" readiness response
  if (readiness === 'overwhelmed') {
    crisisIndicators.push('overwhelmed_state');
  }

  // Check for very low resonance (disconnection from reality/cards)
  if (analysis.overallResonance <= 1.5) {
    crisisIndicators.push('severe_disconnection');
  }

  // Check for multiple blocked/numb emotional states
  const blockedStates = emotions.filter(e =>
    e.emotion === 'numb' || e.energy === 'blocked' || e.energy === 'disconnected'
  );
  if (blockedStates.length >= 2) {
    crisisIndicators.push('emotional_shutdown');
  }

  // Assess severity based on indicators
  if (crisisIndicators.length >= 3) {
    analysis.crisisSignals = {
      detected: true,
      severity: 'severe',
      indicators: crisisIndicators
    };
  } else if (crisisIndicators.length === 2) {
    analysis.crisisSignals = {
      detected: true,
      severity: 'moderate',
      indicators: crisisIndicators
    };
  } else if (crisisIndicators.length === 1) {
    analysis.crisisSignals = {
      detected: true,
      severity: 'mild',
      indicators: crisisIndicators
    };
  }

  return analysis;
}

/**
 * Generate synthesis guidance based on MCQ analysis
 * Tells synthesis engine HOW to write the summary
 */
export function getSynthesisGuidance(mcqAnalysis, mbtiType) {
  const guidance = {
    length: 'long', // 600-1500 words
    emphasisAreas: [],
    tone: mcqAnalysis.synthesesTone || 'balanced',
    actionLevel: mcqAnalysis.actionReadiness || 'medium',
    emotionalDepth: 'high',
    structureType: 'narrative'
  };

  // High resonance = validate their experience heavily
  if (mcqAnalysis.overallResonance >= 4) {
    guidance.emphasisAreas.push('validation', 'depth');
    guidance.tone = 'confirmatory';
  }
  // Low resonance = explore disconnection, reframe
  else if (mcqAnalysis.overallResonance <= 2) {
    guidance.emphasisAreas.push('reframing', 'exploration');
    guidance.tone = 'investigative';
  }

  // High action readiness = directive, concrete
  if (mcqAnalysis.actionReadiness === 'high') {
    guidance.emphasisAreas.push('action_steps', 'urgency');
    guidance.actionLevel = 'high';
  }
  // Low readiness = supportive, processing
  else if (mcqAnalysis.actionReadiness === 'low') {
    guidance.emphasisAreas.push('reflection', 'support');
    guidance.actionLevel = 'low';
  }

  // Pattern recognition strong = synthesis-heavy
  if (mcqAnalysis.patternRecognition === 'reinforcement') {
    guidance.emphasisAreas.push('pattern_weaving');
  }
  else if (mcqAnalysis.patternRecognition === 'tension') {
    guidance.emphasisAreas.push('paradox_exploration');
  }

  // Dominant emotions guide emotional depth
  const blockedEmotions = mcqAnalysis.dominantEmotions.filter(e =>
    e.energy === 'blocked' || e.energy === 'contracted' || e.emotion === 'resistance'
  );
  if (blockedEmotions.length > 0) {
    guidance.emphasisAreas.push('shadow_work', 'compassion');
    guidance.emotionalDepth = 'very_high';
  }

  // MBTI-specific adjustments
  if (mbtiType) {
    const thinking = mbtiType.includes('T');
    const feeling = mbtiType.includes('F');
    const judging = mbtiType.includes('J');
    const perceiving = mbtiType.includes('P');

    if (thinking) {
      guidance.emphasisAreas.push('logic', 'frameworks');
    }
    if (feeling) {
      guidance.emphasisAreas.push('values', 'empathy');
    }
    if (judging) {
      guidance.emphasisAreas.push('clarity', 'closure');
      guidance.structureType = 'structured';
    }
    if (perceiving) {
      guidance.emphasisAreas.push('possibilities', 'flexibility');
      guidance.structureType = 'exploratory';
    }
  }

  return guidance;
}
