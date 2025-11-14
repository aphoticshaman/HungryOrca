/**
 * LUNATIQ AGI ENGINE - Multi-layer tarot interpretation
 * Offline AGI that adapts to personality profile + astrological context
 */

import { CARD_DATABASE } from '../data/cardDatabase';
import { getAstrologicalContext } from './astrology';
import { getDiverseInterpretation } from './diverseWisdom';

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

  // Enrich context with intention and readingType for diverse wisdom
  const enrichedContext = {
    ...context,
    intention,
    readingType
  };

  // LAYER 1: ARCHETYPAL - Universal symbolic meaning
  const archetypal = generateArchetypalLayer(cardData, reversed, position);

  // LAYER 2: CONTEXTUAL - Adapted to reading type and intention
  const contextual = generateContextualLayer(cardData, reversed, position, readingType, intention);

  // LAYER 3: PSYCHOLOGICAL - Shadow work and deeper insights (NOW WITH DIVERSE WISDOM!)
  const psychological = generatePsychologicalLayer(cardData, reversed, position, enrichedContext);

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
 * LAYER 1: ARCHETYPAL - Universal patterns and symbols (EXPANDED)
 */
function generateArchetypalLayer(cardData, reversed, position) {
  const orientation = reversed ? 'reversed' : 'upright';
  const keywords = cardData.keywords?.[orientation] || [];
  const element = cardData.element || 'Spirit';
  const symbols = cardData.symbols || [];

  // EXPANDED core meaning with elemental and symbolic context
  let expandedMeaning = cardData.description || 'A card of transformation and insight.';

  // Add elemental wisdom
  const elementalWisdom = {
    fire: 'The fire element speaks to your passion, willpower, and creative force—the spark that ignites action and drives transformation. This is the energy of becoming, of risk-taking, of burning away what no longer serves.',
    water: 'The water element flows through emotions, intuition, and the subconscious depths. It invites you to feel rather than think, to trust your inner knowing, and to allow yourself to be moved by the currents of your heart.',
    air: 'The air element brings mental clarity, communication, and truth. It asks you to examine your thoughts, speak your reality, and cut through illusion with the sword of intellect and honest dialogue.',
    earth: 'The earth element grounds you in the material world—body, resources, tangible results. This is the energy of building, growing, manifesting, and honoring the physical realm as sacred.',
    spirit: 'Pure spirit energy transcends the elements, speaking to divine connection, fate, and the mysteries beyond the veil. This card calls you to surrender to something larger than yourself.'
  };

  expandedMeaning += ` ${elementalWisdom[element.toLowerCase()] || elementalWisdom.spirit}`;

  // Add symbol layer depth
  if (symbols.length > 0) {
    expandedMeaning += ` Notice the symbols appearing in this card: ${symbols.slice(0, 3).join(', ')}. Each symbol is a gateway to deeper understanding—let them speak to your subconscious and reveal their personal meaning to you.`;
  }

  // Add reversal context if applicable
  if (reversed) {
    expandedMeaning += ` In reversal, this card's energy is inverted, blocked, or internalized. You may be resisting its lesson, experiencing it in shadow form, or needing to address an imbalance before this energy can flow naturally. Reversals aren't "bad"—they're calls to do inner work first.`;
  }

  return {
    name: cardData.name,
    arcana: cardData.arcana,
    element: cardData.element,
    numerology: cardData.numerology,
    symbols: cardData.symbols || [],
    keywords,
    core_meaning: expandedMeaning,
    shadow_aspect: reversed ? cardData.shadow_work : null
  };
}

/**
 * LAYER 2: CONTEXTUAL - Adapted to reading type and intention (EXPANDED)
 */
function generateContextualLayer(cardData, reversed, position, readingType, intention) {
  // Map reading type to interpretation focus with EXPANDED depth
  const focusMap = {
    career: 'your professional path, vocational calling, and how you show up in the world of work',
    romance: 'matters of the heart, intimate connections, and the dance of vulnerability and desire',
    wellness: 'your body\'s wisdom, mental health, energetic vitality, and the sacred practice of self-care',
    finance: 'material abundance, relationship with money, and how you create security in the physical world',
    personal_growth: 'your evolution as a human being, self-knowledge, and the journey of becoming who you\'re meant to be',
    decision: 'the crossroads before you, the paths available, and the consequences rippling from your choice',
    general: 'the overall tapestry of your life, universal wisdom, and guidance from your higher self',
    shadow_work: 'unconscious patterns, repressed aspects, wounds seeking healing, and the gold hiding in your darkness'
  };

  const focus = focusMap[readingType] || focusMap.general;

  // EXPANDED position significance with temporal and energetic context
  const positionLower = position.toLowerCase();
  let positionDepth = `In the ${position} position, this card illuminates ${focus}. `;

  // Add position-specific wisdom
  if (positionLower.includes('past')) {
    positionDepth += 'This is foundation energy—understanding where you\'ve been helps you navigate where you\'re going. Don\'t dwell here, but honor the lessons this card represents in your history.';
  } else if (positionLower.includes('present') || positionLower.includes('current')) {
    positionDepth += 'This is the work of NOW—the most important position in your spread. This energy is active, available, and asking for your conscious engagement in this moment.';
  } else if (positionLower.includes('future') || positionLower.includes('outcome')) {
    positionDepth += 'This shows where current patterns are leading. Remember: the future isn\'t fixed. This is a probability based on your current trajectory—change your actions now, change your outcomes later.';
  } else if (positionLower.includes('challenge') || positionLower.includes('obstacle')) {
    positionDepth += 'This card reveals what\'s standing in your way. Sometimes the obstacle IS the path—this challenge may be exactly what you need for growth, even if it\'s uncomfortable.';
  } else if (positionLower.includes('advice') || positionLower.includes('guidance')) {
    positionDepth += 'This is your cosmic download—direct guidance from the universe about how to navigate your situation. Pay close attention to this card\'s medicine.';
  } else {
    positionDepth += 'This position adds crucial context to your overall reading. Consider how this card\'s energy interacts with the others in your spread.';
  }

  return {
    position_significance: positionDepth,
    intention_alignment: analyzeIntentionAlignment(cardData, intention, reversed),
    reading_type_focus: focus,
    temporal_aspect: analyzeTemporalAspect(position),
    energy_quality: reversed ? 'blocked, internal, or inverted' : 'flowing, external, or direct'
  };
}

/**
 * LAYER 3: PSYCHOLOGICAL - Deep patterns and shadow work (EXPANDED)
 */
function generatePsychologicalLayer(cardData, reversed, position, context) {
  // Get diverse wisdom insights (MBTI, attachment, love languages, etc.)
  const diverseWisdom = getDiverseInterpretation(
    cardData,
    reversed,
    position,
    context.intention || '',
    context.readingType || 'general',
    context
  );

  // EXPANDED shadow work narrative
  let shadowNarrative = cardData.shadow_work || 'This card asks you to examine what you\'ve been avoiding.';
  if (reversed) {
    shadowNarrative += ' In reversal, shadow work becomes ESSENTIAL—you cannot move forward until you face what\'s hiding in the dark. This isn\'t punishment; it\'s invitation. Your shadow holds rejected parts of yourself that need to be reclaimed, not destroyed.';
  } else {
    shadowNarrative += ' Even upright, every card has shadow aspects. Notice where you might be overdoing this energy, using it to bypass deeper work, or projecting its opposite onto others.';
  }

  // EXPANDED integration path
  let integrationNarrative = cardData.integration || 'Begin by acknowledging this energy exists within you.';
  integrationNarrative += ' Integration means making the unconscious conscious. Journal about this card. Notice when its themes show up in your life this week. Ask yourself: How am I this card? Where do I resist being this card? What would change if I fully embodied its medicine?';

  // EXPANDED emotional resonance
  const emotionalDepth = analyzeEmotionalResonance(cardData, reversed);
  const emotionalNarrative = `Emotionally, this card resonates with ${emotionalDepth}. Pay attention to how you FEEL when you look at this card—your emotional response is data. If you feel resistance, explore it. If you feel relief or excitement, lean into that. Your body knows the truth before your mind catches up.`;

  //  EXPANDED growth opportunity
  let growthNarrative = `This card invites you to ${reversed ? 'address resistance or blocks' : 'embrace and embody'} the energy it represents. `;
  if (reversed) {
    growthNarrative += 'The growth here is IN the difficulty—reversals force you to develop capacities you\'ve been avoiding. Trust that you wouldn\'t be shown this challenge if you weren\'t ready to meet it.';
  } else {
    growthNarrative += 'The growth here is in embodiment—you\'re being asked to become this card, not just understand it intellectually. Embody it in your actions, your choices, your way of being in the world.';
  }

  return {
    shadow_work: shadowNarrative,
    integration_path: integrationNarrative,
    emotional_resonance: emotionalNarrative,
    zodiac_connection: analyzeZodiacConnection(cardData, context.zodiacSign),
    growth_opportunity: growthNarrative,
    diverse_wisdom: diverseWisdom // Rich multi-framework insights (MBTI, attachment, love languages, etc.)
  };
}

/**
 * LAYER 4: PRACTICAL - Concrete actions and guidance (EXPANDED)
 */
function generatePracticalLayer(cardData, reversed, position, readingType, intention) {
  const baseAdvice = cardData.advice || 'Trust your intuition and move forward mindfully.';
  const keywords = reversed ? (cardData.keywords?.reversed || []) : (cardData.keywords?.upright || []);

  // EXPANDED focus guidance
  const primaryFocus = keywords[0] || 'Present moment awareness';
  let focusNarrative = `Your primary focus with this card is ${primaryFocus.toLowerCase()}. `;
  if (reversed) {
    focusNarrative += `In reversal, focus on WHERE this energy is blocked in your life. Don't try to force it to flow—first understand the blockage. What fear, belief, or pattern is preventing ${primaryFocus.toLowerCase()} from manifesting naturally? Address the root cause, not the symptoms.`;
  } else {
    focusNarrative += `This energy is available to you RIGHT NOW. Focus on embodying ${primaryFocus.toLowerCase()} in your daily actions. Make it your mantra this week. Notice where you're already doing this, and where you could do it more. Small, consistent actions compound into transformation.`;
  }

  // EXPANDED avoidance guidance
  let avoidanceNarrative = '';
  if (reversed) {
    avoidanceNarrative = `Avoid spiritual bypassing—don't pretend everything is fine when this card is showing you a block that needs attention. Avoid staying stuck in the "why" and move into "what now?" Avoid making big decisions from this reversed energy—clear the block first, then decide.`;
  } else {
    avoidanceNarrative = `Avoid overidentifying with this energy to the point where it becomes unbalanced. Even positive cards can be overdone. Avoid using this card's medicine to bypass necessary shadow work. Avoid assuming this means everything will be easy—upright cards still require your active participation.`;
  }

  // EXPANDED practical advice with reality check
  let expandedAdvice = baseAdvice + ' ';
  expandedAdvice += 'Remember: tarot gives you the map, but you have to walk the path. These action steps are worthless unless you actually DO them. Pick ONE action from the list above and commit to completing it within 72 hours. Massive action beats perfect planning every time.';

  return {
    action_steps: generateActionSteps(cardData, reversed, readingType, intention), // XYZA: Hyper-personalized based on intention entities
    what_to_focus_on: focusNarrative,
    what_to_avoid: avoidanceNarrative,
    timing_guidance: generateTimingGuidance(position),
    practical_advice: expandedAdvice
  };
}

/**
 * LAYER 5: SYNTHESIS - Integrated multi-layer interpretation (EXPANDED)
 */
function generateSynthesis(archetypal, contextual, psychological, practical) {
  // Don't repeat the intention - user already typed it and saw it in contextual layer

  // EXPANDED core message - weaving together all layers
  const primaryKeywords = archetypal.keywords.slice(0, 3).join(', ');
  let coreSynthesis = `${archetypal.name} brings the energy of ${primaryKeywords} into your life right now. `;

  // Add elemental integration
  if (archetypal.element) {
    const elementLower = archetypal.element.toLowerCase();
    if (elementLower === 'fire') coreSynthesis += 'This fire energy asks you to ACT—to take bold steps and trust your inner spark. ';
    else if (elementLower === 'water') coreSynthesis += 'This water energy asks you to FEEL—to trust your intuition and honor your emotional truth. ';
    else if (elementLower === 'air') coreSynthesis += 'This air energy asks you to THINK—to gain clarity, speak truth, and cut through confusion. ';
    else if (elementLower === 'earth') coreSynthesis += 'This earth energy asks you to BUILD—to take practical steps and create tangible results. ';
    else coreSynthesis += 'This spiritual energy asks you to SURRENDER—to trust in something larger than yourself. ';
  }

  coreSynthesis += `${contextual.position_significance.split('.')[0]}. The ${contextual.energy_quality} nature of this card's appearance suggests ${contextual.energy_quality.includes('flowing') ? 'you\'re ready to embody this fully' : 'there\'s inner work to do before external manifestation'}.`;

  // EXPANDED integration - linking psychological and practical
  let integrationSynthesis = `To integrate this card's medicine: ${psychological.integration_path.split('.')[0]}. `;
  integrationSynthesis += `${practical.practical_advice.split('.')[0]}. `;
  integrationSynthesis += 'The gap between insight and transformation is ACTION—knowing is not enough, you must do.';

  // EXPANDED deeper insight - the "so what?"
  let deeperInsight = psychological.shadow_work.split('.').slice(0, 2).join('. ') + '. ';
  deeperInsight += 'This is your edge—the place where growth happens. Lean into the discomfort. The treasure you seek is buried in the territory you\'ve been avoiding.';

  // EXPANDED next steps with urgency
  const firstThreeActions = practical.action_steps.slice(0, 3);
  let nextStepsSynthesis = 'Your immediate next steps: ';
  nextStepsSynthesis += `(1) ${firstThreeActions[0] || 'Reflect on this card\'s message'}. `;
  if (firstThreeActions[1]) nextStepsSynthesis += `(2) ${firstThreeActions[1]}. `;
  if (firstThreeActions[2]) nextStepsSynthesis += `(3) ${firstThreeActions[2]}. `;
  nextStepsSynthesis += 'Do not just read this—ACT within 24 hours while the energy is fresh.';

  return {
    core_message: coreSynthesis,
    integration: integrationSynthesis,
    deeper_insight: deeperInsight,
    next_steps: nextStepsSynthesis
  };
}

// Helper functions

function analyzeIntentionAlignment(cardData, intention, reversed) {
  if (!intention || intention.trim().length === 0) {
    return 'Consider how this card relates to your current situation.';
  }

  const intentionLower = intention.toLowerCase();
  const cardName = cardData.name.toLowerCase();
  const keywords = reversed
    ? (cardData.keywords?.reversed || [])
    : (cardData.keywords?.upright || []);

  // Analyze intention context - GET STRAIGHT TO THE POINT
  let context = '';

  // Detect question type from intention
  if (intentionLower.includes('should i') || intentionLower.includes('can i')) {
    context = reversed
      ? `${cardData.name} reversed suggests reconsidering or addressing blocks before proceeding. The ${keywords.slice(0, 2).join(' and ')} energy is inverted, indicating obstacles or internal resistance you need to work through first.`
      : `${cardData.name} upright indicates ${keywords.slice(0, 2).join(' and ')}, suggesting favorable conditions. The energy supports moving forward, but make sure you're approaching this with intention and awareness.`;
  } else if (intentionLower.includes('how') || intentionLower.includes('what')) {
    context = reversed
      ? `${cardData.name} reversed points to ${keywords.slice(0, 2).join(', ')}, or a need to examine where energy is blocked or misdirected. Look for patterns of resistance or self-sabotage that may be operating beneath conscious awareness.`
      : `${cardData.name} upright illuminates themes of ${keywords.slice(0, 2).join(', ')}. Pay attention to how these energies are showing up in your life right now - they hold the keys to understanding your situation.`;
  } else if (intentionLower.includes('why')) {
    context = reversed
      ? `${cardData.name} reversed suggests this stems from ${keywords.slice(0, 2).join(' or ')}, inverted or blocked energies. Something is operating in shadow - either internal resistance, fear-based patterns, or misaligned actions.`
      : `${cardData.name} upright reveals this is fundamentally about ${keywords.slice(0, 2).join(' and ')}. These are the core themes driving the situation - understanding them will help you see the deeper patterns at play.`;
  } else if (intentionLower.includes('when')) {
    context = reversed
      ? `${cardData.name} reversed suggests delays or the need to resolve ${keywords.slice(0, 2).join(' and ')} issues first. Timing isn't right yet - there's internal work to do before external movement can happen.`
      : `${cardData.name} upright indicates movement around ${keywords.slice(0, 2).join(' and ')}. Pay attention to when these themes intensify - that's when opportunities will appear.`;
  } else {
    // General intention
    context = reversed
      ? `${cardData.name} reversed highlights challenges or inversions in ${keywords.slice(0, 2).join(' and ')}. These areas need healing or course correction before you can move forward effectively.`
      : `${cardData.name} upright brings ${keywords.slice(0, 2).join(' and ')} energy directly into your situation. This is the medicine you need right now - lean into these qualities.`;
  }

  return context;
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

/**
 * XYZA CYCLE 1: Extract semantic entities from intention
 * Enables hyper-specific, context-aware action generation
 */
function extractIntentionEntities(intention) {
  if (!intention || intention.trim().length === 0) {
    return { subjects: [], situations: [], emotions: [], goals: [], timeframes: [], specificity: 'baseline' };
  }

  const intentionLower = intention.toLowerCase();

  // Extract subjects (people, relationships) - EXPANDED DICTIONARY
  const subjects = [];
  const subjectPatterns = [
    // Romantic/intimate
    /my (partner|boyfriend|girlfriend|husband|wife|spouse|fianc[eé]|lover|ex|ex-girlfriend|ex-boyfriend|ex-wife|ex-husband|ex-partner|crush|date)/gi,
    // Professional
    /my (boss|manager|supervisor|coworker|colleague|employee|subordinate|client|customer|vendor|contractor|mentor|coach|advisor|consultant|business partner|investor)/gi,
    /potential customer|prospective client/gi,
    // Family
    /my (mother|mom|father|dad|parent|son|daughter|child|sibling|sister|brother|grandmother|grandma|grandfather|grandpa|grandparent|grandchild|grandson|granddaughter|aunt|uncle|niece|nephew|cousin|in-law|stepmother|stepfather|stepson|stepdaughter)/gi,
    // Social/community
    /my (friend|best friend|neighbor|roommate|enemy|rival|therapist|doctor|lawyer|teacher|landlord)/gi,
    // Contact status
    /ex (who|that|we) (stay|stayed) friends/gi,
    /no contact (with|ex|relationship)/gi,
    /(with|about) ([A-Z][a-z]+)/g, // Names (capitalized words)
    /my relationship|my marriage|my team|my family|my business|my company/gi
  ];
  subjectPatterns.forEach(pattern => {
    const matches = intention.match(pattern);
    if (matches) subjects.push(...matches.map(m => m.trim()));
  });

  // Extract situations (contexts, places, events)
  const situations = [];
  const situationPatterns = [
    /my (job|career|work|business|startup|company|project|role|position) (at|with)? ?([A-Za-z]+)?/gi,
    /(quit|leave|start|launch|join|move to|apply to|negotiate|interview for) ([a-z ]+)/gi,
    /at (work|home|school|university|therapy|counseling)/gi,
    /(interview|meeting|presentation|deadline|decision|choice|opportunity|offer)/gi
  ];
  situationPatterns.forEach(pattern => {
    const matches = intention.match(pattern);
    if (matches) situations.push(...matches.map(m => m.trim()));
  });

  // Extract emotions (feelings, states)
  const emotions = [];
  const emotionPatterns = [
    /(afraid|scared|terrified|anxious|worried|nervous|stressed)/gi,
    /(stuck|trapped|lost|confused|uncertain|unsure)/gi,
    /(angry|frustrated|resentful|bitter|hurt|betrayed)/gi,
    /(hopeful|excited|optimistic|motivated|inspired)/gi,
    /(depressed|sad|lonely|alone|isolated|empty)/gi,
    /(overwhelmed|exhausted|burned out|drained|tired)/gi
  ];
  emotionPatterns.forEach(pattern => {
    const matches = intention.match(pattern);
    if (matches) emotions.push(...matches.map(m => m.trim()));
  });

  // Extract goals (desired outcomes, actions)
  const goals = [];
  const goalPatterns = [
    /should i (quit|leave|start|move|change|apply|ask|tell|confront|end|begin)/gi,
    /want to (quit|leave|start|move|change|apply|ask|tell|confront|end|begin|make|earn|lose|gain|heal|fix|improve)/gi,
    /trying to (quit|leave|start|move|change|apply|ask|tell|confront|end|begin|make|earn|lose|gain|heal|fix|improve)/gi,
    /need to (quit|leave|start|move|change|apply|ask|tell|confront|end|begin|make|earn|lose|gain|heal|fix|improve)/gi
  ];
  goalPatterns.forEach(pattern => {
    const matches = intention.match(pattern);
    if (matches) goals.push(...matches.map(m => m.replace(/(should i|want to|trying to|need to) /gi, '').trim()));
  });

  // Extract timeframes
  const timeframes = [];
  const timePatterns = [
    /(right now|immediately|today|this week|this month|this year|soon|eventually)/gi,
    /(within \d+ (days|weeks|months|years))/gi,
    /(by \w+day|by next (week|month|year))/gi
  ];
  timePatterns.forEach(pattern => {
    const matches = intention.match(pattern);
    if (matches) timeframes.push(...matches.map(m => m.trim()));
  });

  // Calculate specificity level (for XYZA Amplify phase)
  const entityCount = subjects.length + situations.length + emotions.length + goals.length + timeframes.length;
  const specificity = intention.length > 100 && entityCount >= 5 ? 'high' :
                     intention.length > 50 && entityCount >= 3 ? 'medium' : 'baseline';

  return {
    subjects: [...new Set(subjects)], // Remove duplicates
    situations: [...new Set(situations)],
    emotions: [...new Set(emotions)],
    goals: [...new Set(goals)],
    timeframes: [...new Set(timeframes)],
    specificity
  };
}

/**
 * Generate action steps with entity-aware targeting (XYZA enhanced)
 */
function generateActionSteps(cardData, reversed, readingType, intention) {
  const cardName = cardData.name.toLowerCase();
  const element = cardData.element || 'spirit';
  const readingFocus = readingType || 'general';

  // CIA/DIA-level practical actions for real-world application
  const actionMap = {
    career: {
      upright: [
        `Schedule 1-on-1 with decision-maker this week - pitch idea embodying ${element} energy.`,
        `Apply to 3 stretch positions that align with ${cardName} themes.`,
        `Network: reach out to 5 industry contacts, offer value before asking.`,
        `Delegate/automate tasks misaligned with this card's strengths.`,
        `Start revenue-generating side project using ${element} element skills.`,
        `Negotiate raise using market data - schedule meeting within 7 days.`,
        `Join industry community, attend event, or post thought leadership content.`,
        `Take course/certification increasing earning potential by 20%+.`
      ],
      reversed: [
        `Identify 3 workplace blocks to ${cardName} energy - address #1 this week.`,
        `Practice saying "no" to misaligned projects - set boundary with manager.`,
        `Update resume highlighting opposite skills to current limitations.`,
        `Have honest mentor conversation about career plateau or obstacles.`,
        `Eliminate one toxic work relationship or change team dynamics.`,
        `Apply to different role/company - explore complete pivot option.`,
        `Address imposter syndrome: list 10 wins, ask for testimonials.`
      ]
    },
    romance: {
      upright: [
        `Plan ${element}-based date: fire=adventure, water=intimacy, air=deep talk, earth=sensory comfort.`,
        `Have vulnerable conversation expressing 3 specific needs from this card.`,
        `Send appreciation messages showing you notice partner's efforts.`,
        `Try new relationship practice: weekly check-ins, love languages, tantra.`,
        `Meet someone new in ${element}-aligned setting within 3 days.`,
        `Create shared ritual: morning coffee dates, Sunday hikes, monthly staycations.`,
        `Read relationship book together, implement 3 exercises this month.`,
        `Plan surprise embodying ${cardName} energy - execute within 48 hours.`
      ],
      reversed: [
        `Take 72-hour solo retreat to process blocks without contacting partner/dates.`,
        `Set one clear boundary around ${cardName} shadow - communicate lovingly.`,
        `End toxic pattern: breadcrumbing, people-pleasing, unavailability, or love-bombing.`,
        `Book couples therapy or individual session on attachment wounds.`,
        `Have "state of union" talk - express resentments, then co-create solutions.`,
        `Practice opposite action: if you chase, pull back; if you avoid, lean in.`,
        `Delete apps/unfriend exes creating relationship static or comparison.`
      ]
    },
    wellness: {
      upright: [
        `Try ${element} practice: earth=yoga/hiking, air=breathwork, water=swimming, fire=HIIT/dance.`,
        `Meal prep 7 nutrient-dense meals embodying ${cardName} vitality.`,
        `Book bodywork: massage, acupuncture, float tank, cryotherapy, or energy healing.`,
        `Start 21-day challenge: 10k steps, cold showers, meditation, or mobility.`,
        `Join fitness class/wellness community - attend 3 sessions this week.`,
        `Optimize sleep: blackout curtains, magnesium, no screens 90min before bed.`,
        `Track biomarkers: get bloodwork, wearable data, or functional medicine consult.`,
        `Add supplement/superfood supporting ${element} element vitality.`
      ],
      reversed: [
        `Complete rest for 48 hours - no productivity, just deep restoration.`,
        `Eliminate #1 toxic habit: late scrolling, processed food, alcohol, or overwork.`,
        `Get overdue health screening: bloodwork, dental, vision, or specialist.`,
        `Address chronic issue: pain, fatigue, sleep, digestion - see new practitioner.`,
        `Change environment: declutter space, get air purifier, add plants/light.`,
        `Try opposite wellness modality: if high-intensity, do yin; if sedentary, move.`,
        `Fast from stimulation: 24hr phone detox, news detox, or dopamine reset.`
      ]
    },
    finance: {
      upright: [
        `Open high-yield savings, automate $200-500/month transfer from checking.`,
        `Negotiate raise using salary data - aim for 10-20% increase, schedule meeting.`,
        `Start income stream aligned with ${element} talents - earn first $500 in 30 days.`,
        `Invest in skill/tool increasing earning potential: course, software, coaching.`,
        `Review spending, redirect $100-300/month from waste to wealth-building.`,
        `Diversify income: freelance, consulting, investing, or passive income project.`,
        `Network with 3 high-earners in your field - learn money strategies.`,
        `Create value offer: productize skill, raise rates, or upsell existing clients.`
      ],
      reversed: [
        `Face reality: list all debts, income, expenses - no judgment, just data.`,
        `Cut one luxury for 60 days, redirect to emergency fund or debt payoff.`,
        `Sell unused items worth $500+ - create immediate cash flow.`,
        `Get help: financial advisor, debt counselor, or money mindset coach session.`,
        `Challenge money story: do opposite action for 14 days, track results.`,
        `Pause discretionary spending 30 days - save difference, feel abundance.`,
        `Increase income urgently: gig work, sell service, or monetize hobby ASAP.`
      ]
    },
    personal_growth: {
      upright: [
        `Read book on ${cardName} themes, implement 3 insights within 7 days.`,
        `Join mastermind/course/community aligned with this growth path.`,
        `Do one thing that scares you - embody ${cardName} courage publicly.`,
        `Teach skill to someone - sharing accelerates ${element} mastery.`,
        `Create vision/manifestation ritual honoring card's transformational energy.`,
        `Start creative project expressing ${element} element: write, paint, build, perform.`,
        `Mentor someone younger/less experienced in area you've grown.`,
        `Attend retreat, workshop, or immersive experience within 90 days.`
      ],
      reversed: [
        `Identify #1 self-sabotage pattern - interrupt it 5 times this week.`,
        `Work with therapist/coach on ${cardName} shadow blocking growth.`,
        `Make amends: apologize to someone hurt by this card's reversed energy.`,
        `Unfollow/block 10 accounts triggering comparison, FOMO, or smallness.`,
        `Spend full day in nature alone processing what needs to die for rebirth.`,
        `Face avoided conversation/situation - do it within 48 hours.`,
        `Take inventory: stop one activity, relationship, or belief no longer serving you.`
      ]
    },
    decision: {
      upright: [
        `Make pros/cons list using ${cardName} wisdom - what does this card illuminate?`,
        `Interview 3 people who've made similar decisions, ask specific questions.`,
        `Set decision deadline [3-7 days] and commit to honoring outcome.`,
        `Visualize 24 hours with each choice - notice body sensations and energy.`,
        `Take micro-action toward ${element}-aligned choice today.`,
        `Get divination confirmation: tarot, I Ching, or trusted intuitive reading.`,
        `List worst-case scenarios - create mitigation plan for each, reduce fear.`,
        `Trust ${cardName} energy - make decision now, course-correct later if needed.`
      ],
      reversed: [
        `Don't decide yet - gather more intel for minimum 72 hours.`,
        `Examine fear: is resistance intuition (trust) or ego-protection (investigate)?`,
        `Sleep on it for 5 nights - track dreams, morning thoughts, gut feelings.`,
        `Get neutral third-party read: mentor, therapist, or wise advisor outside situation.`,
        `Identify what you're actually avoiding - address that fear first, then decide.`,
        `Create space: pause pressure, extend timeline, or decline forced choice.`,
        `Notice what you're trying to control - surrender, then clarity emerges.`
      ]
    },
    shadow_work: {
      upright: [
        `Free-write on ${cardName} shadow for 20min - no editing, full honesty.`,
        `Do opposite of comfort zone: if withdrawn, be social; if oversharing, be private.`,
        `Work with shadow guide: therapist, coach, or integration specialist.`,
        `Notice 3 projections this week - what you judge in others, you disown in self.`,
        `Create art from shadow: paint, write, dance, or ritualize the darkness.`,
        `Have brutally honest conversation with self or trusted witness.`,
        `Embrace ${element} shadow: fire=rage release, water=grief work, air=speak unspeakable, earth=embody shame.`
      ],
      reversed: [
        `Identify where you're spiritually bypassing pain with positivity - feel it fully.`,
        `Stop running from ${cardName} wound - sit with discomfort for 30min daily.`,
        `Admit where you've been cruel, selfish, or harmful - make repair.`,
        `Work with body: somatic therapy, breathwork, or trauma-release exercise.`,
        `Face addiction/compulsion - get support, attend meeting, or enter treatment.`,
        `End toxic coping: substance, person, or behavior masking real issue.`,
        `Allow breakdown: cry, rage, or collapse in safe container - integration follows.`
      ]
    },
    general: {
      upright: [
        `Take bold ${element}-aligned action today: fire=risk, water=feel, air=speak truth, earth=build.`,
        `Share ${cardName} wisdom with someone who needs it - teach what you're learning.`,
        `Start daily practice honoring this energy: meditation, movement, creativity, service.`,
        `Say yes to opportunity aligned with card themes - even if it scares you.`,
        `Embody ${cardName} publicly: post insight, have conversation, or take visible stand.`,
        `Create accountability: tell 3 people your ${element} intention, report progress weekly.`,
        `Celebrate wins related to this card's energy - acknowledge growth.`
      ],
      reversed: [
        `Pause everything for 24 hours - let nervous system recalibrate.`,
        `Identify where ${cardName} energy is blocked - remove one obstacle this week.`,
        `Ask for help in area where you're stuck - therapist, mentor, or friend.`,
        `Do opposite action: if forcing, surrender; if avoiding, engage; if controlling, trust.`,
        `Release what's complete: relationship, job, identity, or story no longer true.`,
        `Get real with someone about your struggle - vulnerability creates connection.`,
        `Forgive self for ${cardName} shadow - integration over perfection.`
      ]
    }
  };

  const orientation = reversed ? 'reversed' : 'upright';
  const categoryActions = actionMap[readingFocus] || actionMap.general;
  const pool = categoryActions[orientation];

  // XYZA: Extract entities from intention
  const entities = extractIntentionEntities(intention || '');

  // Pick 3 random actions from the pool
  const shuffled = [...pool].sort(() => Math.random() - 0.5);
  let selectedActions = shuffled.slice(0, 3);

  // XYZA ENHANCEMENT: Interpolate entities into actions for specificity
  // This transforms "Have vulnerable conversation" → "Have vulnerable conversation with your partner about your fear of abandonment"
  selectedActions = selectedActions.map(action => {
    let enhanced = action;

    // If high specificity, add entity context
    if (entities.specificity === 'high' || entities.specificity === 'medium') {
      // Add subject context (people, relationships)
      if (entities.subjects.length > 0 && !action.toLowerCase().includes('partner') && !action.toLowerCase().includes('boss')) {
        const subject = entities.subjects[0];
        // Insert subject where contextually appropriate
        if (action.includes('conversation') || action.includes('talk') || action.includes('discuss')) {
          enhanced = enhanced.replace(/conversation/i, `conversation with ${subject}`);
          enhanced = enhanced.replace(/talk/i, `talk with ${subject}`);
        } else if (action.includes('relationship') || action.includes('couples')) {
          enhanced = enhanced.replace(/relationship/i, `relationship with ${subject}`);
        }
      }

      // Add situational context
      if (entities.situations.length > 0) {
        const situation = entities.situations[0];
        // Append situation context at end if not already specific
        if (!enhanced.includes('at work') && !enhanced.includes('job') && (situation.includes('job') || situation.includes('work'))) {
          enhanced += ` (specifically: ${situation})`;
        } else if (situation.includes('quit') || situation.includes('start') || situation.includes('launch')) {
          enhanced += ` - ${situation}`;
        }
      }

      // Add emotional context
      if (entities.emotions.length > 0) {
        const emotion = entities.emotions[0];
        // Address the specific fear/emotion
        if (action.includes('fear') || action.includes('resistance') || action.includes('block')) {
          enhanced = enhanced.replace(/fear/i, `${emotion}`);
          enhanced = enhanced.replace(/resistance/i, `${emotion}`);
        } else if (action.includes('shadow') && !enhanced.includes(emotion)) {
          enhanced += ` - address your ${emotion}`;
        }
      }

      // Add timeframe urgency
      if (entities.timeframes.length > 0) {
        const timeframe = entities.timeframes[0];
        if (!enhanced.includes('this week') && !enhanced.includes('within') && !enhanced.includes('days')) {
          enhanced += ` ${timeframe}`;
        }
      }
    }

    return enhanced;
  });

  return selectedActions;
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

  // Generate spread synthesis (big picture analysis)
  const spreadSummary = generateSpreadSummary(
    interpretations,
    intention,
    spreadType,
    context.readingType || 'general'
  );

  return {
    interpretations,
    astrologicalContext: astroContext,
    spreadType,
    intention,
    summary: generateReadingSummary(interpretations, astroContext),
    spreadSummary // NEW: Big picture synthesis across all cards
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

/**
 * SPREAD SYNTHESIS - "Big Picture" analysis across all cards
 * Generates 200-300 word synthesis showing how all cards relate to the intention
 *
 * @param {Array} interpretations - All card interpretations from the spread
 * @param {string} intention - User's original intention
 * @param {string} spreadType - Type of spread (single_card, three_card, etc.)
 * @param {string} readingType - Reading focus (career, romance, etc.)
 * @returns {Object} - Comprehensive spread synthesis
 */
export function generateSpreadSummary(interpretations, intention, spreadType, readingType) {
  if (!interpretations || interpretations.length === 0) {
    return null;
  }

  // Single card readings don't need spread summary
  if (interpretations.length === 1) {
    return null;
  }

  // Analyze patterns across all cards
  const patterns = analyzeSpreadPatterns(interpretations);

  // Extract key entities from intention
  const entities = extractIntentionEntities(intention || '');

  // Generate narrative synthesis
  const narrative = generateSpreadNarrative(
    interpretations,
    intention,
    spreadType,
    readingType,
    patterns,
    entities
  );

  // Generate integrated action steps from all cards
  const integratedActions = generateIntegratedActions(interpretations, entities);

  return {
    patterns,
    narrative,
    integratedActions,
    keyThemes: patterns.dominantThemes,
    overallOrientation: patterns.energyFlow,
    criticalInsight: generateCriticalInsight(interpretations, patterns, intention)
  };
}

/**
 * Analyze patterns across all cards in the spread
 */
function analyzeSpreadPatterns(interpretations) {
  const totalCards = interpretations.length;
  let reversedCount = 0;
  let majorArcanaCount = 0;
  const elements = { fire: 0, water: 0, air: 0, earth: 0, spirit: 0 };
  const suits = { wands: 0, cups: 0, swords: 0, pentacles: 0, major: 0 };
  const allKeywords = [];

  interpretations.forEach(interp => {
    // Count reversed
    if (interp.reversed) reversedCount++;

    // Count Major Arcana
    const arcana = interp.cardData.arcana;
    if (arcana === 'Major') {
      majorArcanaCount++;
      suits.major++;
    } else {
      // Count suits for Minor Arcana
      const suit = interp.cardData.suit?.toLowerCase();
      if (suit && suits[suit] !== undefined) {
        suits[suit]++;
      }
    }

    // Count elements
    const element = interp.cardData.element?.toLowerCase() || 'spirit';
    if (elements[element] !== undefined) {
      elements[element]++;
    }

    // Collect keywords
    const keywords = interp.reversed
      ? (interp.cardData.keywords?.reversed || [])
      : (interp.cardData.keywords?.upright || []);
    allKeywords.push(...keywords);
  });

  // Calculate dominant element
  const dominantElement = Object.entries(elements)
    .sort((a, b) => b[1] - a[1])[0][0];

  // Calculate dominant suit (if applicable)
  const dominantSuit = Object.entries(suits)
    .filter(([suit, _]) => suit !== 'major')
    .sort((a, b) => b[1] - a[1])[0];

  // Find repeating keyword themes (appear 2+ times)
  const keywordFrequency = {};
  allKeywords.forEach(kw => {
    const key = kw.toLowerCase();
    keywordFrequency[key] = (keywordFrequency[key] || 0) + 1;
  });
  const dominantThemes = Object.entries(keywordFrequency)
    .filter(([_, count]) => count >= 2)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 5)
    .map(([theme, _]) => theme);

  // Energy flow assessment
  const reversedRatio = reversedCount / totalCards;
  const energyFlow = reversedRatio > 0.6 ? 'blocked/internal' :
                     reversedRatio > 0.3 ? 'mixed/transitional' :
                     'flowing/external';

  // Major Arcana intensity
  const majorArcanaRatio = majorArcanaCount / totalCards;
  const intensity = majorArcanaRatio > 0.6 ? 'high' :
                    majorArcanaRatio > 0.3 ? 'moderate' :
                    'low';

  return {
    totalCards,
    reversedCount,
    reversedRatio,
    majorArcanaCount,
    majorArcanaRatio,
    intensity,
    dominantElement,
    elementalBalance: elements,
    dominantSuit: dominantSuit ? dominantSuit[0] : null,
    suitDistribution: suits,
    dominantThemes,
    energyFlow
  };
}

/**
 * Generate narrative synthesis (200-300 words)
 */
function generateSpreadNarrative(interpretations, intention, spreadType, readingType, patterns, entities) {
  const cardNames = interpretations.map(i => i.cardData.name).join(', ');

  // Opening: Cards drawn (no need to repeat the question - they already typed it!)
  let narrative = `You drew ${cardNames}. `;

  // Major Arcana significance
  if (patterns.majorArcanaRatio > 0.5) {
    narrative += `With ${patterns.majorArcanaCount} Major Arcana cards, this is a MAJOR life moment requiring significant attention. These aren't small tweaks—this is deep transformation, destiny-level stuff. `;
  } else if (patterns.majorArcanaCount === 0) {
    narrative += `All Minor Arcana suggests this situation is within your control and more about day-to-day choices than fate. You have agency here. `;
  }

  // Energy flow (reversed ratio)
  if (patterns.energyFlow === 'blocked/internal') {
    narrative += `The majority of cards are reversed, indicating internal blocks, resistance, or the need to address shadow patterns before external progress is possible. You're in an integration phase—don't force forward movement yet. `;
  } else if (patterns.energyFlow === 'flowing/external') {
    narrative += `Most cards upright shows flowing, active energy. The universe is conspiring in your favor—take bold action aligned with these themes. `;
  } else {
    narrative += `Mixed upright/reversed cards suggest you're in transition—some areas flowing, others requiring inner work. `;
  }

  // Elemental dominance
  const elementInsights = {
    fire: 'Fire dominates: passion, action, risk-taking, courage. This requires DOING, not overthinking.',
    water: 'Water dominates: emotion, intuition, feeling, flow. Trust your gut, honor feelings, dive deep.',
    air: 'Air dominates: communication, clarity, truth-telling, mental work. Speak up, get clear, analyze.',
    earth: 'Earth dominates: practical action, building, patience, embodiment. Ground yourself, take concrete steps.',
    spirit: 'Spiritual energy dominates: this is about transcendence, faith, connecting to something larger.'
  };
  narrative += `${elementInsights[patterns.dominantElement]} `;

  // Dominant themes
  if (patterns.dominantThemes.length > 0) {
    narrative += `Key themes repeating across cards: ${patterns.dominantThemes.slice(0, 3).join(', ')}. Pay special attention to these—they're the universe shouting at you. `;
  }

  // Spread-specific insights
  const spreadInsights = {
    three_card: 'Looking at past-present-future: ',
    daily: 'For today: ',
    decision: 'Comparing your two paths: ',
    relationship: 'The relationship dynamics show: ',
    celtic_cross: 'The full Celtic Cross reveals: '
  };
  if (spreadInsights[spreadType]) {
    narrative += spreadInsights[spreadType];

    if (spreadType === 'three_card') {
      narrative += `Your past (${interpretations[0].cardData.name}) brought you here, your present (${interpretations[1].cardData.name}) is the current work, and your future (${interpretations[2].cardData.name}) is where this leads if you stay the course. `;
    } else if (spreadType === 'decision') {
      narrative += `Path A shows ${interpretations[1].cardData.name}, Path B shows ${interpretations[4].cardData.name}. Neither is "right"—both have gifts and challenges. The question isn't which path, but which YOU are you becoming? `;
    }
  }

  // Entity-specific addressing (hyper-personalization)
  if (entities.subjects.length > 0) {
    narrative += `Regarding ${entities.subjects[0]}: the cards suggest ${entities.emotions.length > 0 ? `your ${entities.emotions[0]} is` : 'this relationship is'} central to the current dynamic. `;
  }

  // Closing with directness
  narrative += `Bottom line: ${generateBottomLine(interpretations, patterns, entities)}`;

  return narrative;
}

/**
 * Generate the "bottom line" - direct, actionable conclusion
 */
function generateBottomLine(interpretations, patterns, entities) {
  // High reversed = do inner work first
  if (patterns.reversedRatio > 0.6) {
    return `Don't force external action yet. Do the inner work these cards are demanding—therapy, shadow work, rest, or ending toxic patterns. Once you address blocks, the path forward will be obvious.`;
  }

  // High Major Arcana = major life shift
  if (patterns.majorArcanaRatio > 0.6) {
    return `This is a major life chapter. Trust that what's happening (even if hard) is necessary for your evolution. Surrender to the process while taking practical steps the cards suggest.`;
  }

  // Mostly upright, flowing energy
  if (patterns.energyFlow === 'flowing/external') {
    const actions = interpretations
      .flatMap(i => i.layers.practical.action_steps)
      .slice(0, 2);
    return `The energy is RIPE for action. Start immediately: ${actions[0]} Then: ${actions[1]} Don't overthink—DO.`;
  }

  // Mixed/transitional
  return `You're in transition. Some things need to die (address the reversed cards), while others need to be born (embody the upright ones). Focus on the 2-3 most practical actions from your cards and execute them this week.`;
}

/**
 * Generate integrated actions considering ALL cards together
 */
function generateIntegratedActions(interpretations, entities) {
  // Collect all action steps from all cards
  const allActions = interpretations.flatMap(i => i.layers.practical.action_steps);

  // Prioritize based on patterns
  // 1. Actions addressing blocks (if cards are mostly reversed)
  // 2. Actions that appear in multiple cards
  // 3. Actions with highest specificity

  // For now, return top 5 unique actions
  const uniqueActions = [...new Set(allActions)].slice(0, 5);

  return uniqueActions;
}

/**
 * Generate the single most critical insight from the spread
 */
function generateCriticalInsight(interpretations, patterns, intention) {
  // This is the ONE THING the user must understand

  if (patterns.reversedRatio > 0.7) {
    return `CRITICAL: Stop pushing forward. The blocks ARE the work right now. Address resistance before taking external action.`;
  }

  if (patterns.majorArcanaRatio >= 0.5) {
    const majorCards = interpretations
      .filter(i => i.cardData.arcana === 'Major')
      .map(i => i.cardData.name)
      .join(', ');
    return `CRITICAL: ${majorCards} together signals a destiny-level moment. This isn't casual—it's your soul's curriculum. Pay deep attention.`;
  }

  if (patterns.dominantThemes.length >= 3) {
    return `CRITICAL: The themes ${patterns.dominantThemes.slice(0, 3).join(', ')} keep repeating. The universe is being LOUD about this. Don't ignore the pattern.`;
  }

  // Look for contradiction between cards (past vs future, path A vs B)
  if (interpretations.length >= 2) {
    const firstCard = interpretations[0];
    const lastCard = interpretations[interpretations.length - 1];
    if (firstCard.reversed && !lastCard.reversed) {
      return `CRITICAL: You're moving from ${firstCard.cardData.name} reversed to ${lastCard.cardData.name} upright—this is a healing arc. Trust the transformation.`;
    } else if (!firstCard.reversed && lastCard.reversed) {
      return `CRITICAL: ${firstCard.cardData.name} upright moving to ${lastCard.cardData.name} reversed warns of potential self-sabotage or external obstacles ahead. Course-correct now.`;
    }
  }

  return `CRITICAL: All cards point to the same directive—trust the specific actions they've given you and execute them with full commitment.`;
}
