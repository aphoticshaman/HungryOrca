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
  // Extract the first sentence from contextual.intention_alignment (which now contains the full intention-aware analysis)
  const intentionContext = contextual.intention_alignment || '';

  return {
    core_message: `${intentionContext} ${archetypal.name} embodies ${archetypal.keywords.slice(0, 3).join(', ')} - ${contextual.position_significance}`,
    integration: `${psychological.integration_path} ${practical.practical_advice}`,
    deeper_insight: psychological.shadow_work,
    next_steps: practical.action_steps.join(' Then: ')
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

  // Analyze intention context
  let context = '';

  // Detect question type from intention
  if (intentionLower.includes('should i') || intentionLower.includes('can i')) {
    context = reversed
      ? `Regarding "${intention}" - ${cardData.name} reversed suggests reconsidering or addressing blocks before proceeding. The ${keywords.slice(0, 2).join(' and ')} energy is inverted, indicating obstacles or internal resistance.`
      : `Regarding "${intention}" - ${cardData.name} upright indicates ${keywords.slice(0, 2).join(' and ')}, suggesting favorable conditions for your question.`;
  } else if (intentionLower.includes('how') || intentionLower.includes('what')) {
    context = reversed
      ? `Your question "${intention}" draws ${cardData.name} reversed, pointing to ${keywords.slice(0, 2).join(', ')}, or a need to examine where energy is blocked or misdirected.`
      : `Your question "${intention}" draws ${cardData.name} upright, illuminating themes of ${keywords.slice(0, 2).join(', ')}. This card offers guidance on your inquiry.`;
  } else if (intentionLower.includes('why')) {
    context = reversed
      ? `Asking "${intention}" - ${cardData.name} reversed suggests the reason involves ${keywords.slice(0, 2).join(' or ')}, inverted or blocked energies that need attention.`
      : `Asking "${intention}" - ${cardData.name} upright reveals this is about ${keywords.slice(0, 2).join(' and ')}, core themes requiring your awareness.`;
  } else if (intentionLower.includes('when')) {
    context = reversed
      ? `Your timing question "${intention}" with ${cardData.name} reversed suggests delays or the need to resolve ${keywords.slice(0, 2).join(' and ')} issues first.`
      : `Your timing question "${intention}" with ${cardData.name} upright indicates movement around ${keywords.slice(0, 2).join(' and ')} - pay attention to these themes.`;
  } else {
    // General intention
    context = reversed
      ? `In relation to "${intention}" - ${cardData.name} reversed highlights challenges or inversions in ${keywords.slice(0, 2).join(' and ')}, suggesting areas needing healing or course correction.`
      : `In relation to "${intention}" - ${cardData.name} upright brings ${keywords.slice(0, 2).join(' and ')} energy directly to bear on your situation.`;
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

function generateActionSteps(cardData, reversed, readingType) {
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

  // Pick 3 random actions from the pool
  const shuffled = [...pool].sort(() => Math.random() - 0.5);
  return shuffled.slice(0, 3);
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
