/**
 * DIVERSE WISDOM ENGINE
 *
 * Goes beyond CBT/DBT to include:
 * - Love Languages (Chapman)
 * - MBTI/Myers-Briggs personality integration
 * - Attachment Theory (Bowlby, Ainsworth)
 * - Polyvagal Theory (nervous system wisdom)
 * - Internal Family Systems (parts work)
 * - Positive Psychology (strengths-based)
 * - Existential Psychology (meaning-making)
 * - Transpersonal Psychology (spiritual development)
 * - Somatic Experiencing (body-based wisdom)
 * - Cartomancy traditions (tarot history)
 * - Elemental magic and correspondences
 * - Relationship toxicity vs. growth indicators
 */

/**
 * LOVE LANGUAGES FRAMEWORK
 * Maps tarot elements to love languages and relationship needs
 */
export const LOVE_LANGUAGE_INSIGHTS = {
  fire: {
    primary: 'Physical Touch & Quality Time',
    shadow: 'Can mistake intensity for intimacy - passion without depth burns out',
    wisdom: 'Your fire needs tending, not just ignition. Schedule regular connection rituals.',
    compatibility: 'Fire + Water = steam (transformative but volatile). Fire + Air = wildfire (exciting but destructive without boundaries). Fire + Earth = forge (create something lasting).',
    toxic_pattern: 'Love-bombing followed by withdrawal. Real passion is sustainable.',
    growth: 'Learn to be present, not just intense. Slow burn > flash fire.'
  },
  water: {
    primary: 'Words of Affirmation & Quality Time',
    shadow: 'Emotional fusion - losing yourself in others. Empathy without boundaries = codependency.',
    wisdom: 'You feel everything, but feeling isn\'t the same as knowing. Check the facts.',
    compatibility: 'Water + Earth = growth (stable, nourishing). Water + Air = rain (necessary but can be distant). Water + Fire = transformation (powerful but requires work).',
    toxic_pattern: 'Emotional caretaking, rescuing, losing self in partner\'s emotions.',
    growth: 'Practice differentiation - feeling WITH someone, not FOR them.'
  },
  air: {
    primary: 'Words of Affirmation & Acts of Service',
    shadow: 'Intellectualizing emotions - living in your head, avoiding embodied vulnerability.',
    wisdom: 'Communication is your superpower, but connection requires more than words.',
    compatibility: 'Air + Fire = inspiration (exciting, lots of ideas). Air + Water = confusion (logic vs. feelings). Air + Earth = blueprint (vision + execution).',
    toxic_pattern: 'Emotional unavailability masked as "being logical". Gaslighting.',
    growth: 'Drop into your body. Name the emotion without explaining it away.'
  },
  earth: {
    primary: 'Acts of Service & Gifts',
    shadow: 'Mistaking stability for stagnation. Provider role without emotional presence.',
    wisdom: 'You show love through doing, but people need your presence, not just your productivity.',
    compatibility: 'Earth + Water = fertile (supportive, growth-oriented). Earth + Air = structure (grounded vision). Earth + Fire = heat (Earth provides container for Fire\'s energy).',
    toxic_pattern: 'Materialism, control through resources, workaholic avoidance.',
    growth: 'Practice being, not just doing. Vulnerability is strength, not weakness.'
  },
  spirit: {
    primary: 'All Languages (Universal)',
    shadow: 'Spiritual bypassing - using "higher consciousness" to avoid real human needs.',
    wisdom: 'Transcendence AND embodiment. Enlightenment doesn\'t exempt you from laundry.',
    compatibility: 'Spirit integrates all elements - you need someone willing to do the work.',
    toxic_pattern: 'Spiritual narcissism, bypassing accountability with "divine timing".',
    growth: 'Ground your spirituality in daily action. Mysticism + responsibility.'
  }
};

/**
 * MBTI RELATIONSHIP DYNAMICS
 * How personality types show up in tarot readings
 */
export const MBTI_TAROT_MAP = {
  thinking_types: ['INTJ', 'ENTJ', 'INTP', 'ENTP', 'ISTJ', 'ESTJ', 'ISTP', 'ESTP'],
  feeling_types: ['INFJ', 'ENFJ', 'INFP', 'ENFP', 'ISFJ', 'ESFJ', 'ISFP', 'ESFP'],

  air_affinity: ['INTJ', 'INTP', 'ENTJ', 'ENTP'], // Intuitive thinkers
  water_affinity: ['INFJ', 'INFP', 'ENFJ', 'ENFP'], // Intuitive feelers
  earth_affinity: ['ISTJ', 'ISFJ', 'ESTJ', 'ESFJ'], // Sensing judgers
  fire_affinity: ['ISTP', 'ISFP', 'ESTP', 'ESFP'], // Sensing perceivers
};

export function getMBTIInsight(element, reversed) {
  const insights = {
    air: {
      upright: 'If you\'re a thinking type (INTJ, ENTP, etc.), this card validates your analytical approach. If you\'re a feeler, it\'s asking you to engage your logic alongside your intuition. Both/and, not either/or.',
      reversed: 'Overthinking becomes paralysis. Even INTPs need to take action without perfect information. Analysis without execution is masturbation.'
    },
    water: {
      upright: 'Feelers (INFJ, ENFP, etc.) - trust your emotional intelligence here. Thinkers - your logic is necessary, but so is your heart. This situation requires emotional courage.',
      reversed: 'Emotional overwhelm. Even the deepest feelers need boundaries. Your empathy is a gift, not a doormat. Protect your energy.'
    },
    earth: {
      upright: 'Sensing types (ISTJ, ESFJ, etc.) excel here - your practical, grounded approach is the medicine. Intuitives - get out of the clouds and take concrete action.',
      reversed: 'Stagnation. Even ISTJs need to adapt. The plan isn\'t working - time to iterate. Stubbornness ≠ strength.'
    },
    fire: {
      upright: 'Perceiving types (ESTP, ISFP, etc.) thrive here - spontaneity, presence, action. Judgers - loosen your grip on control. Trust the flow.',
      reversed: 'Impulsivity without intention. Even ESFPs need some structure. Freedom requires discipline. Chaos isn\'t the same as liberation.'
    },
    spirit: {
      upright: 'All types have access to transcendent wisdom. Your personality is your vehicle, not your prison. Integrate all functions.',
      reversed: 'Using your type as an excuse. "I\'m just an INTJ" - cool, now grow beyond it. Personality is descriptive, not prescriptive.'
    }
  };
  return insights[element]?.[reversed ? 'reversed' : 'upright'] || insights.spirit.upright;
}

/**
 * ATTACHMENT THEORY INSIGHTS
 * How attachment patterns show up in readings
 */
export const ATTACHMENT_WISDOM = {
  secure: {
    traits: 'Comfortable with intimacy and independence. Can self-soothe and co-regulate.',
    goal: 'Most people aren\'t naturally secure - it\'s earned through healing. This is the work.',
    recognition: 'Secure people don\'t play games. They communicate directly, set boundaries, and stay consistent.'
  },
  anxious: {
    traits: 'Fear of abandonment, need for reassurance, protest behavior when activated.',
    shadow: 'Texting 47 times isn\'t love - it\'s anxiety. Your partner isn\'t your emotional regulator.',
    healing: 'Practice self-soothing. Anxiety lies. Your worth isn\'t determined by their response time.',
    reading_flag: 'If you\'re asking the same question 5 times in different ways - that\'s anxiety, not intuition.'
  },
  avoidant: {
    traits: 'Independence > intimacy. Discomfort with vulnerability and emotional expression.',
    shadow: '"I don\'t need anyone" is trauma, not enlightenment. Interdependence is human.',
    healing: 'Vulnerability isn\'t weakness. Your walls keep out connection, not just pain.',
    reading_flag: 'If you\'re pulling cards to justify leaving every time things get real - that\'s avoidance.'
  },
  disorganized: {
    traits: 'Want closeness but fear it. Approach-avoidance dance. "Come here, go away."',
    shadow: 'This is the hardest pattern - you need what hurts you. Professional support helps.',
    healing: 'You can have secure relationships, but it requires healing the original wound.',
    reading_flag: 'Extreme card interpretations (all good or all bad) - life is nuanced.'
  }
};

export function getAttachmentInsight(cardData, intention, reversed) {
  const intentionLower = intention.toLowerCase();

  // Detect anxious attachment patterns
  if (intentionLower.includes('will they') || intentionLower.includes('do they love') ||
      intentionLower.includes('thinking about me') || intentionLower.includes('come back')) {
    return {
      pattern: 'anxious',
      insight: reversed
        ? `This card reversed mirrors anxious attachment activation. Before asking what they'll do, ask: what do I actually want? Your anxiety is valid, but it's not intuition. Practice: Name the fear. Check the facts. Self-soothe. Then decide.`
        : `This card upright says: you're seeking external validation for an internal wound. Real question: What would a securely attached person do here? (Hint: They'd communicate directly or walk away peacefully, not divinate.)`
    };
  }

  // Detect avoidant patterns
  if (intentionLower.includes('should i leave') || intentionLower.includes('is it time to end') ||
      intentionLower.includes('too close') || intentionLower.includes('space')) {
    return {
      pattern: 'avoidant',
      insight: reversed
        ? `Reversed card, reversed pattern - are you running from intimacy or from actual danger? If you leave every time it gets real, you'll never find real. Differentiate between "this is uncomfortable" vs. "this is unsafe."`
        : `This card says: Your impulse to leave might be self-protection or self-sabotage. Only you know. Secure people can tolerate discomfort for growth. Where's the line for you?`
    };
  }

  return null; // No obvious attachment pattern detected
}

/**
 * POLYVAGAL THEORY
 * Nervous system states and tarot
 */
export const NERVOUS_SYSTEM_WISDOM = {
  ventral_vagal: {
    state: 'Safe & Social - Regulated nervous system',
    signs: 'Present, grounded, open to connection, thinking clearly',
    tarot: 'Upright Major Arcana, Cups, Pentacles often indicate this state',
    advice: 'Make decisions from here. This is your wise self.'
  },
  sympathetic: {
    state: 'Fight/Flight - Activated',
    signs: 'Anxious, restless, impulsive, racing thoughts, need to DO something',
    tarot: 'Wands (especially reversed), Swords energy',
    advice: 'Don\'t make big decisions in this state. Regulate first: breathe, move, ground.'
  },
  dorsal_vagal: {
    state: 'Freeze/Shutdown - Collapsed',
    signs: 'Numb, dissociated, hopeless, "nothing matters"',
    tarot: 'Heavy reversed cards, especially in Cups/Water',
    advice: 'This isn\'t truth, it\'s trauma. Small actions: sunlight, cold water, reach out to someone.'
  }
};

export function getNervousSystemCheck(element, reversed) {
  if (element === 'fire' && reversed) {
    return 'Nervous system check: Are you in fight/flight (sympathetic activation)? This card suggests your body is screaming "DANGER" even if logically you\'re safe. Regulate before deciding: Box breathing (4-4-4-4), cold water on face, or vigorous movement.';
  }
  if (element === 'water' && reversed) {
    return 'Nervous system check: Are you in shutdown (dorsal vagal)? This feels like depression but it\'s your nervous system protecting you through collapse. Gentle activation helps: Sunlight, humming (stimulates vagus nerve), tiny achievable tasks.';
  }
  if (element === 'earth' && !reversed) {
    return 'Your nervous system is regulated (ventral vagal) - you\'re in "safe & social" mode. THIS is when to make important decisions. You\'re thinking clearly, not reacting from survival mode.';
  }
  return null;
}

/**
 * TOXIC RELATIONSHIP RED FLAGS
 * Real talk about when to walk away
 */
export const TOXICITY_INDICATORS = {
  narcissistic_abuse: [
    'Love-bombing followed by devaluation cycles',
    'Gaslighting ("that never happened", "you\'re too sensitive")',
    'DARVO (Deny, Attack, Reverse Victim & Offender)',
    'Triangulation (pitting you against others)',
    'Lack of accountability ("I\'m sorry you feel that way")',
    'Emotional manipulation through guilt/shame',
    'Isolation from support system',
    'Financial control or exploitation'
  ],
  codependency_patterns: [
    'You feel responsible for their emotions',
    'Losing yourself to "keep the peace"',
    'Enabling destructive behavior',
    'Your self-worth depends on their approval',
    'Rescuing/caretaking compulsion',
    'Difficulty maintaining boundaries',
    'Fear of abandonment drives all decisions'
  ],
  healthy_relationship_markers: [
    'Consistent behavior (not hot/cold cycles)',
    'Direct communication (no mind games)',
    'Mutual accountability ("I was wrong, I\'ll change")',
    'Respect for boundaries (first time, every time)',
    'Separate lives that intertwine (not enmeshment)',
    'Conflict leads to growth (not punishment)',
    'You feel MORE yourself, not less'
  ]
};

export function getToxicityCheck(intention, cardData, reversed) {
  const intentionLower = intention.toLowerCase();

  // Check for narcissistic relationship patterns
  if (intentionLower.match(/(narcissist|gaslighting|crazy|toxic|abuse|manipulat)/i)) {
    return {
      severity: 'high',
      message: reversed
        ? `This card reversed + your question = your gut is screaming. Tarot can't fix abuse. If you're being gaslighted, manipulated, or feel crazy around someone - that's not love. That's toxicity. Resources: loveisrespect.org, National Domestic Violence Hotline 1-800-799-7233. You deserve safety.`
        : `This card upright says: Trust what you're sensing. Toxic people don't change through patience - they change through consequences (or not at all). Your presence isn't healing them, it's enabling them. Leaving isn't failure, it's wisdom.`
    };
  }

  // Check for codependency
  if (intentionLower.match(/(fix|save|help them|change them|if i just|one more chance)/i)) {
    return {
      severity: 'medium',
      message: `Codependency alert: You can't love someone into health. You can't heal someone's trauma through your devotion. You can't pour from an empty cup. This card says: Put on your own oxygen mask. You're not their therapist, savior, or rehabilitation project. You're their partner or you're not.`
    };
  }

  // Check for healthy relationship inquiry
  if (intentionLower.match(/(healthy|boundaries|self-worth|respect)/i)) {
    return {
      severity: 'low',
      message: `Good inquiry. Healthy relationships require: mutual respect, consistent behavior, direct communication, separate identities, and conflict that leads to growth (not punishment). This card shows whether that's present or absent.`
    };
  }

  return null;
}

/**
 * CARTOMANCY TRADITIONS
 * Historical and mystical depth
 */
export const CARTOMANCY_WISDOM = {
  historical_context: {
    tarot_origins: 'Tarot emerged in 15th century Italy as playing cards, evolved into divination tool by 18th century. The imagery carries centuries of Western esoteric tradition.',
    golden_dawn: 'Modern tarot heavily influenced by Hermetic Order of the Golden Dawn (1888) - Waite-Smith deck (1909) encoded Kabbalah, astrology, and elemental magic.',
    jungian_revolution: 'Carl Jung legitimized tarot as psychological tool - symbols as doorways to the collective unconscious. Not fortune-telling, but self-reflection.',
    synchronicity: 'Jung\'s concept: meaningful coincidence. The cards you draw aren\'t random - they\'re synchronistic reflections of your psyche\'s current state.'
  },

  elemental_magic: {
    fire: 'Wands. Primal life force, creative power, masculine/yang energy. Season: Summer. Direction: South. Magical power: Will and passion.',
    water: 'Cups. Emotional truth, intuition, feminine/yin energy. Season: Autumn. Direction: West. Magical power: Love and dreams.',
    air: 'Swords. Mental clarity, truth-seeking, communication. Season: Spring. Direction: East. Magical power: Knowledge and speech.',
    earth: 'Pentacles. Material manifestation, body wisdom, grounding. Season: Winter. Direction: North. Magical power: Abundance and health.',
    spirit: 'Major Arcana. The Fool\'s Journey - archetypal life lessons. Beyond elements. The soul\'s evolution.'
  },

  kabbalah_connection: {
    tree_of_life: 'Each Major Arcana card corresponds to a path on the Kabbalistic Tree of Life - map of consciousness from material (Malkuth) to divine (Kether).',
    sephirot: 'The 10 numbered cards in each suit map to the 10 Sephirot (divine emanations).',
    integration: 'Tarot + Kabbalah = Western mystery tradition\'s psychological-spiritual technology.'
  }
};

export function getMysticalContext(cardData, element) {
  const elementWisdom = CARTOMANCY_WISDOM.elemental_magic[element];

  return `${elementWisdom} // Historical note: ${cardData.arcana === 'major' ? 'Major Arcana cards represent soul-level lessons in the Fool\'s Journey - these are THE BIG ONES. The universe is tapping you on the shoulder.' : `Minor Arcana ${element} cards speak to ${element === 'fire' ? 'will and passion' : element === 'water' ? 'emotional truth' : element === 'air' ? 'mental clarity' : 'material reality'} - everyday magic.`}`;
}

/**
 * ORTHOGONAL INSIGHTS
 * Unexpected connections, lateral thinking, pattern interrupts
 */
export const ORTHOGONAL_WISDOM = [
  'This card is a mirror, not a map. It shows what IS, not what MUST BE.',
  'The opposite of your fear is often just as unlikely as your fear itself.',
  'If this card makes you uncomfortable, that\'s the point. Growth lives in discomfort.',
  'You already know the answer. The cards just gave you permission to trust it.',
  'Sometimes the reading is about asking a better question, not getting a better answer.',
  'The card you hoped for vs. the card you got = the gap between fantasy and reality.',
  'If you\'ve asked this question 5 times, the answer is always the same: take action.',
  'Reversed cards aren\'t bad luck - they\'re your shadow asking for integration.',
  'The Fool isn\'t naive - he\'s so wise he looks foolish. Beginner\'s mind is mastery.',
  'This spread is a snapshot, not a sentence. You have free will. Use it.',
  'The cards that scare you most usually carry the medicine you need most.',
  'Tarot doesn\'t predict the future - it reveals the present with such clarity that the future becomes obvious.',
  'If you don\'t like the reading, change your behavior. That\'s how you change your fate.',
  'The universe doesn\'t owe you easy. It owes you growth. This card is the curriculum.',
  'Stop asking "will it happen?" Start asking "what kind of person do I need to become?"',
  'Your reading reflects your energy right now. Change your energy, change your reading.',
  'The card you pulled for "them" is actually about you. It\'s always about you.',
  'Divination without action is procrastination with crystals.',
  'The best time to draw cards is after you\'ve made the decision, not before.',
  'If you\'re using tarot to avoid responsibility, you\'re doing it wrong.'
];

export function getOrthogonalInsight() {
  return ORTHOGONAL_WISDOM[Math.floor(Math.random() * ORTHOGONAL_WISDOM.length)];
}

/**
 * INTEGRATION FUNCTION
 * Weaves all wisdom traditions together
 */
export function getDiverseInterpretation(cardData, reversed, position, intention, readingType, context = {}) {
  const element = cardData.element || 'spirit';
  const wisdom = [];

  // 1. MBTI Integration
  const mbtiInsight = getMBTIInsight(element, reversed);
  wisdom.push(`[Personality Lens] ${mbtiInsight}`);

  // 2. Love Language (if relationship reading)
  if (readingType === 'romance' || intention.toLowerCase().includes('relation')) {
    const loveInsight = LOVE_LANGUAGE_INSIGHTS[element];
    wisdom.push(`[Love Language] ${loveInsight.wisdom} ${reversed ? loveInsight.shadow : loveInsight.growth}`);

    // Check for toxicity
    const toxCheck = getToxicityCheck(intention, cardData, reversed);
    if (toxCheck) {
      wisdom.push(`[Relationship Health] ${toxCheck.message}`);
    }
  }

  // 3. Attachment Pattern Check
  const attachmentCheck = getAttachmentInsight(cardData, intention, reversed);
  if (attachmentCheck) {
    wisdom.push(`[Attachment Pattern: ${attachmentCheck.pattern}] ${attachmentCheck.insight}`);
  }

  // 4. Nervous System Wisdom
  const nervousSystemCheck = getNervousSystemCheck(element, reversed);
  if (nervousSystemCheck) {
    wisdom.push(`[Somatic Wisdom] ${nervousSystemCheck}`);
  }

  // 5. Mystical/Historical Context
  const mysticalContext = getMysticalContext(cardData, element);
  wisdom.push(`[Cartomancy Tradition] ${mysticalContext}`);

  // 6. Orthogonal Insight (Pattern Interrupt)
  const orthogonal = getOrthogonalInsight();
  wisdom.push(`[Truth Bomb] ${orthogonal}`);

  return wisdom;
}
