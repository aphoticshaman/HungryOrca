/**
 * BALANCED WISDOM MODULE
 *
 * Integrates universal principles of moderation and the Middle Way
 * into tarot synthesis WITHOUT explicit religious language.
 *
 * Based on timeless wisdom:
 * - "All things in moderation" (Greek philosophy)
 * - The Middle Way between extremes (universal principle)
 * - 8 pillars of balanced living (disguised 8-fold path)
 *
 * Purpose: Guide users toward balance, right action, and moderation
 * in a way that resonates with "most reasonable people" regardless
 * of spiritual/religious background.
 */

/**
 * 8 PILLARS OF BALANCED LIVING
 * (Disguised 8-fold path - RIGHT view, intention, speech, action, livelihood, effort, mindfulness, concentration)
 *
 * 1. CLEAR SEEING (Right View)
 *    - See situations as they are, not as you wish/fear them to be
 *    - Recognize patterns without judgment
 *    - Understand cause and effect in your life
 *
 * 2. ALIGNED INTENTION (Right Intention)
 *    - Set intentions from wisdom, not reactivity
 *    - Choose growth over comfort
 *    - Commit to harm reduction (to self and others)
 *
 * 3. CONSCIOUS COMMUNICATION (Right Speech)
 *    - Speak truth without cruelty
 *    - Listen more than you defend
 *    - Use words to heal, not harm
 *
 * 4. INTEGRATED ACTION (Right Action)
 *    - Act in alignment with your values
 *    - Choose courage over convenience
 *    - Make amends when you miss the mark
 *
 * 5. PURPOSEFUL WORK (Right Livelihood)
 *    - Earn your living in ways that honor your integrity
 *    - Contribute value, not just extract it
 *    - Align career with conscience
 *
 * 6. SUSTAINABLE EFFORT (Right Effort)
 *    - Practice consistency over intensity
 *    - Rest is productive
 *    - Avoid both burnout and stagnation
 *
 * 7. PRESENT AWARENESS (Right Mindfulness)
 *    - Notice without narrating
 *    - Feel your feelings without becoming them
 *    - Stay grounded in this moment
 *
 * 8. FOCUSED ATTENTION (Right Concentration)
 *    - Commit to what matters
 *    - Let distractions dissolve
 *    - Channel energy with precision
 */

export const BALANCED_PILLARS = {
  CLEAR_SEEING: {
    name: 'Clear Seeing',
    essence: 'perceive reality without distortion',
    phrases: [
      'See the situation as it is, not as you wish or fear it to be.',
      'Truth reveals itself when you stop arguing with reality.',
      'Understanding patterns is the first step to changing them.',
      'Clarity comes when you release the need to be right.',
      'What you resist persists. What you accept transforms.',
    ],
  },
  ALIGNED_INTENTION: {
    name: 'Aligned Intention',
    essence: 'set intentions from wisdom, not reaction',
    phrases: [
      'Set your intention from wisdom, not from wounds.',
      'Choose growth over comfort, courage over convenience.',
      'Your intention determines your direction.',
      'Act from vision, not from fear.',
      'Commit to creating, not just consuming.',
    ],
  },
  CONSCIOUS_COMMUNICATION: {
    name: 'Conscious Communication',
    essence: 'speak truth without cruelty',
    phrases: [
      'Speak truth, but let kindness guide your tongue.',
      'Listen more than you defend.',
      'Your words can heal or harm—choose with care.',
      'Silence can be wisdom when speech would wound.',
      'Say what you mean. Mean what you say. Don\'t say it mean.',
    ],
  },
  INTEGRATED_ACTION: {
    name: 'Integrated Action',
    essence: 'act in alignment with values',
    phrases: [
      'Let your actions reflect your values, not your anxieties.',
      'Integrity means doing what\'s right even when no one\'s watching.',
      'Small, aligned actions compound into transformation.',
      'You are what you do, not what you say you\'ll do.',
      'Make amends when you miss the mark. Then move forward.',
    ],
  },
  PURPOSEFUL_WORK: {
    name: 'Purposeful Work',
    essence: 'align livelihood with integrity',
    phrases: [
      'Earn your living in ways that honor your integrity.',
      'Your work should feed your soul, not just your bills.',
      'Contribute value, not just extract it.',
      'Career is a vehicle for purpose, not a substitute for it.',
      'Do work that would make your younger self proud.',
    ],
  },
  SUSTAINABLE_EFFORT: {
    name: 'Sustainable Effort',
    essence: 'practice consistency over intensity',
    phrases: [
      'Consistency beats intensity every time.',
      'Rest is productive. Burnout is expensive.',
      'The middle path between overdoing and underdoing.',
      'Small daily disciplines create lasting change.',
      'Marathon pace, not sprint pace.',
    ],
  },
  PRESENT_AWARENESS: {
    name: 'Present Awareness',
    essence: 'stay grounded in this moment',
    phrases: [
      'This moment is the only one you can act in.',
      'Notice without narrating. Feel without becoming.',
      'Presence is the antidote to anxiety.',
      'Be here now. Fully.',
      'Your power lives in the present, not the past or future.',
    ],
  },
  FOCUSED_ATTENTION: {
    name: 'Focused Attention',
    essence: 'commit to what matters',
    phrases: [
      'Where attention goes, energy flows.',
      'Focus is a superpower in a distracted world.',
      'Commit to what matters. Let the rest dissolve.',
      'Deep work creates lasting value.',
      'Single-tasking is the new multitasking.',
    ],
  },
};

/**
 * MODERATION WISDOM
 * "All things in moderation" expressed in various forms
 */
export const MODERATION_WISDOM = [
  'All things in moderation—even moderation itself.',
  'The middle way between extremes is where wisdom lives.',
  'Balance is not standing still. It\'s the dynamic dance between opposites.',
  'Too much of anything—even a good thing—becomes its opposite.',
  'Avoid the extremes. Walk the middle path.',
  'Neither excess nor deprivation. The sweet spot is in between.',
  'Moderation is not mediocrity. It\'s mastery.',
  'The wise person knows when enough is enough.',
  'Extremism in any direction creates suffering.',
  'Find the balance point. That\'s where transformation happens.',
];

/**
 * Get a quantum-selected balanced wisdom phrase for a specific pillar
 */
export function getBalancedWisdom(pillarName, quantumSeed) {
  const pillar = BALANCED_PILLARS[pillarName];
  if (!pillar) return null;

  const index = Math.floor(quantumSeed * pillar.phrases.length);
  return {
    pillar: pillar.name,
    essence: pillar.essence,
    wisdom: pillar.phrases[index],
  };
}

/**
 * Get a moderation principle (quantum-selected)
 */
export function getModerationWisdom(quantumSeed) {
  const index = Math.floor(quantumSeed * MODERATION_WISDOM.length);
  return MODERATION_WISDOM[index];
}

/**
 * Get pillar guidance based on card meaning and user context
 *
 * Maps card themes to relevant pillars:
 * - Swords (Air/Mind) → Clear Seeing, Conscious Communication, Focused Attention
 * - Cups (Water/Emotion) → Aligned Intention, Present Awareness
 * - Wands (Fire/Action) → Integrated Action, Sustainable Effort
 * - Pentacles (Earth/Material) → Purposeful Work, Sustainable Effort
 * - Major Arcana → Context-dependent
 */
export function getPillarGuidance(cardIndex, cardMeaning, quantumSeed) {
  let relevantPillars = [];

  // Swords (50-63): Mental clarity, communication
  if (cardIndex >= 50 && cardIndex <= 63) {
    relevantPillars = ['CLEAR_SEEING', 'CONSCIOUS_COMMUNICATION', 'FOCUSED_ATTENTION'];
  }
  // Cups (36-49): Emotional intelligence, intention
  else if (cardIndex >= 36 && cardIndex <= 49) {
    relevantPillars = ['ALIGNED_INTENTION', 'PRESENT_AWARENESS', 'CONSCIOUS_COMMUNICATION'];
  }
  // Wands (22-35): Action, energy, effort
  else if (cardIndex >= 22 && cardIndex <= 35) {
    relevantPillars = ['INTEGRATED_ACTION', 'SUSTAINABLE_EFFORT', 'ALIGNED_INTENTION'];
  }
  // Pentacles (64-77): Work, material, grounding
  else if (cardIndex >= 64 && cardIndex <= 77) {
    relevantPillars = ['PURPOSEFUL_WORK', 'SUSTAINABLE_EFFORT', 'INTEGRATED_ACTION'];
  }
  // Major Arcana (0-21): Assign based on card themes
  else {
    relevantPillars = selectMajorArcanaPillars(cardIndex);
  }

  // Quantum-select one pillar from relevant options
  const pillarIndex = Math.floor(quantumSeed * relevantPillars.length);
  const selectedPillar = relevantPillars[pillarIndex];

  return getBalancedWisdom(selectedPillar, quantumSeed);
}

/**
 * Select relevant pillars for Major Arcana cards
 */
function selectMajorArcanaPillars(cardIndex) {
  const majorArcanaPillars = {
    0: ['ALIGNED_INTENTION', 'CLEAR_SEEING'], // Fool - new beginnings
    1: ['FOCUSED_ATTENTION', 'CONSCIOUS_COMMUNICATION'], // Magician - manifestation
    2: ['PRESENT_AWARENESS', 'CLEAR_SEEING'], // High Priestess - intuition
    3: ['ALIGNED_INTENTION', 'PURPOSEFUL_WORK'], // Empress - creativity
    4: ['INTEGRATED_ACTION', 'PURPOSEFUL_WORK'], // Emperor - structure
    5: ['CLEAR_SEEING', 'CONSCIOUS_COMMUNICATION'], // Hierophant - tradition
    6: ['ALIGNED_INTENTION', 'INTEGRATED_ACTION'], // Lovers - choice
    7: ['SUSTAINABLE_EFFORT', 'FOCUSED_ATTENTION'], // Chariot - willpower
    8: ['SUSTAINABLE_EFFORT', 'INTEGRATED_ACTION'], // Strength - inner power
    9: ['CLEAR_SEEING', 'PRESENT_AWARENESS'], // Hermit - introspection
    10: ['CLEAR_SEEING', 'ALIGNED_INTENTION'], // Wheel - cycles
    11: ['INTEGRATED_ACTION', 'CLEAR_SEEING'], // Justice - balance
    12: ['PRESENT_AWARENESS', 'SUSTAINABLE_EFFORT'], // Hanged Man - surrender
    13: ['ALIGNED_INTENTION', 'INTEGRATED_ACTION'], // Death - transformation
    14: ['SUSTAINABLE_EFFORT', 'CLEAR_SEEING'], // Temperance - moderation
    15: ['CLEAR_SEEING', 'INTEGRATED_ACTION'], // Devil - bondage
    16: ['ALIGNED_INTENTION', 'INTEGRATED_ACTION'], // Tower - upheaval
    17: ['PRESENT_AWARENESS', 'ALIGNED_INTENTION'], // Star - hope
    18: ['CLEAR_SEEING', 'PRESENT_AWARENESS'], // Moon - illusion
    19: ['FOCUSED_ATTENTION', 'INTEGRATED_ACTION'], // Sun - vitality
    20: ['CLEAR_SEEING', 'INTEGRATED_ACTION'], // Judgment - reckoning
    21: ['INTEGRATED_ACTION', 'CLEAR_SEEING'], // World - completion
  };

  return majorArcanaPillars[cardIndex] || ['CLEAR_SEEING', 'ALIGNED_INTENTION'];
}

/**
 * Generate closing wisdom that incorporates moderation + relevant pillar
 *
 * This is used at the end of synthesis to tie everything together
 * with balanced, non-extreme guidance.
 */
export function generateBalancedClosing(dominantElement, actionReadiness, quantumSeed1, quantumSeed2) {
  const moderationWisdom = getModerationWisdom(quantumSeed1);

  // Select pillar based on action readiness
  let closingPillar;
  if (actionReadiness === 'high') {
    closingPillar = ['INTEGRATED_ACTION', 'SUSTAINABLE_EFFORT'][Math.floor(quantumSeed2 * 2)];
  } else if (actionReadiness === 'low') {
    closingPillar = ['CLEAR_SEEING', 'PRESENT_AWARENESS'][Math.floor(quantumSeed2 * 2)];
  } else {
    closingPillar = ['ALIGNED_INTENTION', 'CONSCIOUS_COMMUNICATION'][Math.floor(quantumSeed2 * 2)];
  }

  const pillarWisdom = getBalancedWisdom(closingPillar, quantumSeed2);

  return {
    moderation: moderationWisdom,
    pillar: pillarWisdom,
  };
}

/**
 * Get all 8 pillars as a summary (for educational purposes or deep dives)
 */
export function getAllPillars() {
  return Object.entries(BALANCED_PILLARS).map(([key, pillar]) => ({
    key,
    name: pillar.name,
    essence: pillar.essence,
  }));
}

/**
 * Integration helpers for synthesis engine
 */
export const BalancedWisdomIntegration = {
  // For card-by-card interpretations
  getCardPillar: (cardIndex, meaning, seed) => getPillarGuidance(cardIndex, meaning, seed),

  // For action steps section
  getActionGuidance: (actionReadiness, seed) => {
    if (actionReadiness === 'high') {
      return getBalancedWisdom('INTEGRATED_ACTION', seed);
    } else if (actionReadiness === 'medium') {
      return getBalancedWisdom('ALIGNED_INTENTION', seed);
    } else {
      return getBalancedWisdom('CLEAR_SEEING', seed);
    }
  },

  // For synthesis closing
  getClosing: (element, readiness, seed1, seed2) =>
    generateBalancedClosing(element, readiness, seed1, seed2),

  // For moderation reminders throughout
  getModeration: (seed) => getModerationWisdom(seed),
};

export default BalancedWisdomIntegration;
