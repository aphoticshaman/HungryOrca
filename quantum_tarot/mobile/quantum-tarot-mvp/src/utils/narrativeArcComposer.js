/**
 * NARRATIVE ARC COMPOSER
 *
 * Professional tarot readings tell a STORY, not a list of card interpretations.
 * They have:
 * - Setup (where you've been)
 * - Rising tension (where you are - the conflict)
 * - Climax (the turning point)
 * - Resolution (where you're going)
 * - Denouement (the wisdom to carry forward)
 *
 * This transforms readings from informational to EMOTIONAL and MEMORABLE.
 * Uses storytelling beats from:
 * - Joseph Campbell's Hero's Journey
 * - Three-act structure (screenwriting)
 * - Freytag's Pyramid (dramatic structure)
 */

/**
 * Identify which Hero's Journey stage each card represents
 */
export function mapCardsToHeroJourney(cards) {
  const journey = [];

  // Safety check
  if (!cards || !Array.isArray(cards)) {
    return journey;
  }

  cards.forEach((card, index) => {
    const position = card.position || '';
    const cardData = card; // Simplified for now

    // Map based on position in spread
    if (index === 0) {
      journey.push({
        card,
        stage: 'ORDINARY_WORLD',
        beat: 'Setup',
        description: 'Where you\'ve been. The familiar territory you\'re leaving or have left.'
      });
    } else if (index === 1) {
      journey.push({
        card,
        stage: 'CROSSING_THRESHOLD',
        beat: 'Conflict',
        description: 'The challenge, obstacle, or transformation you\'re in the midst of.'
      });
    } else if (index === 2) {
      journey.push({
        card,
        stage: 'ROAD_BACK',
        beat: 'Resolution',
        description: 'Where you\'re heading. The potential future if you integrate the lessons.'
      });
    } else if (index === 3) {
      journey.push({
        card,
        stage: 'RETURN_WITH_ELIXIR',
        beat: 'Integration',
        description: 'The wisdom or gift you bring back from this journey.'
      });
    }
  });

  return journey;
}

/**
 * Generate three-act structure framing
 */
export function generateThreeActFrame(cards, quantumSeed) {
  // Safety check
  if (!cards || !Array.isArray(cards) || cards.length === 0) {
    return null;
  }

  const acts = {
    act1: {
      title: 'ACT I: THE SETUP',
      cards: cards.slice(0, Math.ceil(cards.length / 3)),
      description: 'This is your starting point—the world as it was, the patterns you\'ve been living, the roles you\'ve been playing.'
    },
    act2: {
      title: 'ACT II: THE CRUCIBLE',
      cards: cards.slice(Math.ceil(cards.length / 3), Math.ceil(2 * cards.length / 3)),
      description: 'This is where it gets real. The conflict, the test, the dark night of the soul. You can\'t go back, but forward isn\'t clear yet.'
    },
    act3: {
      title: 'ACT III: THE TRANSFORMATION',
      cards: cards.slice(Math.ceil(2 * cards.length / 3)),
      description: 'This is the breakthrough, the integration, the new normal. You\'re not who you were when this started.'
    }
  };

  return acts;
}

/**
 * Generate rising tension narrative beats
 */
export function generateTensionBeats(cards, mcqAnalysis) {
  const beats = [];

  // Safety check
  if (!cards || !Array.isArray(cards)) {
    return beats;
  }

  // Identify conflict indicators
  const hasConflictCard = cards.some(c => {
    // Would check card database for conflict cards
    // For now, simplified
    return c.reversed || (c.cardIndex >= 30 && c.cardIndex <= 35);
  });

  if (hasConflictCard) {
    beats.push({
      type: 'CONFLICT',
      intensity: 'high',
      text: 'Here\'s where the tension spikes. This isn\'t comfortable. It\'s not meant to be.'
    });
  }

  // Check MCQ analysis for emotional peaks
  if (mcqAnalysis?.overallResonance < 2.5) {
    beats.push({
      type: 'RESISTANCE',
      intensity: 'medium',
      text: 'I notice you pulled back from these cards emotionally. That resistance IS the story. What are you protecting yourself from feeling?'
    });
  }

  if (mcqAnalysis?.actionReadiness === 'ready') {
    beats.push({
      type: 'MOMENTUM',
      intensity: 'high',
      text: 'You\'re ready to move. That readiness is the climax—the point where decision becomes action.'
    });
  }

  return beats;
}

/**
 * Generate story transitions between cards
 */
export function generateStoryTransitions(fromCard, toCard, stageChange, quantumSeed) {
  const transitions = {
    ORDINARY_TO_THRESHOLD: [
      'And then the call came...',
      'But life had other plans.',
      'Enter the catalyst:',
      'Just when you thought you knew the terrain, the ground shifted:',
      'The comfortable became uncomfortable:'
    ],
    THRESHOLD_TO_TRIALS: [
      'Once you crossed that threshold, there was no going back.',
      'The real test emerged:',
      'And here\'s where it gets messy:',
      'The universe said, "You think THAT was hard? Watch this:"',
      'Deeper into the unknown:'
    ],
    TRIALS_TO_REVELATION: [
      'But in the darkness, a light:',
      'The breakthrough came through breakdown:',
      'And finally, FINALLY, you saw it:',
      'The lesson crystallized:',
      'From the ashes:'
    ],
    REVELATION_TO_INTEGRATION: [
      'Now the work is bringing this wisdom home:',
      'The return journey begins:',
      'You can\'t unknow what you know now:',
      'The question becomes: how do you live this truth?',
      'Integration looks like:'
    ]
  };

  const key = `${stageChange.from}_TO_${stageChange.to}`;
  const options = transitions[key] || [
    'And then,',
    'Meanwhile,',
    'Next,'
  ];

  const idx = Math.floor(quantumSeed * options.length) % options.length;
  return options[idx] || options[0];
}

/**
 * Generate climax statement (emotional peak of reading)
 */
export function generateClimaxStatement(cards, mcqAnalysis, readingType, quantumSeed) {
  // Identify the "crisis" card or midpoint card
  const climaxIndex = Math.floor(cards.length / 2);
  const climaxCard = cards[climaxIndex];

  const templates = [
    'This is your crossroads. Not the gentle fork in the road—the hard choice between who you\'ve been and who you\'re becoming.',
    'This moment? This is what Joseph Campbell called "the belly of the whale." You\'re being digested by change. It feels like death. It\'s birth.',
    'Here\'s the pivot point. Everything before led here. Everything after flows from your choice NOW.',
    'This is the dark night of the soul in card form. The only way out is through. And you WILL get through.',
    'The crisis point: where the old self dies and the new self hasn\'t fully formed yet. The liminal space. The chrysalis.',
    'This is your Gethsemane moment. "Let this cup pass from me" meets "nevertheless, not my will but thine." The surrender point.',
    'The hero always faces this: the moment when going back is impossible but going forward seems insurmountable. You\'re here.'
  ];

  const idx = Math.floor(quantumSeed * templates.length) % templates.length;
  return `\n\n## THE TURNING POINT\n\n${templates[idx] || templates[0]}\n\n`;
}

/**
 * Generate denouement (the wisdom to carry forward)
 */
export function generateDenouement(cards, chineseZodiac, mbtiType, quantumSeed) {
  const wisdomTemplates = [
    'The story you came in with isn\'t the story you\'re leaving with. That shift—that\'s growth.',
    'Every hero\'s journey ends with a return. You\'re going back to your ordinary world, but you\'re not ordinary anymore.',
    'The cards showed you the map. Walking the territory is yours to do. But now you know the landmarks.',
    'This reading is complete, but your story is still being written. These cards are chapter markers, not the whole book.',
    'You asked the cards a question. They gave you a story instead. Because the answer isn\'t information—it\'s transformation.'
  ];

  const idx = Math.floor(quantumSeed * wisdomTemplates.length) % wisdomTemplates.length;
  return `\n\n## THE STORY'S MORAL\n\n${wisdomTemplates[idx] || wisdomTemplates[0]}\n\n`;
}

/**
 * Analyze dramatic structure of the reading
 */
export function analyzeDramaticStructure(cards) {
  const structure = {
    hasExposition: cards.length > 0,
    hasRisingAction: cards.length >= 2,
    hasClimax: cards.length >= 3,
    hasFallingAction: cards.length >= 4,
    hasResolution: cards.length >= 3,

    exposition: cards[0] || null,
    conflict: cards.length >= 2 ? cards[1] : null,
    climax: cards.length >= 3 ? cards[Math.floor(cards.length / 2)] : null,
    resolution: cards.length >= 3 ? cards[cards.length - 1] : null
  };

  return structure;
}

/**
 * Generate chapter headings for cards
 */
export function generateChapterHeadings(cards, quantumSeed) {
  // Safety check
  if (!cards || !Array.isArray(cards) || cards.length === 0) {
    return [];
  }

  const headingStyles = [
    // Poetic
    ['The World You Knew', 'The Call You Heard', 'The Path You Walk'],
    // Direct
    ['Where You\'ve Been', 'Where You Are', 'Where You\'re Going'],
    // Mythic
    ['The Ordinary World', 'The Threshold Crossing', 'The Return'],
    // Psychological
    ['The Pattern', 'The Break', 'The Integration'],
    // Mystical
    ['The Seed', 'The Death', 'The Rebirth'],
    // Literary
    ['Act I: Setup', 'Act II: Complication', 'Act III: Resolution']
  ];

  const styleIdx = Math.floor(quantumSeed * headingStyles.length) % headingStyles.length;
  const headings = headingStyles[styleIdx] || headingStyles[0]; // Fallback to first style

  return cards.map((card, idx) => {
    if (headings && idx < headings.length) {
      return headings[idx];
    }
    return `Chapter ${idx + 1}`;
  });
}

/**
 * Master function: Compose narrative arc structure
 */
export function composeNarrativeArc(cards, mcqAnalysis, readingType, userProfile, quantumSeed) {
  // Safety check
  if (!cards || !Array.isArray(cards) || cards.length === 0) {
    return {
      structure: null,
      journey: [],
      acts: null,
      tensionBeats: [],
      climax: null,
      denouement: '',
      chapterHeadings: []
    };
  }

  // Ensure userProfile has defaults
  const safeUserProfile = {
    chineseZodiac: userProfile?.chineseZodiac || 'Dragon',
    mbtiType: userProfile?.mbtiType || 'INFP',
    ...userProfile
  };

  const arc = {
    structure: analyzeDramaticStructure(cards),
    journey: mapCardsToHeroJourney(cards),
    acts: cards.length >= 3 ? generateThreeActFrame(cards, quantumSeed) : null,
    tensionBeats: generateTensionBeats(cards, mcqAnalysis),
    climax: cards.length >= 3 ? generateClimaxStatement(cards, mcqAnalysis, readingType, quantumSeed) : null,
    denouement: generateDenouement(cards, safeUserProfile.chineseZodiac, safeUserProfile.mbtiType, quantumSeed * 0.999),
    chapterHeadings: generateChapterHeadings(cards, quantumSeed * 0.777)
  };

  return arc;
}

/**
 * Generate story-based synthesis opener
 */
export function generateStoryOpener(userName, readingType, quantumSeed) {
  const openers = [
    `${userName}, every tarot reading is a story. This one? It's a transformation tale.`,
    `${userName}, let me tell you the story the cards laid out for you about ${readingType}.`,
    `${userName}, what follows isn't just card interpretations—it's your hero's journey in three acts.`,
    `${userName}, the cards spoke to me in story form. Listen for the arc.`,
    `${userName}, this reading has the structure of a myth. You're the protagonist.`,
    `${userName}, buckle up. This reading has a beginning, middle, and end—and you're in the middle.`
  ];

  const idx = Math.floor(quantumSeed * openers.length) % openers.length;
  return openers[idx] || openers[0];
}

/**
 * Helper: Get narrative pacing (fast, medium, slow)
 */
export function getNarrativePacing(cards, mcqAnalysis) {
  // Fast pacing: crisis cards, low resonance (user wants quick answers)
  // Slow pacing: spiritual cards, high resonance (user is integrating)

  const crisisCards = cards.filter(c => c.reversed).length;
  const avgResonance = mcqAnalysis?.overallResonance || 3;

  if (crisisCards >= 2 || avgResonance < 2.5) {
    return 'fast'; // Short, punchy sentences
  } else if (avgResonance >= 4) {
    return 'slow'; // Longer, more contemplative
  }

  return 'medium';
}
